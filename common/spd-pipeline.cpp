#include "spd-pipeline.h"

#include "ggml.h"
#include "llama-ext.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <map>
#include <mutex>
#include <numeric>
#include <thread>
#include <utility>

namespace {

constexpr uint32_t SPD_MAX_STAGE_COUNT = COMMON_SPD_MAX_STAGE_COUNT;
constexpr uint32_t SPD_MAX_ROLLBACK_TOKENS = SPD_MAX_STAGE_COUNT - 1;
constexpr uint32_t SPD_MAX_DRAFT_TOP_K = COMMON_SPD_MAX_DRAFT_TOP_K;

using clock_type = std::chrono::steady_clock;

double seconds_since(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}

int32_t argmax(const float * logits, int32_t n_vocab) {
    int32_t result = 0;
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[result]) {
            result = i;
        }
    }
    return result;
}

struct batch_storage {
    llama_batch batch = {};
    std::vector<llama_token> tokens;
    std::vector<float> embeddings;
    std::vector<llama_pos> positions;
    std::vector<int8_t> logits;
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id> seq_ids;
    std::vector<llama_seq_id *> seq_id_ptrs;

    batch_storage() = default;

    batch_storage(
            int32_t n_tokens,
            const llama_token * token_data,
            const float * embedding_data,
            int32_t n_embd,
            int32_t n_pos_per_embd,
            llama_pos first_pos,
            bool output_all,
            const llama_pos * position_data = nullptr,
            const llama_seq_id * sequence_data = nullptr) {
        set(n_tokens, token_data, embedding_data, n_embd, n_pos_per_embd,
                first_pos, output_all, position_data, sequence_data);
    }

    void set(
            int32_t n_tokens,
            const llama_token * token_data,
            const float * embedding_data,
            int32_t n_embd,
            int32_t n_pos_per_embd,
            llama_pos first_pos,
            bool output_all,
            const llama_pos * position_data = nullptr,
            const llama_seq_id * sequence_data = nullptr) {
        batch = {};
        batch.n_tokens = n_tokens;

        tokens.clear();
        embeddings.clear();
        if (token_data != nullptr) {
            tokens.assign(token_data, token_data + n_tokens);
            batch.token = tokens.data();
        }
        if (embedding_data != nullptr) {
            embeddings.assign(embedding_data, embedding_data + (size_t) n_tokens*n_embd);
            batch.embd = embeddings.data();
        }

        positions.resize((size_t) n_tokens*(embedding_data != nullptr ? n_pos_per_embd : 1));
        logits.resize(n_tokens, 0);
        std::fill(logits.begin(), logits.end(), 0);
        for (int32_t p = 0; p < (embedding_data != nullptr ? n_pos_per_embd : 1); ++p) {
            for (int32_t i = 0; i < n_tokens; ++i) {
                positions[(size_t) p*n_tokens + i] = position_data != nullptr ? position_data[i] : first_pos + i;
            }
        }
        for (int32_t i = 0; i < n_tokens; ++i) {
            logits[i] = output_all || i + 1 == n_tokens;
        }
        batch.pos = positions.data();
        batch.logits = logits.data();

        n_seq_id.clear();
        seq_ids.clear();
        seq_id_ptrs.clear();
        if (sequence_data != nullptr) {
            n_seq_id.assign(n_tokens, 1);
            seq_ids.assign(sequence_data, sequence_data + n_tokens);
            seq_id_ptrs.resize(n_tokens);
            for (int32_t i = 0; i < n_tokens; ++i) {
                seq_id_ptrs[i] = &seq_ids[i];
            }
            batch.n_seq_id = n_seq_id.data();
            batch.seq_id = seq_id_ptrs.data();
        }
    }

    batch_storage(
            const std::vector<llama_token> & selectors,
            const std::vector<float> & features,
            int32_t n_embd,
            const std::vector<llama_pos> & pos) {
        set(selectors, features, n_embd, pos);
    }

    void set(
            const std::vector<llama_token> & selectors,
            const std::vector<float> & features,
            int32_t n_embd,
            const std::vector<llama_pos> & pos,
            const std::vector<llama_seq_id> * sequences = nullptr,
            const std::vector<int8_t> * output_flags = nullptr) {
        batch = {};
        batch.n_tokens = (int32_t) selectors.size();
        tokens = selectors;
        embeddings = features;
        positions = pos;
        logits.resize(selectors.size(), 0);
        if (output_flags != nullptr) {
            logits = *output_flags;
        } else {
            std::fill(logits.begin(), logits.end(), 0);
            if (!logits.empty()) {
                logits.back() = 1;
            }
        }
        batch.token = tokens.data();
        batch.embd = embeddings.data();
        batch.pos = positions.data();
        batch.logits = logits.data();

        n_seq_id.clear();
        seq_ids.clear();
        seq_id_ptrs.clear();
        if (sequences != nullptr) {
            GGML_ASSERT(sequences->size() == selectors.size());
            n_seq_id.assign(selectors.size(), 1);
            seq_ids = *sequences;
            seq_id_ptrs.resize(selectors.size());
            for (size_t i = 0; i < selectors.size(); ++i) {
                seq_id_ptrs[i] = &seq_ids[i];
            }
            batch.n_seq_id = n_seq_id.data();
            batch.seq_id = seq_id_ptrs.data();
        }

        GGML_ASSERT(features.size() == selectors.size()*(size_t) n_embd);
        GGML_ASSERT(pos.size() == selectors.size());
        GGML_ASSERT(output_flags == nullptr || output_flags->size() == selectors.size());
    }
};

class fixed_worker {
public:
    using task_fn = void (*)(void *);

    fixed_worker() : worker([this]() { run(); }) {}

    ~fixed_worker() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        ready.notify_one();
        worker.join();
    }

    void submit(task_fn fn, void * data) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            GGML_ASSERT(fn != nullptr);
            GGML_ASSERT(!pending && !running);
            task = fn;
            task_data = data;
            failure = nullptr;
            pending = true;
        }
        ready.notify_one();
    }

    std::exception_ptr wait() {
        std::unique_lock<std::mutex> lock(mutex);
        finished.wait(lock, [this]() { return !pending && !running; });
        return failure;
    }

private:
    void run() {
        while (true) {
            task_fn fn = nullptr;
            void * data = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex);
                ready.wait(lock, [this]() { return pending || stopping; });
                if (!pending && stopping) {
                    return;
                }
                fn = task;
                data = task_data;
                pending = false;
                running = true;
            }

            std::exception_ptr current_failure;
            try {
                fn(data);
            } catch (...) {
                current_failure = std::current_exception();
            }

            {
                std::lock_guard<std::mutex> lock(mutex);
                failure = current_failure;
                running = false;
            }
            finished.notify_one();
        }
    }

    std::mutex mutex;
    std::condition_variable ready;
    std::condition_variable finished;
    std::thread worker;
    task_fn task = nullptr;
    void * task_data = nullptr;
    std::exception_ptr failure;
    bool pending = false;
    bool running = false;
    bool stopping = false;
};

} // namespace

struct common_spd_pipeline::impl {
    struct snapshot {
        std::vector<std::vector<float>> values;
        std::vector<bool> present;

        explicit snapshot(size_t n = 0) : values(n), present(n, false) {}
    };

    struct entry {
        llama_token token = LLAMA_TOKEN_NULL;
        llama_pos pos = -1;
        std::vector<float> hidden;
        snapshot snap;
    };

    struct speculation_input {
        std::vector<llama_token> selectors;
        std::vector<llama_pos> positions;
        std::vector<float> features;
        llama_pos min_pos = -1;
    };

    struct draft_candidate {
        llama_token token = LLAMA_TOKEN_NULL;
        double log_probability = 0.0;
        uint32_t rank = 0;
    };

    struct tree_chain {
        llama_seq_id seq_id = 0;
        std::vector<entry> pipeline;
        std::map<llama_pos, snapshot> completed;
        std::vector<llama_token> tokens;
        std::vector<uint32_t> draft_ranks;
        double score = 0.0;
        snapshot prev_evicted;
        llama_pos prev_evicted_pos = -1;
        bool has_prev_evicted = false;
    };

    struct stage_decode_job {
        impl * owner;
        uint32_t stage;
        entry * item;
        double * elapsed;
        std::string * error;
        uint8_t * ok;
    };

    struct speculation_job {
        impl * owner;
        const speculation_input * input;
        llama_token * sampled;
        bool * ok;
    };

    struct tree_stage_decode_job {
        impl * owner;
        uint32_t stage;
        std::vector<entry *> items;
        std::vector<llama_seq_id> seq_ids;
        double * elapsed;
        std::string * error;
        uint8_t * ok;
    };

    struct tree_speculation_job {
        impl * owner;
        const std::vector<tree_chain> * chains;
        const std::vector<speculation_input> * inputs;
        std::vector<std::vector<draft_candidate>> * candidates;
        bool * ok;
    };

    llama_model * model_target;
    llama_model * model_spd;
    common_spd_params params;

    int32_t n_embd = 0;
    int32_t n_vocab = 0;
    int32_t n_layers = 0;
    int32_t layers_per_stage = 0;
    int32_t target_n_pos_per_embd = 1;
    int32_t sidecar_n_embd_inp = 0;
    uint32_t stage_count = 0;
    uint32_t rollback_tokens = 0;

    std::vector<int32_t> anchors;
    std::array<llama_context *, SPD_MAX_STAGE_COUNT> stages = {};
    std::array<std::array<std::array<llama_pos, SPD_MAX_ROLLBACK_TOKENS>, SPD_MAX_DRAFT_TOP_K>, SPD_MAX_STAGE_COUNT> checkpoint_pos = {};
    std::array<std::array<llama_pos, SPD_MAX_DRAFT_TOP_K>, SPD_MAX_STAGE_COUNT> stage_tail_pos = {};
    std::array<std::vector<size_t>, SPD_MAX_STAGE_COUNT> stage_resources;
    std::vector<size_t> sidecar_resources;
    std::vector<std::unique_ptr<std::mutex>> resource_mutexes;
    std::array<batch_storage, SPD_MAX_STAGE_COUNT> stage_decode_batches;
    std::array<std::vector<float>, SPD_MAX_STAGE_COUNT> stage_decode_outputs;
    batch_storage head_decode_batch;
    batch_storage embed_decode_batch;
    batch_storage sidecar_decode_batch;
    speculation_input decode_speculation_input;
    std::array<std::unique_ptr<fixed_worker>, SPD_MAX_STAGE_COUNT> stage_workers;
    std::unique_ptr<fixed_worker> sidecar_worker;
    llama_context * head = nullptr;
    llama_context * embed = nullptr;
    llama_context * sidecar = nullptr;
    llama_sampler * head_sampler = nullptr;
    llama_sampler * sidecar_sampler = nullptr;
    std::array<llama_sampler *, SPD_MAX_DRAFT_TOP_K> sidecar_tree_samplers = {};
    bool head_backend_sampling = false;
    bool sidecar_backend_sampling = false;

    std::string last_error;
    bool ready = false;
    bool used = false;
    bool static_decode_fast_path = true;

    uint32_t tree_width() const {
        return params.draft_top_k;
    }

    llama_seq_id checkpoint_seq_id(llama_seq_id active_seq, uint32_t slot) const {
        return (llama_seq_id) tree_width() + active_seq*(llama_seq_id) rollback_tokens + (llama_seq_id) slot;
    }

    impl(llama_model * model_target, llama_model * model_spd, const common_spd_params & params)
        : model_target(model_target), model_spd(model_spd), params(params) {
        if (const char * value = std::getenv("LLAMA_SPD_STATIC_DECODE")) {
            static_decode_fast_path = std::atoi(value) != 0;
        }
        initialize();
    }

    ~impl() {
        sidecar_worker.reset();
        for (auto & worker : stage_workers) {
            worker.reset();
        }
        llama_free(sidecar);
        llama_free(embed);
        llama_free(head);
        for (uint32_t stage = 0; stage < stage_count; ++stage) {
            llama_free(stages[stage]);
        }
        llama_sampler_free(sidecar_sampler);
        for (llama_sampler * sampler : sidecar_tree_samplers) {
            llama_sampler_free(sampler);
        }
        llama_sampler_free(head_sampler);
    }

    static void execute_stage_decode_job(void * data) {
        auto & job = *static_cast<stage_decode_job *>(data);
        *job.ok = job.owner->advance_entry(job.stage, *job.item, *job.elapsed, *job.error);
    }

    static void execute_speculation_job(void * data) {
        auto & job = *static_cast<speculation_job *>(data);
        *job.ok = job.owner->run_speculation(*job.input, *job.sampled);
    }

    static void execute_tree_stage_decode_job(void * data) {
        auto & job = *static_cast<tree_stage_decode_job *>(data);
        *job.ok = job.owner->advance_entries(
                job.stage, job.items, job.seq_ids, *job.elapsed, *job.error);
    }

    static void execute_tree_speculation_job(void * data) {
        auto & job = *static_cast<tree_speculation_job *>(data);
        *job.ok = job.owner->run_tree_speculation(*job.chains, *job.inputs, *job.candidates);
    }

    void fail(std::string message) {
        if (last_error.empty()) {
            last_error = std::move(message);
        }
    }

    llama_sampler * make_greedy_sampler() const {
        llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        return sampler;
    }

    void configure_static_context(llama_context * ctx, bool stable_inputs) const {
        if (!static_decode_fast_path || ctx == nullptr) {
            return;
        }
        llama_set_graph_reuse(ctx, true);
        if (stable_inputs) {
            llama_set_stable_host_inputs(ctx, true);
        }
    }

    std::string execution_resource_key(ggml_backend_dev_t dev) const {
        if (dev == nullptr) {
            return "cpu";
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (reg != nullptr) {
            using endpoint_fn = const char * (*)(ggml_backend_dev_t);
            auto endpoint = (endpoint_fn) ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_rpc_device_endpoint");
            if (endpoint != nullptr) {
                return std::string("rpc:") + endpoint(dev);
            }
        }
        return "device:" + std::to_string((uintptr_t) dev);
    }

    llama_context_params make_context_params(uint32_t n_batch, bool is_sidecar = false) const {
        llama_context_params cp = is_sidecar ? params.sidecar_context : params.target_context;
        cp.n_ctx = params.n_ctx;
        cp.n_batch = n_batch;
        cp.n_ubatch = std::min(n_batch, params.n_ubatch);
        cp.n_seq_max = 1;
        cp.n_rs_seq = 0;
        cp.n_outputs_max = 0;
        cp.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
        cp.embeddings = false;
        cp.kv_unified = false;
        cp.samplers = nullptr;
        cp.n_samplers = 0;
        cp.ctx_other = nullptr;
        cp.no_perf = false;
        if (params.n_threads > 0) {
            cp.n_threads = params.n_threads;
        }
        if (params.n_threads_batch > 0) {
            cp.n_threads_batch = params.n_threads_batch;
        }
        return cp;
    }

    void initialize() {
        if (model_target == nullptr || model_spd == nullptr) {
            fail("SPD requires both a target model and a sidecar model");
            return;
        }
        stage_count = llama_model_spd_stage_count(model_spd);
        if (stage_count < 2 || stage_count > SPD_MAX_STAGE_COUNT) {
            fail("SPD sidecar stage count must be between 2 and " +
                    std::to_string(SPD_MAX_STAGE_COUNT));
            return;
        }
        if (params.draft_top_k == 0 || params.draft_top_k > SPD_MAX_DRAFT_TOP_K) {
            fail("SPD draft tree width must be between 1 and " +
                    std::to_string(SPD_MAX_DRAFT_TOP_K));
            return;
        }
        rollback_tokens = stage_count - 1;
        const uint32_t target_sequence_count = params.draft_top_k*stage_count;
        if (params.n_ctx > UINT32_MAX/target_sequence_count) {
            fail("SPD context length is too large for rollback sequence allocation");
            return;
        }

        n_embd = llama_model_n_embd(model_target);
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_target));
        n_layers = llama_model_n_layer(model_target);
        layers_per_stage = n_layers / (int32_t) stage_count;
        target_n_pos_per_embd = llama_model_n_pos_per_embd(model_target);
        sidecar_n_embd_inp = llama_model_n_embd_inp(model_spd);

        if (static_decode_fast_path) {
            decode_speculation_input.selectors.reserve(stage_count + 1);
            decode_speculation_input.positions.reserve(stage_count + 1);
            decode_speculation_input.features.reserve((size_t) (stage_count + 1)*sidecar_n_embd_inp);
        }

        const uint32_t n_anchor = llama_model_target_layer_ids_n(model_spd);
        const int32_t * anchor_data = llama_model_target_layer_ids(model_spd);
        if (n_layers <= 0 || n_layers % (int32_t) stage_count != 0) {
            fail("SPD target layer count must divide evenly into " +
                    std::to_string(stage_count) + " stages");
            return;
        }
        if (n_anchor == 0 || anchor_data == nullptr) {
            fail("SPD sidecar has no target snapshot anchors");
            return;
        }
        anchors.assign(anchor_data, anchor_data + n_anchor);
        if (anchors.front() != 0 || !std::is_sorted(anchors.begin(), anchors.end())) {
            fail("SPD sidecar snapshot anchors are invalid");
            return;
        }
        if (sidecar_n_embd_inp != n_embd*(int32_t) anchors.size()) {
            fail("SPD sidecar input width does not match target embeddings and snapshot anchors");
            return;
        }
        if (llama_vocab_n_tokens(llama_model_get_vocab(model_spd)) != n_vocab) {
            fail("SPD target and sidecar vocabularies have different sizes");
            return;
        }

        // Independent schedulers may overlap across distinct machines, but
        // schedulers targeting the same physical device or RPC endpoint must
        // not interleave protocol streams. Record every execution resource
        // touched by each logical stage and the sidecar, then lock shared
        // resources in a stable order during concurrent advancement.
        std::map<std::string, size_t> resource_ids;
        for (uint32_t stage = 0; stage < stage_count; ++stage) {
            const int32_t layer_begin = (int32_t) stage*layers_per_stage;
            const int32_t layer_end = layer_begin + layers_per_stage;
            for (int32_t layer = layer_begin; layer < layer_end; ++layer) {
                const std::string key = execution_resource_key(
                        llama_model_layer_device(model_target, layer));
                auto [it, inserted] = resource_ids.emplace(key, resource_ids.size());
                if (inserted) {
                    resource_mutexes.push_back(std::make_unique<std::mutex>());
                }
                if (std::find(stage_resources[stage].begin(), stage_resources[stage].end(), it->second) ==
                    stage_resources[stage].end()) {
                    stage_resources[stage].push_back(it->second);
                }
            }
            std::sort(stage_resources[stage].begin(), stage_resources[stage].end());
        }
        for (int32_t layer = 0; layer < llama_model_n_layer(model_spd); ++layer) {
            const std::string key = execution_resource_key(
                    llama_model_layer_device(model_spd, layer));
            auto [it, inserted] = resource_ids.emplace(key, resource_ids.size());
            if (inserted) {
                resource_mutexes.push_back(std::make_unique<std::mutex>());
            }
            if (std::find(sidecar_resources.begin(), sidecar_resources.end(), it->second) ==
                sidecar_resources.end()) {
                sidecar_resources.push_back(it->second);
            }
        }
        std::sort(sidecar_resources.begin(), sidecar_resources.end());

        for (uint32_t stage = 0; stage < stage_count; ++stage) {
            llama_context_params cp = make_context_params(params.n_batch);
            cp.ctx_type = LLAMA_CONTEXT_TYPE_SPD_STAGE;
            cp.spd_stage = stage;
            cp.spd_stage_count = stage_count;
            cp.n_seq_max = target_sequence_count;
            // llama_context derives n_ctx_seq by dividing total n_ctx by
            // n_seq_max. SPD uses the extra sequence IDs as rollback aliases,
            // but seq 0 must still retain the caller-requested context length.
            // Without this expansion an 8192-token SPD context silently became
            // an effective 1024-token context and diverged on longer prompts.
            cp.n_ctx = params.n_ctx*target_sequence_count;
            // Keep seq 0 on the same per-sequence KV layout as an ordinary
            // target context. Rollback snapshots use seq_cp aliases and do not
            // require a unified cache; forcing one changes long-context target
            // numerics enough to flip close greedy decisions.
            cp.kv_unified = false;
            cp.n_rs_seq = 0;
            cp.embeddings = true;
            stages[stage] = llama_init_from_model(model_target, cp);
            if (stages[stage] == nullptr) {
                fail("failed to initialize target SPD stage " + std::to_string(stage));
                return;
            }
            configure_static_context(stages[stage], true);
        }

        for (int32_t anchor : anchors) {
            const uint32_t stage = std::min<uint32_t>(anchor / layers_per_stage, stage_count - 1);
            llama_set_embeddings_layer_inp(stages[stage], anchor, true);
        }

        llama_context_params hp = make_context_params(std::max<uint32_t>(1, params.n_batch));
        hp.ctx_type = LLAMA_CONTEXT_TYPE_SPD_HEAD;
        head = llama_init_from_model(model_target, hp);
        if (head == nullptr) {
            fail("failed to initialize the target SPD head");
            return;
        }
        if (static_decode_fast_path) {
            head_sampler = make_greedy_sampler();
            head_backend_sampling = llama_set_sampler(head, 0, head_sampler);
            if (!head_backend_sampling) {
                llama_sampler_free(head_sampler);
                head_sampler = nullptr;
            }
        }
        configure_static_context(head, true);

        llama_context_params ep = make_context_params(1);
        ep.ctx_type = LLAMA_CONTEXT_TYPE_SPD_EMBED;
        ep.embeddings = true;
        embed = llama_init_from_model(model_target, ep);
        if (embed == nullptr) {
            fail("failed to initialize the target token-embedding context");
            return;
        }
        configure_static_context(embed, true);

        llama_context_params sp = make_context_params(
                std::max<uint32_t>(params.n_batch, stage_count + 1), true);
        sp.n_seq_max = params.draft_top_k;
        sp.n_ctx = params.n_ctx*params.draft_top_k;
        sidecar = llama_init_from_model(model_spd, sp);
        if (sidecar == nullptr) {
            fail("failed to initialize the SPD sidecar context");
            return;
        }
        if (static_decode_fast_path && params.draft_top_k == 1) {
            sidecar_sampler = make_greedy_sampler();
            sidecar_backend_sampling = llama_set_sampler(sidecar, 0, sidecar_sampler);
            if (!sidecar_backend_sampling) {
                llama_sampler_free(sidecar_sampler);
                sidecar_sampler = nullptr;
            }
        } else if (static_decode_fast_path) {
            for (uint32_t seq = 0; seq < params.draft_top_k; ++seq) {
                llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
                // Softmax precedes top-k so branch scores remain the full
                // sidecar probabilities used by the reference algorithm.
                // The sampled token from the distribution sampler is ignored;
                // top-k gathers only the retained ids, logits, and probabilities.
                llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));
                llama_sampler_chain_add(sampler, llama_sampler_init_top_k((int32_t) params.draft_top_k));
                if (!llama_set_sampler(sidecar, (llama_seq_id) seq, sampler)) {
                    llama_sampler_free(sampler);
                    fail("failed to initialize SPD sidecar backend top-k sampler");
                    return;
                }
                sidecar_tree_samplers[seq] = sampler;
            }
        }
        configure_static_context(sidecar, false);

        if (static_decode_fast_path && params.parallel_stages) {
            for (uint32_t stage = 0; stage < stage_count; ++stage) {
                stage_workers[stage] = std::make_unique<fixed_worker>();
            }
            sidecar_worker = std::make_unique<fixed_worker>();
        }
        ready = true;
    }

    void reset_memories() {
        for (uint32_t stage = 0; stage < stage_count; ++stage) {
            llama_memory_clear(llama_get_memory(stages[stage]), true);
        }
        llama_memory_clear(llama_get_memory(sidecar), true);
        for (auto & stage_positions : checkpoint_pos) {
            for (auto & positions : stage_positions) {
                positions.fill(-1);
            }
        }
        for (auto & tails : stage_tail_pos) {
            tails.fill(-1);
        }
    }

    bool decode_stage(
            uint32_t stage,
            const llama_token * tokens,
            const float * embeddings,
            int32_t n_tokens,
            llama_pos first_pos,
            std::vector<float> & output,
            const std::vector<std::pair<size_t, std::vector<float> *>> & anchor_outputs,
            std::string & error,
            batch_storage * reusable_storage = nullptr,
            const llama_pos * positions = nullptr,
            const llama_seq_id * sequences = nullptr) {
        batch_storage local_storage;
        batch_storage & storage = reusable_storage != nullptr ? *reusable_storage : local_storage;
        storage.set(n_tokens, tokens, embeddings, n_embd, target_n_pos_per_embd,
                first_pos, true, positions, sequences);
        if (llama_decode(stages[stage], storage.batch) != 0) {
            error = "target SPD stage " + std::to_string(stage) + " decode failed";
            return false;
        }

        const float * result = llama_get_embeddings(stages[stage]);
        if (result == nullptr) {
            error = "target SPD stage " + std::to_string(stage) + " produced no hidden state";
            return false;
        }
        output.assign(result, result + (size_t) n_tokens*n_embd);

        for (auto & item : anchor_outputs) {
            const int32_t anchor = anchors[item.first];
            float * data = llama_get_embeddings_layer_inp(stages[stage], anchor);
            if (data == nullptr) {
                error = "target SPD stage " + std::to_string(stage) +
                        " produced no snapshot for anchor " + std::to_string(anchor);
                return false;
            }
            item.second->assign(data, data + (size_t) n_tokens*n_embd);
        }
        return true;
    }

    bool embed_token(llama_token token, std::vector<float> & output) {
        batch_storage local_storage;
        batch_storage & storage = static_decode_fast_path ? embed_decode_batch : local_storage;
        storage.set(1, &token, nullptr, 0, target_n_pos_per_embd, 0, true);
        if (llama_decode(embed, storage.batch) != 0) {
            fail("target token embedding decode failed");
            return false;
        }
        const float * data = llama_get_embeddings(embed);
        if (data == nullptr) {
            fail("target token embedding context produced no embedding");
            return false;
        }
        output.assign(data, data + n_embd);
        return true;
    }

    bool target_head(const std::vector<float> & hidden, llama_token & token) {
        batch_storage local_storage;
        batch_storage & storage = static_decode_fast_path ? head_decode_batch : local_storage;
        storage.set(1, nullptr, hidden.data(), n_embd, target_n_pos_per_embd, 0, true);
        if (llama_decode(head, storage.batch) != 0) {
            fail("target SPD head decode failed");
            return false;
        }
        if (head_backend_sampling) {
            token = llama_get_sampled_token_ith(head, -1);
            if (token == LLAMA_TOKEN_NULL) {
                fail("target SPD head produced no sampled token");
                return false;
            }
            return true;
        }
        const float * logits = llama_get_logits_ith(head, -1);
        if (logits == nullptr) {
            fail("target SPD head produced no logits");
            return false;
        }
        token = argmax(logits, n_vocab);
        return true;
    }

    size_t depth_to_anchor(int32_t depth) const {
        const int32_t available_hf = std::min(n_layers, depth*layers_per_stage);
        size_t result = 0;
        for (size_t i = 1; i < anchors.size(); ++i) {
            if (anchors[i] > available_hf) {
                break;
            }
            result = i;
        }
        return result;
    }

    size_t choose_anchor(const snapshot & snap, int32_t search_hi) const {
        for (int32_t depth = search_hi; depth >= 0; --depth) {
            const size_t selector = depth_to_anchor(depth);
            bool complete = true;
            for (size_t i = 0; i <= selector; ++i) {
                complete = complete && snap.present[i];
            }
            if (complete) {
                return selector;
            }
        }
        return 0;
    }

    void append_feature_row(std::vector<float> & features, const snapshot & snap, size_t selector) const {
        const size_t old_size = features.size();
        features.resize(old_size + sidecar_n_embd_inp, 0.0f);
        float * dst = features.data() + old_size;
        for (size_t i = 0; i <= selector; ++i) {
            std::memcpy(dst + i*n_embd, snap.values[i].data(), (size_t) n_embd*sizeof(float));
        }
    }

    bool sidecar_decode(
            const std::vector<llama_token> & selectors,
            const std::vector<float> & features,
            const std::vector<llama_pos> & positions,
            llama_token * sampled,
            batch_storage * reusable_storage = nullptr) {
        std::vector<std::unique_lock<std::mutex>> resource_locks;
        resource_locks.reserve(sidecar_resources.size());
        for (size_t resource : sidecar_resources) {
            resource_locks.emplace_back(*resource_mutexes[resource]);
        }
        batch_storage local_storage;
        batch_storage & storage = reusable_storage != nullptr ? *reusable_storage : local_storage;
        storage.set(selectors, features, sidecar_n_embd_inp, positions);
        if (llama_decode(sidecar, storage.batch) != 0) {
            fail("SPD sidecar decode failed");
            return false;
        }
        if (sampled != nullptr) {
            if (sidecar_backend_sampling) {
                *sampled = llama_get_sampled_token_ith(sidecar, -1);
                if (*sampled == LLAMA_TOKEN_NULL) {
                    fail("SPD sidecar produced no sampled token");
                    return false;
                }
                return true;
            }
            const float * logits = llama_get_logits_ith(sidecar, -1);
            if (logits == nullptr) {
                fail("SPD sidecar produced no logits");
                return false;
            }
            *sampled = argmax(logits, n_vocab);
        }
        return true;
    }

    bool prefill_sidecar(const std::vector<std::vector<float>> & anchor_data, int32_t n_tokens) {
        const int32_t prefill_len = std::max(0, n_tokens - (int32_t) stage_count + 1);
        const int32_t chunk_size = std::max<int32_t>(1, params.n_batch);
        const size_t selector = anchors.size() - 1;

        for (int32_t begin = 0; begin < prefill_len; begin += chunk_size) {
            const int32_t count = std::min(chunk_size, prefill_len - begin);
            std::vector<llama_token> selectors(count, (llama_token) selector);
            std::vector<llama_pos> positions(count);
            std::vector<float> features((size_t) count*sidecar_n_embd_inp, 0.0f);

            for (int32_t row = 0; row < count; ++row) {
                positions[row] = begin + row;
                float * dst = features.data() + (size_t) row*sidecar_n_embd_inp;
                for (size_t ai = 0; ai < anchors.size(); ++ai) {
                    const float * src = anchor_data[ai].data() + (size_t) (begin + row)*n_embd;
                    std::memcpy(dst + ai*n_embd, src, (size_t) n_embd*sizeof(float));
                }
            }
            if (!sidecar_decode(selectors, features, positions, nullptr)) {
                return false;
            }
        }
        return true;
    }

    bool prepare_speculation(
            const std::vector<entry> & pipeline,
            const std::map<llama_pos, snapshot> & completed,
            const snapshot * prev_evicted,
            llama_pos prev_evicted_pos,
            speculation_input & input) {
        if (pipeline.empty()) {
            fail("SPD pipeline is empty before speculation");
            return false;
        }

        const llama_pos newest_pos = pipeline.front().pos;
        const llama_pos oldest_needed = newest_pos - stage_count + 1;
        if (oldest_needed < 0) {
            fail("SPD prompt is too short to construct a complete speculation window");
            return false;
        }

        input.selectors.clear();
        input.positions.clear();
        input.features.clear();
        input.min_pos = -1;
        const bool has_evicted = prev_evicted != nullptr;

        if (has_evicted) {
            const size_t selector = choose_anchor(*prev_evicted, stage_count);
            input.selectors.push_back((llama_token) selector);
            input.positions.push_back(prev_evicted_pos);
            append_feature_row(input.features, *prev_evicted, selector);
        }

        for (llama_pos pos = oldest_needed; pos <= newest_pos; ++pos) {
            const snapshot * snap = nullptr;
            for (const entry & item : pipeline) {
                if (item.pos == pos) {
                    snap = &item.snap;
                    break;
                }
            }
            if (snap == nullptr) {
                auto complete_it = completed.find(pos);
                if (complete_it == completed.end()) {
                    fail("missing completed SPD snapshot at position " + std::to_string(pos));
                    return false;
                }
                snap = &complete_it->second;
            }

            const int32_t nominal_depth = newest_pos - pos;
            const int32_t search_hi = nominal_depth == 0 ? 0 : has_evicted ? stage_count - 1 : stage_count;
            const size_t selector = choose_anchor(*snap, search_hi);
            input.selectors.push_back((llama_token) selector);
            input.positions.push_back(pos);
            append_feature_row(input.features, *snap, selector);
        }

        input.min_pos = input.positions.front();
        return true;
    }

    bool run_speculation(const speculation_input & input, llama_token & sampled) {
        if (!llama_memory_seq_rm(llama_get_memory(sidecar), 0, input.min_pos, -1)) {
            fail("failed to crop SPD sidecar cache at position " + std::to_string(input.min_pos));
            return false;
        }
        return sidecar_decode(input.selectors, input.features, input.positions, &sampled,
                static_decode_fast_path ? &sidecar_decode_batch : nullptr);
    }

    bool run_tree_speculation(
            const std::vector<tree_chain> & chains,
            const std::vector<speculation_input> & inputs,
            std::vector<std::vector<draft_candidate>> & candidates) {
        if (chains.size() != inputs.size() || chains.empty()) {
            fail("invalid SPD tree speculation batch");
            return false;
        }

        std::vector<std::unique_lock<std::mutex>> resource_locks;
        resource_locks.reserve(sidecar_resources.size());
        for (size_t resource : sidecar_resources) {
            resource_locks.emplace_back(*resource_mutexes[resource]);
        }

        std::vector<llama_token> selectors;
        std::vector<llama_pos> positions;
        std::vector<llama_seq_id> sequences;
        std::vector<float> features;
        std::vector<int8_t> output_flags;
        std::vector<int32_t> output_indices;
        size_t total_rows = 0;
        for (const auto & input : inputs) {
            total_rows += input.selectors.size();
        }
        selectors.reserve(total_rows);
        positions.reserve(total_rows);
        sequences.reserve(total_rows);
        features.reserve(total_rows*(size_t) sidecar_n_embd_inp);
        output_flags.reserve(total_rows);
        output_indices.reserve(chains.size());

        for (size_t ci = 0; ci < chains.size(); ++ci) {
            const auto & input = inputs[ci];
            if (input.selectors.empty() || input.min_pos < 0) {
                fail("empty SPD tree speculation input");
                return false;
            }
            if (!llama_memory_seq_rm(
                    llama_get_memory(sidecar), chains[ci].seq_id, input.min_pos, -1)) {
                fail("failed to crop SPD sidecar tree cache at position " +
                        std::to_string(input.min_pos));
                return false;
            }
            for (size_t row = 0; row < input.selectors.size(); ++row) {
                selectors.push_back(input.selectors[row]);
                positions.push_back(input.positions[row]);
                sequences.push_back(chains[ci].seq_id);
                output_flags.push_back(row + 1 == input.selectors.size());
            }
            output_indices.push_back((int32_t) selectors.size() - 1);
            features.insert(features.end(), input.features.begin(), input.features.end());
        }

        batch_storage local_storage;
        batch_storage & storage = static_decode_fast_path ? sidecar_decode_batch : local_storage;
        storage.set(selectors, features, sidecar_n_embd_inp, positions, &sequences, &output_flags);
        if (llama_decode(sidecar, storage.batch) != 0) {
            fail("SPD sidecar tree decode failed");
            return false;
        }

        candidates.assign(chains.size(), {});
        const uint32_t width = tree_width();
        for (size_t ci = 0; ci < chains.size(); ++ci) {
            if (static_decode_fast_path) {
                const int32_t output_index = output_indices[ci];
                const float * probs = llama_get_sampled_probs_ith(sidecar, output_index);
                const llama_token * token_ids = llama_get_sampled_candidates_ith(sidecar, output_index);
                const uint32_t count = llama_get_sampled_probs_count_ith(sidecar, output_index);
                if (probs == nullptr || token_ids == nullptr || count < width) {
                    fail("SPD sidecar backend top-k produced incomplete candidates for branch " +
                            std::to_string(ci));
                    return false;
                }

                std::vector<uint32_t> order(count);
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(),
                        [&](uint32_t a, uint32_t b) { return probs[a] > probs[b]; });
                candidates[ci].reserve(width);
                for (uint32_t rank = 0; rank < width; ++rank) {
                    const uint32_t index = order[rank];
                    candidates[ci].push_back({
                        token_ids[index], std::log(std::max((double) probs[index], 1e-30)), rank,
                    });
                }
            } else {
                const float * logits = llama_get_logits_ith(sidecar, output_indices[ci]);
                if (logits == nullptr) {
                    fail("SPD sidecar tree produced no logits for branch " + std::to_string(ci));
                    return false;
                }

                std::vector<int32_t> order(n_vocab);
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + width, order.end(),
                        [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
                const float max_logit = *std::max_element(logits, logits + n_vocab);
                double exp_sum = 0.0;
                for (int32_t i = 0; i < n_vocab; ++i) {
                    exp_sum += std::exp((double) logits[i] - max_logit);
                }
                const double log_normalizer = (double) max_logit + std::log(exp_sum);
                candidates[ci].reserve(width);
                for (uint32_t rank = 0; rank < width; ++rank) {
                    const llama_token token = (llama_token) order[rank];
                    candidates[ci].push_back({ token, (double) logits[token] - log_normalizer, rank });
                }
            }
        }
        return true;
    }

    bool speculate(
            const std::vector<entry> & pipeline,
            const std::map<llama_pos, snapshot> & completed,
            const snapshot * prev_evicted,
            llama_pos prev_evicted_pos,
            llama_token & sampled) {
        speculation_input input;
        return prepare_speculation(pipeline, completed, prev_evicted, prev_evicted_pos, input) &&
                run_speculation(input, sampled);
    }

    bool rollback(llama_pos target_pos) {
        for (uint32_t stage = 0; stage < stage_count; ++stage) {
            const llama_pos restore_pos = target_pos - 1;
            if (stage_tail_pos[stage][0] == restore_pos) {
                continue;
            }

            llama_seq_id restore_seq = -1;
            for (uint32_t i = 0; i < rollback_tokens; ++i) {
                if (checkpoint_pos[stage][0][i] == restore_pos) {
                    restore_seq = checkpoint_seq_id(0, i);
                    break;
                }
            }
            if (restore_seq < 0) {
                fail("target SPD stage " + std::to_string(stage) +
                        " has no checkpoint for position " + std::to_string(restore_pos));
                return false;
            }

            llama_memory_t memory = llama_get_memory(stages[stage]);
            if (!llama_memory_seq_rm(memory, 0, -1, -1)) {
                fail("failed to clear target SPD stage " + std::to_string(stage) + " before restore");
                return false;
            }
            llama_memory_seq_cp(memory, restore_seq, 0, -1, -1);
            stage_tail_pos[stage][0] = restore_pos;
        }
        if (!llama_memory_seq_rm(llama_get_memory(sidecar), 0, target_pos, -1)) {
            fail("failed to roll back SPD sidecar");
            return false;
        }
        return true;
    }

    bool prefill_target(
            const std::vector<llama_token> & prompt,
            std::vector<float> & hidden,
            std::vector<std::vector<float>> & anchor_data) {
        const int32_t n_prompt = (int32_t) prompt.size();
        const int32_t chunk_size = std::max<int32_t>(1, params.n_batch);
        const int32_t n_chunks = (n_prompt + chunk_size - 1)/chunk_size;
        hidden.resize((size_t) n_prompt*n_embd);
        anchor_data.assign(anchors.size(), std::vector<float>((size_t) n_prompt*n_embd));

        struct prefill_task_result {
            bool ok;
            std::string error;
        };

        // A target-prefill task depends on both the previous stage for the
        // same chunk and the previous chunk for the same stage. Execute the
        // stage/chunk grid by antidiagonals so both dependencies have finished
        // while distinct stages can keep distinct accelerators busy together.
        for (int32_t wave = 0; wave < n_chunks + (int32_t) stage_count - 1; ++wave) {
            std::vector<std::future<prefill_task_result>> work;
            for (uint32_t stage = 0; stage < stage_count; ++stage) {
                const int32_t chunk = wave - (int32_t) stage;
                if (chunk < 0 || chunk >= n_chunks) {
                    continue;
                }

                work.push_back(std::async(std::launch::async, [&, stage, chunk]() {
                    std::vector<std::unique_lock<std::mutex>> resource_locks;
                    resource_locks.reserve(stage_resources[stage].size());
                    for (size_t resource : stage_resources[stage]) {
                        resource_locks.emplace_back(*resource_mutexes[resource]);
                    }

                    const int32_t begin = chunk*chunk_size;
                    const int32_t count = std::min(chunk_size, n_prompt - begin);
                    std::vector<float> chunk_output;
                    std::vector<std::vector<float>> chunk_anchors(anchors.size());
                    std::vector<std::pair<size_t, std::vector<float> *>> wanted;
                    for (size_t ai = 0; ai < anchors.size(); ++ai) {
                        if ((uint32_t) std::min<int32_t>(anchors[ai] / layers_per_stage, stage_count - 1) == stage) {
                            wanted.push_back({ ai, &chunk_anchors[ai] });
                        }
                    }

                    std::string error;
                    const llama_token * tokens = stage == 0 ? prompt.data() + begin : nullptr;
                    const float * embeddings = stage == 0 ? nullptr : hidden.data() + (size_t) begin*n_embd;
                    if (!decode_stage(stage, tokens, embeddings, count, begin, chunk_output, wanted, error)) {
                        return prefill_task_result { false, std::move(error) };
                    }
                    std::copy(chunk_output.begin(), chunk_output.end(), hidden.begin() + (size_t) begin*n_embd);
                    for (auto & item : wanted) {
                        std::copy(chunk_anchors[item.first].begin(), chunk_anchors[item.first].end(),
                                anchor_data[item.first].begin() + (size_t) begin*n_embd);
                    }
                    return prefill_task_result { true, {} };
                }));
            }

            for (auto & task : work) {
                prefill_task_result task_result = task.get();
                if (!task_result.ok) {
                    fail(std::move(task_result.error));
                    return false;
                }
            }
        }

        for (uint32_t stage = 0; stage < stage_count; ++stage) {
            stage_tail_pos[stage][0] = n_prompt - 1;
        }
        return true;
    }

    snapshot snapshot_at(const std::vector<std::vector<float>> & anchor_data, int32_t pos) const {
        snapshot snap(anchors.size());
        for (size_t ai = 0; ai < anchors.size(); ++ai) {
            const float * begin = anchor_data[ai].data() + (size_t) pos*n_embd;
            snap.values[ai].assign(begin, begin + n_embd);
            snap.present[ai] = true;
        }
        return snap;
    }

    entry make_entry(llama_token token, llama_pos pos, std::vector<float> hidden) const {
        entry item;
        item.token = token;
        item.pos = pos;
        item.hidden = std::move(hidden);
        item.snap = snapshot(anchors.size());
        item.snap.values[0] = item.hidden;
        item.snap.present[0] = true;
        return item;
    }

    bool advance_entry(uint32_t stage, entry & item, double & elapsed, std::string & error) {
        std::vector<std::unique_lock<std::mutex>> resource_locks;
        resource_locks.reserve(stage_resources[stage].size());
        for (size_t resource : stage_resources[stage]) {
            resource_locks.emplace_back(*resource_mutexes[resource]);
        }
        const auto start = clock_type::now();

        const uint32_t checkpoint_slot = (uint32_t) (item.pos % rollback_tokens);
        const llama_seq_id checkpoint_seq = checkpoint_seq_id(0, checkpoint_slot);
        llama_memory_seq_cp(llama_get_memory(stages[stage]), 0, checkpoint_seq, -1, -1);
        checkpoint_pos[stage][0][checkpoint_slot] = stage_tail_pos[stage][0];

        std::vector<float> local_output;
        std::vector<float> & output = static_decode_fast_path
                ? stage_decode_outputs[stage]
                : local_output;
        std::vector<std::vector<float>> captured(anchors.size());
        std::vector<std::pair<size_t, std::vector<float> *>> wanted;
        for (size_t ai = 0; ai < anchors.size(); ++ai) {
            if ((uint32_t) std::min<int32_t>(anchors[ai] / layers_per_stage, stage_count - 1) == stage) {
                wanted.push_back({ ai, &captured[ai] });
            }
        }

        const llama_token * token = stage == 0 ? &item.token : nullptr;
        const float * embd = stage == 0 ? nullptr : item.hidden.data();
        if (!decode_stage(stage, token, embd, 1, item.pos, output, wanted, error,
                static_decode_fast_path ? &stage_decode_batches[stage] : nullptr)) {
            return false;
        }
        if (static_decode_fast_path && params.draft_top_k == 1) {
            item.hidden.swap(output);
        } else {
            item.hidden = std::move(output);
        }
        stage_tail_pos[stage][0] = item.pos;
        for (auto & captured_item : wanted) {
            item.snap.values[captured_item.first] = std::move(captured[captured_item.first]);
            item.snap.present[captured_item.first] = true;
        }
        elapsed = seconds_since(start);
        return true;
    }

    void fork_tree_sequence(llama_seq_id source, llama_seq_id destination) {
        if (source == destination) {
            return;
        }
        for (uint32_t stage = 0; stage < stage_count; ++stage) {
            llama_memory_t memory = llama_get_memory(stages[stage]);
            llama_memory_seq_rm(memory, destination, -1, -1);
            llama_memory_seq_cp(memory, source, destination, -1, -1);
            stage_tail_pos[stage][destination] = stage_tail_pos[stage][source];
            for (uint32_t slot = 0; slot < rollback_tokens; ++slot) {
                const llama_seq_id source_checkpoint = checkpoint_seq_id(source, slot);
                const llama_seq_id destination_checkpoint = checkpoint_seq_id(destination, slot);
                llama_memory_seq_rm(memory, destination_checkpoint, -1, -1);
                llama_memory_seq_cp(memory, source_checkpoint, destination_checkpoint, -1, -1);
                checkpoint_pos[stage][destination][slot] = checkpoint_pos[stage][source][slot];
            }
        }
        llama_memory_t sidecar_memory = llama_get_memory(sidecar);
        llama_memory_seq_rm(sidecar_memory, destination, -1, -1);
        llama_memory_seq_cp(sidecar_memory, source, destination, -1, -1);
    }

    bool rollback_tree(llama_seq_id seq, llama_pos target_pos) {
        for (uint32_t stage = 0; stage < stage_count; ++stage) {
            const llama_pos restore_pos = target_pos - 1;
            if (stage_tail_pos[stage][seq] == restore_pos) {
                continue;
            }

            llama_seq_id restore_seq = -1;
            for (uint32_t slot = 0; slot < rollback_tokens; ++slot) {
                if (checkpoint_pos[stage][seq][slot] == restore_pos) {
                    restore_seq = checkpoint_seq_id(seq, slot);
                    break;
                }
            }
            if (restore_seq < 0) {
                fail("target SPD tree stage " + std::to_string(stage) +
                        " has no checkpoint for position " + std::to_string(restore_pos));
                return false;
            }

            llama_memory_t memory = llama_get_memory(stages[stage]);
            if (!llama_memory_seq_rm(memory, seq, -1, -1)) {
                fail("failed to clear target SPD tree sequence before restore");
                return false;
            }
            llama_memory_seq_cp(memory, restore_seq, seq, -1, -1);
            stage_tail_pos[stage][seq] = restore_pos;
        }
        if (!llama_memory_seq_rm(llama_get_memory(sidecar), seq, target_pos, -1)) {
            fail("failed to roll back SPD sidecar tree sequence");
            return false;
        }
        return true;
    }

    bool advance_entries(
            uint32_t stage,
            const std::vector<entry *> & items,
            const std::vector<llama_seq_id> & seq_ids,
            double & elapsed,
            std::string & error) {
        if (items.size() != seq_ids.size() || items.empty()) {
            error = "invalid target SPD tree stage batch";
            return false;
        }

        std::vector<std::unique_lock<std::mutex>> resource_locks;
        resource_locks.reserve(stage_resources[stage].size());
        for (size_t resource : stage_resources[stage]) {
            resource_locks.emplace_back(*resource_mutexes[resource]);
        }
        const auto start = clock_type::now();

        for (size_t i = 0; i < items.size(); ++i) {
            const llama_seq_id seq = seq_ids[i];
            const uint32_t slot = (uint32_t) (items[i]->pos % rollback_tokens);
            const llama_seq_id checkpoint_seq = checkpoint_seq_id(seq, slot);
            llama_memory_seq_rm(llama_get_memory(stages[stage]), checkpoint_seq, -1, -1);
            llama_memory_seq_cp(llama_get_memory(stages[stage]), seq, checkpoint_seq, -1, -1);
            checkpoint_pos[stage][seq][slot] = stage_tail_pos[stage][seq];
        }

        std::vector<llama_token> tokens;
        std::vector<float> embeddings;
        std::vector<llama_pos> positions;
        positions.reserve(items.size());
        if (stage == 0) {
            tokens.reserve(items.size());
        } else {
            embeddings.reserve(items.size()*(size_t) n_embd);
        }
        for (entry * item : items) {
            positions.push_back(item->pos);
            if (stage == 0) {
                tokens.push_back(item->token);
            } else {
                embeddings.insert(embeddings.end(), item->hidden.begin(), item->hidden.end());
            }
        }

        std::vector<float> local_output;
        std::vector<float> & output = static_decode_fast_path
                ? stage_decode_outputs[stage]
                : local_output;
        std::vector<std::vector<float>> captured(anchors.size());
        std::vector<std::pair<size_t, std::vector<float> *>> wanted;
        for (size_t ai = 0; ai < anchors.size(); ++ai) {
            if ((uint32_t) std::min<int32_t>(anchors[ai] / layers_per_stage, stage_count - 1) == stage) {
                wanted.push_back({ ai, &captured[ai] });
            }
        }

        if (!decode_stage(
                stage,
                stage == 0 ? tokens.data() : nullptr,
                stage == 0 ? nullptr : embeddings.data(),
                (int32_t) items.size(),
                positions.front(),
                output,
                wanted,
                error,
                static_decode_fast_path ? &stage_decode_batches[stage] : nullptr,
                positions.data(),
                seq_ids.data())) {
            return false;
        }

        for (size_t i = 0; i < items.size(); ++i) {
            entry & item = *items[i];
            item.hidden.assign(
                    output.begin() + i*(size_t) n_embd,
                    output.begin() + (i + 1)*(size_t) n_embd);
            stage_tail_pos[stage][seq_ids[i]] = item.pos;
            for (auto & captured_item : wanted) {
                const auto & values = captured[captured_item.first];
                item.snap.values[captured_item.first].assign(
                        values.begin() + i*(size_t) n_embd,
                        values.begin() + (i + 1)*(size_t) n_embd);
                item.snap.present[captured_item.first] = true;
            }
        }
        elapsed = seconds_since(start);
        return true;
    }

    void synchronize_all() {
        for (uint32_t stage = 0; stage < stage_count; ++stage) {
            llama_synchronize(stages[stage]);
        }
        llama_synchronize(head);
        llama_synchronize(embed);
        llama_synchronize(sidecar);
    }

    bool generate_linear(const std::vector<llama_token> & prompt, int32_t n_predict, common_spd_result & result) {
        result = {};
        last_error.clear();
        if (prompt.empty()) {
            fail("SPD prompt must not be empty");
            return false;
        }
        if (n_predict <= 0) {
            fail("SPD prediction count must be positive");
            return false;
        }
        if ((uint64_t) prompt.size() + n_predict + stage_count >= params.n_ctx) {
            fail("SPD prompt and prediction count exceed the configured context");
            return false;
        }

        reset_memories();
        const int32_t n_prompt = (int32_t) prompt.size();
        const auto prefill_start = clock_type::now();

        std::vector<float> prompt_hidden;
        std::vector<std::vector<float>> prompt_anchors;
        if (!prefill_target(prompt, prompt_hidden, prompt_anchors) ||
            !prefill_sidecar(prompt_anchors, n_prompt)) {
            return false;
        }
        if (static_decode_fast_path) {
            llama_synchronize(sidecar);
            llama_set_stable_host_inputs(sidecar, true);
        }

        std::map<llama_pos, snapshot> completed;
        const int32_t retained_begin = std::max(0, n_prompt - (int32_t) stage_count + 1);
        for (int32_t pos = retained_begin; pos < n_prompt; ++pos) {
            completed.emplace(pos, snapshot_at(prompt_anchors, pos));
        }

        std::vector<float> last_hidden(
                prompt_hidden.begin() + (size_t) (n_prompt - 1)*n_embd,
                prompt_hidden.begin() + (size_t) n_prompt*n_embd);
        llama_token first_token;
        if (!target_head(last_hidden, first_token)) {
            return false;
        }

        std::vector<float> first_embedding;
        if (!embed_token(first_token, first_embedding)) {
            return false;
        }

        result.prefill_seconds = seconds_since(prefill_start);
        result.tokens.push_back(first_token);
        result.accepted.push_back(true);
        result.n_accepted = 1;

        std::vector<entry> pipeline;
        pipeline.push_back(make_entry(first_token, n_prompt, std::move(first_embedding)));
        llama_pos next_position = n_prompt + 1;
        llama_pos verified_up_to = n_prompt + 1;
        snapshot prev_evicted;
        llama_pos prev_evicted_pos = -1;
        bool has_prev_evicted = false;

        const auto decode_start = clock_type::now();
        while (verified_up_to - n_prompt < n_predict) {
            ++result.decode_steps;

            llama_token pending_draft = LLAMA_TOKEN_NULL;
            const llama_pos oldest_needed = pipeline.front().pos - stage_count + 1;
            bool has_pending_draft = false;
            speculation_input local_pending_input;
            speculation_input & pending_input = static_decode_fast_path
                    ? decode_speculation_input
                    : local_pending_input;
            if (oldest_needed >= 0) {
                if (!prepare_speculation(pipeline, completed,
                        has_prev_evicted ? &prev_evicted : nullptr,
                        prev_evicted_pos, pending_input)) {
                    return false;
                }
                has_pending_draft = true;
            }

            const size_t active_count = pipeline.size();
            std::array<double, SPD_MAX_STAGE_COUNT> elapsed = {};
            std::array<std::string, SPD_MAX_STAGE_COUNT> errors;
            std::array<uint8_t, SPD_MAX_STAGE_COUNT> ok = {};
            std::array<stage_decode_job, SPD_MAX_STAGE_COUNT> stage_jobs = {};
            std::future<bool> speculation_work;
            speculation_job sidecar_job = {};
            bool speculation_ok = true;
            bool sidecar_job_submitted = false;
            std::exception_ptr worker_failure;
            if (has_pending_draft) {
                // The features were copied before target stages mutate their
                // snapshots, so an independently placed sidecar can decode
                // them while the target pipeline advances.
                if (static_decode_fast_path && params.parallel_stages) {
                    sidecar_job = { this, &pending_input, &pending_draft, &speculation_ok };
                    sidecar_worker->submit(execute_speculation_job, &sidecar_job);
                    sidecar_job_submitted = true;
                } else if (params.parallel_stages) {
                    speculation_work = std::async(std::launch::async, [&]() {
                        return run_speculation(pending_input, pending_draft);
                    });
                } else {
                    speculation_ok = run_speculation(pending_input, pending_draft);
                }
            }

            if (static_decode_fast_path && params.parallel_stages && active_count > 1) {
                for (size_t stage = 0; stage < active_count; ++stage) {
                    stage_jobs[stage] = {
                        this,
                        (uint32_t) stage,
                        &pipeline[stage],
                        &elapsed[stage],
                        &errors[stage],
                        &ok[stage],
                    };
                    stage_workers[stage]->submit(execute_stage_decode_job, &stage_jobs[stage]);
                }
                for (size_t stage = 0; stage < active_count; ++stage) {
                    std::exception_ptr failure = stage_workers[stage]->wait();
                    if (worker_failure == nullptr && failure != nullptr) {
                        worker_failure = failure;
                    }
                }
            } else if (params.parallel_stages && active_count > 1) {
                std::vector<std::future<bool>> work;
                work.reserve(active_count);
                for (size_t stage = 0; stage < active_count; ++stage) {
                    work.push_back(std::async(std::launch::async, [&, stage]() {
                        return advance_entry((uint32_t) stage, pipeline[stage], elapsed[stage], errors[stage]);
                    }));
                }
                for (size_t stage = 0; stage < active_count; ++stage) {
                    ok[stage] = work[stage].get();
                }
            } else {
                for (size_t stage = 0; stage < active_count; ++stage) {
                    ok[stage] = advance_entry((uint32_t) stage, pipeline[stage], elapsed[stage], errors[stage]);
                }
            }

            if (sidecar_job_submitted) {
                std::exception_ptr failure = sidecar_worker->wait();
                if (worker_failure == nullptr && failure != nullptr) {
                    worker_failure = failure;
                }
            }
            if (speculation_work.valid()) {
                speculation_ok = speculation_work.get();
            }
            if (worker_failure != nullptr) {
                try {
                    std::rethrow_exception(worker_failure);
                } catch (const std::exception & error) {
                    fail(std::string("SPD decode worker failed: ") + error.what());
                } catch (...) {
                    fail("SPD decode worker failed");
                }
                return false;
            }
            for (size_t stage = 0; stage < active_count; ++stage) {
                result.stage_compute_seconds += elapsed[stage];
                if (!ok[stage]) {
                    fail(errors[stage]);
                    return false;
                }
            }
            if (!speculation_ok) {
                return false;
            }

            if (pipeline.size() >= stage_count) {
                entry completed_entry = std::move(pipeline.back());
                completed[completed_entry.pos] = completed_entry.snap;
                const llama_pos target_pos = completed_entry.pos + 1;
                const int32_t generated_index = target_pos - n_prompt;

                if (generated_index < (int32_t) result.tokens.size()) {
                    llama_token verified_token;
                    if (!target_head(completed_entry.hidden, verified_token)) {
                        return false;
                    }

                    if (result.tokens[generated_index] != verified_token) {
                        result.tokens.resize(generated_index);
                        result.accepted.resize(generated_index);
                        result.tokens.push_back(verified_token);
                        result.accepted.push_back(false);
                        ++result.n_rejected;

                        if (!rollback(target_pos)) {
                            return false;
                        }
                        for (auto it = completed.lower_bound(target_pos); it != completed.end();) {
                            it = completed.erase(it);
                        }

                        std::vector<float> corrected_embedding;
                        if (!embed_token(verified_token, corrected_embedding)) {
                            return false;
                        }
                        pipeline.clear();
                        pipeline.push_back(make_entry(verified_token, target_pos, std::move(corrected_embedding)));
                        next_position = target_pos + 1;
                        verified_up_to = target_pos + 1;
                        has_prev_evicted = false;
                        continue;
                    }

                    ++result.n_accepted;
                    verified_up_to = target_pos + 1;
                }

                prev_evicted = std::move(completed_entry.snap);
                prev_evicted_pos = completed_entry.pos;
                has_prev_evicted = true;
                pipeline.pop_back();
            }

            if (!has_pending_draft) {
                if (!speculate(pipeline, completed,
                        has_prev_evicted ? &prev_evicted : nullptr,
                        prev_evicted_pos, pending_draft)) {
                    return false;
                }
            }

            result.tokens.push_back(pending_draft);
            result.accepted.push_back(true);
            std::vector<float> next_embedding;
            if (!embed_token(pending_draft, next_embedding)) {
                return false;
            }
            pipeline.insert(pipeline.begin(), make_entry(pending_draft, next_position, std::move(next_embedding)));
            ++next_position;

            // Drafts can run stage_count - 1 positions ahead of the verified
            // frontier. A rejection restarts there and must be able to rebuild
            // the preceding stage_count-row sidecar window, so retention follows
            // verified position rather than the newest speculative position.
            const llama_pos keep_from = std::max<llama_pos>(0, verified_up_to - stage_count);
            for (auto it = completed.begin(); it != completed.end() && it->first < keep_from;) {
                it = completed.erase(it);
            }
        }

        result.decode_seconds = seconds_since(decode_start);
        result.tokens.resize(n_predict);
        result.accepted.resize(n_predict);
        return true;
    }

    bool generate_tree(const std::vector<llama_token> & prompt, int32_t n_predict, common_spd_result & result) {
        result = {};
        last_error.clear();
        if (prompt.empty()) {
            fail("SPD prompt must not be empty");
            return false;
        }
        if (n_predict <= 0) {
            fail("SPD prediction count must be positive");
            return false;
        }
        if ((uint64_t) prompt.size() + n_predict + stage_count >= params.n_ctx) {
            fail("SPD prompt and prediction count exceed the configured context");
            return false;
        }

        reset_memories();
        const int32_t n_prompt = (int32_t) prompt.size();
        const auto prefill_start = clock_type::now();

        std::vector<float> prompt_hidden;
        std::vector<std::vector<float>> prompt_anchors;
        if (!prefill_target(prompt, prompt_hidden, prompt_anchors) ||
            !prefill_sidecar(prompt_anchors, n_prompt)) {
            return false;
        }
        if (static_decode_fast_path) {
            llama_synchronize(sidecar);
            llama_set_stable_host_inputs(sidecar, true);
        }

        std::map<llama_pos, snapshot> completed;
        const int32_t retained_begin = std::max(0, n_prompt - (int32_t) stage_count + 1);
        for (int32_t pos = retained_begin; pos < n_prompt; ++pos) {
            completed.emplace(pos, snapshot_at(prompt_anchors, pos));
        }

        std::vector<float> last_hidden(
                prompt_hidden.begin() + (size_t) (n_prompt - 1)*n_embd,
                prompt_hidden.begin() + (size_t) n_prompt*n_embd);
        llama_token first_token;
        if (!target_head(last_hidden, first_token)) {
            return false;
        }
        std::vector<float> first_embedding;
        if (!embed_token(first_token, first_embedding)) {
            return false;
        }

        result.prefill_seconds = seconds_since(prefill_start);
        result.tokens.push_back(first_token);
        result.accepted.push_back(true);
        result.n_accepted = 1;

        tree_chain initial;
        initial.seq_id = 0;
        initial.pipeline.push_back(make_entry(first_token, n_prompt, std::move(first_embedding)));
        initial.completed = std::move(completed);
        initial.tokens.push_back(first_token);
        initial.draft_ranks.push_back(0);

        std::vector<tree_chain> chains;
        chains.push_back(std::move(initial));
        llama_pos verified_up_to = n_prompt + 1;

        const auto decode_start = clock_type::now();
        while ((int32_t) result.tokens.size() < n_predict) {
            ++result.decode_steps;

            std::vector<speculation_input> pending_inputs(chains.size());
            for (size_t ci = 0; ci < chains.size(); ++ci) {
                tree_chain & chain = chains[ci];
                if (!prepare_speculation(
                        chain.pipeline,
                        chain.completed,
                        chain.has_prev_evicted ? &chain.prev_evicted : nullptr,
                        chain.prev_evicted_pos,
                        pending_inputs[ci])) {
                    return false;
                }
            }

            std::vector<std::vector<draft_candidate>> pending_candidates;
            bool speculation_ok = true;
            bool sidecar_job_submitted = false;
            std::future<bool> speculation_work;
            tree_speculation_job sidecar_job = {
                this, &chains, &pending_inputs, &pending_candidates, &speculation_ok,
            };
            if (static_decode_fast_path && params.parallel_stages) {
                sidecar_worker->submit(execute_tree_speculation_job, &sidecar_job);
                sidecar_job_submitted = true;
            } else if (params.parallel_stages) {
                speculation_work = std::async(std::launch::async, [&]() {
                    return run_tree_speculation(chains, pending_inputs, pending_candidates);
                });
            } else {
                speculation_ok = run_tree_speculation(chains, pending_inputs, pending_candidates);
            }

            size_t active_stages = 0;
            for (const auto & chain : chains) {
                active_stages = std::max(active_stages, chain.pipeline.size());
            }
            std::array<double, SPD_MAX_STAGE_COUNT> elapsed = {};
            std::array<std::string, SPD_MAX_STAGE_COUNT> errors;
            std::array<uint8_t, SPD_MAX_STAGE_COUNT> ok = {};
            std::array<tree_stage_decode_job, SPD_MAX_STAGE_COUNT> stage_jobs = {};
            for (size_t stage = 0; stage < active_stages; ++stage) {
                stage_jobs[stage].owner = this;
                stage_jobs[stage].stage = (uint32_t) stage;
                stage_jobs[stage].elapsed = &elapsed[stage];
                stage_jobs[stage].error = &errors[stage];
                stage_jobs[stage].ok = &ok[stage];
                for (tree_chain & chain : chains) {
                    if (stage < chain.pipeline.size()) {
                        stage_jobs[stage].items.push_back(&chain.pipeline[stage]);
                        stage_jobs[stage].seq_ids.push_back(chain.seq_id);
                    }
                }
            }

            std::exception_ptr worker_failure;
            if (static_decode_fast_path && params.parallel_stages && active_stages > 1) {
                for (size_t stage = 0; stage < active_stages; ++stage) {
                    stage_workers[stage]->submit(execute_tree_stage_decode_job, &stage_jobs[stage]);
                }
                for (size_t stage = 0; stage < active_stages; ++stage) {
                    std::exception_ptr failure = stage_workers[stage]->wait();
                    if (worker_failure == nullptr && failure != nullptr) {
                        worker_failure = failure;
                    }
                }
            } else if (params.parallel_stages && active_stages > 1) {
                std::vector<std::future<bool>> work;
                work.reserve(active_stages);
                for (size_t stage = 0; stage < active_stages; ++stage) {
                    work.push_back(std::async(std::launch::async, [&, stage]() {
                        return advance_entries(
                                (uint32_t) stage,
                                stage_jobs[stage].items,
                                stage_jobs[stage].seq_ids,
                                elapsed[stage],
                                errors[stage]);
                    }));
                }
                for (size_t stage = 0; stage < active_stages; ++stage) {
                    ok[stage] = work[stage].get();
                }
            } else {
                for (size_t stage = 0; stage < active_stages; ++stage) {
                    ok[stage] = advance_entries(
                            (uint32_t) stage,
                            stage_jobs[stage].items,
                            stage_jobs[stage].seq_ids,
                            elapsed[stage],
                            errors[stage]);
                }
            }

            if (sidecar_job_submitted) {
                std::exception_ptr failure = sidecar_worker->wait();
                if (worker_failure == nullptr && failure != nullptr) {
                    worker_failure = failure;
                }
            }
            if (speculation_work.valid()) {
                speculation_ok = speculation_work.get();
            }
            if (worker_failure != nullptr) {
                try {
                    std::rethrow_exception(worker_failure);
                } catch (const std::exception & error) {
                    fail(std::string("SPD tree decode worker failed: ") + error.what());
                } catch (...) {
                    fail("SPD tree decode worker failed");
                }
                return false;
            }
            for (size_t stage = 0; stage < active_stages; ++stage) {
                result.stage_compute_seconds += elapsed[stage];
                if (!ok[stage]) {
                    fail(errors[stage]);
                    return false;
                }
            }
            if (!speculation_ok) {
                return false;
            }

            std::map<llama_seq_id, std::vector<draft_candidate>> candidates_by_sequence;
            for (size_t ci = 0; ci < chains.size(); ++ci) {
                candidates_by_sequence.emplace(chains[ci].seq_id, std::move(pending_candidates[ci]));
            }

            struct verification {
                size_t chain_index;
                llama_pos target_pos;
                int32_t generated_index;
                llama_token proposed;
                llama_token target;
                bool accepted;
                uint32_t rank;
            };
            std::vector<verification> verifications;
            for (size_t ci = 0; ci < chains.size(); ++ci) {
                tree_chain & chain = chains[ci];
                if (chain.pipeline.size() < stage_count) {
                    continue;
                }
                entry completed_entry = std::move(chain.pipeline.back());
                chain.pipeline.pop_back();
                chain.completed[completed_entry.pos] = completed_entry.snap;

                const llama_pos target_pos = completed_entry.pos + 1;
                const int32_t generated_index = target_pos - n_prompt;
                if (target_pos != verified_up_to || generated_index < 0 ||
                    generated_index >= (int32_t) chain.tokens.size()) {
                    fail("SPD tree verification frontier is inconsistent");
                    return false;
                }
                llama_token target_token;
                if (!target_head(completed_entry.hidden, target_token)) {
                    return false;
                }
                const llama_token proposed = chain.tokens[generated_index];
                verifications.push_back({
                    ci,
                    target_pos,
                    generated_index,
                    proposed,
                    target_token,
                    proposed == target_token,
                    chain.draft_ranks[generated_index],
                });

                chain.prev_evicted = std::move(completed_entry.snap);
                chain.prev_evicted_pos = completed_entry.pos;
                chain.has_prev_evicted = true;
            }

            if (!verifications.empty()) {
                const verification * chosen = nullptr;
                for (const auto & candidate : verifications) {
                    if (candidate.accepted &&
                        (chosen == nullptr || chains[candidate.chain_index].score > chains[chosen->chain_index].score)) {
                        chosen = &candidate;
                    }
                }

                if (chosen != nullptr) {
                    tree_chain surviving = std::move(chains[chosen->chain_index]);
                    result.tokens.push_back(chosen->proposed);
                    result.accepted.push_back(true);
                    ++result.n_accepted;
                    if (chosen->rank > 0) {
                        ++result.n_branch_rescued;
                    }
                    verified_up_to = chosen->target_pos + 1;
                    chains.clear();
                    chains.push_back(std::move(surviving));
                } else {
                    const verification * best = &verifications.front();
                    for (const auto & candidate : verifications) {
                        if (chains[candidate.chain_index].score > chains[best->chain_index].score) {
                            best = &candidate;
                        }
                    }
                    tree_chain surviving = std::move(chains[best->chain_index]);
                    if (!rollback_tree(surviving.seq_id, best->target_pos)) {
                        return false;
                    }
                    for (auto it = surviving.completed.lower_bound(best->target_pos);
                         it != surviving.completed.end();) {
                        it = surviving.completed.erase(it);
                    }
                    surviving.tokens.resize(best->generated_index);
                    surviving.tokens.push_back(best->target);
                    surviving.draft_ranks.resize(best->generated_index);
                    surviving.draft_ranks.push_back(0);
                    surviving.score = 0.0;
                    surviving.has_prev_evicted = false;
                    surviving.prev_evicted_pos = -1;

                    std::vector<float> corrected_embedding;
                    if (!embed_token(best->target, corrected_embedding)) {
                        return false;
                    }
                    surviving.pipeline.clear();
                    surviving.pipeline.push_back(make_entry(
                            best->target, best->target_pos, std::move(corrected_embedding)));

                    result.tokens.push_back(best->target);
                    result.accepted.push_back(false);
                    ++result.n_rejected;
                    verified_up_to = best->target_pos + 1;
                    chains.clear();
                    chains.push_back(std::move(surviving));
                    continue;
                }
            }

            if ((int32_t) result.tokens.size() >= n_predict) {
                break;
            }

            struct expansion {
                size_t parent_index;
                draft_candidate candidate;
                double score;
                llama_seq_id destination = -1;
            };
            std::vector<expansion> expansions;
            for (size_t ci = 0; ci < chains.size(); ++ci) {
                auto candidates_it = candidates_by_sequence.find(chains[ci].seq_id);
                if (candidates_it == candidates_by_sequence.end()) {
                    fail("missing SPD tree candidates for retained branch");
                    return false;
                }
                for (const auto & candidate : candidates_it->second) {
                    expansions.push_back({ ci, candidate, chains[ci].score + candidate.log_probability, -1 });
                }
            }
            std::sort(expansions.begin(), expansions.end(),
                    [](const expansion & a, const expansion & b) { return a.score > b.score; });
            if (expansions.size() > tree_width()) {
                expansions.resize(tree_width());
            }
            if (expansions.empty()) {
                fail("SPD tree produced no branch expansions");
                return false;
            }

            std::array<bool, SPD_MAX_DRAFT_TOP_K> retained_source = {};
            for (auto & expansion : expansions) {
                const llama_seq_id source = chains[expansion.parent_index].seq_id;
                if (!retained_source[source]) {
                    expansion.destination = source;
                    retained_source[source] = true;
                }
            }
            std::vector<llama_seq_id> free_sequences;
            for (llama_seq_id seq = 0; seq < (llama_seq_id) tree_width(); ++seq) {
                if (!retained_source[seq]) {
                    free_sequences.push_back(seq);
                }
            }
            size_t free_index = 0;
            for (auto & expansion : expansions) {
                if (expansion.destination >= 0) {
                    continue;
                }
                if (free_index >= free_sequences.size()) {
                    fail("SPD tree sequence assignment overflow");
                    return false;
                }
                expansion.destination = free_sequences[free_index++];
            }

            std::vector<tree_chain> next_chains;
            next_chains.reserve(expansions.size());
            for (const auto & expansion : expansions) {
                const tree_chain & parent = chains[expansion.parent_index];
                if (expansion.destination != parent.seq_id) {
                    fork_tree_sequence(parent.seq_id, expansion.destination);
                }
                tree_chain child = parent;
                child.seq_id = expansion.destination;
                child.score = expansion.score;
                const llama_pos next_position = n_prompt + (llama_pos) child.tokens.size();
                child.tokens.push_back(expansion.candidate.token);
                child.draft_ranks.push_back(expansion.candidate.rank);

                std::vector<float> embedding;
                if (!embed_token(expansion.candidate.token, embedding)) {
                    return false;
                }
                child.pipeline.insert(child.pipeline.begin(), make_entry(
                        expansion.candidate.token, next_position, std::move(embedding)));

                const llama_pos keep_from = std::max<llama_pos>(0, verified_up_to - stage_count);
                for (auto it = child.completed.begin();
                     it != child.completed.end() && it->first < keep_from;) {
                    it = child.completed.erase(it);
                }
                next_chains.push_back(std::move(child));
            }
            chains = std::move(next_chains);
        }

        result.decode_seconds = seconds_since(decode_start);
        result.tokens.resize(n_predict);
        result.accepted.resize(n_predict);
        return true;
    }

    bool generate(const std::vector<llama_token> & prompt, int32_t n_predict, common_spd_result & result) {
        return tree_width() == 1
                ? generate_linear(prompt, n_predict, result)
                : generate_tree(prompt, n_predict, result);
    }
};

common_spd_pipeline::common_spd_pipeline(
        llama_model * model_target,
        llama_model * model_spd,
        const common_spd_params & params)
    : pimpl(std::make_unique<impl>(model_target, model_spd, params)) {}

common_spd_pipeline::~common_spd_pipeline() = default;

bool common_spd_pipeline::valid() const {
    return pimpl->ready;
}

const std::string & common_spd_pipeline::error() const {
    return pimpl->last_error;
}

uint32_t common_spd_pipeline::stage_count() const {
    return pimpl->stage_count;
}

bool common_spd_pipeline::generate(
        const std::vector<llama_token> & prompt,
        int32_t n_predict,
        common_spd_result & result) {
    if (!pimpl->ready) {
        return false;
    }
    if (pimpl->used) {
        // Hybrid recurrent rollback aliases are deliberately short-lived.
        // Recreate only the contexts between requests while retaining both
        // loaded model objects and their weight buffers. This avoids carrying
        // backend cache/alias state across requests, which is unsafe for the
        // staged RPC schedulers and previously caused repeat-request crashes.
        pimpl->synchronize_all();
        llama_model * model_target = pimpl->model_target;
        llama_model * model_spd = pimpl->model_spd;
        common_spd_params params = pimpl->params;
        pimpl = std::make_unique<impl>(model_target, model_spd, params);
        if (!pimpl->ready) {
            return false;
        }
    }

    pimpl->used = true;
    return pimpl->generate(prompt, n_predict, result);
}
