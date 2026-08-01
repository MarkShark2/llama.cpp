#include "spd-pipeline.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama-ext.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <future>
#include <map>
#include <mutex>
#include <thread>
#include <utility>

namespace {

constexpr uint32_t SPD_MAX_STAGE_COUNT = COMMON_SPD_MAX_STAGE_COUNT;
constexpr uint32_t SPD_MAX_ROLLBACK_TOKENS = SPD_MAX_STAGE_COUNT - 1;

using clock_type = std::chrono::steady_clock;

double seconds_since(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}

struct scope_timer {
    double & total;
    clock_type::time_point start;

    explicit scope_timer(double & total) : total(total), start(clock_type::now()) {}
    ~scope_timer() { total += seconds_since(start); }
};

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

    batch_storage() = default;

    batch_storage(
            int32_t n_tokens,
            const llama_token * token_data,
            const float * embedding_data,
            int32_t n_embd,
            int32_t n_pos_per_embd,
            llama_pos first_pos,
            bool output_all) {
        set(n_tokens, token_data, embedding_data, n_embd, n_pos_per_embd, first_pos, output_all);
    }

    void set(
            int32_t n_tokens,
            const llama_token * token_data,
            const float * embedding_data,
            int32_t n_embd,
            int32_t n_pos_per_embd,
            llama_pos first_pos,
            bool output_all) {
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
                positions[(size_t) p*n_tokens + i] = first_pos + i;
            }
        }
        for (int32_t i = 0; i < n_tokens; ++i) {
            logits[i] = output_all || i + 1 == n_tokens;
        }
        batch.pos = positions.data();
        batch.logits = logits.data();
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
            const std::vector<llama_pos> & pos) {
        batch = {};
        batch.n_tokens = (int32_t) selectors.size();
        tokens = selectors;
        embeddings = features;
        positions = pos;
        logits.resize(selectors.size(), 0);
        std::fill(logits.begin(), logits.end(), 0);
        if (!logits.empty()) {
            logits.back() = 1;
        }
        batch.token = tokens.data();
        batch.embd = embeddings.data();
        batch.pos = positions.data();
        batch.logits = logits.data();

        GGML_ASSERT(features.size() == selectors.size()*(size_t) n_embd);
        GGML_ASSERT(pos.size() == selectors.size());
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

    llama_model * model_target;
    llama_model * model_spd;
    common_spd_params params;

    int32_t n_embd = 0;
    int32_t n_embd_boundary = 0; // width of the state handed between stages
    int32_t n_vocab = 0;
    int32_t n_layers = 0;
    int32_t layers_per_stage = 0;
    int32_t target_n_pos_per_embd = 1;
    int32_t sidecar_n_embd_inp = 0;
    uint32_t stage_count = 0;
    uint32_t rollback_tokens = 0;

    std::vector<int32_t> anchors;
    // [fork] LLAMA_SPD_HOST_TAPS=0 restores the graph-tap readback on every
    // stage (the pre-2026-08-01 behaviour) for A/B measurement.
    const bool host_taps_enabled = [] {
        const char * v = getenv("LLAMA_SPD_HOST_TAPS");
        return v == nullptr || atoi(v) != 0;
    }();
    std::array<llama_context *, SPD_MAX_STAGE_COUNT> stages = {};
    std::array<std::array<llama_pos, SPD_MAX_ROLLBACK_TOKENS>, SPD_MAX_STAGE_COUNT> checkpoint_pos = {};
    std::array<llama_pos, SPD_MAX_STAGE_COUNT> stage_tail_pos = {};
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
    bool head_backend_sampling = false;
    bool sidecar_backend_sampling = false;

    std::string last_error;
    bool ready = false;
    bool used = false;
    bool static_decode_fast_path = true;
    bool light_rollback = false;
    bool timing_enabled = false;

    // LLAMA_SPD_DUMP=<dir>: write every speculation input (selectors,
    // positions, features) and every verification outcome for offline replay
    // through the reference torch model. Capped so a long request cannot fill
    // the disk; the cap counts speculation steps.
    std::string dump_dir;
    int32_t dump_limit = 200;
    int32_t dump_step = 0;
    int32_t dump_prefill_rows_left = 4096;
    FILE * dump_index = nullptr;

    // LLAMA_SPD_PEER=1: ship prefill stage boundaries board-to-board through
    // the deployed RPC peer machinery instead of hairpinning ~2x16.8 MB per
    // cell through the client's per-board links serialized against compute.
    // One record per CONSUMING stage. `staging` is a same-shape twin of the
    // consumer graph's embd input on the same device: the producer pushes into
    // it (safe to apply on arrival -- nothing reads it), and the consumer
    // fences + does an on-server copy into the real input right before its
    // decode. The consumed/pushed chunk handshake bounds the producer to one
    // chunk ahead so staging is never overwritten before it is drained.
    // All non-buffer fields are guarded by prefill_target's progress mutex.
    struct peer_boundary {
        ggml_context * staging_ctx = nullptr;
        ggml_backend_buffer_t staging_buf = nullptr;
        // two parity slots so the producer can run up to two chunks ahead --
        // a single slot couples adjacent stages tightly enough that ordinary
        // board-to-board variance propagates as stalls in both directions
        ggml_tensor * staging[2] = { nullptr, nullptr };
        uint64_t ordinals[2] = { 0, 0 };  // delivery fence target per slot
        ggml_tensor * inp = nullptr;      // consumer graph's wide embd input
        bool ready = false;               // consumer-side setup complete
        bool dead  = false;               // route refused or a transfer failed
        int32_t  pushed_chunk = -1;       // last chunk delivered by push
        int32_t  consumed_chunk = -1;     // last chunk the consumer finished

        // Decode twins. A decode step moves one token, so the prefill staging
        // pair (a full n_batch chunk, 16 MiB) is the wrong shape: the push
        // requires ggml_nbytes(src) == ggml_nbytes(dst). `inp_dec` is a
        // 1-token alias of the persistent boundary's first column -- the
        // on-server copy target, since a 1-token graph reads column 0.
        //
        // The slot is chosen by the *entry position*, not a step counter:
        // adjacent stages hold entries whose positions differ by one, so
        // parity alone keeps the producer and consumer off the same slot, and
        // a stale slot cannot be mistaken for a live one after a rejection
        // resets the pipeline (`pushed_pos` is cleared there as well, because
        // a refill can legitimately reissue the same positions).
        ggml_tensor * staging_dec[2] = { nullptr, nullptr };
        ggml_tensor * inp_dec = nullptr;
        uint64_t  dec_ordinals[2] = { 0, 0 };
        llama_pos pushed_pos[2] = { -1, -1 };
        bool dec_ready = false;
    };
    std::array<peer_boundary, SPD_MAX_STAGE_COUNT> peer_links;
    bool peer_boundaries_enabled = false;
    // per-producing-stage: a decode push is outstanding, so this stage's next
    // graph must be ordered behind the push's read of the output tensor.
    //
    // A flag, not the tensor: ggml_backend_rpc_sync_peer_guard only needs the
    // tensor to find its *endpoint*, and holding the previous step's t_embd
    // across steps is a use-after-free -- a graph rebuild (~25 of 1567 steps
    // here) frees it, and the guard then reads a dead tensor, resolves no
    // endpoint, and silently skips the fence. The producer's next graph then
    // overwrites the boundary while the push is still reading it, which shows
    // up as a rare, nondeterministic wrong token. Derive the probe fresh.
    std::array<bool, SPD_MAX_STAGE_COUNT> peer_push_pending = {};

    // A rejection clears the pipeline and refills from the verified position,
    // so positions already delivered can be reissued with different hidden
    // states. Anything still sitting in a staging slot is stale by definition.
    void peer_decode_invalidate() {
        for (auto & pb : peer_links) {
            pb.pushed_pos[0] = -1;
            pb.pushed_pos[1] = -1;
        }
        peer_push_pending.fill(false);
    }

    struct phase_timing {
        double prepare = 0.0;
        double stage_wall = 0.0;
        double sidecar_extra = 0.0;
        double head = 0.0;
        double embed = 0.0;
        double rollback = 0.0;
        double step_total = 0.0;
        double sidecar_run = 0.0;
        uint64_t head_calls = 0;
        uint64_t embed_calls = 0;
        uint64_t sidecar_calls = 0;
        std::array<double, SPD_MAX_STAGE_COUNT> stage_busy = {};
        std::array<double, SPD_MAX_STAGE_COUNT> stage_lock = {};
        std::array<double, SPD_MAX_STAGE_COUNT> stage_decode = {};
        std::array<double, SPD_MAX_STAGE_COUNT> stage_read = {};
        // [fork] the per-stage decode work that sits OUTSIDE llama_decode, split
        // by protocol round trip. The guard, the fence and the staging->input
        // copy are each a blocking command on the endpoint's single command
        // loop, so `busy - decode` is three round trips plus the host-side tap,
        // not bookkeeping. Sized before deciding what to cut.
        std::array<double, SPD_MAX_STAGE_COUNT> stage_guard = {};
        std::array<double, SPD_MAX_STAGE_COUNT> stage_fence = {};
        std::array<double, SPD_MAX_STAGE_COUNT> stage_copy  = {};
        std::array<double, SPD_MAX_STAGE_COUNT> stage_push  = {};
        std::array<double, SPD_MAX_STAGE_COUNT> stage_tap   = {};
        std::array<uint64_t, SPD_MAX_STAGE_COUNT> stage_calls = {};
        // prefill is a separate regime from decode: chunks of n_batch tokens
        // walked as a stage/chunk grid, so it gets its own counters
        double prefill_wall = 0.0;
        std::array<double, SPD_MAX_STAGE_COUNT> prefill_busy = {};
        std::array<double, SPD_MAX_STAGE_COUNT> prefill_decode = {};
        std::array<double, SPD_MAX_STAGE_COUNT> prefill_read = {};
        std::array<double, SPD_MAX_STAGE_COUNT> prefill_dep_wait = {};
        std::array<double, SPD_MAX_STAGE_COUNT> prefill_lock = {};
        std::array<uint64_t, SPD_MAX_STAGE_COUNT> prefill_calls = {};
        // peer-boundary costs: fence+copy on the consumer, push on the producer
        std::array<double, SPD_MAX_STAGE_COUNT> prefill_peer_in = {};
        std::array<double, SPD_MAX_STAGE_COUNT> prefill_push = {};
        std::array<uint64_t, SPD_MAX_STAGE_COUNT> prefill_pushes = {};
        // decode-side peer boundary: pushes issued per producing stage. A zero
        // here with LLAMA_SPD_PEER=1 means the pair fell back to the hairpin.
        std::array<uint64_t, SPD_MAX_STAGE_COUNT> decode_pushes = {};
        // busy time and call count bucketed by how many stages ran in the step
        std::array<std::array<double, SPD_MAX_STAGE_COUNT + 1>, SPD_MAX_STAGE_COUNT> busy_by_active = {};
        std::array<std::array<uint64_t, SPD_MAX_STAGE_COUNT + 1>, SPD_MAX_STAGE_COUNT> calls_by_active = {};
        // Orchestration cost, exactly: the step's stage-phase wall minus the
        // slowest stage that actually ran in it. Bucketed by active count so a
        // flat profile reads as dispatch overhead and a rising one reads as the
        // cost of waiting on the slowest of N jittery stages. Taking the max
        // per step (not the max of the per-stage means) keeps the two apart.
        std::array<double,   SPD_MAX_STAGE_COUNT + 1> gap_by_active   = {};
        std::array<uint64_t, SPD_MAX_STAGE_COUNT + 1> gap_calls       = {};
    };
    phase_timing timing;
    size_t timing_active_count = 0;

    impl(llama_model * model_target, llama_model * model_spd, const common_spd_params & params)
        : model_target(model_target), model_spd(model_spd), params(params) {
        if (const char * value = std::getenv("LLAMA_SPD_STATIC_DECODE")) {
            static_decode_fast_path = std::atoi(value) != 0;
        }
        if (const char * value = std::getenv("LLAMA_SPD_TIMING")) {
            timing_enabled = std::atoi(value) != 0;
        }
        if (const char * value = std::getenv("LLAMA_SPD_DUMP")) {
            dump_dir = value;
        }
        if (const char * value = std::getenv("LLAMA_SPD_DUMP_LIMIT")) {
            dump_limit = std::atoi(value);
        }
        if (const char * value = std::getenv("LLAMA_SPD_PEER")) {
            peer_boundaries_enabled = std::atoi(value) != 0;
        }
        initialize();
    }

    ~impl() {
        sidecar_worker.reset();
        for (auto & worker : stage_workers) {
            worker.reset();
        }
        for (auto & pb : peer_links) {
            if (pb.staging_buf != nullptr) {
                ggml_backend_buffer_free(pb.staging_buf);
            }
            if (pb.staging_ctx != nullptr) {
                ggml_free(pb.staging_ctx);
            }
        }
        llama_free(sidecar);
        llama_free(embed);
        llama_free(head);
        for (uint32_t stage = 0; stage < stage_count; ++stage) {
            llama_free(stages[stage]);
        }
        llama_sampler_free(sidecar_sampler);
        llama_sampler_free(head_sampler);
        if (dump_index != nullptr) {
            std::fclose(dump_index);
        }
    }

    static void execute_stage_decode_job(void * data) {
        auto & job = *static_cast<stage_decode_job *>(data);
        *job.ok = job.owner->advance_entry(job.stage, *job.item, *job.elapsed, *job.error);
    }

    static void execute_speculation_job(void * data) {
        auto & job = *static_cast<speculation_job *>(data);
        *job.ok = job.owner->run_speculation(*job.input, *job.sampled);
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
        rollback_tokens = stage_count - 1;
        if (params.n_ctx > UINT32_MAX/stage_count) {
            fail("SPD context length is too large for rollback sequence allocation");
            return;
        }

        // Attention KV entries are position-addressed, so a rejection can rewind
        // a stage with llama_memory_seq_rm alone (rollback depth is at most
        // stage_count - 1 and the iswa cache keeps n_ubatch slack past the SWA
        // window). Recurrent/hybrid targets mutate state in place and still need
        // the cross-stream checkpoint copies.
        light_rollback = !llama_model_is_recurrent(model_target) && !llama_model_is_hybrid(model_target);
        if (const char * value = std::getenv("LLAMA_SPD_SEQCP_ROLLBACK")) {
            if (std::atoi(value) != 0) {
                light_rollback = false;
            }
        }

        n_embd = llama_model_n_embd(model_target);
        // What a stage hands the next one. Equal to n_embd for every
        // single-stream architecture, but DeepSeek-V4 carries hc_mult
        // hyper-connection streams between layers, so a mid-trunk boundary is
        // that much wider. Anchors stay n_embd wide -- they are stream means.
        n_embd_boundary = (int32_t) llama_model_n_embd_spd_boundary(model_target);
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_target));
        n_layers = llama_model_n_layer(model_target);
        // ceil, with the remainder on the last stage: 43 layers over 9 stages is
        // 5x8 + 3. Matches llama_context's slicing and the trainer's presets.
        layers_per_stage = (n_layers + (int32_t) stage_count - 1) / (int32_t) stage_count;
        target_n_pos_per_embd = llama_model_n_pos_per_embd(model_target);
        sidecar_n_embd_inp = llama_model_n_embd_inp(model_spd);

        if (static_decode_fast_path) {
            decode_speculation_input.selectors.reserve(stage_count + 1);
            decode_speculation_input.positions.reserve(stage_count + 1);
            decode_speculation_input.features.reserve((size_t) (stage_count + 1)*sidecar_n_embd_inp);
        }

        const uint32_t n_anchor = llama_model_target_layer_ids_n(model_spd);
        const int32_t * anchor_data = llama_model_target_layer_ids(model_spd);
        if (n_layers <= 0 || layers_per_stage*((int32_t) stage_count - 1) >= n_layers) {
            fail("SPD target layer count " + std::to_string(n_layers) + " leaves an empty stage at " +
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
            const int32_t layer_end = std::min(layer_begin + layers_per_stage, n_layers);
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
            if (!light_rollback) {
                cp.n_seq_max = stage_count;
                // llama_context derives n_ctx_seq by dividing total n_ctx by
                // n_seq_max. SPD uses the extra sequence IDs as rollback aliases,
                // but seq 0 must still retain the caller-requested context length.
                // Without this expansion an 8192-token SPD context silently became
                // an effective 1024-token context and diverged on longer prompts.
                cp.n_ctx = params.n_ctx*stage_count;
            }
            // Keep seq 0 on the same per-sequence KV layout as an ordinary
            // target context. Rollback snapshots use seq_cp aliases and do not
            // require a unified cache; forcing one changes long-context target
            // numerics enough to flip close greedy decisions.
            cp.kv_unified = false;
            // Rollback depth is at most stage_count - 1. Architectures whose
            // cache can only rewind a bounded suffix -- DeepSeek-V4, whose
            // compressor rings are built n_rs_seq rows wider than their read
            // window -- need that budget declared up front, or seq_rm refuses
            // every partial removal and the pipeline falls back to copying
            // checkpoints across sequences on every stage of every step. Over a
            // 9-stage RPC fabric that fallback costs more than the speculation
            // saves. Ignored by caches that rewind by position alone.
            cp.n_rs_seq = rollback_tokens;
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
        sidecar = llama_init_from_model(model_spd, sp);
        if (sidecar == nullptr) {
            fail("failed to initialize the SPD sidecar context");
            return;
        }
        if (static_decode_fast_path) {
            sidecar_sampler = make_greedy_sampler();
            sidecar_backend_sampling = llama_set_sampler(sidecar, 0, sidecar_sampler);
            if (!sidecar_backend_sampling) {
                llama_sampler_free(sidecar_sampler);
                sidecar_sampler = nullptr;
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
        for (auto & positions : checkpoint_pos) {
            positions.fill(-1);
        }
        stage_tail_pos.fill(-1);
    }

    // Which stage owns anchor `ai`, i.e. whose graph taps it as a layer input.
    uint32_t anchor_stage(size_t ai) const {
        return (uint32_t) std::min<int32_t>(anchors[ai]/layers_per_stage, stage_count - 1);
    }

    // An anchor sitting exactly on a stage's first layer is that stage's own
    // boundary input, meaned over the hc streams -- the identical quantity the
    // host computes from item.hidden the moment the PREVIOUS stage finishes
    // (see the end_layer loop in the decode path). Reading the graph tap too
    // costs a second blocking GET_TENSOR per stage per step and then
    // overwrites the host value with the same numbers. DSV4-s9 anchors are
    // [0,5,...,40], so this covers 8 of the 9 stages; anchor 0 has no previous
    // stage and keeps its readback.
    //
    // Decode only. Prefill has no host-side end_layer tap (and with peer
    // boundaries it never reads the hidden state back at all), so it still
    // consumes the graph tap on every stage.
    bool anchor_is_host_derived(size_t ai) const {
        if (!host_taps_enabled) {
            return false;
        }
        const uint32_t stage = anchor_stage(ai);
        if (stage == 0) {
            return false;
        }
        return anchors[ai] == std::min<int32_t>((int32_t) stage*layers_per_stage, n_layers);
    }

    // true when every anchor this stage taps is host-derived, so the decode can
    // skip the readback entirely
    bool stage_taps_all_host_derived(uint32_t stage) const {
        bool any = false;
        for (size_t ai = 0; ai < anchors.size(); ++ai) {
            if (anchor_stage(ai) != stage) {
                continue;
            }
            if (!anchor_is_host_derived(ai)) {
                return false;
            }
            any = true;
        }
        return any;
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
            double * t_decode = nullptr,
            double * t_read = nullptr,
            bool read_output = true) {
        batch_storage local_storage;
        batch_storage & storage = reusable_storage != nullptr ? *reusable_storage : local_storage;
        storage.set(n_tokens, tokens, embeddings, n_embd_boundary, target_n_pos_per_embd, first_pos, true);
        const auto decode_start = clock_type::now();
        if (llama_decode(stages[stage], storage.batch) != 0) {
            error = "target SPD stage " + std::to_string(stage) + " decode failed";
            return false;
        }
        const auto read_start = clock_type::now();
        if (t_decode != nullptr) {
            *t_decode += std::chrono::duration<double>(read_start - decode_start).count();
        }

        if (read_output) {
            const float * result = llama_get_embeddings(stages[stage]);
            if (result == nullptr) {
                error = "target SPD stage " + std::to_string(stage) + " produced no hidden state";
                return false;
            }
            output.assign(result, result + (size_t) n_tokens*n_embd_boundary);
        }

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
        if (t_read != nullptr) {
            *t_read += seconds_since(read_start);
        }
        return true;
    }

    bool embed_token(llama_token token, std::vector<float> & output) {
        const auto start = clock_type::now();
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
        timing.embed += seconds_since(start);
        ++timing.embed_calls;
        return true;
    }

    bool target_head(const std::vector<float> & hidden, llama_token & token) {
        scope_timer head_timer(timing.head);
        ++timing.head_calls;
        batch_storage local_storage;
        batch_storage & storage = static_decode_fast_path ? head_decode_batch : local_storage;
        storage.set(1, nullptr, hidden.data(), n_embd_boundary, target_n_pos_per_embd, 0, true);
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

    // ---- LLAMA_SPD_DUMP support ----------------------------------------
    // Row files carry the exact sidecar batch (selectors, positions,
    // features); index.jsonl carries the event stream (prefill/spec/verify)
    // so a torch replay can rebuild the sidecar's sequence state per step.

    FILE * dump_open_index() {
        if (dump_dir.empty()) {
            return nullptr;
        }
        if (dump_index == nullptr) {
            std::filesystem::create_directories(dump_dir);
            dump_index = std::fopen((dump_dir + "/index.jsonl").c_str(), "a");
        }
        return dump_index;
    }

    void dump_rows(const char * kind,
            const std::vector<llama_token> & selectors,
            const std::vector<llama_pos> & positions,
            const std::vector<float> & features,
            llama_token sampled) {
        FILE * index = dump_open_index();
        if (index == nullptr) {
            return;
        }
        char name[64];
        snprintf(name, sizeof(name), "%s_%06d.bin", kind, dump_step);
        FILE * fh = std::fopen((dump_dir + "/" + name).c_str(), "wb");
        if (fh == nullptr) {
            return;
        }
        const int32_t n_rows = (int32_t) selectors.size();
        std::fwrite(&n_rows, sizeof(n_rows), 1, fh);
        std::fwrite(&sidecar_n_embd_inp, sizeof(sidecar_n_embd_inp), 1, fh);
        std::fwrite(selectors.data(), sizeof(llama_token), selectors.size(), fh);
        std::fwrite(positions.data(), sizeof(llama_pos), positions.size(), fh);
        std::fwrite(features.data(), sizeof(float), features.size(), fh);
        std::fclose(fh);
        fprintf(index, "{\"kind\":\"%s\",\"step\":%d,\"file\":\"%s\",\"n_rows\":%d,"
                "\"pos0\":%d,\"sampled\":%d}\n",
                kind, dump_step, name, n_rows,
                positions.empty() ? -1 : (int) positions.front(), (int) sampled);
        fflush(index);
    }

    void dump_verify(llama_pos pos, llama_token drafted, llama_token verified) {
        FILE * index = dump_open_index();
        if (index == nullptr || dump_step >= dump_limit) {
            return;
        }
        fprintf(index, "{\"kind\":\"verify\",\"step\":%d,\"pos\":%d,\"drafted\":%d,"
                "\"verified\":%d,\"accepted\":%s}\n",
                dump_step, (int) pos, (int) drafted, (int) verified,
                drafted == verified ? "true" : "false");
        fflush(index);
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
            if (!dump_dir.empty() && dump_prefill_rows_left >= count) {
                dump_prefill_rows_left -= count;
                dump_rows("prefill", selectors, positions, features, LLAMA_TOKEN_NULL);
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
        scope_timer sidecar_timer(timing.sidecar_run);
        ++timing.sidecar_calls;
        if (!llama_memory_seq_rm(llama_get_memory(sidecar), 0, input.min_pos, -1)) {
            fail("failed to crop SPD sidecar cache at position " + std::to_string(input.min_pos));
            return false;
        }
        const bool ok = sidecar_decode(input.selectors, input.features, input.positions, &sampled,
                static_decode_fast_path ? &sidecar_decode_batch : nullptr);
        if (ok && !dump_dir.empty() && dump_step < dump_limit) {
            dump_rows("spec", input.selectors, input.positions, input.features, sampled);
            ++dump_step;
        }
        return ok;
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
            if (stage_tail_pos[stage] == restore_pos) {
                continue;
            }

            if (light_rollback) {
                // Position-addressed attention cache: dropping every cell at or
                // past the rejected position restores the checkpoint exactly.
                if (!llama_memory_seq_rm(llama_get_memory(stages[stage]), 0, target_pos, -1)) {
                    fail("failed to rewind target SPD stage " + std::to_string(stage) +
                            " to position " + std::to_string(restore_pos));
                    return false;
                }
                stage_tail_pos[stage] = restore_pos;
                continue;
            }

            llama_seq_id restore_seq = -1;
            for (uint32_t i = 0; i < rollback_tokens; ++i) {
                if (checkpoint_pos[stage][i] == restore_pos) {
                    restore_seq = (llama_seq_id) i + 1;
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
            stage_tail_pos[stage] = restore_pos;
        }
        if (!llama_memory_seq_rm(llama_get_memory(sidecar), 0, target_pos, -1)) {
            fail("failed to roll back SPD sidecar");
            return false;
        }
        return true;
    }

    // Consumer-side peer-boundary setup for `stage`: capture the graph's embd
    // input tensor, learn the endpoint's session id, and allocate the
    // same-shape staging twin on the same device. Runs on the stage's worker
    // thread (which owns that endpoint's main socket), after its first
    // full-chunk decode so the input tensor exists.
    void setup_peer_boundary(peer_boundary & pb, uint32_t stage, std::mutex & progress_mutex) {
        ggml_tensor * inp = llama_spd_peer_inp_tensor(stages[stage]);
        const char * why = nullptr;
        if (inp == nullptr) {
            why = "no input tensor on the last graph";
        } else if (inp->buffer == nullptr) {
            why = "input tensor has no buffer";
        } else if (inp->type != GGML_TYPE_F32) {
            why = "input tensor is not f32";
        } else if (!ggml_backend_rpc_sync_peer_prepare(inp)) {
            why = "peer prepare refused (endpoint / env / session)";
        }
        if (why != nullptr) {
            fprintf(stderr, "SPD peer: stage %u boundary setup failed: %s%s%s\n",
                    stage, why,
                    inp != nullptr && inp->buffer != nullptr ? ", buffer=" : "",
                    inp != nullptr && inp->buffer != nullptr ? ggml_backend_buffer_name(inp->buffer) : "");
        }
        bool ok = why == nullptr;
        if (ok && pb.staging[0] == nullptr) {
            ggml_init_params ip = { 8*ggml_tensor_overhead(), nullptr, /*no_alloc =*/ true };
            pb.staging_ctx = ggml_init(ip);
            ggml_tensor * st0 = pb.staging_ctx == nullptr ? nullptr :
                    ggml_new_tensor_2d(pb.staging_ctx, GGML_TYPE_F32, inp->ne[0], inp->ne[1]);
            ggml_tensor * st1 = st0 == nullptr ? nullptr :
                    ggml_new_tensor_2d(pb.staging_ctx, GGML_TYPE_F32, inp->ne[0], inp->ne[1]);
            // decode-sized twins + the 1-token destination alias. The alias is
            // a view, so ggml_backend_alloc_ctx_tensors_from_buft binds it to
            // the boundary's own buffer instead of giving it storage.
            ggml_tensor * dec0 = st1 == nullptr ? nullptr :
                    ggml_new_tensor_2d(pb.staging_ctx, GGML_TYPE_F32, inp->ne[0], 1);
            ggml_tensor * dec1 = dec0 == nullptr ? nullptr :
                    ggml_new_tensor_2d(pb.staging_ctx, GGML_TYPE_F32, inp->ne[0], 1);
            ggml_tensor * dec_dst = dec1 == nullptr ? nullptr :
                    ggml_view_2d(pb.staging_ctx, inp, inp->ne[0], 1, inp->nb[1], 0);
            pb.staging_buf = dec_dst == nullptr ? nullptr :
                    ggml_backend_alloc_ctx_tensors_from_buft(pb.staging_ctx,
                            ggml_backend_buffer_get_type(inp->buffer));
            if (pb.staging_buf != nullptr) {
                pb.staging[0] = st0;
                pb.staging[1] = st1;
                pb.staging_dec[0] = dec0;
                pb.staging_dec[1] = dec1;
                pb.inp_dec        = dec_dst;
                pb.dec_ready = dec_dst->buffer != nullptr && dec_dst->data != nullptr &&
                               ggml_nbytes(dec_dst) == ggml_nbytes(dec0);
                if (!pb.dec_ready) {
                    // decode keeps working on the hairpin, but silently: say so
                    fprintf(stderr, "SPD peer: stage %u decode boundary unavailable "
                            "(alias buffer=%p data=%p nbytes=%zu vs %zu)\n", stage,
                            (void *) dec_dst->buffer, dec_dst->data,
                            ggml_nbytes(dec_dst), ggml_nbytes(dec0));
                }
            } else {
                if (pb.staging_ctx != nullptr) {
                    ggml_free(pb.staging_ctx);
                    pb.staging_ctx = nullptr;
                }
                ok = false;
            }
        }
        std::lock_guard<std::mutex> lock(progress_mutex);
        if (ok) {
            pb.inp = inp;
            pb.ready = true;
        } else {
            pb.dead = true;
        }
    }

    bool prefill_target(
            const std::vector<llama_token> & prompt,
            std::vector<float> & hidden,
            std::vector<std::vector<float>> & anchor_data) {
        const int32_t n_prompt = (int32_t) prompt.size();
        const int32_t chunk_size = std::max<int32_t>(1, params.n_batch);
        const int32_t n_chunks = (n_prompt + chunk_size - 1)/chunk_size;
        hidden.resize((size_t) n_prompt*n_embd_boundary);
        anchor_data.assign(anchors.size(), std::vector<float>((size_t) n_prompt*n_embd));
        for (auto & pb : peer_links) {
            pb.pushed_chunk   = -1;
            pb.ordinals[0]    = 0;
            pb.ordinals[1]    = 0;
            pb.consumed_chunk = -1;
        }

        // A target-prefill cell (stage, chunk) depends on the previous stage
        // for the same chunk and on the previous chunk for the same stage.
        // One worker per stage walking chunks in order satisfies the second
        // dependency implicitly and waits on a progress counter for the first.
        //
        // This used to walk the grid by antidiagonals with a join at the end of
        // every wave. The dependencies were satisfied, but the join is stronger
        // than they are: a wave costs the *slowest* cell in it, so every fast
        // stage sits idle until the slow one lands, on every wave. Per-cell
        // dependencies let a stage run ahead into the next chunk instead.
        const auto prefill_start = clock_type::now();
        std::mutex progress_mutex;
        std::condition_variable progress_cv;
        std::vector<int32_t> chunks_done((size_t) stage_count, 0);
        bool aborted = false;

        std::vector<std::thread> workers;
        workers.reserve(stage_count);
        for (uint32_t stage = 0; stage < stage_count; ++stage) {
            workers.emplace_back([&, stage]() {
                std::vector<std::pair<size_t, std::vector<float> *>> wanted;
                std::vector<float> chunk_output;
                std::vector<std::vector<float>> chunk_anchors(anchors.size());
                // set after a deferred-ack push; the next decode must be
                // ordered behind the push's local read of the output tensor
                ggml_tensor * guard_probe = nullptr;

                for (int32_t chunk = 0; chunk < n_chunks; ++chunk) {
                    const auto wait_start = clock_type::now();
                    {
                        std::unique_lock<std::mutex> lock(progress_mutex);
                        progress_cv.wait(lock, [&] {
                            return aborted || stage == 0 || chunks_done[stage - 1] > chunk;
                        });
                        if (aborted) {
                            return;
                        }
                    }
                    const auto lock_start = clock_type::now();

                    const int32_t begin = chunk*chunk_size;
                    const int32_t count = std::min(chunk_size, n_prompt - begin);
                    wanted.clear();
                    for (size_t ai = 0; ai < anchors.size(); ++ai) {
                        if ((uint32_t) std::min<int32_t>(anchors[ai] / layers_per_stage, stage_count - 1) == stage) {
                            wanted.push_back({ ai, &chunk_anchors[ai] });
                        }
                    }

                    // Peer-boundary roles for this cell, decided under the
                    // progress lock: does the input arrive by push (consumer),
                    // does the output leave by push (producer). Partial final
                    // chunks stay on the host path (different graph shape).
                    peer_boundary * pb_in  = stage > 0 ? &peer_links[stage] : nullptr;
                    peer_boundary * pb_out = stage + 1 < stage_count ? &peer_links[stage + 1] : nullptr;
                    bool peer_in = false, peer_out = false;
                    if (peer_boundaries_enabled) {
                        std::lock_guard<std::mutex> lock(progress_mutex);
                        peer_in  = pb_in  != nullptr && pb_in->ready && !pb_in->dead &&
                                   pb_in->pushed_chunk >= chunk && pb_in->inp != nullptr;
                        peer_out = pb_out != nullptr && pb_out->ready && !pb_out->dead &&
                                   count == chunk_size;
                    }

                    std::string error;
                    bool ok = false;
                    const auto busy_start = clock_type::now();
                    {
                        std::vector<std::unique_lock<std::mutex>> resource_locks;
                        resource_locks.reserve(stage_resources[stage].size());
                        for (size_t resource : stage_resources[stage]) {
                            resource_locks.emplace_back(*resource_mutexes[resource]);
                        }
                        ok = true;
                        if (guard_probe != nullptr) {
                            ggml_backend_rpc_sync_peer_guard(guard_probe);
                            guard_probe = nullptr;
                        }
                        if (peer_in) {
                            // stall the endpoint's command loop until the push
                            // has been applied, then move it from staging into
                            // the graph's input tensor on-server
                            const auto peer_start = clock_type::now();
                            ggml_tensor * slot = pb_in->staging[chunk & 1];
                            ok = ggml_backend_rpc_sync_peer_fence(slot, pb_in->ordinals[chunk & 1]);
                            if (ok) {
                                ggml_backend_tensor_copy(slot, pb_in->inp);
                            } else {
                                error = "SPD peer fence failed for stage " + std::to_string(stage);
                            }
                            timing.prefill_peer_in[stage] += seconds_since(peer_start);
                        }
                        if (ok) {
                            llama_set_spd_peer_io(stages[stage], peer_in, peer_out, false);
                            const llama_token * tokens = stage == 0 ? prompt.data() + begin : nullptr;
                            const float * embeddings = stage == 0
                                    ? nullptr
                                    : hidden.data() + (size_t) begin*n_embd_boundary;
                            ok = decode_stage(stage, tokens, embeddings, count, begin, chunk_output, wanted, error,
                                    nullptr, &timing.prefill_decode[stage], &timing.prefill_read[stage],
                                    /*read_output =*/ !peer_out);
                            llama_set_spd_peer_io(stages[stage], false, false, false);
                        }
                        if (ok && peer_in) {
                            // a rebuild would have read a different input tensor
                            // than the copy filled; that cell would be garbage,
                            // so surface it loudly instead
                            if (llama_spd_peer_inp_tensor(stages[stage]) != pb_in->inp) {
                                error = "SPD peer boundary input tensor moved under stage " + std::to_string(stage);
                                ok = false;
                            }
                        } else if (ok && pb_in != nullptr && pb_in->ready && count == chunk_size) {
                            // host-path cell on a peer-capable stage: keep the
                            // captured input tensor current for the next push
                            std::lock_guard<std::mutex> lock(progress_mutex);
                            pb_in->inp = llama_spd_peer_inp_tensor(stages[stage]);
                        }
                    }
                    timing.prefill_dep_wait[stage] += std::chrono::duration<double>(lock_start - wait_start).count();
                    timing.prefill_lock[stage] += std::chrono::duration<double>(busy_start - lock_start).count();
                    timing.prefill_busy[stage] += seconds_since(busy_start);
                    ++timing.prefill_calls[stage];

                    if (!ok) {
                        {
                            std::lock_guard<std::mutex> lock(progress_mutex);
                            aborted = true;
                            fail(std::move(error));
                        }
                        progress_cv.notify_all();
                        return;
                    }

                    // Producer side: ship the boundary straight to the next
                    // stage's staging tensor. The graph is known complete (the
                    // anchor readback inside llama_decode is a sync round-trip
                    // behind GRAPH_COMPUTE on the same socket).
                    bool pushed = false;
                    if (peer_out) {
                        const auto push_start = clock_type::now();
                        {
                            // never overwrite a staging slot before the
                            // consumer drained it (two slots -> two ahead)
                            std::unique_lock<std::mutex> lock(progress_mutex);
                            progress_cv.wait(lock, [&] {
                                return aborted || pb_out->consumed_chunk >= chunk - 2;
                            });
                            if (aborted) {
                                return;
                            }
                        }
                        ggml_tensor * out = llama_spd_peer_out_tensor(stages[stage]);
                        ggml_tensor * slot = pb_out->staging[chunk & 1];
                        uint64_t ordinal = 0;
                        if (out != nullptr && slot != nullptr &&
                            ggml_nbytes(out) == ggml_nbytes(slot) &&
                            ggml_backend_rpc_sync_peer_push(out, slot, &ordinal)) {
                            {
                                std::lock_guard<std::mutex> lock(progress_mutex);
                                pb_out->ordinals[chunk & 1] = ordinal;
                                pb_out->pushed_chunk = chunk;
                            }
                            pushed = true;
                            guard_probe = out;
                            ++timing.prefill_pushes[stage];
                        } else {
                            // recover the boundary for the host path and stop
                            // trying this pair
                            {
                                std::lock_guard<std::mutex> lock(progress_mutex);
                                pb_out->dead = true;
                            }
                            if (out == nullptr) {
                                {
                                    std::lock_guard<std::mutex> lock(progress_mutex);
                                    aborted = true;
                                    fail("SPD peer push failed with no output tensor to recover from");
                                }
                                progress_cv.notify_all();
                                return;
                            }
                            chunk_output.resize((size_t) count*n_embd_boundary);
                            ggml_backend_tensor_get(out, chunk_output.data(), 0,
                                    chunk_output.size()*sizeof(float));
                        }
                        timing.prefill_push[stage] += seconds_since(push_start);
                    }

                    if (!pushed) {
                        std::copy(chunk_output.begin(), chunk_output.end(),
                                hidden.begin() + (size_t) begin*n_embd_boundary);
                    }
                    for (auto & item : wanted) {
                        std::copy(chunk_anchors[item.first].begin(), chunk_anchors[item.first].end(),
                                anchor_data[item.first].begin() + (size_t) begin*n_embd);
                    }

                    // consumer-side setup, once, after the first full-chunk
                    // decode so the input tensor exists
                    if (peer_boundaries_enabled && stage > 0 && count == chunk_size) {
                        bool need_setup;
                        {
                            std::lock_guard<std::mutex> lock(progress_mutex);
                            need_setup = !pb_in->ready && !pb_in->dead;
                        }
                        if (need_setup) {
                            setup_peer_boundary(*pb_in, stage, progress_mutex);
                        }
                    }

                    {
                        std::lock_guard<std::mutex> lock(progress_mutex);
                        chunks_done[stage] = chunk + 1;
                        if (pb_in != nullptr) {
                            pb_in->consumed_chunk = chunk;
                        }
                    }
                    progress_cv.notify_all();
                }
            });
        }
        for (auto & worker : workers) {
            worker.join();
        }
        timing.prefill_wall += seconds_since(prefill_start);
        if (aborted) {
            return false;
        }

        if (timing_enabled) {
            fprintf(stderr, "SPD timing: prefill %d tok in %d chunks of %d, wall %.2fs (%.1f t/s)\n",
                    n_prompt, n_chunks, chunk_size, timing.prefill_wall,
                    timing.prefill_wall > 0.0 ? n_prompt/timing.prefill_wall : 0.0);
            for (uint32_t stage = 0; stage < stage_count; ++stage) {
                if (timing.prefill_calls[stage] == 0) {
                    continue;
                }
                fprintf(stderr,
                        "SPD timing: prefill stage %u: busy %.2fs/%" PRIu64 " calls (%.0f ms/call) | "
                        "decode %.2fs | read %.2fs | dep-wait %.2fs | lock-wait %.2fs | "
                        "peer-in %.2fs | push %.2fs/%" PRIu64 "\n",
                        stage, timing.prefill_busy[stage], timing.prefill_calls[stage],
                        1e3*timing.prefill_busy[stage]/(double) timing.prefill_calls[stage],
                        timing.prefill_decode[stage], timing.prefill_read[stage],
                        timing.prefill_dep_wait[stage], timing.prefill_lock[stage],
                        timing.prefill_peer_in[stage], timing.prefill_push[stage],
                        timing.prefill_pushes[stage]);
            }
        }

        for (uint32_t stage = 0; stage < stage_count; ++stage) {
            stage_tail_pos[stage] = n_prompt - 1;
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
        const auto lock_start = clock_type::now();
        std::vector<std::unique_lock<std::mutex>> resource_locks;
        resource_locks.reserve(stage_resources[stage].size());
        for (size_t resource : stage_resources[stage]) {
            resource_locks.emplace_back(*resource_mutexes[resource]);
        }
        const auto start = clock_type::now();
        timing.stage_lock[stage] += std::chrono::duration<double>(start - lock_start).count();
        ++timing.stage_calls[stage];

        // [fork] Direct stage-to-stage boundary transfer during decode.
        //
        // Without this the hyper-connection-wide residual makes a round trip
        // through the client every step: 64 KiB read off this stage, 64 KiB
        // written back up to the next one, over the head's single NIC and
        // serialized against every other stage doing the same. The push moves
        // it board-to-board instead, and the consumer skips the upload.
        //
        // The readback stays: this stage's end-of-stage anchor tap is the
        // stream mean of the boundary and the sidecar needs it one step before
        // the next stage would capture it at its own input.
        //
        // Records belong to the CONSUMING stage, so `stage`'s inbound link is
        // peer_links[stage] and its outbound one is peer_links[stage + 1].
        peer_boundary * pb_in  = peer_boundaries_enabled && stage > 0
                ? &peer_links[stage] : nullptr;
        peer_boundary * pb_out = peer_boundaries_enabled && stage + 1 < stage_count
                ? &peer_links[stage + 1] : nullptr;
        const int slot = (int) (item.pos & 1);

        // Order this stage's next graph behind the previous push's read of the
        // output tensor -- the graph below overwrites exactly that tensor. The
        // probe only names the endpoint, so the live previous-graph output
        // works and nothing has to survive a rebuild.
        if (peer_push_pending[stage]) {
            scope_timer guard_timer(timing.stage_guard[stage]);
            ggml_tensor * probe = llama_spd_peer_out_tensor(stages[stage]);
            if (probe == nullptr || !ggml_backend_rpc_sync_peer_guard(probe)) {
                error = "SPD peer guard failed for decode stage " + std::to_string(stage);
                return false;
            }
            peer_push_pending[stage] = false;
        }

        const bool peer_in = pb_in != nullptr && pb_in->ready && pb_in->dec_ready &&
                             !pb_in->dead && pb_in->pushed_pos[slot] == item.pos;
        if (peer_in) {
            // stall the endpoint's command loop until the push landed, then
            // move it from staging into the boundary the graph reads
            {
                scope_timer fence_timer(timing.stage_fence[stage]);
                if (!ggml_backend_rpc_sync_peer_fence(pb_in->staging_dec[slot], pb_in->dec_ordinals[slot])) {
                    error = "SPD peer fence failed for decode stage " + std::to_string(stage);
                    return false;
                }
            }
            scope_timer copy_timer(timing.stage_copy[stage]);
            ggml_backend_tensor_copy(pb_in->staging_dec[slot], pb_in->inp_dec);
        }

        if (!light_rollback) {
            const uint32_t checkpoint_slot = (uint32_t) (item.pos % rollback_tokens);
            const llama_seq_id checkpoint_seq = (llama_seq_id) checkpoint_slot + 1;
            llama_memory_seq_cp(llama_get_memory(stages[stage]), 0, checkpoint_seq, -1, -1);
            checkpoint_pos[stage][checkpoint_slot] = stage_tail_pos[stage];
        }

        std::vector<float> local_output;
        std::vector<float> & output = static_decode_fast_path
                ? stage_decode_outputs[stage]
                : local_output;
        std::vector<std::vector<float>> captured(anchors.size());
        std::vector<std::pair<size_t, std::vector<float> *>> wanted;
        for (size_t ai = 0; ai < anchors.size(); ++ai) {
            if (anchor_stage(ai) == stage && !anchor_is_host_derived(ai)) {
                wanted.push_back({ ai, &captured[ai] });
            }
        }

        const llama_token * token = stage == 0 ? &item.token : nullptr;
        const float * embd = stage == 0 ? nullptr : item.hidden.data();
        llama_set_spd_peer_io(stages[stage], peer_in, /*skip_out =*/ false,
                /*skip_layer_inp =*/ stage_taps_all_host_derived(stage));
        const bool decoded = decode_stage(stage, token, embd, 1, item.pos, output, wanted, error,
                static_decode_fast_path ? &stage_decode_batches[stage] : nullptr,
                &timing.stage_decode[stage], &timing.stage_read[stage]);
        llama_set_spd_peer_io(stages[stage], false, false, false);
        if (!decoded) {
            return false;
        }
        if (peer_in && llama_spd_peer_inp_tensor(stages[stage]) != pb_in->inp) {
            // a rebuild would have read a different tensor than the copy
            // filled, so that step would be garbage -- surface it loudly
            error = "SPD peer boundary input tensor moved under decode stage " + std::to_string(stage);
            return false;
        }

        // Producer side: ship this boundary straight to the next stage's
        // staging slot. Issued before the host-side tap maths below so the
        // transfer overlaps it. A failure is not fatal -- clearing `dead`
        // ownership on the link puts the pair back on the host path for good.
        if (pb_out != nullptr && pb_out->ready && pb_out->dec_ready && !pb_out->dead) {
            scope_timer push_timer(timing.stage_push[stage]);
            ggml_tensor * out  = llama_spd_peer_out_tensor(stages[stage]);
            ggml_tensor * dst  = pb_out->staging_dec[slot];
            uint64_t ordinal = 0;
            if (out != nullptr && ggml_nbytes(out) == ggml_nbytes(dst) &&
                ggml_backend_rpc_sync_peer_push(out, dst, &ordinal)) {
                pb_out->dec_ordinals[slot] = ordinal;
                pb_out->pushed_pos[slot]   = item.pos;
                peer_push_pending[stage]   = true;
                ++timing.decode_pushes[stage];
            } else {
                pb_out->dead = true;
                fprintf(stderr, "SPD peer: decode push %u -> %u failed, staying on the host path\n",
                        stage, stage + 1);
            }
        }
        if (static_decode_fast_path) {
            item.hidden.swap(output);
        } else {
            item.hidden = std::move(output);
        }
        stage_tail_pos[stage] = item.pos;
        for (auto & captured_item : wanted) {
            item.snap.values[captured_item.first] = std::move(captured[captured_item.first]);
            item.snap.present[captured_item.first] = true;
        }

        // The boundary this stage just produced is the residual entering the
        // next stage's first layer, i.e. the anchor tap at this stage's end.
        // Stage graphs only capture taps at their own input, so without this a
        // token that has completed d stages exposes d anchors where the head
        // was trained on d+1 (reference depth_to_avail: depth d = anchors
        // through the end of stage d-1), and every in-flight speculation row
        // falls back to one aggregation pattern shallower than trained. The
        // tap is the stream mean of the boundary; flat-residual targets have
        // one stream and take a plain copy.
        {
            scope_timer tap_timer(timing.stage_tap[stage]);
            const int32_t end_layer = std::min<int32_t>((int32_t) (stage + 1)*layers_per_stage, n_layers);
            for (size_t ai = 0; ai < anchors.size(); ++ai) {
                if (anchors[ai] != end_layer) {
                    continue;
                }
                const int32_t streams = n_embd_boundary/n_embd;
                std::vector<float> & tap = item.snap.values[ai];
                tap.assign(item.hidden.begin(), item.hidden.begin() + n_embd);
                for (int32_t st = 1; st < streams; ++st) {
                    const float * src = item.hidden.data() + (size_t) st*n_embd;
                    for (int32_t j = 0; j < n_embd; ++j) {
                        tap[j] += src[j];
                    }
                }
                if (streams > 1) {
                    const float scale = 1.0f/(float) streams;
                    for (int32_t j = 0; j < n_embd; ++j) {
                        tap[j] *= scale;
                    }
                }
                item.snap.present[ai] = true;
                break;
            }
        }
        elapsed = seconds_since(start);
        timing.stage_busy[stage] += elapsed;
        if (timing_active_count <= SPD_MAX_STAGE_COUNT) {
            timing.busy_by_active[stage][timing_active_count] += elapsed;
            ++timing.calls_by_active[stage][timing_active_count];
        }
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

    bool generate(const std::vector<llama_token> & prompt, int32_t n_predict, common_spd_result & result) {
        result = {};
        last_error.clear();
        timing = {};
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

        // Decode-side peer boundaries do not inherit prefill's setup: a prompt
        // shorter than one n_batch chunk never takes prefill's peer path at all
        // (it needs count == chunk_size), so nothing would ever be allocated
        // and every decode step would quietly stay on the hairpin. The input
        // tensor a decode push targets is the context's persistent boundary,
        // which exists from context creation, so setup needs no prior decode.
        // Safe on this thread: prefill's workers have joined and the stage
        // threads have not started.
        if (peer_boundaries_enabled) {
            std::mutex setup_mutex;
            for (uint32_t stage = 1; stage < stage_count; ++stage) {
                peer_boundary & pb = peer_links[stage];
                if (!pb.ready && !pb.dead) {
                    setup_peer_boundary(pb, stage, setup_mutex);
                }
            }
        }
        peer_decode_invalidate();

        std::map<llama_pos, snapshot> completed;
        const int32_t retained_begin = std::max(0, n_prompt - (int32_t) stage_count + 1);
        for (int32_t pos = retained_begin; pos < n_prompt; ++pos) {
            completed.emplace(pos, snapshot_at(prompt_anchors, pos));
        }

        std::vector<float> last_hidden(
                prompt_hidden.begin() + (size_t) (n_prompt - 1)*n_embd_boundary,
                prompt_hidden.begin() + (size_t) n_prompt*n_embd_boundary);
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
            scope_timer step_timer(timing.step_total);
            ++result.decode_steps;

            llama_token pending_draft = LLAMA_TOKEN_NULL;
            const llama_pos oldest_needed = pipeline.front().pos - stage_count + 1;
            bool has_pending_draft = false;
            speculation_input local_pending_input;
            speculation_input & pending_input = static_decode_fast_path
                    ? decode_speculation_input
                    : local_pending_input;
            if (oldest_needed >= 0) {
                scope_timer prepare_timer(timing.prepare);
                if (!prepare_speculation(pipeline, completed,
                        has_prev_evicted ? &prev_evicted : nullptr,
                        prev_evicted_pos, pending_input)) {
                    return false;
                }
                has_pending_draft = true;
            }

            const size_t active_count = pipeline.size();
            timing_active_count = active_count;
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

            const auto stage_phase_start = clock_type::now();
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
            const double stage_phase_wall = seconds_since(stage_phase_start);
            timing.stage_wall += stage_phase_wall;
            if (timing_enabled && active_count > 0 && active_count <= SPD_MAX_STAGE_COUNT) {
                double slowest = 0.0;
                for (size_t stage = 0; stage < active_count; ++stage) {
                    slowest = std::max(slowest, elapsed[stage]);
                }
                timing.gap_by_active[active_count] += stage_phase_wall - slowest;
                ++timing.gap_calls[active_count];
            }

            {
                scope_timer sidecar_wait_timer(timing.sidecar_extra);
                if (sidecar_job_submitted) {
                    std::exception_ptr failure = sidecar_worker->wait();
                    if (worker_failure == nullptr && failure != nullptr) {
                        worker_failure = failure;
                    }
                }
                if (speculation_work.valid()) {
                    speculation_ok = speculation_work.get();
                }
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

                    dump_verify(target_pos, result.tokens[generated_index], verified_token);
                    if (result.tokens[generated_index] != verified_token) {
                        result.tokens.resize(generated_index);
                        result.accepted.resize(generated_index);
                        result.tokens.push_back(verified_token);
                        result.accepted.push_back(false);
                        ++result.n_rejected;

                        {
                            scope_timer rollback_timer(timing.rollback);
                            if (!rollback(target_pos)) {
                                return false;
                            }
                        }
                        for (auto it = completed.lower_bound(target_pos); it != completed.end();) {
                            it = completed.erase(it);
                        }

                        std::vector<float> corrected_embedding;
                        if (!embed_token(verified_token, corrected_embedding)) {
                            return false;
                        }
                        pipeline.clear();
                        peer_decode_invalidate();
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
        if (timing_enabled && result.decode_steps > 0) {
            const double other = timing.step_total - timing.prepare - timing.stage_wall -
                    timing.sidecar_extra - timing.head - timing.embed - timing.rollback;
            fprintf(stderr,
                    "SPD timing: steps=%" PRIu64 " decode=%.2fs (%.1f ms/step) rollback_mode=%s | "
                    "prepare %.2fs | stages(wall) %.2fs | sidecar-extra %.2fs | head %.2fs/%" PRIu64 " | "
                    "embed %.2fs/%" PRIu64 " | rollback %.2fs/%" PRIu64 " | other %.2fs\n",
                    result.decode_steps, result.decode_seconds,
                    1e3*result.decode_seconds/(double) result.decode_steps,
                    light_rollback ? "seq_rm" : "seq_cp",
                    timing.prepare, timing.stage_wall, timing.sidecar_extra,
                    timing.head, timing.head_calls,
                    timing.embed, timing.embed_calls,
                    timing.rollback, result.n_rejected,
                    other);
            fprintf(stderr, "SPD timing: sidecar_run %.2fs/%" PRIu64 " calls (%.1f ms/call)\n",
                    timing.sidecar_run, timing.sidecar_calls,
                    timing.sidecar_calls > 0 ? 1e3*timing.sidecar_run/(double) timing.sidecar_calls : 0.0);
            for (uint32_t stage = 0; stage < stage_count; ++stage) {
                if (timing.stage_calls[stage] == 0) {
                    continue;
                }
                fprintf(stderr,
                        "SPD timing: stage %u: busy %.2fs/%" PRIu64 " calls (%.1f ms/call) | "
                        "decode %.2fs | read %.2fs | lock-wait %.2fs | peer-push %" PRIu64 " | graphs_reused %d\n",
                        stage, timing.stage_busy[stage], timing.stage_calls[stage],
                        1e3*timing.stage_busy[stage]/(double) timing.stage_calls[stage],
                        timing.stage_decode[stage], timing.stage_read[stage], timing.stage_lock[stage],
                        timing.decode_pushes[stage],
                        llama_perf_context(stages[stage]).n_reused);
                const double calls = (double) timing.stage_calls[stage];
                fprintf(stderr,
                        "SPD timing: stage %u outside-decode ms/call: guard %.2f | fence %.2f | copy %.2f | "
                        "push %.2f | tap %.2f | sum %.2f (busy-decode %.2f)\n",
                        stage,
                        1e3*timing.stage_guard[stage]/calls, 1e3*timing.stage_fence[stage]/calls,
                        1e3*timing.stage_copy[stage]/calls,  1e3*timing.stage_push[stage]/calls,
                        1e3*timing.stage_tap[stage]/calls,
                        1e3*(timing.stage_guard[stage] + timing.stage_fence[stage] + timing.stage_copy[stage] +
                             timing.stage_push[stage] + timing.stage_tap[stage])/calls,
                        1e3*(timing.stage_busy[stage] - timing.stage_decode[stage])/calls);
            }
            fprintf(stderr, "SPD timing: graphs_reused sidecar %d | head %d | embed %d\n",
                    llama_perf_context(sidecar).n_reused,
                    llama_perf_context(head).n_reused,
                    llama_perf_context(embed).n_reused);
            for (uint32_t stage = 0; stage < stage_count; ++stage) {
                char row[512];
                size_t off = (size_t) snprintf(row, sizeof(row), "SPD timing: stage %u ms/call by active:", stage);
                for (uint32_t a = 1; a <= stage_count && off < sizeof(row) - 32; ++a) {
                    const uint64_t n = timing.calls_by_active[stage][a];
                    if (n == 0) {
                        off += (size_t) snprintf(row + off, sizeof(row) - off, " a%u=-", a);
                    } else {
                        off += (size_t) snprintf(row + off, sizeof(row) - off, " a%u=%.1f/%" PRIu64,
                                a, 1e3*timing.busy_by_active[stage][a]/(double) n, n);
                    }
                }
                fprintf(stderr, "%s\n", row);
            }
            {
                char row[512];
                size_t off = (size_t) snprintf(row, sizeof(row), "SPD timing: orchestration gap ms/step by active:");
                double gap_total = 0.0;
                uint64_t gap_steps = 0;
                for (uint32_t a = 1; a <= stage_count && off < sizeof(row) - 32; ++a) {
                    const uint64_t n = timing.gap_calls[a];
                    gap_total += timing.gap_by_active[a];
                    gap_steps += n;
                    if (n == 0) {
                        off += (size_t) snprintf(row + off, sizeof(row) - off, " a%u=-", a);
                    } else {
                        off += (size_t) snprintf(row + off, sizeof(row) - off, " a%u=%.2f/%" PRIu64,
                                a, 1e3*timing.gap_by_active[a]/(double) n, n);
                    }
                }
                fprintf(stderr, "%s | total %.2fs (%.2f ms/step)\n", row, gap_total,
                        gap_steps > 0 ? 1e3*gap_total/(double) gap_steps : 0.0);
            }
        }
        result.tokens.resize(n_predict);
        result.accepted.resize(n_predict);
        return true;
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
