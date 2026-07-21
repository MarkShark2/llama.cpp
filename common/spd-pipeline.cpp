#include "spd-pipeline.h"

#include "ggml.h"
#include "llama-ext.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <future>
#include <map>
#include <mutex>
#include <utility>

namespace {

constexpr uint32_t SPD_STAGE_COUNT = COMMON_SPD_STAGE_COUNT;
constexpr uint32_t SPD_ROLLBACK_TOKENS = SPD_STAGE_COUNT - 1;

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

    batch_storage(
            int32_t n_tokens,
            const llama_token * token_data,
            const float * embedding_data,
            int32_t n_embd,
            int32_t n_pos_per_embd,
            llama_pos first_pos,
            bool output_all) {
        batch.n_tokens = n_tokens;

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
        batch.n_tokens = (int32_t) selectors.size();
        tokens = selectors;
        embeddings = features;
        positions = pos;
        logits.resize(selectors.size(), 0);
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

    llama_model * model_target;
    llama_model * model_spd;
    common_spd_params params;

    int32_t n_embd = 0;
    int32_t n_vocab = 0;
    int32_t n_layers = 0;
    int32_t layers_per_stage = 0;
    int32_t target_n_pos_per_embd = 1;
    int32_t sidecar_n_embd_inp = 0;

    std::vector<int32_t> anchors;
    std::array<llama_context *, SPD_STAGE_COUNT> stages = {};
    std::array<std::array<llama_pos, SPD_ROLLBACK_TOKENS>, SPD_STAGE_COUNT> checkpoint_pos = {};
    std::array<llama_pos, SPD_STAGE_COUNT> stage_tail_pos = {};
    std::array<std::vector<size_t>, SPD_STAGE_COUNT> stage_resources;
    std::vector<std::unique_ptr<std::mutex>> resource_mutexes;
    llama_context * head = nullptr;
    llama_context * embed = nullptr;
    llama_context * sidecar = nullptr;

    std::string last_error;
    bool ready = false;
    bool used = false;

    impl(llama_model * model_target, llama_model * model_spd, const common_spd_params & params)
        : model_target(model_target), model_spd(model_spd), params(params) {
        initialize();
    }

    ~impl() {
        llama_free(sidecar);
        llama_free(embed);
        llama_free(head);
        for (llama_context * ctx : stages) {
            llama_free(ctx);
        }
    }

    void fail(std::string message) {
        if (last_error.empty()) {
            last_error = std::move(message);
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
        if (params.n_ctx > UINT32_MAX/SPD_STAGE_COUNT) {
            fail("SPD context length is too large for rollback sequence allocation");
            return;
        }

        n_embd = llama_model_n_embd(model_target);
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_target));
        n_layers = llama_model_n_layer(model_target);
        layers_per_stage = n_layers / (int32_t) SPD_STAGE_COUNT;
        target_n_pos_per_embd = llama_model_n_pos_per_embd(model_target);
        sidecar_n_embd_inp = llama_model_n_embd_inp(model_spd);

        const uint32_t n_anchor = llama_model_target_layer_ids_n(model_spd);
        const int32_t * anchor_data = llama_model_target_layer_ids(model_spd);
        if (n_layers <= 0 || n_layers % (int32_t) SPD_STAGE_COUNT != 0) {
            fail("SPD target layer count must divide evenly into eight stages");
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
        // touched by each logical stage and lock shared resources in a stable
        // order during stage advancement.
        std::map<std::string, size_t> resource_ids;
        for (uint32_t stage = 0; stage < SPD_STAGE_COUNT; ++stage) {
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

        for (uint32_t stage = 0; stage < SPD_STAGE_COUNT; ++stage) {
            llama_context_params cp = make_context_params(params.n_batch);
            cp.ctx_type = LLAMA_CONTEXT_TYPE_SPD_STAGE;
            cp.spd_stage = stage;
            cp.spd_stage_count = SPD_STAGE_COUNT;
            cp.n_seq_max = SPD_STAGE_COUNT;
            // llama_context derives n_ctx_seq by dividing total n_ctx by
            // n_seq_max. SPD uses the extra sequence IDs as rollback aliases,
            // but seq 0 must still retain the caller-requested context length.
            // Without this expansion an 8192-token SPD context silently became
            // an effective 1024-token context and diverged on longer prompts.
            cp.n_ctx = params.n_ctx*SPD_STAGE_COUNT;
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
        }

        for (int32_t anchor : anchors) {
            const uint32_t stage = std::min<uint32_t>(anchor / layers_per_stage, SPD_STAGE_COUNT - 1);
            llama_set_embeddings_layer_inp(stages[stage], anchor, true);
        }

        llama_context_params hp = make_context_params(std::max<uint32_t>(1, params.n_batch));
        hp.ctx_type = LLAMA_CONTEXT_TYPE_SPD_HEAD;
        head = llama_init_from_model(model_target, hp);
        if (head == nullptr) {
            fail("failed to initialize the target SPD head");
            return;
        }

        llama_context_params ep = make_context_params(1);
        ep.ctx_type = LLAMA_CONTEXT_TYPE_SPD_EMBED;
        ep.embeddings = true;
        embed = llama_init_from_model(model_target, ep);
        if (embed == nullptr) {
            fail("failed to initialize the target token-embedding context");
            return;
        }

        llama_context_params sp = make_context_params(
                std::max<uint32_t>(params.n_batch, SPD_STAGE_COUNT + 1), true);
        sidecar = llama_init_from_model(model_spd, sp);
        if (sidecar == nullptr) {
            fail("failed to initialize the SPD sidecar context");
            return;
        }
        ready = true;
    }

    void reset_memories() {
        for (llama_context * ctx : stages) {
            llama_memory_clear(llama_get_memory(ctx), true);
        }
        llama_memory_clear(llama_get_memory(sidecar), true);
        for (auto & positions : checkpoint_pos) {
            positions.fill(-1);
        }
        stage_tail_pos.fill(-1);
    }

    bool decode_stage(
            uint32_t stage,
            const llama_token * tokens,
            const float * embeddings,
            int32_t n_tokens,
            llama_pos first_pos,
            std::vector<float> & output,
            std::vector<std::pair<size_t, std::vector<float> *>> anchor_outputs,
            std::string & error) {
        batch_storage storage(n_tokens, tokens, embeddings, n_embd, target_n_pos_per_embd, first_pos, true);
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
        batch_storage storage(1, &token, nullptr, 0, target_n_pos_per_embd, 0, true);
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
        batch_storage storage(1, nullptr, hidden.data(), n_embd, target_n_pos_per_embd, 0, true);
        if (llama_decode(head, storage.batch) != 0) {
            fail("target SPD head decode failed");
            return false;
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
            llama_token * sampled) {
        batch_storage storage(selectors, features, sidecar_n_embd_inp, positions);
        if (llama_decode(sidecar, storage.batch) != 0) {
            fail("SPD sidecar decode failed");
            return false;
        }
        if (sampled != nullptr) {
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
        const int32_t prefill_len = std::max(0, n_tokens - (int32_t) SPD_STAGE_COUNT + 1);
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

    bool speculate(
            const std::vector<entry> & pipeline,
            const std::map<llama_pos, snapshot> & completed,
            const snapshot * prev_evicted,
            llama_pos prev_evicted_pos,
            llama_token & sampled) {
        if (pipeline.empty()) {
            fail("SPD pipeline is empty before speculation");
            return false;
        }

        const llama_pos newest_pos = pipeline.front().pos;
        const llama_pos oldest_needed = newest_pos - SPD_STAGE_COUNT + 1;
        if (oldest_needed < 0) {
            fail("SPD prompt is too short to construct a complete speculation window");
            return false;
        }

        std::map<llama_pos, const snapshot *> active;
        for (const entry & item : pipeline) {
            active[item.pos] = &item.snap;
        }

        std::vector<llama_token> selectors;
        std::vector<llama_pos> positions;
        std::vector<float> features;
        const bool has_evicted = prev_evicted != nullptr;

        if (has_evicted) {
            const size_t selector = choose_anchor(*prev_evicted, SPD_STAGE_COUNT);
            selectors.push_back((llama_token) selector);
            positions.push_back(prev_evicted_pos);
            append_feature_row(features, *prev_evicted, selector);
        }

        for (llama_pos pos = oldest_needed; pos <= newest_pos; ++pos) {
            const snapshot * snap = nullptr;
            auto active_it = active.find(pos);
            if (active_it != active.end()) {
                snap = active_it->second;
            } else {
                auto complete_it = completed.find(pos);
                if (complete_it == completed.end()) {
                    fail("missing completed SPD snapshot at position " + std::to_string(pos));
                    return false;
                }
                snap = &complete_it->second;
            }

            const int32_t nominal_depth = newest_pos - pos;
            const int32_t search_hi = nominal_depth == 0 ? 0 : has_evicted ? SPD_STAGE_COUNT - 1 : SPD_STAGE_COUNT;
            const size_t selector = choose_anchor(*snap, search_hi);
            selectors.push_back((llama_token) selector);
            positions.push_back(pos);
            append_feature_row(features, *snap, selector);
        }

        const llama_pos min_pos = positions.front();
        if (!llama_memory_seq_rm(llama_get_memory(sidecar), 0, min_pos, -1)) {
            fail("failed to crop SPD sidecar cache at position " + std::to_string(min_pos));
            return false;
        }
        return sidecar_decode(selectors, features, positions, &sampled);
    }

    bool rollback(llama_pos target_pos) {
        for (uint32_t stage = 0; stage < SPD_STAGE_COUNT; ++stage) {
            const llama_pos restore_pos = target_pos - 1;
            if (stage_tail_pos[stage] == restore_pos) {
                continue;
            }

            llama_seq_id restore_seq = -1;
            for (uint32_t i = 0; i < SPD_ROLLBACK_TOKENS; ++i) {
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

    bool prefill_target(
            const std::vector<llama_token> & prompt,
            std::vector<float> & hidden,
            std::vector<std::vector<float>> & anchor_data) {
        const int32_t n_prompt = (int32_t) prompt.size();
        const int32_t chunk_size = std::max<int32_t>(1, params.n_batch);
        hidden.resize((size_t) n_prompt*n_embd);
        anchor_data.assign(anchors.size(), std::vector<float>((size_t) n_prompt*n_embd));

        for (uint32_t stage = 0; stage < SPD_STAGE_COUNT; ++stage) {
            for (int32_t begin = 0; begin < n_prompt; begin += chunk_size) {
                const int32_t count = std::min(chunk_size, n_prompt - begin);
                std::vector<float> chunk_output;
                std::vector<std::vector<float>> chunk_anchors(anchors.size());
                std::vector<std::pair<size_t, std::vector<float> *>> wanted;
                for (size_t ai = 0; ai < anchors.size(); ++ai) {
                    if ((uint32_t) std::min<int32_t>(anchors[ai] / layers_per_stage, SPD_STAGE_COUNT - 1) == stage) {
                        wanted.push_back({ ai, &chunk_anchors[ai] });
                    }
                }

                std::string error;
                const llama_token * tokens = stage == 0 ? prompt.data() + begin : nullptr;
                const float * embeddings = stage == 0 ? nullptr : hidden.data() + (size_t) begin*n_embd;
                if (!decode_stage(stage, tokens, embeddings, count, begin, chunk_output, wanted, error)) {
                    fail(std::move(error));
                    return false;
                }
                std::copy(chunk_output.begin(), chunk_output.end(), hidden.begin() + (size_t) begin*n_embd);
                for (auto & item : wanted) {
                    std::copy(chunk_anchors[item.first].begin(), chunk_anchors[item.first].end(),
                            anchor_data[item.first].begin() + (size_t) begin*n_embd);
                }
            }
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
        std::vector<std::unique_lock<std::mutex>> resource_locks;
        resource_locks.reserve(stage_resources[stage].size());
        for (size_t resource : stage_resources[stage]) {
            resource_locks.emplace_back(*resource_mutexes[resource]);
        }
        const auto start = clock_type::now();

        const uint32_t checkpoint_slot = (uint32_t) (item.pos % SPD_ROLLBACK_TOKENS);
        const llama_seq_id checkpoint_seq = (llama_seq_id) checkpoint_slot + 1;
        llama_memory_seq_cp(llama_get_memory(stages[stage]), 0, checkpoint_seq, -1, -1);
        checkpoint_pos[stage][checkpoint_slot] = stage_tail_pos[stage];

        std::vector<float> output;
        std::vector<std::vector<float>> captured(anchors.size());
        std::vector<std::pair<size_t, std::vector<float> *>> wanted;
        for (size_t ai = 0; ai < anchors.size(); ++ai) {
            if ((uint32_t) std::min<int32_t>(anchors[ai] / layers_per_stage, SPD_STAGE_COUNT - 1) == stage) {
                wanted.push_back({ ai, &captured[ai] });
            }
        }

        const llama_token * token = stage == 0 ? &item.token : nullptr;
        const float * embd = stage == 0 ? nullptr : item.hidden.data();
        if (!decode_stage(stage, token, embd, 1, item.pos, output, wanted, error)) {
            return false;
        }
        item.hidden = std::move(output);
        stage_tail_pos[stage] = item.pos;
        for (auto & captured_item : wanted) {
            item.snap.values[captured_item.first] = std::move(captured[captured_item.first]);
            item.snap.present[captured_item.first] = true;
        }
        elapsed = seconds_since(start);
        return true;
    }

    void synchronize_all() {
        for (llama_context * ctx : stages) {
            llama_synchronize(ctx);
        }
        llama_synchronize(head);
        llama_synchronize(embed);
        llama_synchronize(sidecar);
    }

    bool generate(const std::vector<llama_token> & prompt, int32_t n_predict, common_spd_result & result) {
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
        if ((uint64_t) prompt.size() + n_predict + SPD_STAGE_COUNT >= params.n_ctx) {
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

        std::map<llama_pos, snapshot> completed;
        const int32_t retained_begin = std::max(0, n_prompt - (int32_t) SPD_STAGE_COUNT + 1);
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
            const llama_pos oldest_needed = pipeline.front().pos - SPD_STAGE_COUNT + 1;
            bool has_pending_draft = false;
            if (oldest_needed >= 0) {
                if (!speculate(pipeline, completed,
                        has_prev_evicted ? &prev_evicted : nullptr,
                        prev_evicted_pos, pending_draft)) {
                    return false;
                }
                has_pending_draft = true;
            }

            const size_t active_count = pipeline.size();
            std::vector<double> elapsed(active_count, 0.0);
            std::vector<std::string> errors(active_count);
            std::vector<uint8_t> ok(active_count, 0);

            if (params.parallel_stages && active_count > 1) {
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

            for (size_t stage = 0; stage < active_count; ++stage) {
                result.stage_compute_seconds += elapsed[stage];
                if (!ok[stage]) {
                    fail(errors[stage]);
                    return false;
                }
            }

            if (pipeline.size() >= SPD_STAGE_COUNT) {
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

            // Drafts can run seven positions ahead of the verified frontier. A
            // rejection restarts at that frontier and must be able to rebuild
            // the preceding eight-row sidecar window, so retention follows the
            // verified position rather than the newest speculative position.
            const llama_pos keep_from = std::max<llama_pos>(0, verified_up_to - SPD_STAGE_COUNT);
            for (auto it = completed.begin(); it != completed.end() && it->first < keep_from;) {
                it = completed.erase(it);
            }
        }

        result.decode_seconds = seconds_since(decode_start);
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
