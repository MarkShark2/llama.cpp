#include "models.h"
#include "gguf.h"
#include "llama-kv-cache.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

extern "C" {
void ggml_mul_mat_id_set_k_tri(struct ggml_tensor * a, int32_t k_block);
}

namespace {

class llm_graph_input_spd : public llm_graph_input_i {
public:
    llm_graph_input_spd(int64_t n_embd, int64_t n_aggr) : n_embd(n_embd), n_aggr(n_aggr) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(tokens != nullptr);
        GGML_ASSERT(embd != nullptr);
        GGML_ASSERT(embd->ne[0] == n_embd);

        if (ubatch->embd != nullptr) {
            GGML_ASSERT(ubatch->token != nullptr);
            for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
                GGML_ASSERT(ubatch->token[i] >= 0 && ubatch->token[i] < n_aggr);
            }
            ggml_backend_tensor_set(tokens, ubatch->token, 0, ubatch->n_tokens*ggml_element_size(tokens));
            ggml_backend_tensor_set(embd, ubatch->embd, 0, ubatch->n_tokens*n_embd*ggml_element_size(embd));
            return;
        }

        fallback_ids.assign(ubatch->n_tokens, n_aggr - 1);
        fallback_embd.assign(ubatch->n_tokens*n_embd, 0.0f);
        ggml_backend_tensor_set(tokens, fallback_ids.data(), 0, fallback_ids.size()*ggml_element_size(tokens));
        ggml_backend_tensor_set(embd, fallback_embd.data(), 0, fallback_embd.size()*ggml_element_size(embd));
    }

    bool can_reuse(const llm_graph_params & params) override {
        return params.ubatch.token != nullptr &&
               params.ubatch.embd  != nullptr &&
               tokens != nullptr &&
               embd   != nullptr &&
               tokens->ne[0] == params.ubatch.n_tokens &&
               embd->ne[1]   == params.ubatch.n_tokens;
    }

    ggml_tensor * tokens = nullptr;
    ggml_tensor * embd = nullptr;

private:
    const int64_t n_embd;
    const int64_t n_aggr;
    std::vector<llama_token> fallback_ids;
    std::vector<float> fallback_embd;
};

// Shared-block bank: pattern e aggregates as (sum_{k<=e} blk_k x_k) * scale_e +
// bias_e, so a position's sum only ever grows by the anchors that arrived
// since the last step. The sums live in a device-resident ring
// (cparams.spd_aggr_state) and the host sends only the new anchor vectors,
// through the plan (llama_spd_aggr_set_plan) rather than batch.embd. The
// batch carries the pattern selectors as tokens.
class llm_graph_input_spd_shared : public llm_graph_input_i {
public:
    llm_graph_input_spd_shared(int64_t n_embd, int64_t n_aggr, const llama_spd_aggr_plan * plan, int64_t c, int64_t changed_count)
        : n_embd(n_embd), n_aggr(n_aggr), plan(plan), c(c), changed_count(changed_count) {}

    // widest new-anchor run in the ubatch; every row is padded to it
    static int64_t width(const llama_ubatch & ubatch, const llama_spd_aggr_plan * plan, int64_t n_aggr) {
        if (ubatch.token == nullptr || ubatch.pos == nullptr || plan == nullptr) {
            return n_aggr;
        }
        int32_t widest = 1;
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            const int32_t row = plan->find(ubatch.pos[i]);
            if (row >= 0) {
                widest = std::max(widest, plan->n_new[row]);
            }
        }
        return widest;
    }

    static int64_t count_changed(const llama_ubatch & ubatch, const llama_spd_aggr_plan * plan) {
        int64_t count = 0;
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            const int32_t row = ubatch.pos && plan ? plan->find(ubatch.pos[i]) : -1;
            count += row < 0 || plan->n_new[row] > 0;
        }
        return count;
    }

    void set_input(const llama_ubatch * ubatch) override {
        const uint32_t n_rows  = ubatch->n_tokens;
        const int32_t  scratch = (int32_t) plan->n_slots;

        pattern.assign(n_rows, (int32_t) n_aggr - 1);
        feat_data.assign((size_t) n_rows*c*n_embd, 0.0f);
        ids_data.assign((size_t) n_rows*c, 0);
        slot_data.assign(n_rows, scratch);
        keep_data.assign(n_rows, 0.0f);
        raw_ids_data.assign((size_t) n_rows*c, (int32_t) n_aggr);
        raw_valid_data.assign((size_t) n_rows*n_aggr, 0.0f);
        changed_data.clear();
        position_data.resize((size_t) n_rows*c);
        for (uint32_t i = 0; i < n_rows; ++i) {
            const int32_t row = ubatch->pos ? plan->find(ubatch->pos[i]) : -1;
            if (row < 0 || plan->n_new[row] > 0) {
                changed_data.push_back(i);
            }
            std::fill_n(position_data.data() + i*c, c, ubatch->pos ? ubatch->pos[i] : 0);
        }

        for (uint32_t i = 0; ubatch->token != nullptr && i < n_rows; ++i) {
            const int32_t row = plan->find(ubatch->pos[i]);
            if (row < 0) {
                // not described by the plan: contributes nothing and leaves the
                // ring alone (warmup, or a decode the host did not plan)
                continue;
            }
            const int32_t held  = plan->held[row];
            const int32_t first = std::max(held, 0);
            const int32_t n_new = plan->n_new[row];
            GGML_ASSERT(n_new <= c);
            GGML_ASSERT(ubatch->token[i] == first + n_new - 1);
            GGML_ASSERT(ubatch->token[i] >= 0 && ubatch->token[i] < n_aggr);

            pattern[i] = ubatch->token[i];
            std::fill_n(raw_valid_data.data() + i*n_aggr, n_aggr, -INFINITY);
            for (int32_t k = 0; k <= pattern[i]; ++k) {
                raw_valid_data[(size_t) i*n_aggr + k] = 0.0f;
            }
            if (held >= 0) {
                slot_data[i] = (int32_t) (ubatch->pos[i] % (llama_pos) plan->n_slots);
                keep_data[i] = held > 0 ? 1.0f : 0.0f;
            }
            for (int32_t t = 0; t < n_new; ++t) {
                ids_data[(size_t) i*c + t] = first + t;
                raw_ids_data[(size_t) i*c + t] = first + t;
            }
            if (n_new > 0) {
                std::memcpy(feat_data.data() + (size_t) i*c*n_embd, plan->feat.data() + plan->off[row],
                        (size_t) n_new*n_embd*sizeof(float));
            }
        }

        ggml_backend_tensor_set(tokens, pattern.data(),   0, ggml_nbytes(tokens));
        ggml_backend_tensor_set(feat,   feat_data.data(), 0, ggml_nbytes(feat));
        ggml_backend_tensor_set(ids,    ids_data.data(),  0, ggml_nbytes(ids));
        ggml_backend_tensor_set(slots,  slot_data.data(), 0, ggml_nbytes(slots));
        ggml_backend_tensor_set(keep,   keep_data.data(), 0, ggml_nbytes(keep));
        if (changed) {
            ggml_backend_tensor_set(changed, changed_data.data(), 0, ggml_nbytes(changed));
        }
        if (raw_ids) {
            ggml_backend_tensor_set(raw_positions, position_data.data(), 0, ggml_nbytes(raw_positions));
            ggml_backend_tensor_set(raw_ids, raw_ids_data.data(), 0, ggml_nbytes(raw_ids));
            ggml_backend_tensor_set(raw_valid, raw_valid_data.data(), 0, ggml_nbytes(raw_valid));
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        return tokens != nullptr &&
               tokens->ne[0] == params.ubatch.n_tokens &&
               params.cparams.spd_aggr_plan == plan &&
               width(params.ubatch, plan, n_aggr) == c &&
               count_changed(params.ubatch, plan) == changed_count;
    }

    ggml_tensor * tokens = nullptr; // I32 [n_rows]            pattern selector
    ggml_tensor * feat   = nullptr; // F32 [n_embd, c, n_rows] new anchors, zero-padded
    ggml_tensor * ids    = nullptr; // I32 [c, n_rows]         their anchor indices
    ggml_tensor * slots  = nullptr; // I32 [n_rows]            ring slot, n_slots = scratch
    ggml_tensor * keep   = nullptr; // F32 [1, n_rows]         0 = start the sum over
    ggml_tensor * changed = nullptr;
    ggml_tensor * raw_positions = nullptr;
    ggml_tensor * raw_ids = nullptr;
    ggml_tensor * raw_valid = nullptr;

private:
    const int64_t n_embd;
    const int64_t n_aggr;
    const llama_spd_aggr_plan * plan;
    const int64_t c;
    const int64_t changed_count;
    std::vector<int32_t> pattern;
    std::vector<float>   feat_data;
    std::vector<int32_t> ids_data;
    std::vector<int32_t> slot_data;
    std::vector<float>   keep_data;
    std::vector<int32_t> changed_data;
    std::vector<int32_t> position_data;
    std::vector<int32_t> raw_ids_data;
    std::vector<float> raw_valid_data;
};

}

void llama_model_spd::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    // qwen35-style sidecars carry rope sections (IMROPE); others use
    // standard NEOX rope and omit them
    hparams.rope_sections.fill(0);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);
    ml.get_key(LLM_KV_SPD_CHECKPOINT_VERSION, checkpoint_version);
    ml.get_key(LLM_KV_SPD_STAGE_COUNT, stage_count);
    ml.get_key(LLM_KV_SPD_USE_DEEPEST, use_deepest);

    const int64_t head_key = gguf_find_key(ml.metadata, "spd.head_architecture");
    if (head_key >= 0) {
        if (gguf_get_kv_type(ml.metadata, head_key) != GGUF_TYPE_STRING) {
            throw std::runtime_error("SPD head_architecture must be a string");
        }
        head_architecture = gguf_get_val_str(ml.metadata, head_key);
        if (head_architecture != "glm_final_raw_anchor_reader_v1" && head_architecture != "glm_bank_ffn_norm_v1") {
            throw std::runtime_error("unsupported SPD reader architecture: " + head_architecture);
        }
        auto get_u32 = [&](const char * name) {
            const int64_t key = gguf_find_key(ml.metadata, name);
            if (key < 0 || gguf_get_kv_type(ml.metadata, key) != GGUF_TYPE_UINT32) {
                throw std::runtime_error(std::string("SPD requires uint32 metadata: ") + name);
            }
            return gguf_get_val_u32(ml.metadata, key);
        };
        auto get_f32 = [&](const char * name) {
            const int64_t key = gguf_find_key(ml.metadata, name);
            if (key < 0 || gguf_get_kv_type(ml.metadata, key) != GGUF_TYPE_FLOAT32) {
                throw std::runtime_error(std::string("SPD requires float32 metadata: ") + name);
            }
            return gguf_get_val_f32(ml.metadata, key);
        };
        const int64_t mask_key = gguf_find_key(ml.metadata, "spd.mask_semantics");
        if (mask_key < 0 || gguf_get_kv_type(ml.metadata, mask_key) != GGUF_TYPE_STRING ||
                std::strcmp(gguf_get_val_str(ml.metadata, mask_key), "pre_step_snapshot_v1") != 0) {
            throw std::runtime_error("SPD readers require pre_step_snapshot_v1 masks");
        }
        reader_head_count = get_u32("spd.reader.head_count");
        reader_head_dim = get_u32("spd.reader.head_dim");
        reader_rope_freq_base = get_f32("spd.reader.rope_freq_base");
        const float reader_eps = get_f32("spd.reader.rms_norm_eps");
        raw_head_count = get_u32("spd.raw_reader.head_count");
        raw_head_dim = get_u32("spd.raw_reader.head_dim");
        if (reader_head_count != 8 || reader_head_dim != 64 || raw_head_count != 4 || raw_head_dim != 32 ||
                reader_rope_freq_base != 10000.0f || reader_eps != hparams.f_norm_rms_eps || hparams.n_layer() != 2) {
            throw std::runtime_error("unsupported SPD reader dimensions, rotary base or normalization");
        }
        if (head_architecture == "glm_bank_ffn_norm_v1") {
            bank_correction_rank = get_u32("spd.bank_correction_rank");
            if (bank_correction_rank != 64) {
                throw std::runtime_error("SPD bank correction requires rank 64");
            }
        }
    }

    const int64_t target_arch_key = gguf_find_key(ml.metadata, "spd.target_architecture");
    if (target_arch_key >= 0 && gguf_get_kv_type(ml.metadata, target_arch_key) != GGUF_TYPE_STRING) {
        throw std::runtime_error("SPD target_architecture must be a string");
    }
    const int64_t stage_layers_key = gguf_find_key(ml.metadata, "spd.stage_layers");
    if (stage_layers_key >= 0) {
        if (gguf_get_kv_type(ml.metadata, stage_layers_key) != GGUF_TYPE_ARRAY ||
                gguf_get_arr_type(ml.metadata, stage_layers_key) != GGUF_TYPE_UINT32) {
            throw std::runtime_error("SPD stage_layers must be a uint32 array");
        }
        ml.get_arr("spd.stage_layers", stage_layers);
        if (stage_layers.size() != stage_count ||
                std::find(stage_layers.begin(), stage_layers.end(), 0) != stage_layers.end()) {
            throw std::runtime_error("SPD stage_layers must contain one positive layer count per stage");
        }
    }

    // The sidecar is trained on fixed-length windows cut out of the corpus
    // (train_offline.py --chunk), so it has never attended across more than
    // that many positions. RoPE is relative, so *where* a training window sat
    // in its document is not something the head could have learned -- the span
    // is the whole of it. Serving it against a full context therefore runs it
    // outside its training distribution as soon as the conversation is longer
    // than the chunk: on the DSV4 head (chunk 1024) measured acceptance fell
    // from 34-41% below ~8k of context to 2-7% above ~18k, which is the
    // difference between SPD being a 20% speedup and a slowdown.
    //
    // So bound the sidecar's attention to the span it was trained on.
    // LLAMA_SWA_TYPE_STANDARD masks every key more than n_swa positions back,
    // which is exactly the training window. The per-layer SWA pattern is left
    // dense on purpose: is_swa(il) also selects the separate SWA rope
    // frequency, and a windowed-but-not-interleaved model wants one plain
    // windowed cache, not the base/SWA pair llama_kv_cache_iswa builds.
    uint32_t train_span = 0;
    ml.get_key(LLM_KV_SPD_TRAIN_SPAN, train_span, false);
    if (const char * value = std::getenv("LLAMA_SPD_SPAN")) {
        // sidecars converted before the key existed do not carry a span, and
        // the window is worth sweeping against a fixed head; 0 disables it
        train_span = (uint32_t) std::max(0, std::atoi(value));
    }

    hparams.n_swa    = train_span;
    hparams.swa_type = train_span > 0 ? LLAMA_SWA_TYPE_STANDARD : LLAMA_SWA_TYPE_NONE;

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("SPD model requires target_layers in GGUF metadata");
    }
    if (checkpoint_version != 11) {
        throw std::runtime_error("SPD model requires checkpoint version 11");
    }
    if (stage_count == 0 || target_layer_ids.empty()) {
        throw std::runtime_error("SPD model requires non-empty target stages and aggregation anchors");
    }
    if (!use_deepest) {
        throw std::runtime_error("SPD model requires a checkpoint trained with deepest snapshots");
    }
    if (!std::is_sorted(target_layer_ids.begin(), target_layer_ids.end()) || target_layer_ids.front() != 0) {
        throw std::runtime_error("SPD target_layers must be sorted and begin at zero");
    }

    hparams.n_embd_inp_impl = hparams.n_embd * target_layer_ids.size();
    type = LLM_TYPE_UNKNOWN;

    LLAMA_LOG_INFO("%s: SPD checkpoint v%u, stages = %u, aggregation types = %zu\n",
            __func__, checkpoint_version, stage_count, target_layer_ids.size());
    if (hparams.n_swa > 0) {
        LLAMA_LOG_INFO("%s: SPD trained attention span = %u (sidecar attention is windowed)\n",
                __func__, hparams.n_swa);
    } else {
        LLAMA_LOG_WARN("%s: SPD sidecar declares no trained attention span, so it will attend over "
                "the whole context. Acceptance collapses once a request is longer than the window "
                "the head was trained on. Reconvert with tools/spd/convert_spd_to_gguf.py, or set "
                "LLAMA_SPD_SPAN to the trainer's --chunk.\n", __func__);
    }
}

void llama_model_spd::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_aggr = target_layer_ids.size();
    if (ml->get_tensor_meta(tn(LLM_TENSOR_SPD_AGGR_BLK, "weight").str().c_str()) != nullptr) {
        spd_aggr_blk   = create_tensor(tn(LLM_TENSOR_SPD_AGGR_BLK,   "weight"), { n_embd, n_embd, n_aggr }, 0);
        // no ".weight" suffix: llama-quantize keeps these f32 the way it keeps d2t
        spd_aggr_scale = create_tensor(tn(LLM_TENSOR_SPD_AGGR_SCALE), { n_embd, n_aggr }, 0);
        spd_aggr_bias  = create_tensor(tn(LLM_TENSOR_SPD_AGGR_BIAS),  { n_embd, n_aggr }, 0);
    } else {
        spd_aggr = create_tensor(tn(LLM_TENSOR_SPD_AGGR, "weight"), { n_embd*n_aggr, n_embd, n_aggr }, 0);
    }

    if (!head_architecture.empty()) {
        if (spd_aggr_blk == nullptr) {
            throw std::runtime_error("SPD readers require the shared aggregation bank");
        }
        const int64_t reader_width = reader_head_count*reader_head_dim;
        const int64_t raw_width = raw_head_count*raw_head_dim;
        spd_target_mem_norm = create_tensor(tn(LLM_TENSOR_SPD_TARGET_MEM_NORM, "weight"), { n_embd }, 0);
        spd_target_k = create_tensor(tn(LLM_TENSOR_SPD_TARGET_K, "weight"), { n_embd, reader_width }, 0);
        spd_target_v = create_tensor(tn(LLM_TENSOR_SPD_TARGET_V, "weight"), { n_embd, reader_width }, 0);
        spd_target_k_norm = create_tensor(tn(LLM_TENSOR_SPD_TARGET_K_NORM, "weight"), { reader_head_dim }, 0);
        spd_raw_norm = create_tensor(tn(LLM_TENSOR_SPD_RAW_NORM), { n_embd, n_aggr }, 0);
        spd_raw_k = create_tensor(tn(LLM_TENSOR_SPD_RAW_K, "weight"), { n_embd, raw_width }, 0);
        spd_raw_v = create_tensor(tn(LLM_TENSOR_SPD_RAW_V, "weight"), { n_embd, raw_width }, 0);
        spd_raw_stage_k = create_tensor(tn(LLM_TENSOR_SPD_RAW_STAGE_K), { raw_width, n_aggr }, 0);
        spd_raw_stage_v = create_tensor(tn(LLM_TENSOR_SPD_RAW_STAGE_V), { raw_width, n_aggr }, 0);
        spd_raw_k_norm = create_tensor(tn(LLM_TENSOR_SPD_RAW_K_NORM, "weight"), { raw_head_dim }, 0);
        if (bank_correction_rank > 0) {
            spd_bank_norm = create_tensor(tn(LLM_TENSOR_SPD_BANK_NORM), { n_embd, n_aggr }, 0);
            spd_bank_gate = create_tensor(tn(LLM_TENSOR_SPD_BANK_GATE), { 1, n_aggr }, 0);
            spd_bank_correction_a = create_tensor(tn(LLM_TENSOR_SPD_BANK_CORRECTION_A, "weight"), { n_embd, bank_correction_rank, n_aggr }, 0);
            spd_bank_correction_u = create_tensor(tn(LLM_TENSOR_SPD_BANK_CORRECTION_U, "weight"), { bank_correction_rank, n_embd, n_aggr }, 0);
        }
    }

    const ggml_tensor * d2t_meta = ml->get_tensor_meta(tn(LLM_TENSOR_D2T).str().c_str());
    if (d2t_meta == nullptr || d2t_meta->type != GGML_TYPE_I64) {
        throw std::runtime_error("SPD model requires an I64 d2t tensor");
    }
    const int64_t n_draft_vocab = d2t_meta->ne[0];
    d2t = create_tensor(tn(LLM_TENSOR_D2T), { n_draft_vocab }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_draft_vocab }, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        const int64_t n_ff_layer = hparams.n_ff(i);
        if (!head_architecture.empty()) {
            const int64_t reader_width = reader_head_count*reader_head_dim;
            layer.spd_reader_norm = create_tensor(tn(LLM_TENSOR_SPD_READER_NORM, "weight", i), { n_embd }, 0);
            layer.spd_reader_q = create_tensor(tn(LLM_TENSOR_SPD_READER_Q, "weight", i), { n_embd, reader_width }, 0);
            layer.spd_reader_q_norm = create_tensor(tn(LLM_TENSOR_SPD_READER_Q_NORM, "weight", i), { reader_head_dim }, 0);
            layer.spd_reader_o = create_tensor(tn(LLM_TENSOR_SPD_READER_O, "weight", i), { reader_width, n_embd }, 0);
            layer.spd_reader_stage_gate = create_tensor(tn(LLM_TENSOR_SPD_READER_STAGE_GATE, nullptr, i), { n_embd, n_aggr }, 0);
            if (i == n_layer - 1) {
                const int64_t raw_width = raw_head_count*raw_head_dim;
                layer.spd_raw_reader_norm = create_tensor(tn(LLM_TENSOR_SPD_RAW_READER_NORM, "weight", i), { n_embd }, 0);
                layer.spd_raw_reader_q = create_tensor(tn(LLM_TENSOR_SPD_RAW_READER_Q, "weight", i), { n_embd, raw_width }, 0);
                layer.spd_raw_reader_q_norm = create_tensor(tn(LLM_TENSOR_SPD_RAW_READER_Q_NORM, "weight", i), { raw_head_dim }, 0);
                layer.spd_raw_reader_o = create_tensor(tn(LLM_TENSOR_SPD_RAW_READER_O, "weight", i), { raw_width, n_embd }, 0);
            }
        }

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k*n_head }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_v_gqa }, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k*n_head, n_embd }, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff_layer }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff_layer, n_embd }, 0);
        layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff_layer }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_spd::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_spd::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const auto & spd = static_cast<const llama_model_spd &>(model);
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int64_t n_aggr = model.target_layer_ids.size();
    const bool readers = model.spd_target_k != nullptr;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    std::vector<ggml_tensor *> target_views(n_layer, nullptr);
    ggml_tensor * raw_k = nullptr;
    ggml_tensor * raw_v = nullptr;
    ggml_tensor * raw_valid = nullptr;
    ggml_tensor * changed_rows = nullptr;
    ggml_tensor * ring_slots = nullptr;
    auto round_bank = [&](ggml_tensor * x) {
        return readers ? ggml_cast(ctx0, ggml_cast(ctx0, x, GGML_TYPE_BF16), GGML_TYPE_F32) : x;
    };
    auto norm_f32 = [&](ggml_tensor * x, ggml_tensor * weight) {
        return ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), weight);
    };
    auto reader_norm = [&](ggml_tensor * x, ggml_tensor * weight) {
        return round_bank(norm_f32(x, weight));
    };
    auto reader_rope = [&](ggml_tensor * x, ggml_tensor * pos) {
        return ggml_rope_ext(ctx0, x, pos, nullptr, x->ne[0], GGML_ROPE_TYPE_NEOX,
                n_ctx_orig, spd.reader_rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    };

    ggml_tensor * inpL = nullptr;
    if (model.spd_aggr_blk != nullptr) {
        ggml_tensor * state = cparams.spd_aggr_state;
        const llama_spd_aggr_plan * plan = cparams.spd_aggr_plan;
        GGML_ASSERT(state != nullptr && plan != nullptr);
        const int64_t c = llm_graph_input_spd_shared::width(ubatch, plan, n_aggr);
        const int64_t changed_count = llm_graph_input_spd_shared::count_changed(ubatch, plan);
        auto inp = std::make_unique<llm_graph_input_spd_shared>(n_embd, n_aggr, plan, c, changed_count);
        if (readers && changed_count > 0) {
            inp->changed = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, changed_count);
            ggml_set_input(inp->changed);
            changed_rows = inp->changed;
        }
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        inp->feat   = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, c, n_tokens);
        inp->ids    = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, c, n_tokens);
        inp->slots  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        inp->keep   = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
        ggml_set_input(inp->tokens);
        ggml_set_input(inp->feat);
        ggml_set_input(inp->ids);
        ggml_set_input(inp->slots);
        ggml_set_input(inp->keep);
        res->t_inp_tokens = inp->tokens;
        ring_slots = inp->slots;
        ggml_tensor * flat_ids = ggml_reshape_1d(ctx0, inp->ids, c*n_tokens);
        ggml_tensor * features = round_bank(inp->feat);
        if (model.spd_bank_norm) {
            ggml_tensor * flat = ggml_reshape_2d(ctx0, features, n_embd, c*n_tokens);
            ggml_tensor * normalized = norm_f32(flat, ggml_get_rows(ctx0, model.spd_bank_norm, flat_ids));
            ggml_tensor * gate = ggml_get_rows(ctx0, model.spd_bank_gate, flat_ids);
            features = round_bank(ggml_add(ctx0, flat, ggml_mul(ctx0, gate, ggml_sub(ctx0, normalized, flat))));
            features = ggml_reshape_3d(ctx0, features, n_embd, c, n_tokens);
        }
        ggml_tensor * contrib = round_bank(ggml_mul_mat_id(ctx0, model.spd_aggr_blk, features, inp->ids));
        auto prefix = [&](ggml_tensor * terms, ggml_tensor * ring, bool rounded) {
            GGML_ASSERT(ring != nullptr);
            const int64_t width = terms->ne[0];
            ggml_tensor * acc = ggml_mul(ctx0, ggml_get_rows(ctx0, ring, inp->slots), inp->keep);
            for (int64_t t = 0; t < c; ++t) {
                ggml_tensor * term = ggml_view_2d(ctx0, terms, width, n_tokens, terms->nb[2], t*terms->nb[1]);
                acc = ggml_add(ctx0, acc, term);
                if (rounded) {
                    acc = round_bank(acc);
                }
            }
            ggml_build_forward_expand(gf, ggml_set_rows(ctx0, ring, acc, inp->slots));
            return acc;
        };
        ggml_tensor * acc;
        if (readers) {
            acc = prefix(contrib, state, true);
        } else {
            ggml_tensor * sum = ggml_view_2d(ctx0, contrib, n_embd, n_tokens, contrib->nb[2], 0);
            for (int64_t t = 1; t < c; ++t) {
                sum = ggml_add(ctx0, sum, ggml_view_2d(ctx0, contrib, n_embd, n_tokens, contrib->nb[2], t*contrib->nb[1]));
            }
            acc = ggml_add(ctx0, ggml_mul(ctx0, ggml_get_rows(ctx0, state, inp->slots), inp->keep), sum);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx0, state, acc, inp->slots));
        }
        ggml_tensor * scale = ggml_get_rows(ctx0, model.spd_aggr_scale, inp->tokens);
        inpL = round_bank(ggml_mul(ctx0, acc, scale));
        inpL = round_bank(ggml_add(ctx0, inpL, ggml_get_rows(ctx0, model.spd_aggr_bias, inp->tokens)));
        if (model.spd_bank_correction_a) {
            ggml_tensor * low = round_bank(ggml_mul_mat_id(ctx0, model.spd_bank_correction_a, features, inp->ids));
            low = prefix(low, cparams.spd_aggr_correction_state, true);
            low = ggml_mul_mat_id(ctx0, model.spd_bank_correction_u,
                    ggml_reshape_3d(ctx0, low, low->ne[0], 1, n_tokens),
                    ggml_reshape_2d(ctx0, inp->tokens, 1, n_tokens));
            low = ggml_reshape_2d(ctx0, round_bank(low), n_embd, n_tokens);
            inpL = round_bank(ggml_add(ctx0, inpL, low));
        }
        if (readers) {
            GGML_ASSERT(cparams.spd_aggr_view_state.size() == (size_t) n_layer);
            for (int il = 0; il < n_layer; ++il) {
                ggml_tensor * gate = ggml_get_rows(ctx0, model.layers[il].spd_reader_stage_gate, flat_ids);
                gate = ggml_reshape_3d(ctx0, gate, n_embd, c, n_tokens);
                ggml_tensor * view = prefix(ggml_mul(ctx0, contrib, gate), cparams.spd_aggr_view_state[il], false);
                target_views[il] = round_bank(ggml_add(ctx0, inpL, ggml_mul(ctx0, scale, view)));
            }
        }
        if (model.spd_raw_k) {
            const int64_t width = spd.raw_head_count*spd.raw_head_dim;
            inp->raw_positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, c*n_tokens);
            ggml_set_input(inp->raw_positions);
            inp->raw_ids = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, c, n_tokens);
            inp->raw_valid = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_aggr, n_tokens);
            ggml_set_input(inp->raw_ids);
            ggml_set_input(inp->raw_valid);
            raw_valid = inp->raw_valid;
            ggml_tensor * raw = ggml_reshape_2d(ctx0, round_bank(inp->feat), n_embd, c*n_tokens);
            raw = reader_norm(raw, ggml_get_rows(ctx0, model.spd_raw_norm, flat_ids));
            ggml_tensor * key = round_bank(ggml_add(ctx0, round_bank(build_lora_mm(model.spd_raw_k, raw)),
                    ggml_get_rows(ctx0, model.spd_raw_stage_k, flat_ids)));
            ggml_tensor * value = round_bank(ggml_add(ctx0, round_bank(build_lora_mm(model.spd_raw_v, raw)),
                    ggml_get_rows(ctx0, model.spd_raw_stage_v, flat_ids)));
            key = ggml_reshape_3d(ctx0, key, spd.raw_head_dim, spd.raw_head_count, c*n_tokens);
            key = reader_rope(reader_norm(key, model.spd_raw_k_norm), inp->raw_positions);
            auto update_raw = [&](ggml_tensor * cur, ggml_tensor * ring) {
                GGML_ASSERT(ring != nullptr);
                ggml_tensor * previous = ggml_mul(ctx0, ggml_get_rows(ctx0, ring, inp->slots), inp->keep);
                previous = ggml_reshape_3d(ctx0, previous, width, n_aggr, n_tokens);
                ggml_tensor * scratch = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, width, 1, n_tokens), 0.0f);
                previous = ggml_concat(ctx0, previous, scratch, 1);
                cur = ggml_set_rows(ctx0, previous, ggml_reshape_3d(ctx0, cur, width, c, n_tokens), inp->raw_ids);
                cur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, width, n_aggr, n_tokens, cur->nb[1], cur->nb[2], 0));
                cur = ggml_reshape_2d(ctx0, cur, width*n_aggr, n_tokens);
                ggml_build_forward_expand(gf, ggml_set_rows(ctx0, ring, cur, inp->slots));
                return cur;
            };
            raw_k = update_raw(key, cparams.spd_raw_k_state);
            raw_v = update_raw(value, cparams.spd_raw_v_state);
        }
        res->add_input(std::move(inp));
    } else {
        auto inp = std::make_unique<llm_graph_input_spd>(hparams.n_embd_inp(), n_aggr);
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
        ggml_set_input(inp->tokens);
        ggml_set_input(inp->embd);
        res->t_inp_tokens = inp->tokens;
        inpL = ggml_mul_mat_id(ctx0, model.spd_aggr,
                ggml_reshape_3d(ctx0, inp->embd, hparams.n_embd_inp(), 1, n_tokens),
                ggml_reshape_2d(ctx0, inp->tokens, 1, n_tokens));
        ggml_mul_mat_id_set_k_tri(inpL, hparams.n_embd_inp()/n_aggr);
        inpL = ggml_reshape_2d(ctx0, inpL, n_embd, n_tokens);
        res->add_input(std::move(inp));
    }
    cb(inpL, "spd_aggr", -1);
    res->t_inp_embd = inpL;
    ggml_build_forward_expand(gf, inpL);

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);
    const float kq_scale = 1.0f/std::sqrt(float(n_embd_head));
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        res->t_layer_inp[il] = inpL;
        const bool retain = readers && il == n_layer - 1 && inp_out_ids && inp_out_ids->ne[0] > 0 &&
                inp_attn->get_kq_mask()->ne[3] == 1;
        const int64_t nq = retain ? inp_out_ids->ne[0] : n_tokens;
        ggml_tensor * qpos = inp_pos;
        ggml_tensor * mask = inp_attn->get_kq_mask();
        if (retain) {
            qpos = ggml_reshape_1d(ctx0, ggml_get_rows(ctx0,
                    ggml_reshape_2d(ctx0, inp_pos, 1, n_tokens), inp_out_ids), nq);
            const int64_t ns = mask->ne[3];
            mask = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, mask, mask->ne[0], n_tokens), inp_out_ids);
            mask = ggml_reshape_4d(ctx0, mask, mask->ne[0], nq/ns, 1, ns);
            if (cparams.flash_attn) {
                mask = ggml_cast(ctx0, mask, GGML_TYPE_F16);
            }
        }
        ggml_tensor * inpSA = retain ? ggml_get_rows(ctx0, inpL, inp_out_ids) : inpL;
        ggml_tensor * cur = build_norm(inpL, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_tensor * Qcur = build_lora_mm(layer.wq, retain ? ggml_get_rows(ctx0, cur, inp_out_ids) : cur);
        ggml_tensor * Kcur = build_lora_mm(layer.wk, cur);
        ggml_tensor * Vcur = build_lora_mm(layer.wv, cur);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, nq);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
        Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        if (hparams.use_mrope()) {
            Qcur = ggml_rope_multi(ctx0, Qcur, qpos, nullptr, n_rot, sections, rope_type,
                    n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr, n_rot, sections, rope_type,
                    n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        } else {
            Qcur = ggml_rope_ext(ctx0, Qcur, qpos, nullptr, n_rot, rope_type,
                    n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type,
                    n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        }
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        llm_graph_input_attn_kv selected = *inp_attn;
        selected.self_kq_mask_cnv = mask;
        cur = build_attn(&selected, layer.wo, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        if (!readers && il == n_layer - 1 && inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        if (readers) {
            const auto * cache = inp_attn->mctx;
            const int64_t width = spd.reader_head_dim*spd.reader_head_count;
            ggml_tensor * key_ring = cparams.spd_target_k_state[il];
            ggml_tensor * value_ring = cparams.spd_target_v_state[il];
            ggml_tensor * key = ggml_get_rows(ctx0, key_ring, ring_slots);
            ggml_tensor * value = ggml_get_rows(ctx0, value_ring, ring_slots);
            if (changed_rows) {
                const int64_t nc = changed_rows->ne[0];
                ggml_tensor * z = reader_norm(ggml_get_rows(ctx0, target_views[il], changed_rows), model.spd_target_mem_norm);
                ggml_tensor * positions = ggml_reshape_1d(ctx0, ggml_get_rows(ctx0,
                        ggml_reshape_2d(ctx0, inp_pos, 1, n_tokens), changed_rows), nc);
                ggml_tensor * new_key = ggml_reshape_3d(ctx0, round_bank(build_lora_mm(model.spd_target_k, z)),
                        spd.reader_head_dim, spd.reader_head_count, nc);
                new_key = reader_rope(reader_norm(new_key, model.spd_target_k_norm), positions);
                key = ggml_set_rows(ctx0, key, ggml_reshape_2d(ctx0, new_key, width, nc), changed_rows);
                value = ggml_set_rows(ctx0, value, round_bank(build_lora_mm(model.spd_target_v, z)), changed_rows);
                ggml_build_forward_expand(gf, ggml_set_rows(ctx0, key_ring, key, ring_slots));
                ggml_build_forward_expand(gf, ggml_set_rows(ctx0, value_ring, value, ring_slots));
            }
            ggml_build_forward_expand(gf, cache->cpy_spd_aux(ctx0, key,
                    inp_attn->get_k_idxs(), il, llama_spd_cache_kind::TARGET_K));
            ggml_build_forward_expand(gf, cache->cpy_spd_aux(ctx0, value,
                    inp_attn->get_k_idxs(), il, llama_spd_cache_kind::TARGET_V));
            auto cached_heads = [&](llama_spd_cache_kind kind, int64_t dim, int64_t heads) {
                ggml_tensor * bank = cache->get_spd_aux(ctx0, il, kind);
                if (kind == llama_spd_cache_kind::RAW_K || kind == llama_spd_cache_kind::RAW_V) {
                    dim = bank->ne[0]/(heads*n_aggr);
                }
                return ggml_view_4d(ctx0, bank, dim, heads, bank->ne[0]/(dim*heads)*bank->ne[1], bank->ne[2],
                        dim*ggml_element_size(bank), dim*heads*ggml_element_size(bank), bank->nb[2], 0);
            };
            auto read = [&](ggml_tensor * norm, ggml_tensor * wq, ggml_tensor * qnorm, ggml_tensor * wo,
                    int64_t dim, int64_t heads, ggml_tensor * k, ggml_tensor * v, ggml_tensor * legal) {
                ggml_tensor * q = round_bank(build_lora_mm(wq, reader_norm(ffn_inp, norm)));
                q = ggml_reshape_3d(ctx0, q, dim, heads, nq);
                q = reader_rope(reader_norm(q, qnorm), qpos);
                const int64_t attention_dim = k->ne[0];
                GGML_ASSERT(attention_dim >= dim && v->ne[0] == attention_dim);
                if (attention_dim != dim) {
                    q = ggml_pad(ctx0, q, attention_dim - dim, 0, 0, 0);
                }
                ggml_tensor * out = build_attn_mha(q, k, v, nullptr, legal, nullptr, nullptr,
                        0, 1.0f/std::sqrt(float(dim)), il);
                if (attention_dim != dim) {
                    out = ggml_reshape_3d(ctx0, out, attention_dim, heads, nq);
                    out = ggml_view_3d(ctx0, out, dim, heads, nq, out->nb[1], out->nb[2], 0);
                    out = ggml_cont_2d(ctx0, out, dim*heads, nq);
                }
                return build_lora_mm(wo, out);
            };
            ggml_tensor * delta = read(layer.spd_reader_norm, layer.spd_reader_q, layer.spd_reader_q_norm,
                    layer.spd_reader_o, spd.reader_head_dim, spd.reader_head_count,
                    cached_heads(llama_spd_cache_kind::TARGET_K, spd.reader_head_dim, spd.reader_head_count),
                    cached_heads(llama_spd_cache_kind::TARGET_V, spd.reader_head_dim, spd.reader_head_count), mask);
            ggml_tensor * inherited = ggml_add(ctx0, ffn_inp, delta);
            if (layer.spd_raw_reader_q) {
                const int64_t raw_cache_dim = cache->get_spd_aux(ctx0, il, llama_spd_cache_kind::RAW_K)->ne[0]/
                        (spd.raw_head_count*n_aggr);
                auto pad_raw = [&](ggml_tensor * current) {
                    if (raw_cache_dim == spd.raw_head_dim) {
                        return current;
                    }
                    GGML_ASSERT(raw_cache_dim > spd.raw_head_dim);
                    current = ggml_reshape_3d(ctx0, current, spd.raw_head_dim, spd.raw_head_count*n_aggr, n_tokens);
                    current = ggml_pad(ctx0, current, raw_cache_dim - spd.raw_head_dim, 0, 0, 0);
                    return ggml_reshape_2d(ctx0, current, raw_cache_dim*spd.raw_head_count*n_aggr, n_tokens);
                };
                ggml_build_forward_expand(gf, cache->cpy_spd_aux(ctx0, pad_raw(raw_k), inp_attn->get_k_idxs(), il, llama_spd_cache_kind::RAW_K));
                ggml_build_forward_expand(gf, cache->cpy_spd_aux(ctx0, pad_raw(raw_v), inp_attn->get_k_idxs(), il, llama_spd_cache_kind::RAW_V));
                ggml_build_forward_expand(gf, cache->cpy_spd_aux(ctx0, raw_valid, inp_attn->get_k_idxs(), il, llama_spd_cache_kind::RAW_VALID));
                ggml_tensor * valid = ggml_cont(ctx0, cache->get_spd_aux(ctx0, il, llama_spd_cache_kind::RAW_VALID));
                const int64_t nk = valid->ne[1];
                const int64_t ns = valid->ne[2];
                valid = ggml_reshape_4d(ctx0, valid, n_aggr*nk, 1, 1, ns);
                ggml_tensor * legal = ggml_cast(ctx0, mask, GGML_TYPE_F32);
                legal = ggml_reshape_4d(ctx0, legal, 1, nk, nq/ns, ns);
                legal = ggml_repeat(ctx0, legal, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_aggr, nk, nq/ns, ns));
                legal = ggml_reshape_4d(ctx0, legal, n_aggr*nk, nq/ns, 1, ns);
                legal = ggml_add(ctx0, legal, valid);
                if (cparams.flash_attn) {
                    legal = ggml_cast(ctx0, legal, GGML_TYPE_F16);
                }
                delta = read(layer.spd_raw_reader_norm, layer.spd_raw_reader_q, layer.spd_raw_reader_q_norm,
                        layer.spd_raw_reader_o, spd.raw_head_dim, spd.raw_head_count,
                        cached_heads(llama_spd_cache_kind::RAW_K, spd.raw_head_dim, spd.raw_head_count),
                        cached_heads(llama_spd_cache_kind::RAW_V, spd.raw_head_dim, spd.raw_head_count), legal);
                inherited = ggml_add(ctx0, inherited, delta);
            }
            ffn_inp = inherited;
        }
        cb(ffn_inp, "ffn_inp", il);
        cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cur = build_ffn(cur,
                layer.ffn_up, nullptr, nullptr,
                layer.ffn_gate, nullptr, nullptr,
                layer.ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        if (readers && !retain && il == n_layer - 1 && inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }
        inpL = cur;
    }
    ggml_tensor * cur = inpL;
    res->t_h_nextn = cur;
    res->t_embd = cur;
    cur = build_lora_mm(model.output, cur);
    const int64_t n_draft_vocab = cur->ne[0];
    const int64_t n_outputs = cur->ne[1];
    const int64_t n_vocab = model.vocab.n_tokens();
    ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_outputs), -INFINITY);
    cur = ggml_set_rows(ctx0, logits,
            ggml_reshape_3d(ctx0, cur, 1, n_draft_vocab, n_outputs),
            ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1, 1));
    cur = ggml_reshape_2d(ctx0, cur, n_vocab, n_outputs);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
