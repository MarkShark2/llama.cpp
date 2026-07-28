#include "models.h"

#include <algorithm>
#include <cmath>

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

}

void llama_model_spd::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    // qwen35-style sidecars carry rope sections (IMROPE); gemma4-style
    // sidecars use standard NEOX rope and omit them
    hparams.rope_sections.fill(0);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);
    ml.get_key(LLM_KV_SPD_CHECKPOINT_VERSION, checkpoint_version);
    ml.get_key(LLM_KV_SPD_STAGE_COUNT, stage_count);
    ml.get_key(LLM_KV_SPD_USE_DEEPEST, use_deepest);

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
}

void llama_model_spd::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_aggr = target_layer_ids.size();
    spd_aggr = create_tensor(tn(LLM_TENSOR_SPD_AGGR, "weight"), { n_embd*n_aggr, n_embd, n_aggr }, 0);

    const ggml_tensor * d2t_meta = ml->get_tensor_meta(tn(LLM_TENSOR_D2T).str().c_str());
    if (d2t_meta == nullptr || d2t_meta->type != GGML_TYPE_I64) {
        throw std::runtime_error("SPD model requires an I64 d2t tensor");
    }
    const int64_t n_draft_vocab = d2t_meta->ne[0];
    d2t = create_tensor(tn(LLM_TENSOR_D2T), { n_draft_vocab }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_draft_vocab }, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k*n_head }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_v_gqa }, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k*n_head, n_embd }, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_spd::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_spd::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int64_t n_aggr = model.target_layer_ids.size();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    auto inp = std::make_unique<llm_graph_input_spd>(hparams.n_embd_inp(), n_aggr);
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->tokens);
    ggml_set_input(inp->embd);
    res->t_inp_tokens = inp->tokens;

    ggml_tensor * aggr_ids = ggml_reshape_2d(ctx0, inp->tokens, 1, n_tokens);
    ggml_tensor * aggr_inp = ggml_reshape_3d(ctx0, inp->embd, hparams.n_embd_inp(), 1, n_tokens);
    ggml_tensor * inpL = ggml_mul_mat_id(ctx0, model.spd_aggr, aggr_inp, aggr_ids);
    inpL = ggml_reshape_2d(ctx0, inpL, n_embd, n_tokens);
    cb(inpL, "spd_aggr", -1);
    res->t_inp_embd = inpL;
    res->add_input(std::move(inp));
    ggml_build_forward_expand(gf, inpL);

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    const float kq_scale = 1.0f/std::sqrt(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;
        ggml_tensor * cur = build_norm(inpL, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * Qcur = build_lora_mm(layer.wq, cur);
        ggml_tensor * Kcur = build_lora_mm(layer.wk, cur);
        ggml_tensor * Vcur = build_lora_mm(layer.wv, cur);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        if (hparams.use_mrope()) {
            Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
        } else {
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
        }
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        cur = build_attn(inp_attn, layer.wo, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
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
