#pragma once

#include "s2_backend.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace s2 {

// A rank-r LoRA update for one base tensor: delta = b @ (a @ x). The alpha/r
// scale is folded into b when the adapter is uploaded, so the graph does not
// need a separate scale node. Both null when no adapter targets the tensor.
struct LoraPair {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    bool active() const { return a != nullptr && b != nullptr; }
};

struct ModelLayer {
    ggml_tensor * attention_norm = nullptr;
    ggml_tensor * ffn_norm       = nullptr;
    ggml_tensor * q_norm         = nullptr;
    ggml_tensor * k_norm         = nullptr;
    ggml_tensor * wqkv           = nullptr;
    ggml_tensor * wo             = nullptr;
    ggml_tensor * w1             = nullptr;
    ggml_tensor * w2             = nullptr;
    ggml_tensor * w3             = nullptr;

    LoraPair wqkv_lora;
    LoraPair wo_lora;
    LoraPair w1_lora;
    LoraPair w2_lora;
    LoraPair w3_lora;
};

struct ModelHParams {
    int32_t context_length     = 0;
    int32_t vocab_size         = 0;
    int32_t embedding_length   = 0;
    int32_t feed_forward_length = 0;
    int32_t block_count        = 0;
    int32_t head_count         = 0;
    int32_t head_count_kv      = 0;
    int32_t codebook_size      = 0;
    int32_t num_codebooks      = 0;
    int32_t semantic_begin_id  = 0;
    int32_t semantic_end_id    = 0;
    float   rope_freq_base     = 10000.0f;
    float   rms_norm_eps       = 1e-5f;
    bool    tie_word_embeddings = true;
    bool    attention_qk_norm  = false;
    bool    scale_codebook_embeddings = false;

    int32_t fast_context_length     = 0;
    int32_t fast_embedding_length   = 0;
    int32_t fast_feed_forward_length = 0;
    int32_t fast_block_count        = 0;
    int32_t fast_head_count         = 0;
    int32_t fast_head_count_kv      = 0;
    int32_t fast_head_dim           = 0;
    float   fast_rope_freq_base     = 10000.0f;
    float   fast_rms_norm_eps       = 1e-5f;
    bool    fast_attention_qk_norm  = false;
    bool    fast_has_project_in     = false;
    bool    has_fast_decoder        = false;
};

struct ModelWeights {
    ggml_context * ctx_w = nullptr;
    std::vector<ggml_backend_buffer_t> model_bufs_cpu;
    std::vector<ggml_backend_buffer_t> model_bufs_gpu;

    ggml_context * ctx_lora = nullptr;
    std::vector<ggml_backend_buffer_t> lora_bufs;

    ggml_tensor * embeddings           = nullptr;
    ggml_tensor * codebook_embeddings  = nullptr;
    ggml_tensor * norm                 = nullptr;
    ggml_tensor * fast_project_in      = nullptr;
    ggml_tensor * fast_embeddings      = nullptr;
    ggml_tensor * fast_norm            = nullptr;
    ggml_tensor * fast_output          = nullptr;

    LoraPair fast_embeddings_lora;
    LoraPair fast_output_lora;

    std::vector<ModelLayer> layers;
    std::vector<ModelLayer> fast_layers;
};

struct StepResult {
    std::vector<float> hidden;
    std::vector<float> logits;
};

class SlowARModel {
public:
    SlowARModel();
    ~SlowARModel();

    bool load(const std::string & gguf_path, int32_t gpu_device = -1, BackendType backend_type = BackendType::CPU, int32_t n_gpu_layers = -1);

    bool load_shared(gguf_context * gguf_ctx, const std::string & gguf_path, int32_t gpu_device = -1, BackendType backend_type = BackendType::CPU, int32_t n_gpu_layers = -1);

    bool read_tensor_data(const std::string & gguf_path, gguf_context * gguf_ctx);

    // Applies a GGUF LoRA adapter on top of already-loaded weights. Base tensors
    // are left untouched, so this works against a quantized model. Pass a
    // negative scale to use the adapter's own alpha/rank.
    bool load_lora(const std::string & lora_path, float scale = -1.0f);

    ggml_context * weights_ctx() { return weights_.ctx_w; }
    const std::unordered_set<ggml_tensor *> & weight_tensor_set() const { return weight_tensor_set_; }

    bool init_kv_cache(int32_t max_seq_len);

    void reset();

    void clear_kv_cache();

private:
    bool eval_cached(const std::vector<int32_t> & flat_tokens,
        int32_t n_tokens, int32_t n_threads,
        StepResult & result);
public:
    bool prefill_fast(const std::vector<int32_t> & flat_tokens, int32_t n_tokens,
        int32_t n_threads, StepResult & result);

    bool prefill(const std::vector<int32_t> & flat_tokens, int32_t n_tokens,
                 int32_t n_threads, StepResult & result);

    bool step(const std::vector<int32_t> & flat_tokens, int32_t n_threads,
              StepResult & result);

    bool fast_decode(const std::vector<float> & hidden,
                     const std::vector<int32_t> & prefix_codes,
                     int32_t n_threads,
                     std::vector<float> & logits_out);

    const ModelHParams & hparams() const { return hparams_; }

    ModelWeights   weights_;

private:
    ModelHParams   hparams_;
    ggml_backend_t backend_cpu_  = nullptr;
    ggml_backend_t backend_gpu_  = nullptr;
    ggml_backend_sched_t sched_       = nullptr;
    ggml_backend_sched_t fast_sched_  = nullptr;
    ggml_context * ctx_kv_      = nullptr;
    ggml_backend_buffer_t kv_buf_ = nullptr;
    ggml_tensor *  memory_k_   = nullptr;
    ggml_tensor *  memory_v_   = nullptr;
    int32_t        max_seq_len_ = 0;
    int32_t        n_gpu_layers_ = -1;
    int32_t        n_past_     = 0;
    size_t         ctx_size_   = 0;
    std::vector<uint8_t> ctx_buf_;
    size_t         fast_ctx_size_ = 0;
    std::vector<uint8_t> fast_ctx_buf_;

    std::unordered_set<ggml_tensor *> weight_tensor_set_;

};

}
