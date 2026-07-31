#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    uint32_t spd_stage;       // zero-based logical SPD stage
    uint32_t spd_stage_count; // total logical SPD stages
    uint32_t spd_layer_start; // first target layer owned by this stage (inclusive)
    uint32_t spd_layer_end;   // last target layer owned by this stage (exclusive)
    // Width of one input row this context accepts, i.e. the stride of
    // batch.embd. Normally hparams.n_embd_inp(), but an SPD stage is handed the
    // previous stage's raw residual, and on DeepSeek-V4 that residual is
    // hc_mult streams wide rather than n_embd. 0 = not yet resolved.
    uint32_t n_embd_inp_ctx;
    // Width of one row of the embeddings output. Differs from n_embd_inp_ctx on
    // an SPD stage 0, which is fed tokens but still emits the wide residual.
    uint32_t n_embd_out_ctx;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    // [fork, SPD peer boundaries] toggled per decode by the SPD pipeline when
    // a stage boundary moves board-to-board instead of through the client:
    // skip the embd input upload (the data was peer-pushed and locally copied
    // into the graph's input tensor) and/or the embd output readback (the
    // boundary leaves via a peer push, the host never needs it).
    bool spd_peer_skip_inp = false;
    bool spd_peer_skip_out = false;
    // persistent device-resident boundary input for an SPD stage, allocated by
    // the context; build_inp_embd references it like a weight so the scheduler
    // neither stages it on the CPU nor re-uploads it per eval
    struct ggml_tensor * spd_boundary_inp = nullptr;

    std::vector<bool> embeddings_layer_inp; // [n_layer() + 1] extract input embeddings for layer; slot n_layer = output of the final layer

    enum llama_context_type ctx_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;
};
