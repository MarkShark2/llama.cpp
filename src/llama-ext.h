#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);
LLAMA_API int32_t llama_model_n_pos_per_embd(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);
LLAMA_API ggml_backend_dev_t llama_model_layer_device(const struct llama_model * model, int32_t layer);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

// [fork, SPD collect] per-row (seq_id, pos) metadata of the layer-input tap
// buffers for the last llama_decode call, in the buffers' physical row order
// (concatenated ubatch order, NOT user-batch order). Returns the row count;
// the arrays are valid until the next llama_decode on this context.
LLAMA_API int32_t llama_get_embeddings_layer_inp_rows(struct llama_context * ctx, const llama_seq_id ** seq_ids, const llama_pos ** pos);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

// Allow callers with stable, reusable input storage to opt a context back into
// graph reuse and asynchronous host input staging.
LLAMA_API void llama_set_graph_reuse(struct llama_context * ctx, bool value);
LLAMA_API void llama_set_stable_host_inputs(struct llama_context * ctx, bool value);

// [fork, SPD peer boundaries] the last graph's raw (un-narrowed) embd input
// tensor and embd output tensor -- the device-resident endpoints of a stage
// boundary. Valid until the context builds a different graph; callers must
// re-fetch after every decode. And the per-decode host-transfer skips: with
// skip_inp the embd upload in set_input is suppressed (the data was placed on
// device by a peer push + local copy), with skip_out the embeddings readback
// in decode is suppressed (the boundary leaves via a peer push), and with
// skip_layer_inp the layer-input tap readback is suppressed (the host derives
// those anchors from the previous stage's boundary instead).
LLAMA_API struct ggml_tensor * llama_spd_peer_inp_tensor(struct llama_context * ctx);
LLAMA_API struct ggml_tensor * llama_spd_peer_out_tensor(struct llama_context * ctx);
LLAMA_API void llama_set_spd_peer_io(struct llama_context * ctx, bool skip_inp, bool skip_out, bool skip_layer_inp);

// [fork, SPD peer boundaries] synchronous peer-push entry points, implemented
// in ggml-rpc.cpp (exported from the RPC backend, declared here so the SPD
// pipeline can drive them; see the "Synchronous peer push" block there for
// the ordering contract).
extern "C" {
bool ggml_backend_rpc_sync_peer_prepare(const struct ggml_tensor * dst_probe);
bool ggml_backend_rpc_sync_peer_push(const struct ggml_tensor * src, const struct ggml_tensor * dst, uint64_t * ordinal_out);
bool ggml_backend_rpc_sync_peer_fence(const struct ggml_tensor * dst_probe, uint64_t ordinal);
bool ggml_backend_rpc_sync_peer_guard(const struct ggml_tensor * src_probe);
// [fork, chained decode] ordinal-scoped async-read fences (see ggml-rpc.cpp):
// a completed read also proves the graph that produced its tensor retired
uint64_t ggml_backend_rpc_read_ordinal(ggml_backend_t backend);
void     ggml_backend_rpc_read_wait(ggml_backend_t backend, uint64_t ordinal);
bool     ggml_backend_is_rpc(ggml_backend_t backend);
// [fork, imatrix] reduce a matmul activation to sum(x^2) per column (per
// expert when ids != NULL) on the daemon that already holds it, instead of
// GETting the whole activation. sums is [src1->ne[0] * n_mat], counts is
// [n_mat] (may be NULL). Returns false when the tensor is not RPC-resident or
// the daemon predates proto patch 3 - the caller must then do it the old way.
bool ggml_backend_rpc_imatrix_sqsum(
        const struct ggml_tensor * src1,
        const struct ggml_tensor * ids,
        int64_t n_mat, int64_t src0_ne2, int64_t src0_ne3,
        float * sums, int64_t * counts);
}

// [fork] chained cohort decode (LLAMA_DECODE_CHAIN=G): the caller decodes
// seq-aligned cohorts of up to G single-token sequences as separate
// llama_decode calls held in flight together. Output rows are keyed by seq
// id; the cohort's lane is (first_seq/G) % LLAMA_DECODE_LANES. Wait on the
// lane, then read that cohort's rows - the other cohorts keep flowing.
extern "C" {
LLAMA_API void          llama_chain_lane_sync (struct llama_context * ctx, int32_t lane);
LLAMA_API const float * llama_chain_logits_row(struct llama_context * ctx, int32_t row);
LLAMA_API const float * llama_chain_tap_row   (struct llama_context * ctx, uint32_t lid, int32_t row);
// lane the last llama_decode staged as a chain call, or -1 if it took the
// classic path - callers MUST check this after every chained submit (a silent
// fallback would leave the seq-keyed rows stale)
LLAMA_API int32_t       llama_chain_last_lane (struct llama_context * ctx);
// one-shot opt-in consumed by the NEXT llama_decode: without it no call is
// ever staged chained (classic callers read packed rows via output_ids and
// must never be hijacked into seq-keyed staging). `lane` is the decode lane
// the call rides - cohorts are repacked per cycle, so the caller names it.
LLAMA_API void          llama_chain_arm      (struct llama_context * ctx, int32_t lane);
// internal (llama-kv-cache-dsv4.cpp): armed calls suppress the
// LLAMA_DECODE_SPLIT cut for the duration of init_batch
void dsv4_decode_split_suppress(bool on);
}

// [PipeDec] deferred verify group (requires GGML_PIPEDEC_STAGE2 eligibility):
// llama_pipedec_defer marks the NEXT llama_decode as a deferred group member -
// its stage-2 token lanes are submitted but no head runs and no logits are
// produced. The next regular stage-2 decode closes the group: one LM head over
// all group rows, logits indexed by group row [0..n). llama_pipedec_abort
// drains and discards an open group without running the head (the caller
// removes the tokens' KV itself if unwanted).
LLAMA_API void     llama_pipedec_defer  (struct llama_context * ctx);
LLAMA_API void     llama_pipedec_abort  (struct llama_context * ctx);
LLAMA_API uint32_t llama_pipedec_group_n(struct llama_context * ctx);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);
// returns the number of target pipeline stages encoded by an SPD sidecar, or zero for other architectures
LLAMA_API uint32_t        llama_model_spd_stage_count   (const struct llama_model * model);
// width of one token's state as handed from one SPD stage to the next. This is
// n_embd for every architecture whose residual is a single stream, but
// DeepSeek-V4 carries hc_mult hyper-connection streams between layers and a
// mid-trunk boundary is that much wider.
LLAMA_API uint32_t        llama_model_n_embd_spd_boundary(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);
