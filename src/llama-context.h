#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <array>
#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    // [fork] restore the worst-case galloc reserve if a narrow graph shrank it
    void galloc_restore_worstcase();

    // [fork] reserve a decode lane's galloc at its widest shape on creation
    void decode_lane_reserve(uint32_t lane);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    // [fork, SPD collect] row metadata of embd_layer_inp for the last decode
    int32_t get_embeddings_layer_inp_rows(const llama_seq_id ** seq_ids, const llama_pos ** pos);

    // [fork] chained cohort decode (LLAMA_DECODE_CHAIN=G): output rows are
    // keyed by seq id, so a caller holding several cohort calls in flight can
    // wait on one lane and read that cohort's rows without draining the rest
    void          chain_lane_sync(int32_t lane);
    const float * chain_logits_row(int32_t row) const;
    const float * chain_tap_row(uint32_t lid, int32_t row) const;
    int32_t       chain_last_lane_get() const { return chain_last_lane; }
    void          chain_arm(int32_t lane) { chain_armed_lane = lane; }

    // [fork, SPD peer boundaries] the persistent boundary input tensor (or the
    // last graph's raw embd input as a fallback) and the last graph's embd
    // output tensor, for direct device-to-device boundary pushes; and the
    // per-decode host-transfer skip toggles
    ggml_tensor * spd_peer_inp_tensor() const;
    ggml_tensor * spd_peer_out_tensor() const;
    void          set_spd_peer_io(bool skip_inp, bool skip_out, bool skip_layer_inp);

    ggml_context         * spd_boundary_ctx = nullptr;
    ggml_backend_buffer_t  spd_boundary_buf = nullptr;

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    bool set_mtp_dsa_index_share(bool enabled);
    bool set_mtp_dsa_selection(const int32_t * data, size_t size);
    const int32_t * get_mtp_dsa_selection(size_t * size);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);
    void set_graph_reuse(bool value);

    // [fork] drop every cached graph and scheduler split. Needed after the
    // model's weight buffers have moved (RPC hibernation): a reused graph
    // still carries the old device pointers, and over RPC the server's graph
    // cache is keyed on exactly those pointers.
    void invalidate_graphs();
    void set_stable_host_inputs(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    // Stage 2 body submission uses one token-sized scheduler lane per in-flight
    // speculative token. This keeps every lane's graph and activations stable
    // without multiplying the much larger prompt/prefill graph buffers.
    llm_graph_result * process_ubatch_pipedec_body(
                const llama_ubatch & ubatch,
            llama_memory_context_i * mctx,
                           uint32_t   lane,
                           uint32_t   total,
                       ggml_status & ret);

    // [fork] decode lane pool (LLAMA_DECODE_LANES): one persistent scheduler +
    // graph per decode-group ubatch, reused across steps with no drain. Safe
    // because each lane's tensors are private, at most one graph per lane is in
    // flight (all lanes retire at the per-step llama_get_* synchronize), and
    // cross-lane ordering rides each backend's stream/socket FIFO — the safety
    // argument the rejected copy-slot rotation never had.
    llm_graph_result * process_ubatch_decode_lane(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                           uint32_t   lane,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    // [fork, PipeDec] deferred verify group: a decode marked deferred submits its
    // stage-2 token lanes and returns without draining or running the LM head;
    // the next regular stage-2 decode closes the group (one head over all rows).
    // Lets the caller overlap MTP drafting with the first lane's pipeline walk.
    void     pipedec_defer_mark();  // next decode() call is a deferred group member
    void     pipedec_abort_group(); // drain + discard in-flight group rows (no head)
    uint32_t pipedec_group_n() const { return pipedec_group_tokens; }

    // [fork, PipeDec tree] see llama-ext.h
    int32_t       pipedec_tree_enable (bool value);
    int32_t       pipedec_tree_submit (const llama_batch & batch, int32_t lane);
    int32_t       pipedec_tree_close  (int32_t lane, int32_t row);
    void          pipedec_tree_discard(int32_t lane);
    const float * pipedec_tree_h      (int32_t lane, int32_t row);
    int32_t       pipedec_tree_commit (llama_seq_id seq_src, llama_seq_id seq_dst);
    // one LM-head graph over n_rows host rows; logits land in rows [0, n_rows)
    int32_t       pipedec_run_head    (const float * rows, uint32_t n_rows);
    static constexpr uint32_t pipedec_tree_max_lanes() { return PIPEDEC_STAGE2_MAX_LANES; }
    static constexpr uint32_t pipedec_tree_max_rows () { return PIPEDEC_TREE_MAX_ROWS; }
    int64_t pipedec_tree_wait_us() const { return pipedec_tree_t_wait_us; }
    int64_t pipedec_tree_head_us() const { return pipedec_tree_t_head_us; }

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out,
 llama_state_seq_flags   flags = LLAMA_STATE_SEQ_FLAGS_NONE);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count,
 llama_state_seq_flags   flags = LLAMA_STATE_SEQ_FLAGS_NONE);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers. res_sched is the
    // scheduler the ubatch actually ran on (a decode lane's tensors are not
    // resolvable against the shared scheduler); the ubatch supplies the
    // per-row (seq_id, pos) metadata published to collectors.
    void extract_layer_inputs(const llm_graph_result * res, ggml_backend_sched_t res_sched, const llama_ubatch & ubatch, size_t token_offset, bool chain_rows = false);

    // [fork, PipeDec] stage-2 variant: async-GET each enabled layer-input row of a
    // single-token body lane into the stable per-group buffers (published to
    // embd_layer_inp at group close, mirroring pipedec_group_h -> embd_nextn).
    void extract_layer_inputs_pipedec(const llm_graph_result * res, ggml_backend_sched_t lane_sched, uint32_t lane);

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype,
                ggml_backend_sched_t       sched_override = nullptr,
                           uint32_t         pipedec_lane = 0,
                           uint32_t         pipedec_total = 0) const;

    llm_graph_cb graph_get_cb(ggml_backend_sched_t sched_override = nullptr) const;

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    std::vector<int32_t> mtp_dsa_sel_raw;
    std::vector<float> mtp_dsa_sel_mask;
    std::vector<llama_seq_id> mtp_dsa_sel_seq;
    std::vector<int32_t> mtp_dsa_sel;
    size_t mtp_dsa_sel_width = 0;

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;

    // [fork, SPD collect] (seq_id, pos) per embd_layer_inp row of the LAST
    // llama_decode call, in the buffers' physical row order (concatenated
    // ubatch order - NOT user-batch order: split_equal merges prompts and the
    // grouped decode split reorders sequences, so collectors must map rows by
    // this metadata rather than by batch index)
    std::vector<llama_seq_id> embd_layer_inp_seq;
    std::vector<llama_pos>    embd_layer_inp_pos;

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    static constexpr uint32_t PIPEDEC_STAGE2_MAX_LANES = 16;
    static constexpr uint32_t PIPEDEC_TREE_MAX_ROWS    = 8;
    // [fork] one scheduler per (lane, level rows). A graph is reused only for
    // its own ubatch shape, and a tree lane alternates 1-row restart levels
    // with width-row levels: one scheduler per lane rebuilt and reallocated
    // the whole split graph on every shape change (~9 ms per level).
    // Classic stage-2 lanes are one token wide and live in slot 0.
    std::array<std::array<ggml_backend_sched_ptr, PIPEDEC_TREE_MAX_ROWS>, PIPEDEC_STAGE2_MAX_LANES> sched_pipedec_body;

    // [fork, PipeDec tree] one level per lane. rows = tokens the lane carries,
    // busy = its body graph or row GETs may still run. Rows stay readable after
    // close until the lane is reused.
    bool     pipedec_tree_enabled = false;
    std::array<uint32_t, PIPEDEC_STAGE2_MAX_LANES> pipedec_tree_lane_rows{};
    std::array<bool,     PIPEDEC_STAGE2_MAX_LANES> pipedec_tree_lane_busy{};
    // the row GET's read fence: waiting on it proves this lane's graph retired
    // on the last stage without draining the other lanes' queued work
    std::array<ggml_backend_t, PIPEDEC_STAGE2_MAX_LANES> pipedec_tree_lane_backend{};
    std::array<uint64_t,       PIPEDEC_STAGE2_MAX_LANES> pipedec_tree_lane_fence{};
    void pipedec_tree_lane_wait(int32_t lane);
    // set by close(): the logits came from the head, which already waited on
    // its device, so synchronize() (llama_get_logits_ith calls it) must not
    // drain the fabric behind the levels still in flight
    bool     pipedec_tree_logits_fresh = false;
    int64_t  pipedec_tree_t_wait_us = 0;
    int64_t  pipedec_tree_t_head_us = 0;
    ggml_backend_sched_ptr sched_pipedec_copy; // recurrent state copy at commit

    float * pipedec_tree_row(int32_t lane, int32_t row);

    // [fork] decode lane pool, grown lazily to the number of decode-group
    // ubatches per step (see process_ubatch_decode_lane)
    std::vector<ggml_backend_sched_ptr> sched_decode_lane;

    // [fork] chained cohort decode: per-lane read fences (backend, GET-stream
    // ordinal). A lane sync waits only on these - an endpoint-global backend
    // synchronize would fence every younger cohort's graphs too.
    std::vector<std::vector<std::pair<ggml_backend_t, uint64_t>>> chain_lane_reads;
    std::vector<ggml_backend_t> chain_read_backends; // scratch, current call
    int32_t chain_last_lane = -1;
    // one-shot per-call opt-in (llama_chain_arm): without it a classic caller's
    // single-token decode would be silently staged seq-keyed while the caller
    // reads packed rows via output_ids. Value = the lane this call rides
    // (cohorts are repacked from whatever seqs are generating, so the lane
    // cannot be derived from the seq ids); -1 = not armed.
    int32_t chain_armed_lane = -1;

    void chain_read_note(ggml_backend_t backend);

    // Stage 2 keeps the deferred output head on a separate scheduler so
    // switching to the tiny head graph does not evict the reusable 4k-node
    // decoder-body graph from the primary scheduler every iteration.
    ggml_backend_sched_ptr sched_pipedec_head;

    bool sched_need_reserve = true;

    // [fork] signature of the last worst-case graph reserved by memory_update().
    // That reserve re-splits and re-allocates the whole graph across every
    // backend; over an RPC fabric it costs seconds, and its three inputs never
    // move for the life of the context. See memory_update() for the argument.
    // [fork] galloc's plan is not a high-water mark - see
    // galloc_restore_worstcase(). Tracks the scheduler's re-plan epoch so a
    // shrunk reserve can be restored before it costs another fabric drain.
    int      galloc_epoch_seen     = 0;

    bool     mem_reserve_valid     = false;
    uint32_t mem_reserve_n_tokens  = 0;
    uint32_t mem_reserve_n_seqs    = 0;
    uint32_t mem_reserve_n_outputs = 0;
    int64_t  mem_reserve_skipped   = 0;

    // Avoid repeating the Stage 2 activation banner for every speculative
    // verification batch handled by this context.
    bool pipedec_stage2_reported = false;

    // [fork, PipeDec] deferred verify group state. pipedec_group_h has fixed
    // capacity (PIPEDEC_STAGE2_MAX_LANES rows) and is never resized once
    // allocated: deferred hidden-row GETs snapshot their destination address.
    bool               pipedec_defer_next   = false;
    uint32_t           pipedec_group_tokens = 0;
    std::vector<float> pipedec_group_h;

    // [fork, PipeDec] per-layer DFlash feature taps for the deferred group, one
    // fixed lane-indexed buffer per enabled layer (same stability contract as
    // pipedec_group_h). Indexed by layer id; unused layers stay empty.
    std::vector<std::vector<float>> pipedec_group_layer_inp;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;
    std::array<std::array<llm_graph_result_ptr, PIPEDEC_TREE_MAX_ROWS>, PIPEDEC_STAGE2_MAX_LANES> gf_res_pipedec_body;
    llm_graph_result_ptr gf_res_pipedec_head;
    std::vector<llm_graph_result_ptr> gf_res_decode_lane; // [fork] decode lane pool

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE (1 = batch graphs only, 2 = every graph)
    bool graph_reuse_disable     = false;
    bool graph_reuse_disable_all = false;
    bool stable_host_inputs = false;

    // disabling graph reuse is a pipelined-*prefill* knob: rebuilding per ubatch is what keeps
    // the RPC pipeline full, because the reuse path has to drain it before set_inputs. Graphs
    // that carry fewer tokens than a full ubatch are token generation (or a single-ubatch
    // prompt) -- there is nothing to overlap and a rebuild per token costs ~30x throughput --
    // so they keep reusing unless LLAMA_GRAPH_REUSE_DISABLE=2 asks for the old behaviour.
    bool graph_reuse_allowed(const llama_ubatch & ubatch) const {
        if (!graph_reuse_disable) {
            return true;
        }
        if (graph_reuse_disable_all) {
            return false;
        }
        // decode-shaped only (one token per sequence): prompt chunks must
        // rebuild, because the reuse path drains the pipeline under
        // pipeline_parallel, and a merged multi-sequence prompt ubatch can
        // land just under n_ubatch and masquerade as token generation (the
        // old `n_tokens < n_ubatch` test serialized concurrent prefill at
        // one full pipeline walk per ubatch once plan bucketing made the
        // shapes stable enough to actually reuse)
        return ubatch.n_seq_tokens == 1;
    }

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
