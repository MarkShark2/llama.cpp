#pragma once

#include "llama.h"
#include "common.h"

#include <functional>

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   n_past;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // [fork, PipeDec] streamed draft-lane submission: invoked right after draft
    // token `result[depth]` is sampled, while the draft loop continues. The
    // callback may submit the token's verify lane immediately (deferred group
    // member) so it pipelines behind the earlier lanes instead of waiting for
    // the whole draft. Return false to stop being called for this draft()
    // (remaining tokens then ride the regular closing decode).
    // Never invoked for a token that could be the draft's last (depth == n_max-1),
    // so the closing decode always has at least one token to carry.
    std::function<bool(llama_token id, int32_t depth)> on_draft_token;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// [fork, PipeDec probe] top candidates (best first) the draft sampler saw at draft
// step `step` of the most recent draft for this seq; nullptr if unavailable
const std::vector<llama_token> * common_speculative_dbg_topk(common_speculative * spec, llama_seq_id seq_id, int step);

// [fork, PipeDec tree] the draft-mtp implementation's carry-over hidden row for
// seq_id: the target's h at the last position process() saw, the row that pairs
// with the next token. nullptr without a draft-mtp implementation. set_ replaces it.
const float * common_speculative_mtp_pending_h    (common_speculative * spec, llama_seq_id seq_id);
void          common_speculative_mtp_set_pending_h(common_speculative * spec, llama_seq_id seq_id, const float * h);

// [fork, PipeDec tree] how the loaded drafter feeds the prediction tree:
//   MTP_ROW - draft-mtp: one head step per level, chained on the head's own hidden row
//   BLOCK   - draft-dflash / draft-dspark: one noise-block forward yields the candidates
//             of block_size consecutive levels (valid along the in-graph argmax chain);
//             committed tokens are handed back as target-layer features for KV injection
enum common_spec_tree_kind {
    COMMON_SPEC_TREE_NONE = 0,
    COMMON_SPEC_TREE_MTP_ROW,
    COMMON_SPEC_TREE_BLOCK,
};

struct common_spec_tree_cand {
    llama_token tok;
    float       p;
};

common_spec_tree_kind common_speculative_tree_kind(common_speculative * spec);

// block drafters: run the noise block on `seq` anchored at (tok, pos), after dropping
// whatever `seq` held from pos on. The n_prefix tokens already drafted for
// pos + 1 .. pos + n_prefix ride in the block as real tokens ahead of the
// masks, so the masks predict from the deepest of them (the drafter has never
// seen the target's state for them, only their embeddings). out[j] = top
// candidates for position pos + 1 + j. Returns the number of levels drafted,
// < 0 on error.
int32_t common_speculative_tree_block_draft(
        common_speculative * spec, llama_seq_id seq, llama_token tok, llama_pos pos,
        const llama_token * prefix, int32_t n_prefix, int32_t n_cand,
        std::vector<std::vector<common_spec_tree_cand>> & out);

// block drafters: the target layers whose input features the drafter injects, in the
// order the feature row is laid out (n_layers x n_embd_tgt floats per row)
int32_t common_speculative_tree_feat_layers(common_speculative * spec, const int32_t ** ids);
int32_t common_speculative_tree_feat_width (common_speculative * spec);

// block drafters: inject n_rows committed positions into `seq` from their target
// features (rows of tree_feat_width floats), replacing the noise cells there
bool    common_speculative_tree_inject(
        common_speculative * spec, llama_seq_id seq, int32_t n_rows, const llama_pos * pos, const float * feats);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
