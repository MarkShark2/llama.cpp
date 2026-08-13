#pragma once

#include "llama.h"

#include "common.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Upper bound on SPD pipeline stages. Only sizes std::arrays of per-stage
// bookkeeping, so raising it costs a little static storage and nothing else.
// DeepSeek-V4 runs 9 stages (43 layers over the 8 BC-250s plus the 3080 Ti),
// which the previous limit of 8 rejected outright.
constexpr uint32_t COMMON_SPD_MAX_STAGE_COUNT = 16;

struct common_spd_params {
    uint32_t n_ctx = 4096;
    uint32_t n_batch = 512;
    uint32_t n_ubatch = 512;
    int32_t n_threads = 0;
    int32_t n_threads_batch = 0;
    bool parallel_stages = true;

    // Context templates keep SPD's target stages and sidecar aligned with the
    // caller's cache types, Flash Attention, RoPE, and offload settings.
    llama_context_params target_context = llama_context_default_params();
    llama_context_params sidecar_context = llama_context_default_params();
};

// Per-request generation controls. SPD samples with a bare argmax and never
// builds a common_sampler, so anything the sampler chain would normally do has
// to be handed to the pipeline explicitly.
struct common_spd_gen_params {
    int32_t n_predict = -1;   // must be > 0; the caller resolves its own default
    bool    ignore_eos = false;

    // Mirrors the matching common_params_sampling fields. The budget sampler is
    // driven on the finalized token stream, and while it forces, its token
    // replaces the target head's argmax -- so the forced sequence is what
    // actually enters the KV cache and what drafts are verified against.
    std::vector<llama_token>              reasoning_budget_start;
    std::vector<std::vector<llama_token>> reasoning_budget_end;
    std::vector<llama_token>              reasoning_budget_forced;
    int32_t                               reasoning_budget_tokens = -1;

    // Tokens the chat template already placed in the prompt, from
    // common_sampler_prefill_tokens(). The budget and grammar samplers are
    // advanced past these before generation, exactly as common_sampler does --
    // seeding either from the whole prompt instead arms it on any user message
    // that contains a literal start tag or a fragment of the output format.
    std::vector<llama_token>              generation_prompt_tokens;

    // A grammar: a user GBNF, a JSON schema, or the tool-call grammar the chat
    // template compiled from the request's tools. Constrains the target head's
    // argmax, which is where SPD selects -- so the output is what plain greedy
    // decoding under the same grammar would have produced.
    //
    // The sidecar drafts unconstrained. A draft is still accepted only if it
    // equals the constrained argmax, so nothing invalid can slip through; a
    // grammar just costs acceptance for as long as it is actually binding.
    std::string                           grammar;
    bool                                  grammar_lazy = false;
    std::vector<common_grammar_trigger>   grammar_triggers;
    // common_grammar_needs_prefill(): true for output-format and tool-call
    // grammars, false for a user-supplied one, which must not be prefilled.
    bool                                  grammar_needs_prefill = false;

    // Reuse the prefix of `prompt` the stage caches already hold instead of
    // re-prefilling it. Off restores the clear-everything behaviour.
    bool prefix_reuse = true;

    // How many context checkpoints to retain (--ctx-checkpoints). A DSV4 target
    // rejects any cache crop deeper than its rollback slack, so a checkpoint
    // restore is the *only* way a chat turn can reuse the previous turn --
    // 0 means every request re-prefills its whole prompt.
    int32_t n_ctx_checkpoints = 4;

    // Minimum spacing between checkpoints taken during prefill, in tokens
    // (--checkpoint-min-step). Same meaning as on the ordinary server path.
    int32_t checkpoint_min_step = 0;

    // Polled between prefill chunks; decode is covered by the token callback
    // returning false. Returning true abandons the request: generate() returns
    // false with no error message, result.cancelled is set, and the caches are
    // reset like any other unfinished request. This is the only way a client
    // cancel can reach a request whose caller is blocked inside generate().
    std::function<bool()> should_cancel;
};

// Invoked as each token becomes final -- the target has verified it, so no
// later rejection can retract it, which is what makes it safe to send onward.
// `index` counts generated tokens from 0 and never skips or repeats. Returning
// false stops generation at that token; the caller uses this for stop strings
// and any limit it evaluates itself.
using common_spd_token_callback = std::function<bool(llama_token token, size_t index)>;

struct common_spd_result {
    std::vector<llama_token> tokens;
    std::vector<bool> accepted;

    uint64_t decode_steps = 0;
    uint64_t n_accepted = 0;
    uint64_t n_rejected = 0;

    // Prompt tokens served from the resident caches vs. actually prefilled.
    int32_t n_prompt_reused = 0;
    int32_t n_prompt_processed = 0;

    // How many leading tokens (prompt followed by generated) the caches still
    // hold when the request ends. A request always stops with entries partway
    // up the stage ladder, and those cells are dropped, so this is short of
    // prompt + generated by a few positions.
    //
    // The caller has to trim its own token record to this before handing the
    // state to the prompt cache: a record longer than the state it describes
    // would let a later restore claim resident tokens that are not there.
    int32_t n_cached_tokens = 0;

    // should_cancel abandoned the request during prefill. generate() returned
    // false, but there is no error to report and nothing to send.
    bool cancelled = false;

    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    double stage_compute_seconds = 0.0;
};

struct spd_state_io;

// Single-sequence, greedy Speculative Pipeline Decoding (SPD) controller.
//
// The target model is executed as independently cached stages. Their count is
// read from the sidecar GGUF metadata (currently four or eight). The SPD
// sidecar consumes the trained staircase snapshots and proposes one token per
// pipeline step. Completed target tokens verify those proposals; a rejection
// rewinds both attention and recurrent target state before restarting the
// staircase at the corrected token.
class common_spd_pipeline {
public:
    common_spd_pipeline(llama_model * model_target, llama_model * model_spd, const common_spd_params & params);
    ~common_spd_pipeline();

    common_spd_pipeline(const common_spd_pipeline &) = delete;
    common_spd_pipeline & operator=(const common_spd_pipeline &) = delete;

    bool valid() const;
    const std::string & error() const;
    uint32_t stage_count() const;

    // The pipeline's whole state as one serializable unit -- every target stage
    // plus the sidecar, framed together.
    //
    // This is what lets SPD go through the ordinary server mechanics instead of
    // reimplementing them: hand this to common_prompt_checkpoint and
    // server_prompt_cache in place of a llama_context and --ctx-checkpoints,
    // --checkpoint-min-step, --cache-ram and --cache-disk all work unchanged.
    // The stages are meaningless apart, so they move together or not at all.
    common_state_seq_io & state_io();

    // Tell the pipeline what its caches now hold after the layers above have
    // written state into them through state_io() -- a prompt-cache restore.
    // Without this the pipeline would keep matching new prompts against the
    // token record of the state that was just overwritten, and reuse a prefix
    // that is no longer there. Pass an empty vector for "assume nothing".
    void note_resident_prefix(const std::vector<llama_token> & tokens);

    bool generate(const std::vector<llama_token> & prompt,
            const common_spd_gen_params & gparams,
            common_spd_result & result,
            const common_spd_token_callback & on_token = nullptr);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;

    std::unique_ptr<common_state_seq_io> pimpl_state_io;

    // reaches into pimpl to fan a state get/set across the stages
    friend struct spd_state_io;
};
