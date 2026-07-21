#pragma once

#include "llama.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

constexpr uint32_t COMMON_SPD_MAX_STAGE_COUNT = 8;
constexpr uint32_t COMMON_SPD_MAX_DRAFT_TOP_K = 4;

struct common_spd_params {
    uint32_t n_ctx = 4096;
    uint32_t n_batch = 512;
    uint32_t n_ubatch = 512;
    int32_t n_threads = 0;
    int32_t n_threads_batch = 0;
    bool parallel_stages = true;
    uint32_t draft_top_k = 1;

    // Context templates keep SPD's target stages and sidecar aligned with the
    // caller's cache types, Flash Attention, RoPE, and offload settings.
    llama_context_params target_context = llama_context_default_params();
    llama_context_params sidecar_context = llama_context_default_params();
};

struct common_spd_result {
    std::vector<llama_token> tokens;
    std::vector<bool> accepted;

    uint64_t decode_steps = 0;
    uint64_t n_accepted = 0;
    uint64_t n_rejected = 0;
    uint64_t n_branch_rescued = 0;

    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    double stage_compute_seconds = 0.0;
};

// Single-sequence, greedy Speculative Pipeline Decoding (SPD) controller.
//
// The target model is executed as independently cached stages. Their count is
// read from the sidecar GGUF metadata (currently four or eight). The SPD
// sidecar consumes the trained staircase snapshots and proposes one or more
// tokens per pipeline step. At draft_top_k > 1, each retained path owns a
// target/sidecar KV sequence and target stages batch the live tree branches.
// Completed target tokens verify those proposals; a rejection rewinds both
// attention and recurrent target state before restarting the staircase at the
// corrected token.
class common_spd_pipeline {
public:
    common_spd_pipeline(llama_model * model_target, llama_model * model_spd, const common_spd_params & params);
    ~common_spd_pipeline();

    common_spd_pipeline(const common_spd_pipeline &) = delete;
    common_spd_pipeline & operator=(const common_spd_pipeline &) = delete;

    bool valid() const;
    const std::string & error() const;
    uint32_t stage_count() const;

    bool generate(const std::vector<llama_token> & prompt, int32_t n_predict, common_spd_result & result);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
