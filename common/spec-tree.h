#pragma once

// [fork, PipeDec tree] the paper's dynamic prediction tree (arXiv 2504.04104)
// over the stage-2 lane pipeline.
//
// Every tree node is one token at one position in its own seq id. A level is
// all nodes at one depth; it is submitted as one lane and its rows walk the
// pipeline while later levels are drafted and submitted behind it. When the
// oldest level completes, the LM head runs on the root's row only, the caller
// samples the target token x, and advance() prunes: x among the root's children
// keeps that child's subtree (everything else dies, in the queue and in
// flight), otherwise the whole tree dies and x becomes the new root.
//
// The drafter is the draft-mtp head: each node carries the hidden row its
// children are drafted from (the target's true row at a restart, the head's own
// output row further down), and a level is expanded in one batched draft
// decode over the frontier.

#include "llama.h"
#include "common.h"
#include "sampling.h"

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

struct common_spec_tree_params {
    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr; // draft-mtp context

    int32_t depth  = 2;  // levels in flight, i.e. how far the tree runs ahead of the root
    int32_t width  = 4;  // max nodes per level
    int32_t branch = 0;  // max children per node (0 = width)
    int32_t lanes  = 8;  // lane ring, needs depth + 2

    llama_seq_id seq_base = 1; // first tree seq id; lanes*width ids follow

    float p_min = 0.0f; // drop draft candidates below this probability
};

struct common_spec_tree_stats {
    int64_t n_steps     = 0; // levels closed
    int64_t n_hits      = 0; // x was among the root's children
    int64_t n_children  = 0; // children offered over all closed levels
    int64_t n_levels    = 0; // levels submitted
    int64_t n_rows      = 0; // rows submitted
    int64_t n_restarts  = 0;

    int64_t t_draft_us  = 0;
    int64_t t_submit_us = 0;
    int64_t t_wait_us   = 0;
    int64_t t_head_us   = 0;
};

struct common_spec_tree_advance {
    bool    hit        = false;
    int32_t n_children = 0;
};

class common_spec_tree {
public:
    explicit common_spec_tree(const common_spec_tree_params & params);
    ~common_spec_tree();

    // start a tree: root_tok sits at root_pos, its prefix lives in parent_seq on
    // both contexts, h_in is the target's hidden row at root_pos - 1
    bool begin(llama_token root_tok, llama_pos root_pos, llama_seq_id parent_seq, const float * h_in);

    bool    active()     const { return root >= 0; }
    int32_t n_inflight() const { return (int32_t) levels.size(); }
    int32_t depth()      const { return params.depth; }

    // true while another level can still be submitted this timestep
    bool can_submit() const;

    // draft the next level if none is pending and submit it; returns rows
    // submitted, 0 when the tree cannot grow, < 0 on error
    int32_t submit_next();

    // wait for the oldest level and run the head on the root's row; the caller
    // then samples from llama_get_logits_ith(ctx_tgt, 0). < 0 on error
    int32_t close_oldest();

    // prune with the sampled token
    common_spec_tree_advance advance(llama_token x);

    // teardown: every lane is discarded and every seq but the trunk is freed.
    // Returns the seq holding the last committed state, its position and the
    // target hidden row at that position; the caller commits and frees it.
    llama_seq_id finish(llama_pos * trunk_pos, const float ** trunk_h);

    const common_spec_tree_stats & stats() const { return st; }

    std::string summary() const;

private:
    struct cand {
        llama_token tok;
        float       logp;
        bool        used;
    };

    struct node {
        llama_token  tok;
        llama_pos    pos;
        int32_t      parent;
        int32_t      level;
        llama_seq_id parent_seq;
        llama_seq_id seq   = -1;
        int32_t      lane  = -1;
        int32_t      row   = -1;
        float        logp  = 0.0f;
        bool         alive = true;
        bool         expanded = false;

        std::vector<int32_t> children;
        std::vector<cand>    cands;
        std::vector<float>   h_in; // draft input row for this node's children
    };

    struct level {
        int32_t              lane;
        int32_t              depth;
        std::vector<int32_t> nodes; // row order
    };

    int32_t new_node();
    void    free_node(int32_t id);
    void    free_seq(llama_seq_id seq);
    void    kill(int32_t id);
    void    kill_subtree_except(int32_t parent, int32_t keep);

    bool    expand();  // batched draft decode over the frontier's unexpanded live nodes
    void    select();  // fill pending from the frontier's unused candidates

    int32_t alloc_lane() const;

    common_spec_tree_params params;
    common_spec_tree_stats  st;

    int32_t n_embd = 0;

    std::vector<node>    nodes;
    std::vector<int32_t> free_nodes;

    int32_t root = -1;

    std::deque<level> levels;       // in flight, oldest first
    std::vector<int32_t> pending;   // drafted, not submitted (row order)
    int32_t pending_depth = 0;

    std::vector<llama_seq_id> hold_seqs; // freed after the next submit

    std::vector<bool> lane_used;
    int32_t ring_cursor = 0;

    // draft side
    llama_batch batch_dft;
    llama_batch batch_tgt;
    std::vector<common_sampler_ptr> smpls;      // one per draft row
    std::vector<llama_sampler *>    backend_chains; // one per tree seq
    std::vector<float>              h_out;      // scratch: draft output rows
    std::vector<float>              trunk_h_buf;
};
