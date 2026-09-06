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
//
// Chain mode (a block drafter such as draft-dspark, `spec` set): width 1, and
// every level lives in the slot's own seq, so the cache needs neither unified
// mode nor per-node seqs. The drafter's one noise-block forward, anchored at
// the root, yields the candidates of the next block_size positions; the tokens
// past the deepest level in flight extend the chain. When the root closes its
// layer-input taps are injected into the drafter's cache (its context for the
// next block), a hit advances the root, a miss drops the suffix from the cache
// (a bounded partial rollback, n_rs_seq >= depth) and restarts from x.

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct common_spec_tree_params {
    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr; // draft-mtp context, or the block drafter's

    // chain mode: the speculative context whose block drafter feeds the tree
    common_speculative * spec = nullptr;

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

    // chain mode
    int64_t n_blocks    = 0; // block drafts
    int64_t n_block_new = 0; // chain tokens taken from them
    int64_t n_agree     = 0; // block positions that matched a level already in flight
    int64_t n_disagree  = 0;
    int64_t n_preempt   = 0; // suffixes in flight replaced by a fresh block's tokens
    int64_t n_preempt_levels = 0;
    int64_t n_soft_hits = 0; // x matched the fresh block's first token before its level was submitted
    int64_t n_stale     = 0; // background blocks dropped: a restart happened while they ran
    int64_t t_inject_us = 0;
    int64_t t_draft_bg_us   = 0; // block drafts on the worker thread (off the step)
    int64_t t_draft_join_us = 0; // what the step still waited for them
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
    void    sync_seq(llama_seq_id parent_seq, llama_seq_id seq); // give seq the parent's cells above its synced trunk
    void    clear_seqs(llama_seq_id keep);                      // empty every tree seq but keep, forget the trunk marks
    void    kill(int32_t id);
    void    kill_subtree_except(int32_t parent, int32_t keep);

    bool    expand();  // batched draft decode over the frontier's unexpanded live nodes
    void    select();  // fill pending from the frontier's unused candidates

    // chain mode
    bool    chain_expand();          // one block from the root (joined or drafted now), folded into the chain
    bool    chain_apply();           // fold draft_out into the chain: compare, preempt, extend
    void    chain_draft_start();     // queue the root's block on the worker
    bool    chain_draft_join();      // wait for it; false when none was pending
    void    chain_preempt_at(int32_t id); // kill the lineage from node id on, queued and in flight
    bool    chain_inject(int32_t id); // a closed level's taps into the drafter's cache
    int32_t chain_push(int32_t parent, llama_token tok); // node for the next chain token
    int32_t chain_node_at(llama_pos pos) const;          // lineage node at pos, -1 if none

    int32_t alloc_lane() const;

    common_spec_tree_params params;
    common_spec_tree_stats  st;

    // per tree seq (index seq - seq_base): the position through which the seq
    // still holds the trunk's cells, -1 when it holds nothing. The trunk below
    // the root never changes inside one tree, so a lane only ever needs the
    // cells above this mark trimmed and re-copied: O(depth) per level instead
    // of a walk over the whole context on every submit and kill.
    std::vector<llama_pos> synced;

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

    // chain mode
    bool         chain      = false;
    llama_seq_id chain_seq  = -1; // the slot's seq: every node lives here
    llama_pos    feat_pos   = -1; // the drafter's cache holds injected target taps through here
    int32_t      deepest    = -1; // last node of the lineage (root when nothing is past it)
    int32_t      chain_take = 0;  // chain tokens taken per block (0 = all past the deepest level)
    bool         chain_dry  = false; // the last block added nothing: wait for the root to move
    // the drafter's context has one owner: a worker thread running the tap
    // injections and the block drafts in the order they were queued. A block
    // is queued when the chain queue is down to one token (right after that
    // token's level goes out, so it runs under the close wait) and folded in
    // when the queue is empty or, at close, once it has finished. A block
    // anchored at an older root still extends the chain; one drafted before
    // a restart is stale and dropped.
    struct chain_job {
        enum kind_t { INJECT, BLOCK } kind;
        llama_pos          pos;
        llama_token        tok;
        int64_t            epoch;
        int64_t            id;      // BLOCK: superseded by any later block
        std::vector<float> feats;   // INJECT
        std::vector<llama_token> prefix; // BLOCK: the lineage in flight past the anchor
    };
    std::thread             worker;
    std::mutex              worker_m;
    std::condition_variable worker_cv;   // main -> worker: a job was queued, or stop
    std::condition_variable worker_done; // worker -> main: a job finished
    std::deque<chain_job>   jobs;
    bool                    worker_stop = false;
    bool                    worker_busy = false;
    void chain_worker_loop();
    void chain_enqueue(chain_job job);
    void chain_worker_flush();           // wait until every queued job ran
    bool chain_draft_done();             // the queued block has its result

    // the block job's result (worker_m): the newest the worker finished. A
    // block not yet started is dropped when a newer one is queued; a finished
    // one is folded in whether or not a newer one is already running (its
    // extension past the deepest level is still valid), so the chain never
    // waits for the worker unless the queue is empty.
    int64_t   draft_id_latest  = 0; // queued
    int64_t   draft_id_done    = 0; // finished (result in draft_out)
    int64_t   draft_id_applied = 0; // folded in
    bool      draft_pending() const { return draft_id_latest > draft_id_applied; }
    int32_t   draft_rc    = 0;
    int64_t   draft_us    = 0;
    llama_pos draft_pos   = -1; // the root the block was anchored at
    int64_t   draft_epoch = 0;
    int64_t   chain_epoch = 0;  // bumped by a restart
    std::vector<std::vector<common_spec_tree_cand>> draft_out;
    // preempt (GGML_PIPEDEC_CHAIN_PREEMPT=1): a block every step; where it
    // disagrees with a level in flight, that suffix dies and the fresh tokens
    // take its place. Measured a loss on DSV4/DSpark (the resubmitted level
    // waits the whole pipeline again, like a miss would), off by default.
    int32_t      chain_preempt = 0;
    // prefix (GGML_PIPEDEC_CHAIN_PREFIX=1): a block every step, the
    // levels in flight past the root ride in it as real tokens ahead of the
    // masks, so the new chain tokens come out of the block's first mask
    // positions instead of its tail; the queued tokens are replaced by every
    // fresh block, the levels in flight never are. Measured no gain on
    // DSV4/DSpark: its acceptance is ~60% per token in every slot.
    int32_t      chain_prefix = 0;
    // async (GGML_PIPEDEC_CHAIN_ASYNC=1): queue the next block as soon as the
    // queue is down to one token, so it runs under the close wait. Measured
    // within noise of drafting when the queue is empty (the restart's block
    // is the one that matters and nothing hides it), off by default.
    int32_t      chain_async  = 0;
    std::deque<llama_token>  chain_toks; // drafted past the deepest level, not yet submitted
    std::vector<int32_t>     feat_layers;
    int32_t                  n_feat     = 0; // one feature row
    int32_t                  n_embd_tgt = 0; // one tap
    std::vector<float>       feat_buf;

    // draft side
    llama_batch batch_dft;
    llama_batch batch_tgt;
    std::vector<common_sampler_ptr> smpls;      // one per draft row
    std::vector<llama_sampler *>    backend_chains; // one per tree seq
    std::vector<float>              h_out;      // scratch: draft output rows
    std::vector<float>              trunk_h_buf;
};
