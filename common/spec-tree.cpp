#include "spec-tree.h"

#include "log.h"
#include "../src/llama-ext.h" // llama_pipedec_tree_*, llama_get_embeddings_nextn_ith

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool spec_tree_trace() {
    static const bool on = getenv("GGML_PIPEDEC_TREE_TRACE") && atoi(getenv("GGML_PIPEDEC_TREE_TRACE")) != 0;
    return on;
}

#define TREE_TRC(...) do { if (spec_tree_trace()) { LOG_INF("[tree] " __VA_ARGS__); } } while (0)

common_spec_tree::common_spec_tree(const common_spec_tree_params & params) : params(params) {
    GGML_ASSERT(params.ctx_tgt && params.ctx_dft);
    GGML_ASSERT(params.width >= 1 && params.width <= llama_pipedec_tree_n_rows_max());
    GGML_ASSERT(params.lanes >= params.depth + 2 && params.lanes <= llama_pipedec_tree_n_lanes());

    if (this->params.branch <= 0) {
        this->params.branch = this->params.width;
    }

    n_embd = llama_model_n_embd_out(llama_get_model(params.ctx_dft));
    GGML_ASSERT(n_embd == llama_model_n_embd_out(llama_get_model(params.ctx_tgt)));

    const int32_t width = this->params.width;

    // the draft batch carries both the token and the hidden row of every node
    batch_dft = llama_batch_init(width, n_embd, 1);
    batch_dft.token = (llama_token *) malloc(sizeof(llama_token) * width);
    batch_tgt = llama_batch_init(width, 0, 1);

    smpls.resize(width);
    for (auto & s : smpls) {
        common_params_sampling sparams;
        sparams.no_perf  = true;
        sparams.top_k    = std::max(16, this->params.branch);
        sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
        s.reset(common_sampler_init(llama_get_model(params.ctx_dft), sparams));
    }

    // backend top-k on the draft device for every tree seq, CPU fallback otherwise
    const int32_t n_tree_seq = this->params.lanes * width;
    backend_chains.assign(n_tree_seq, nullptr);
    for (int32_t i = 0; i < n_tree_seq; ++i) {
        const llama_seq_id seq = this->params.seq_base + i;
        llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(std::max(16, this->params.branch)));
        if (!llama_set_sampler(params.ctx_dft, seq, chain)) {
            llama_sampler_free(chain);
            chain = nullptr;
        }
        backend_chains[i] = chain;
    }

    lane_used.assign(this->params.lanes, false);
    synced.assign(n_tree_seq, -1);

    h_out.resize((size_t) width * n_embd);

    LOG_INF("%s: PipeDec tree: depth=%d width=%d branch=%d lanes=%d seq_base=%d p_min=%.2f backend_topk=%s\n",
            __func__, this->params.depth, width, this->params.branch, this->params.lanes,
            (int) this->params.seq_base, this->params.p_min, backend_chains[0] ? "yes" : "no");
}

common_spec_tree::~common_spec_tree() {
    for (size_t i = 0; i < backend_chains.size(); ++i) {
        if (backend_chains[i]) {
            llama_set_sampler(params.ctx_dft, params.seq_base + (llama_seq_id) i, nullptr);
            llama_sampler_free(backend_chains[i]);
        }
    }
    if (batch_dft.token) {
        free(batch_dft.token);
        batch_dft.token = nullptr;
    }
    llama_batch_free(batch_dft);
    llama_batch_free(batch_tgt);
}

// ids are never reused inside one tree: a lane in flight may still name a dead
// node, and the alive flag must keep telling the truth about that id
int32_t common_spec_tree::new_node() {
    const int32_t id = (int32_t) nodes.size();
    nodes.emplace_back();
    return id;
}

void common_spec_tree::free_node(int32_t id) {
    nodes[id].alive = false;
    nodes[id].children.clear();
    nodes[id].cands.clear();
    nodes[id].h_in.clear();
    nodes[id].h_in.shrink_to_fit();
}

// a freed lane keeps the trunk cells it already shares (they stay valid for the
// rest of this tree); only its own branch above the mark goes
void common_spec_tree::free_seq(llama_seq_id seq) {
    if (seq < 0) {
        return;
    }
    const int32_t   i  = (int32_t) (seq - params.seq_base);
    const llama_pos p0 = (i >= 0 && i < (int32_t) synced.size()) ? synced[i] + 1 : 0;
    llama_memory_seq_trim(llama_get_memory(params.ctx_tgt), seq, p0);
    llama_memory_seq_trim(llama_get_memory(params.ctx_dft), seq, p0);
}

void common_spec_tree::sync_seq(llama_seq_id parent_seq, llama_seq_id seq) {
    const int32_t i = (int32_t) (seq - params.seq_base);
    GGML_ASSERT(i >= 0 && i < (int32_t) synced.size());

    // the parent's cells through root.pos - 1 are the trunk, shared by every
    // seq in the tree; whatever this seq holds through its own mark is that
    // same trunk, so only the span above the mark moves
    const llama_pos p0 = synced[i] + 1;

    llama_memory_seq_trim(llama_get_memory(params.ctx_tgt), seq, p0);
    llama_memory_seq_trim(llama_get_memory(params.ctx_dft), seq, p0);
    llama_memory_seq_cp(llama_get_memory(params.ctx_tgt), parent_seq, seq, p0, -1);
    llama_memory_seq_cp(llama_get_memory(params.ctx_dft), parent_seq, seq, p0, -1);

    synced[i] = std::max<llama_pos>(synced[i], nodes[root].pos - 1);
}

void common_spec_tree::clear_seqs(llama_seq_id keep) {
    for (size_t i = 0; i < synced.size(); ++i) {
        const llama_seq_id seq = params.seq_base + (llama_seq_id) i;
        if (seq != keep && synced[i] >= 0) {
            llama_memory_seq_rm(llama_get_memory(params.ctx_tgt), seq, -1, -1);
            llama_memory_seq_rm(llama_get_memory(params.ctx_dft), seq, -1, -1);
        }
        synced[i] = -1;
    }
}

// the bloodline dies: the node, its seq, its children, and any queued
// descendants. Rows already in flight keep computing but are never read.
void common_spec_tree::kill(int32_t id) {
    if (id < 0 || !nodes[id].alive) {
        return;
    }
    auto children = nodes[id].children; // copy: free_node clears it
    for (int32_t c : children) {
        kill(c);
    }
    pending.erase(std::remove(pending.begin(), pending.end(), id), pending.end());
    free_seq(nodes[id].seq);
    free_node(id);
}

void common_spec_tree::kill_subtree_except(int32_t parent, int32_t keep) {
    auto children = nodes[parent].children;
    for (int32_t c : children) {
        if (c != keep) {
            kill(c);
        }
    }
    nodes[parent].children.clear();
    if (keep >= 0) {
        nodes[parent].children.push_back(keep);
    }
}

bool common_spec_tree::begin(llama_token root_tok, llama_pos root_pos, llama_seq_id parent_seq, const float * h_in) {
    // a previous tree must have been finished
    GGML_ASSERT(root < 0 && levels.empty() && pending.empty());

    nodes.clear();
    free_nodes.clear();
    hold_seqs.clear();
    std::fill(lane_used.begin(), lane_used.end(), false);
    std::fill(synced.begin(), synced.end(), -1);
    ring_cursor = 0;

    const int32_t id = new_node();
    auto & n = nodes[id];
    n.tok        = root_tok;
    n.pos        = root_pos;
    n.parent     = -1;
    n.level      = 0;
    n.parent_seq = parent_seq;
    n.logp       = 0.0f;
    n.h_in.assign(h_in, h_in + n_embd);

    root = id;
    pending = { id };
    pending_depth = 0;

    TREE_TRC("begin root tok=%d pos=%d parent_seq=%d\n", (int) root_tok, (int) root_pos, (int) parent_seq);
    return true;
}

int32_t common_spec_tree::alloc_lane() const {
    for (int32_t k = 0; k < params.lanes; ++k) {
        const int32_t lane = (ring_cursor + k) % params.lanes;
        if (!lane_used[lane]) {
            return lane;
        }
    }
    return -1;
}

bool common_spec_tree::can_submit() const {
    if (root < 0) {
        return false;
    }
    if (n_inflight() >= params.depth) {
        return false;
    }
    if (!pending.empty()) {
        return true;
    }
    if (levels.empty()) {
        return false;
    }
    // something on the frontier still has candidates or was never expanded
    for (int32_t id : levels.back().nodes) {
        const auto & n = nodes[id];
        if (!n.alive) {
            continue;
        }
        if (!n.expanded) {
            return true;
        }
        for (const auto & c : n.cands) {
            if (!c.used) {
                return true;
            }
        }
    }
    return false;
}

// batched draft decode over the frontier's live, unexpanded nodes: each row
// yields the node's top candidates and the hidden row its children draft from
bool common_spec_tree::expand() {
    if (levels.empty()) {
        return true;
    }

    std::vector<int32_t> rows;
    for (int32_t id : levels.back().nodes) {
        if (nodes[id].alive && !nodes[id].expanded) {
            rows.push_back(id);
        }
    }
    if (rows.empty()) {
        return true;
    }
    GGML_ASSERT((int32_t) rows.size() <= params.width);

    const int64_t t0 = ggml_time_us();

    common_batch_clear(batch_dft);
    for (int32_t id : rows) {
        const auto & n = nodes[id];
        common_batch_add(batch_dft, n.tok, n.pos, { n.seq }, true);
        std::memcpy(batch_dft.embd + (size_t) (batch_dft.n_tokens - 1) * n_embd, n.h_in.data(), (size_t) n_embd * sizeof(float));
    }

    const int32_t rc = llama_decode(params.ctx_dft, batch_dft);
    if (rc != 0) {
        LOG_ERR("%s: draft decode failed rc=%d (rows=%zu pos=%d)\n", __func__, rc, rows.size(), (int) nodes[rows[0]].pos);
        return false;
    }

    for (size_t i = 0; i < rows.size(); ++i) {
        auto & n = nodes[rows[i]];

        common_sampler_reset(smpls[i].get());
        common_sampler_sample(smpls[i].get(), params.ctx_dft, (int) i, true);
        const auto * cur_p = common_sampler_get_candidates(smpls[i].get(), true);

        const float * h = llama_get_embeddings_nextn_ith(params.ctx_dft, (int) i);
        std::memcpy(h_out.data() + i * n_embd, h, (size_t) n_embd * sizeof(float));

        n.cands.clear();
        for (int k = 0; k < (int) cur_p->size && k < params.branch; ++k) {
            const float p = cur_p->data[k].p;
            if (p < params.p_min || p <= 0.0f) {
                break;
            }
            n.cands.push_back({ cur_p->data[k].id, n.logp + std::log(p), false });
        }
        n.expanded = true;
    }

    // children draft from their parent's draft output row
    for (size_t i = 0; i < rows.size(); ++i) {
        auto & n = nodes[rows[i]];
        n.h_in.assign(h_out.begin() + i * n_embd, h_out.begin() + (i + 1) * n_embd);
    }

    st.t_draft_us += ggml_time_us() - t0;
    return true;
}

// the paper's layer generation: keep the width best cumulative-probability
// candidates over the whole frontier. Also refills a queued level after pruning.
void common_spec_tree::select() {
    if (levels.empty()) {
        return;
    }
    const int32_t room = params.width - (int32_t) pending.size();
    if (room <= 0) {
        return;
    }

    struct pick { int32_t parent; int32_t k; float logp; };
    std::vector<pick> picks;
    for (int32_t id : levels.back().nodes) {
        const auto & n = nodes[id];
        if (!n.alive) {
            continue;
        }
        for (int32_t k = 0; k < (int32_t) n.cands.size(); ++k) {
            if (!n.cands[k].used) {
                picks.push_back({ id, k, n.cands[k].logp });
            }
        }
    }
    if (picks.empty()) {
        return;
    }
    std::sort(picks.begin(), picks.end(), [](const pick & a, const pick & b) { return a.logp > b.logp; });
    if ((int32_t) picks.size() > room) {
        picks.resize(room);
    }

    const int32_t depth = levels.back().depth + 1;
    for (const auto & p : picks) {
        nodes[p.parent].cands[p.k].used = true;

        // new_node() may grow the vector: copy what the child needs first
        const llama_token  tok        = nodes[p.parent].cands[p.k].tok;
        const llama_pos    pos        = nodes[p.parent].pos + 1;
        const llama_seq_id parent_seq = nodes[p.parent].seq;
        const std::vector<float> h_in = nodes[p.parent].h_in;

        const int32_t id = new_node();
        auto & c = nodes[id];
        c.tok        = tok;
        c.pos        = pos;
        c.parent     = p.parent;
        c.level      = depth;
        c.parent_seq = parent_seq;
        c.logp       = p.logp;
        c.h_in       = h_in;

        nodes[p.parent].children.push_back(id);
        pending.push_back(id);
    }
    pending_depth = depth;
}

int32_t common_spec_tree::submit_next() {
    if (root < 0 || n_inflight() >= params.depth) {
        return 0;
    }

    if (pending.empty()) {
        if (!expand()) {
            return -1;
        }
        select();
        if (pending.empty()) {
            return 0;
        }
    }

    const int32_t lane = alloc_lane();
    if (lane < 0) {
        LOG_ERR("%s: no free lane\n", __func__);
        return -1;
    }

    const int64_t t0 = ggml_time_us();

    // the level's rows take the lane's seq block, ascending: the recurrent cache
    // wants one contiguous cell run per ubatch
    common_batch_clear(batch_tgt);
    int64_t t_seq_rm = 0, t_seq_tgt = 0, t_seq_dft = 0;
    for (size_t i = 0; i < pending.size(); ++i) {
        auto & n = nodes[pending[i]];
        n.seq  = params.seq_base + lane * params.width + (llama_seq_id) i;
        n.lane = lane;
        n.row  = (int32_t) i;

        int64_t ta = ggml_time_us();
        sync_seq(n.parent_seq, n.seq);
        int64_t tb = ggml_time_us();
        t_seq_rm += tb - ta;
        (void) t_seq_tgt; (void) t_seq_dft;

        common_batch_add(batch_tgt, n.tok, n.pos, { n.seq }, true);
    }

    const int64_t t1 = ggml_time_us();
    const int32_t rc = llama_pipedec_tree_submit(params.ctx_tgt, &batch_tgt, lane);
    const int64_t t2 = ggml_time_us();
    TREE_TRC("submit-cost rows=%zu seq_rm=%.2f seq_cp_tgt=%.2f seq_cp_dft=%.2f ctx_submit=%.2f ms\n",
            pending.size(), t_seq_rm/1000.0, t_seq_tgt/1000.0, t_seq_dft/1000.0, (t2 - t1)/1000.0);
    if (rc != 0) {
        LOG_ERR("%s: level submit failed rc=%d (lane=%d rows=%zu depth=%d)\n", __func__, rc, lane, pending.size(), pending_depth);
        for (int32_t id : pending) {
            free_seq(nodes[id].seq);
            nodes[id].seq = -1;
            nodes[id].lane = -1;
        }
        return -1;
    }

    levels.push_back({ lane, pending_depth, pending });
    lane_used[lane] = true;
    ring_cursor = (lane + 1) % params.lanes;

    st.n_levels += 1;
    st.n_rows   += (int64_t) pending.size();
    st.t_submit_us += ggml_time_us() - t0;

    TREE_TRC("submit lane=%d depth=%d rows=%zu inflight=%d\n", lane, pending_depth, pending.size(), n_inflight());

    pending.clear();

    // a restart root has been submitted: its parent's seq is no longer needed
    for (llama_seq_id s : hold_seqs) {
        free_seq(s);
    }
    hold_seqs.clear();

    return (int32_t) levels.back().nodes.size();
}

int32_t common_spec_tree::close_oldest() {
    if (levels.empty()) {
        return -1;
    }
    const auto & lv = levels.front();

    int32_t row = -1;
    for (size_t i = 0; i < lv.nodes.size(); ++i) {
        if (lv.nodes[i] == root) {
            row = (int32_t) i;
            break;
        }
    }
    GGML_ASSERT(row >= 0 && "the root must be in the oldest level");
    GGML_ASSERT(nodes[root].row == row);

    const int32_t rc = llama_pipedec_tree_close(params.ctx_tgt, lv.lane, row);
    if (rc != 0) {
        LOG_ERR("%s: level close failed rc=%d (lane=%d row=%d)\n", __func__, rc, lv.lane, row);
        return -1;
    }

    int64_t t_wait = 0, t_head = 0;
    llama_pipedec_tree_close_timing(params.ctx_tgt, &t_wait, &t_head);
    st.t_wait_us += t_wait;
    st.t_head_us += t_head;
    st.n_steps   += 1;

    TREE_TRC("close lane=%d depth=%d row=%d wait=%.1fms head=%.1fms\n", lv.lane, lv.depth, row, t_wait/1000.0, t_head/1000.0);
    return 0;
}

common_spec_tree_advance common_spec_tree::advance(llama_token x) {
    common_spec_tree_advance res;

    GGML_ASSERT(!levels.empty());
    const level lv = levels.front();
    levels.pop_front();
    lane_used[lv.lane] = false;

    node & R = nodes[root];
    GGML_ASSERT(R.lane == lv.lane);

    int32_t hit = -1;
    for (int32_t c : R.children) {
        if (nodes[c].alive) {
            res.n_children++;
            if (hit < 0 && nodes[c].tok == x) {
                hit = c;
            }
        }
    }
    st.n_children += res.n_children;

    // the target's true hidden row at the root (the lane is drained: close ran)
    const float * h_true = llama_pipedec_tree_h(params.ctx_tgt, lv.lane, R.row);
    GGML_ASSERT(h_true != nullptr);

    if (hit >= 0) {
        res.hit = true;
        st.n_hits += 1;

        // the other bloodlines die, queued or in flight
        kill_subtree_except(root, hit);

        {
            node & C = nodes[hit];
            // children not drafted yet can use the true row instead of the chained one
            if (!C.expanded) {
                C.h_in.assign(h_true, h_true + n_embd);
            }
            C.parent = -1;
        }

        // R is consumed: C's lane already shares R's cells and gathered its state
        free_seq(R.seq);
        R.children.clear();
        free_node(root);
        root = hit;

        // a queued level may have lost rows to the pruning: refill it
        if (!pending.empty()) {
            select();
        }

        TREE_TRC("hit x=%d children=%d inflight=%d pending=%zu\n", (int) x, res.n_children, n_inflight(), pending.size());
    } else {
        // miss: the whole tree dies and x restarts it under R's seq
        st.n_restarts += 1;

        std::vector<float> h_copy(h_true, h_true + n_embd);
        const llama_pos    pos  = R.pos + 1;
        const llama_seq_id rseq = R.seq;

        for (const auto & l : levels) {
            llama_pipedec_tree_discard(params.ctx_tgt, l.lane);
            lane_used[l.lane] = false;
        }
        levels.clear();

        auto children = R.children;
        for (int32_t c : children) {
            kill(c);
        }
        GGML_ASSERT(pending.empty());

        // R's seq feeds x's lane and is freed after that submit
        hold_seqs.push_back(rseq);
        R.seq = -1;
        free_node(root);

        const int32_t id = new_node();
        node & X = nodes[id];
        X.tok        = x;
        X.pos        = pos;
        X.parent     = -1;
        X.level      = 0;
        X.parent_seq = rseq;
        X.logp       = 0.0f;
        X.h_in       = std::move(h_copy);

        root = id;
        pending = { id };
        pending_depth = 0;

        TREE_TRC("miss x=%d children=%d restart pos=%d\n", (int) x, res.n_children, (int) pos);
    }

    return res;
}

llama_seq_id common_spec_tree::finish(llama_pos * trunk_pos, const float ** trunk_h) {
    if (root < 0) {
        return -1;
    }

    node & R = nodes[root];

    llama_seq_id trunk;
    llama_pos    tpos;
    if (R.lane >= 0) {
        // the committed state ends at the root; read its row before the lanes go
        const float * h = llama_pipedec_tree_h(params.ctx_tgt, R.lane, R.row);
        GGML_ASSERT(h != nullptr);
        trunk_h_buf.assign(h, h + n_embd);
        trunk = R.seq;
        tpos  = R.pos;
    } else {
        // the root was never submitted: the state ends at its parent
        trunk_h_buf = R.h_in;
        trunk = R.parent_seq;
        tpos  = R.pos - 1;
    }

    for (const auto & l : levels) {
        llama_pipedec_tree_discard(params.ctx_tgt, l.lane);
        lane_used[l.lane] = false;
    }
    levels.clear();

    auto children = R.children;
    for (int32_t c : children) {
        kill(c);
    }
    pending.clear();

    for (llama_seq_id s : hold_seqs) {
        if (s != trunk) {
            free_seq(s);
        }
    }
    hold_seqs.clear();

    free_node(root);
    root = -1;
    nodes.clear();
    free_nodes.clear();

    // the next request rewrites the slot below the marks, so the trunk cells
    // the lanes kept sharing go now; the caller empties the trunk seq itself
    clear_seqs(trunk);

    if (trunk_pos) {
        *trunk_pos = tpos;
    }
    if (trunk_h) {
        *trunk_h = trunk_h_buf.data();
    }

    TREE_TRC("finish trunk_seq=%d trunk_pos=%d\n", (int) trunk, (int) tpos);
    return trunk;
}

std::string common_spec_tree::summary() const {
    const double steps = st.n_steps > 0 ? (double) st.n_steps : 1.0;
    char buf[512];
    snprintf(buf, sizeof(buf),
            "tree: steps=%lld hits=%lld (%.1f%%) children/step=%.2f restarts=%lld levels=%lld rows=%lld | "
            "per step ms: draft %.2f submit %.2f wait %.2f head %.2f",
            (long long) st.n_steps, (long long) st.n_hits, 100.0 * st.n_hits / steps,
            (double) st.n_children / steps, (long long) st.n_restarts, (long long) st.n_levels, (long long) st.n_rows,
            st.t_draft_us / 1000.0 / steps, st.t_submit_us / 1000.0 / steps,
            st.t_wait_us / 1000.0 / steps, st.t_head_us / 1000.0 / steps);
    return buf;
}
