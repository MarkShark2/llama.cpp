# PipeDec on llama.cpp RPC — design notes (branch `feat/pipedec`)

Goal: fill the pipeline bubble when a model is layer-split across RPC nodes ("The House"),
adapting PipeDec (arXiv 2504.04104). Measured baseline (Step-3.7-Flash, 5-stage House,
MTP n_max=3): decode ~9 t/s, of which the **verify pass is ~294 ms (87%) and runs strictly
serial across the 5 stages** — one device active at a time. That serial verify is the bubble.

## Why it's serial today

llama.cpp already has a pipeline-parallel scheduler (`ggml_backend_sched` with `n_copies`
double-buffering + events) that overlaps layer-split stages across micro-batches. It is
gated in `llama-context.cpp` on every device advertising `caps.async && caps.events`.
**The RPC backend advertises `async=false, events=false`** and provides no event or
async-copy ops (`ggml-rpc.cpp`), so the gate fails and RPC pipelines run fully serial:
each stage boundary blocks the scheduler thread on a `GET_TENSOR` (star topology — every
inter-node hop hairpins through the client).

## Phase 1 — async + events in the RPC backend (this branch)

Make each RPC endpoint behave like an async "stream" so the existing scheduler can overlap
stages. Key observation: **each endpoint already has one persistent, strictly in-order
socket** (server serves one client, processes commands FIFO). That socket *is* a stream.

Add a per-endpoint **worker thread** (`rpc_stream`) that owns the socket and runs queued
tasks in order. All decode-hot-path backend ops become non-blocking enqueues so the
scheduler thread never blocks on one endpoint while another could compute:

- `graph_compute`  → serialize on caller (cheap), enqueue the socket send (already
  fire-and-forget). Non-blocking.
- `synchronize`    → drain this endpoint's queue (+ one round-trip so the last compute is
  known complete). Blocks host.
- `event_record(ev, be)` → reset `ev`, enqueue a task on `be`'s stream that signals `ev`.
- `event_wait(be, ev)`   → enqueue a task on `be`'s stream that blocks until `ev` signaled
  (cross-stream ordering; does not block the scheduler thread).
- `event_synchronize(ev)`→ host waits until `ev` signaled.
- `cpy_tensor_async(A,B,src,dst)` → star copy: enqueue `GET src` on A's stream into a
  shared host buffer, enqueue `SET dst` on B's stream that waits for that buffer. Both
  run off the scheduler thread, so A can start the next micro-batch while B consumes.
- `set_tensor_async` / `get_tensor_async` → enqueue SET/GET on the endpoint stream.

Invariant: the worker is the **sole** user of its socket. Non-hot-path ops (buffer
alloc/free, weight upload during load, `get_device_memory`) drain the stream first so the
socket is never touched concurrently. Events are `{mutex, cv, bool signaled}`, reset on
record (safe because the scheduler issues record→wait in program order on one thread).

Then advertise `async=true, events=true`, wire `event_new/free/synchronize` at the device
level, and relax the `llama-context` gate so it engages for RPC layer-splits.

**Phase-1 test (independent of the tree work):** prompt/prefill already emits many
micro-batches, so pipelining should overlap stages during prompt processing —
visible as higher prompt t/s and concurrent per-backend times in `GGML_SCHED_TIMING=1`.

## Route decision (2026-07-17): ggml `pipeline_parallel` is a dead end here

Forcing ggml's built-in `pipeline_parallel` (via a `LLAMA_PIPELINE_PARALLEL` env override,
since `--fit` sets tensor-buft overrides and trips the `!has_tensor_overrides()` gate)
blows the CUDA0 compute buffer up to **~260 GiB**. Cause: `n_copies` (=4) duplicates every
cross-backend boundary tensor and pins it input+output (non-reusable, `ggml-backend.cpp`
~1364), and in the RPC **star** topology every hop hairpins through the client (CUDA0), so
all those pinned copies pile up there. fit's own `memory_breakdown` shows CUDA0's `compute`
column moving *inversely* with its layer count (14 layers → 348 MiB; 0 layers → 281 GiB).
`pipeline_parallel` assumes direct peer links; it is the wrong vehicle for RPC-star + fit.
**Abandoned.** (The `LLAMA_PIPELINE_PARALLEL` override + `GGML_SCHED_PROBE` diagnostic and
the async RPC backend from Phase 1 remain, all env-gated / off by default.)

## Route B — custom software pipeline over the async RPC streams

Keep `--fit`'s exact placement and **serial-sized** per-device compute buffers (NO
`n_copies`). Get stage overlap from the Phase-1 per-endpoint async streams.

Use `--fit-whole-layers` with `--fit on` for PipeDec profiles. It keeps fit dynamic—model
metadata, context size, current free memory, and per-device `--fit-target` margins are still
probed at every load—but treats MoE layers as indivisible. The result is a contiguous tensor
split with no per-tensor/CPU expert overrides, avoiding the cross-stage expert traffic that
previous fit output introduced. PipeDec also suppresses llama.cpp's built-in four-copy
pipeline scheduler automatically, because a clean tensor split no longer has overrides that
would otherwise disable that scheduler heuristic.

Key memory insight: each device (client CUDA0 + each RPC node) owns its own compute buffer.
If micro-batches are pipelined so that at any instant each device is working on a *different*
micro-batch, each device only ever holds **one** micro-batch's working set = the serial fit
buffer. Within a single node's in-order stream, micro-batches run sequentially and reuse
that one buffer safely: the boundary GET (stage output → client) is ordered before the next
micro-batch's compute overwrites it. So overlap is free of buffer duplication — the opposite
of `pipeline_parallel`'s pile-up.

Mechanism: submit the speculative-verify tree as several **micro-batch graphs** through the
async sched **without synchronizing between them**, so consecutive micro-batches queue on
each node's stream and different nodes run different micro-batches concurrently. Latency of
the verify drops from Σ(stage compute) toward the bottleneck stage; draft folds in as stage 0.

Open implementation questions (design before coding):
- Driving multiple in-flight graphs through `ggml_backend_sched` without it synchronizing or
  reusing a buffer a still-pending micro-batch needs (may need to bypass the sched for the
  pipeline and drive the streams directly, PipeDec central-scheduler style).
- KV-cache correctness across pipelined tree micro-batches (two-level cache in the paper).
- Where to hook: the server spec loop (`server-context.cpp`, the `llama_decode(ctx_tgt,
  batch_view)` verify) vs. deeper in `llama-context`.

Validation rule: at every increment, compare memory against the **stock baseline**
(recorded 2026-07-17): load ws ~6.8 GB / commit ~43 GB; ready+decode ws ~11.4 GB /
commit ~52 GB / ~21 GB sys used; page file FLAT ~1.2 GB. Danger only if commit → ~90 GB+
or the page file grows. Stock decode = 9.67 t/s (the number to beat).

## Route B milestone 1 — non-blocking sched walk (2026-07-18)

Implemented, env-gated `GGML_PIPEDEC=1` (requires `GGML_RPC_ASYNC=1`):

- **ggml-rpc.cpp** (correctness fixes, active whenever async is on):
  - every stream task now snapshots what it needs **at enqueue time** (serialized
    rpc_tensor / wire message / struct copy) — graph tensor structs are rewritten by
    the next ubatch's build before the task runs;
  - host-visible-src `cpy_tensor_async` records an event on the src backend at enqueue
    (submission order) and the worker waits it before reading — previously it raced the
    src backend's still-running compute;
  - `get_tensor_async` routes through the endpoint stream (never touches the socket
    concurrently with the worker) and is **deferred**: data lands when the stream
    reaches it, callers' later synchronize completes it. The per-ubatch MTP nextn
    read from the *last* stage no longer stalls the walk;
  - RPC → non-RPC copies return false (scheduler sync fallback).
- **ggml-backend.cpp sched** (gated `GGML_PIPEDEC=1`, n_copies==1 only):
  - FLAG_INPUT copies use `tensor_set_async` (snapshot semantics, stream-ordered) —
    no host-side synchronize;
  - the pre-copy "wait for backend before overwriting input" synchronize is skipped
    (every copy path lands on the dst stream/socket in submission order);
  - when the batch is big enough that ~every expert is used, host-spilled MoE expert
    tensors are uploaded whole + async instead of the router-ids read (which stalls
    the walk on the split backend per spilled layer);
  - `GGML_PIPEDEC_TRACE=1` logs per-walk time split (inp/moe_ids/moe_cpy/fallback/
    submit) + per-tensor blocking-fallback hits > 5ms.

**Result: memory identical to stock** (ws ~11.6 GB, commit ~53 GB, pagefile flat).
Correct output. Decode ~unchanged (expected: one verify graph is still a serial
chain). Prefill ~unchanged (75 t/s) — and the trace shows **why**, which is the real
finding:

### Finding: prefill on this config is placement-bound, not bubble-bound

fit's spillover scatters MoE experts across the pipeline: early-layer experts →
CUDA_Host (6.2 GB, re-uploaded each 512-tok ubatch since ~all 288 experts are used),
**layer-40 experts → CPU**, layer-41-44 experts → CUDA0 while those layers' attention
stays on shredder. The walk trace shows ~5.7 s/ubatch on the submission thread:
`ffn_moe_swiglu-40` (CPU ← shredder) alone absorbs ~4.3 s — the CPU split executes
synchronously on the sched thread and needs the terminal stage's output mid-graph, so
no ubatch overlap is possible regardless of stream asynchrony. The remaining ~1 s is
the layer-41-44 shredder↔CUDA0 expert ping-pong through the blocking sync fallback.

Unlocking prefill overlap would need (a) an async CPU-backend proxy (worker thread +
events, like rpc_stream) so CPU splits don't block the walk, and (b) the staged
pinned-buffer async RPC→CUDA copy path. Both are secondary to the decode goal.

## Next — decode: cross-iteration pipeline (the real PipeDec)

Within one verify pass the stages are serial by data dependency; splitting the 4-token
verify into 1-token chained graphs hits the **CUDA0 head cycle**: CUDA0 hosts stage 0
*and* the output head, so in stream order `head(t0)` blocks `stage0(t1)` and the
pipeline collapses to depth 1. The paper's answer is the one that works: pipeline
across *timesteps* with the dynamic tree — feed tree level k+1 into stage 0 while
level k is mid-pipeline, defer accept/prune decisions by the pipeline depth, and keep
the head off the per-level critical path.

### Measured decode anatomy (2026-07-18, walk trace during 200-tok generation)

- verify walk ≈ **190-210 ms** for a 4-token verify graph (34 splits, 4092 nodes);
  baseline "verify 294 ms" has compressed to ~200 ms under async submission.
- of that, `submit` ≈ **50 ms** — CUDA0 appears in many small splits (L0-12, L13
  experts, L41-44 experts, head), each its own graph launch.
- the walk's blocking waits concentrate at the placement pathologies: `ffn_moe_probs-13`
  (CUDA0 ← fedora, L13 attn/router on fedora but experts on CUDA0) ~35 ms, then
  `ffn_moe_swiglu-40` (CPU ← shredder, L40 experts on CPU) ~100 ms — the latter absorbs
  the remaining upstream chain time (first blocking point), it is not additive.
- draft (MTP head, CUDA0) walks: ~3-7 ms each, 3 per iteration.
- effective avg stage time ≈ 40 ms (Σ≈200 ms over ~5 stages); max stage ~60 ms.

**Projection**: steady-state cross-iteration pipeline emits one tree level per
max-stage time (~60 ms). Because the BC-250 stages are weight-streaming-bound,
t_stage(8 tok) ≈ t_stage(1 tok) — so *wide* tree levels are nearly free and raise
accepted-tokens-per-level; realistic ceiling ~1.5-2x decode (→ 15-20 t/s), consistent
with the original estimate discounted for depth-5 and heterogeneity.

### Hook points identified

- server verify: `tools/server/server-context.cpp` `decode()` (~line 3647) —
  `llama_decode(ctx_tgt, batch_view)` + `common_speculative_process`.
- MTP draft: `common/speculative.cpp` `common_speculative_impl_draft_mtp` (~line 1204)
  — feeds `llama_get_embeddings_nextn(ctx_tgt)` rows into per-head draft decodes on
  ctx_dft via `llama_set_nextn_layer_offset`.
- deferred head needs a new head-only graph type (input: host-provided hidden rows →
  final norm → lm_head → logits) plus target decodes that request nextn embeddings but
  no logits (the nextn read is already async/deferred after milestone 1).
- config-level counterpart worth testing first: keep fit's late-layer expert spillover
  off the CPU (L40) and minimize mid-chain straddles (L13, L41-44) — these shape the
  serial verify floor for every route.

## Stage 2 groundwork — token-body lanes + deferred head (2026-07-18, env-gated OFF)

`GGML_PIPEDEC_STAGE2=1` (step35 only, narrow eligibility, everything else falls through
to the normal decoder): an MTP verification batch is split into per-token decoder-body
graphs, each on its own persistent scheduler lane (`sched_pipedec_body[lane]`, stable
graphs/activations, no buffer races), bodies stop at `h_nextn`; the hidden rows are then
fed to one batched output-norm/LM-head graph on a separate persistent head scheduler.
The lanes opt into the scheduler's new stable-host-inputs mode (async dst-stream upload
of host-backed split inputs — exported `extern "C"` from ggml-backend.cpp, deliberately
NOT in ggml-backend.h so the public header, a dependency of every CUDA object, stays
untouched).

**Measured on the 6-stage config** (below): correct output, body_submit ~9 ms (the
stable-input fix removed the progressive serialization), body_drain ~105 ms, head
~11 ms → verify ~120-135 ms per 4-token group, decode 7.7 t/s. The plain batched
verify graph does the same 4 tokens in one ~80 ms walk — batching is nearly free on
weight-streaming-bound stages, so **chain-splitting a single iteration cannot beat the
batched graph**. The lanes + deferred head only pay off with *cross-iteration* overlap
(submit iteration k+1's draft/bodies before draining iteration k). Stage 2 therefore
ships OFF; the machinery is the foundation for the real pipeline.

## Config change that mattered more (2026-07-18): whole-layer fit, 6 stages

- shredder's daemon now exposes **both** GTX 1080s (`device: CUDA0,CUDA1` in
  `serve.rpc-shredder.json`) → RPC3/RPC4 on one endpoint; with `--device
  CUDA0,RPC0,RPC1,RPC2,RPC3,RPC4 --fit-target 1024,512,512,512,512,512`.
- `--fit-whole-layers` + the extra ~8 GB VRAM ⇒ **no spill at all**: CUDA0 18.6 GB,
  fedora ×3 13.1 GB, shredder 6.5/5.3 GB, CPU only the 413 MB token embedding. The
  CUDA_Host expert re-upload, the L40 CPU trampoline, and the straddled layers are gone.
- Result (normal path, `GGML_PIPEDEC=1`, stage2 off): **prefill 92 t/s** (was 75),
  **decode 13.8-15.6 t/s** at 55-85 % MTP acceptance (was 9.7 stock). This placement is
  the new baseline every pipelining experiment must beat.

## Design-deciding measurements (2026-07-18, 6-stage no-spill config)

- Per-iteration: draft 13.4 + **verify 146.2** (submit 5, rest = drain) + spec_proc 12.7 +
  sample 8.1 + pre 13.7 ≈ 194 ms for 2.82 tokens/iter.
- **Width is NOT free**: batched-bench per-forward latency at widths 1/2/4/8/16 =
  62/113/139/272/642 ms. MoE expert streaming scales with tokens — the BC-250 stages are
  expert-streaming-bound. Wide (batched) tree levels are therefore expensive; STPP-style
  trees lose as batches. (Gotcha: step35 `chain_heads` clamps `n_max` to its 3 MTP heads —
  a `--spec-draft-n-max 8` run still drafts 3.)
- **Top-K probe** (always-on, logs `spec tree probe` every 32 rejections): at rejections the
  target's token is in the draft's top-2 ~40-46 %, top-3 ~53-58 % → a K=3 sibling level
  lifts per-level acceptance 0.56 → ~0.8. The uplift is real if width can be made cheap.
- **Stage-2 lanes are vindicated on graph time**: 105 ms body drain + 11 ms head beats the
  ~135-146 ms batched verify — stage overlap dodges the expert-streaming width cost. The
  7.7 t/s regression was pure overhead (spec_proc 43 ms vs 12.7, per-lane syncs).

Next: (1) fix stage-2 overheads (spec_proc / lane-sync granularity) → ~16.5 t/s projected;
(2) pipeline the MTP draft steps into the lane pipeline; (3) add K=2-3 first-level sibling
branches as *extra lanes* (width via pipeline, not batch), with tree attention via scratch
seq-ids (branch in own seq, prefix seq_cp'd in, accepted branch seq_cp'd back).

## The ping-storm fix (2026-07-18, commit a09fcbcfd) — stage 2 becomes the fast path

The stage-2 "overhead" was almost entirely **redundant RPC barrier pings**:
`ggml_backend_rpc_synchronize` always did a GET_ALIGNMENT round-trip;
`llama_context::synchronize()` fires it per sched × per backend (~10 scheds under
stage 2), and every logits/embeddings accessor triggers a context synchronize.
`rpc_stream` now tracks `dirty_seq` (newest enqueued task) vs `barrier_seq` (covered
by the last completed ping) and skips the round-trip on quiescent streams.

Same prompt, same ~53 % acceptance, measured end-to-end:

| mode | before | after | per-iter after |
|---|---:|---:|---|
| stage-2 lanes | 7.7 t/s | **21.9 t/s** | draft 7.7 + verify 116.8 (drain ~80, head ~11) + spec_proc 3.1 + sample 0.4 + pre 8.1 ≈ 136 ms |
| batched verify | 13.8 t/s | 18.6 t/s | verify ~136 ms dominates |

spec_proc 43 → 3.1 ms, sample 8 → 0.4 ms. The lanes beat the batched graph by ~17 %,
exactly the body-drain vs batched-walk gap measured earlier. `GGML_PIPEDEC_STAGE2=1`
is now the production mode for House serving (launch script `-Stage2 1`). Remaining
floor: the ~80 ms lane pipeline (≈ (n_stages + n_tokens − 1) × stage time) — next
levers are draft-step overlap and sibling-branch lanes (see above).

## Deferred verify group — draft overlap (2026-07-18, commit 70b0dda1e)

`GGML_PIPEDEC_DEFER=1` (requires STAGE2): before running the MTP draft, the server
submits the sampled token's stage-2 lane as a *deferred group member*
(`llama_pipedec_defer` + a 1-token `llama_decode`: lanes submitted, no drain, no
head). The MTP draft steps then overlap that lane's 6-stage walk. The iteration's
regular decode carries only the draft tokens and *closes* the group: one drain, one
LM head over all group rows, logits indexed by group row (identity output mapping;
`spec_i_batch` holds group rows). Fork API: `llama_pipedec_defer/abort/group_n` in
`llama-ext.h`. Stage-2 hidden rows always land in the fixed-capacity
`pipedec_group_h` buffer (deferred GETs snapshot their destination address) and are
copied into `embd_nextn` at close.

Safety rails: empty draft → `llama_pipedec_abort` + `seq_rm` of the deferred row +
combined fallback; deferral only fires when the pass's batch will contain nothing
but the drafting slot's tokens (one generating+drafting slot, all others IDLE —
n_parallel auto-defaults to 4 slots); `release()` aborts any open group; a
non-eligible decode with a group open aborts it defensively.

Measured (same prompt, temp 0): output **bit-identical** in all modes; 21.0 →
**21.6 t/s**. The draft's serial 7-8 ms moves into the pipeline shadow (its wall
time grows to ~14 ms from CUDA0-stream contention with lane 0 but is overlapped).
The decode-without-drain + group-close machinery is the foundation for
cross-iteration overlap.

## Proto v5 deployed + marginal-lane cost (2026-07-18 evening)

Multi-slot RPC graph cache (proto v5, bcd103546) deployed to all daemons (fedora ×3 +
shredder rebuilt + restarted). One fix on the way: `extern "C" GGML_API` is invalid on
GCC (GGML_API expands to `... extern` on Linux) — block-form `extern "C" { }` instead
(4c7150ce2). New baseline with the cache: **23.5-25.2 t/s** (was 21.6).

**Marginal-lane cost** (DraftMax 1 vs 3, steady-state): marginal verify 65 ms @ 2 lanes
vs 93 ms @ 4 lanes → **~14 ms per extra lane** (≈ one stage time, as the
(n_stages + n_lanes − 1) × t_stage model predicts; t_stage ≈ 10 ms).

**Sibling-branch lanes are rejected on this data**: a K=2 sibling chain of depth 3 adds
3 lanes (+42 ms, +45 % verify) for only ~10-13 % more accepted tokens (probe coverage
40-46 % of level-0 rejections × rejection rate). Width is not free even via lanes —
every extra lane pays a full stage time. Shelved unless per-lane cost drops a lot.

## Streamed draft lanes (2026-07-18, env-gated OFF — groundwork)

`GGML_PIPEDEC_STREAM=1` (requires DEFER): `common_speculative_draft_params.on_draft_token`
callback fires as each MTP draft token is sampled; the server submits that token's lane
as another deferred group member mid-draft, so lanes stream into the pipeline instead of
lump-submitting after the 16 ms draft loop. Safety: never fires for a token that could be
the draft's last (n_max_cap gate); if the draft stops early right after a streamed token,
`handle_last_sampled_token` aborts the group + seq_rm + combined fallback.

Measured: **loses ~3 t/s** (21.4-21.8 vs 24-25 off). The pipeline effect is real — the
closing lane's body_drain drops 67 → 53 ms — but each streamed 1-token `llama_decode`
costs ~10 ms of CPU-side submission (balloc/memory init, graph build+alloc), serial inside
the draft loop (draft 15 → 39 ms) and it also pushes `pre` 19 → 43 ms. The stagger gain
(~10 ms/lane) cannot beat a ~10 ms/lane submission cost. **Ships OFF.** To revisit:
make lane submission cheap (prebuilt per-lane graph with input rebind, skip balloc/mctx
for the 1-token steady case) — then streaming becomes nearly free and should win ~10 ms.

Note: stream on/off produce *different* (both-deterministic) outputs at temp 0 — KV padding
/ ubatch-boundary numerics, not a correctness issue (per-mode output is bit-stable across
server restarts).

## Later — dynamic prediction tree + pruning + tree attention

The paper's accuracy machinery: keep all stages fed with tokens likely to survive
verification so the filled pipeline emits *accepted* tokens.
