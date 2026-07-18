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

## Later — dynamic prediction tree + pruning + tree attention

The paper's accuracy machinery: keep all stages fed with tokens likely to survive
verification so the filled pipeline emits *accepted* tokens.
