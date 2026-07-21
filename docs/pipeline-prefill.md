# Pipelined chunked prefill over RPC — notes (branch `feat/pipeline-prefill`)

Goal: overlap prefill ubatches across a layer-split multi-endpoint RPC pipeline ("The
House": RTX 3090 + 3× BC-250 over Vulkan RPC + shredder 2× GTX 1080 over CUDA RPC = 6
stages). llama.cpp's existing pipeline parallelism (`ggml_backend_sched`, `n_copies`
input slots + per-(backend, slot) events) already expresses this; with `feat/rpc-async`
the RPC devices advertise the async+events caps that gate it. Enabling it alone did
nothing — prefill stayed at the sequential baseline. Two hidden serialization points had
to go.

Measured on Step-3.7-Flash IQ3_XXS (45 layers, 288-expert MoE), ~9.8k-token prompts,
`--batch-size 8192`, ubatch 512, `--parallel 1`, July 20 2026:

| configuration | prefill t/s |
|---|---:|
| sequential (no pipelining) | 123.1 |
| pipelined, before fixes | ~126 |
| pipelined, both fixes, warm | **306** |
| pipelined, both fixes, first request per process | ~127 |

## Fix 1 — INPUT copies drained the pipeline (`ggml-backend.cpp`)

With `n_copies > 1` the scheduler's INPUT branch copied host-side inputs (`kq_mask`,
positions, …) to each split backend via `ggml_backend_event_synchronize` + a blocking
ordered `tensor_copy`. On an RPC endpoint that copy queues behind the endpoint's
in-flight compute, so every ubatch waited for the whole pipeline once per stage.

The PipeDec decode walk already had the safe alternative for `n_copies == 1`: a staged
`set_tensor_async` (the RPC async set snapshots the payload at call time and lands on the
destination stream FIFO-ordered behind the previous graph). The same ordering argument
holds for `n_copies > 1` — `input_cpy` is the current copy slot, and the async set lands
behind the graph that last read that slot — so the INPUT branch now uses it whenever the
backend has `set_tensor_async` and the input lives in a host buffer.

## Fix 2 — GET_ALLOC_SIZE round trips (`ggml-rpc.cpp`)

`ggml_gallocr` asks the buffer type for the alloc size of every FLASH_ATTN_EXT and
MUL_MAT_ID node on **every graph allocation**; for RPC that was a blocking ordered round
trip (~70 per ubatch), each queued behind ~1 s of in-flight compute. Invisible when
sequential (the queue is empty), fatal when pipelined.

The responses are now cached client-side, keyed by endpoint, device, and the full
type/op/ne/nb signature of the tensor and all of its sources (this resolves the upstream
`TODO: cache the alloc responses`). Steady state makes zero GET_ALLOC_SIZE calls.
Because the cache is per-process and attention shapes depend on n_kv, **the first
request after a server start pays every miss** (~127 t/s) and later requests run warm.
Fixed-length chunks (an SPD data-collection loop) only pay this once.

## Knobs and diagnostics (all env-gated, default off)

- `GGML_RPC_ASYNC=1` — required; RPC async streams/events (from `feat/rpc-async`).
- `LLAMA_PIPELINE_PARALLEL=1` — force the pipeline-parallel heuristic on.
- `LLAMA_GRAPH_REUSE_DISABLE=1` — required for throughput: the graph-reuse path
  synchronizes (full drain) before every reused graph.
- `GGML_SCHED_COPIES=N` — runtime pipeline depth 1..`GGML_SCHED_MAX_COPIES` (compile
  ceiling raised 4 → 8; it is a **CMake cache variable**, the source `#ifndef` is dead —
  `build-llama.cmd` configures it). Default 4; 8 measured no gain (see below).
- `GGML_SCHED_SUBMIT_TRACE=1` — raw-stderr `[submit]`/`[pu]`/`[pu-alloc]`/`[sched-alloc]`
  per-graph phase timing plus a REALLOC-drain marker; `GGML_SCHED_TIMING=1` — per-backend
  per-split table, now mirrored to stderr (the server log filter hides lib INFO lines).
- The `pipeline_parallel=` decision is printed unconditionally on stderr at context
  creation for the same reason.

## Findings / limits

- Per-stage cost for a 512-token graph: CUDA0 ~350 ms, each BC-250 ~1000–1030 ms,
  shredder ~255 + ~230 ms. Sum ≈ 3.85 s → **ceiling ≈ 500 t/s** with perfect overlap;
  306 t/s means the BC-250s are already near-saturated. The residual per-endpoint serial
  cost is server-side per-graph deserialize/build on the BC-250 CPUs — graphs differ
  every ubatch (n_kv grows) so the proto-v5 graph cache never hits. An n_kv-tolerant
  server-side graph cache is the next lever and needs a daemon redeploy.
- Deeper pipelines don't help: depth 8 ≈ depth 4. The bottleneck is per-endpoint serial
  work, not in-flight slots.
- **Decode under this mode is unusable** (~0.2 t/s): per-token graphs rebuild and
  re-allocate every step. This is a prefill/data-collection mode; serve with the normal
  PipeDec config.
- A reuse-path variant that rotated copy slots instead of synchronizing
  (`LLAMA_PIPELINE_REUSE` + `ggml_backend_sched_advance_copy`) **corrupted output** and
  was removed before commit. If graph reuse under pipelining is ever revisited, the
  input-slot rotation needs a real safety argument for the non-INPUT split boundaries.
- Output verified: greedy completion on a real prompt is byte-identical with pipelining
  on and off.
