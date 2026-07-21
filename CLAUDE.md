IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work on upstream llama.cpp code.

---

# Fork notes — BC-250 cluster build of llama.cpp

This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) maintained to run large MoE models across a small heterogeneous cluster. This section documents the fork; everything above/around it is upstream.

## Hardware & topology

- **Desktop (this machine, Windows 11):** RTX 3090 (`sm_86`). Build host and the `llama-server` RPC client for "The House". Source lives at `llama_loader\llamacpp-src`.
- **Cluster:** 3× Fedora nodes on BC-250 AMD APUs (**UMA** — system RAM shared with the GPU, host memory scarce). SSH aliases `fedora` / `fedora2` / `fedora3`; scripts under `/home/mark/llama-scripts`; Vulkan backend. Wired into a bridged **10 GbE fabric** (`10.10.10.1/.2/.3`, MTU 9000; ~5–6.4 Gbit/s usable — the BC-250 PCIe 2.0 x2 link is the ceiling). RPC daemons at `10.10.10.1:50052`, `.2:50053`, `.3:50054`.
- **shredder:** Ubuntu box, 2× GTX 1080 (`sm_61`, Pascal), CUDA-in-docker builds; RPC daemon at `192.168.4.60:50052` (~4.5 Gbit/s link).

**Current workload ("The House"):** one big MoE model (e.g. Step-3.7-Flash IQ3_XXS ~68.5 GB, or Hy3 295B ternary) layer-split across 3090 + the 3 BC-250s + shredder over RPC, launched by the Llama Loader app. See `llama_loader\CLAUDE.md` for the cluster/network details.

## Git branch strategy

Goal: stay easy to merge against fast-moving upstream, keep each feature isolated, have one branch we actually build.

- **`master`** — a clean mirror of `origin/master` (ggml-org). Never commit fork work here. Only fast-forward it:
  `git fetch origin master && git checkout master && git merge --ff-only origin/master`
- **`feat/*`** — one branch per feature, based off `master`, **code only** (no fork docs/config), so each stays rebasable and potentially upstreamable. Current:
  - **`feat/disk-cache`** — the disk cache-streaming feature (see below).
  - **`feat/ssm-dstate-96`** — mamba2 `SSM_SCAN` with `d_state == 96` in the CUDA and Vulkan backends. The Puzzle NAS shrank Nemotron-H's SSM state from 128 to 96 (`nemotron_h_moe.ssm.state_size = 96` in the GGUF); upstream kernels only ship d_state 128/256, so the op fell back to CPU on CUDA and hard-crashed Vulkan RPC workers because the RPC client's `supports_op` is a `return true` stub (upstream TODO in `ggml-rpc.cpp`) that never consults the remote backend. That stub is a standing footgun: *any* op a node's Vulkan backend rejects will abort that node's rpc-server instead of falling back. Upstreamable.
  - **`feat/vk-uma-mem`** — clamp Vulkan reported free memory to host `MemAvailable` on Linux UMA devices. The BC-250 advertises a 16.5 GiB GTT budget on a 15.2 GiB board (the budget double-counts RAM the OS is using), so `--fit`, proportional tensor splits, and the rpc-server memory advertisement all overcommit and the node's rpc-server gets OOM-killed mid-load. Upstreamable.
  - **`feat/puzzle-port`** — Nemotron-3-Puzzle-75B-A9B support (per-layer heterogeneous MoE + mamba2 hybrid + MTP head), tracked from [YanissAmz/llama.cpp `puzzle-port`](https://github.com/YanissAmz/llama.cpp/tree/puzzle-port) rather than authored here; refresh by fetching that branch, not by rebasing onto our `master`. Runs the `Puzzle-75B-A9B-UD-IQ4-XL.gguf` on `G:\lm_studio_models`. MTP speculative decoding (`--spec-type draft-mtp`) loads but is currently *slower* than plain decoding (mamba2 states can't roll back yet) — leave it off.
  - **`feat/rpc-robust`** — survive transient RPC connect failures instead of crashing. Root cause of "the C5 bug" (llama-server dies with 0xC0000005 / `GGML_ASSERT(buft)` at ggml-backend.cpp:39 mid-load): the rpc-server serves **one client at a time** and its listen backlog was **1**, so concurrent connects (the app's fabric probes + a multi-endpoint `--fit` client) get refused; `get_socket` returned null, `ggml_backend_rpc_buffer_type` returned a null buft, and the loader asserted or AV'd. Changes: (a) `get_socket` retries connect+HELLO up to 5× with 250 ms-step backoff, logging every attempt; (b) a failed HELLO handshake returns false (retryable) instead of `GGML_ABORT`; (c) server listen backlog 1 → 16 (**server-side — daemons only pick this up when their binaries are redeployed and restarted**); (d) `make_gpu_buft_list` throws a clean "RPC endpoint unreachable?" error on a null device buft instead of asserting deep in the loader. Upstreamable.
  - **`feat/hy3-port`** — Hy3 / Hunyuan 3 (`hy_v3`) support (299B MoE, 17B active, 192 experts, MTP head), tracked from [satindergrewal/llama.cpp `hy3-mtp`](https://github.com/satindergrewal/llama.cpp/tree/hy3-mtp) (upstream PR #25395); refresh by fetching that branch. Runs `satgeze/Hy3-1M-GGUF` (`hy3-1M-MTP-IQ2_M.gguf`, 93 GB) — this is "The House" model, split across the 3090 + BC-250 cluster + shredder over RPC. No ggml/backend changes, so RPC workers don't need new kernels — only the client needs the arch. Merge note: on `integration` the branch's `hy-v3.cpp` needed adapting to puzzle-port's per-layer `n_ff_exp(il)` accessor (upstream has a plain `n_ff_exp` field). The GGUF's MTP head loads; the model card recommends `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75` (the p-min gate matters: without it acceptance is ~39% and MTP is a net slowdown) — untested here, off by default.
  - **`feat/rpc-cache`** — per-model control of the rpc-server tensor cache + LRU size cap. Client: `--rpc-cache` on llama-server (env `LLAMA_ARG_RPC_CACHE`) opts the process in; only then are large tensors offered via SET_TENSOR_HASH, and the server only writes cache entries for uploads offered as a hash first (no wire-format change — old/new client/server mixes stay compatible; cache *reads* work for everyone). Server: `ggml-rpc-server --cache-limit SIZE` (plain MiB or K/M/G/T) evicts least-recently-used entries at startup and after each write; cache hits bump the entry mtime as the LRU clock. Deployed with daemon args in `serve.rpc-cluster.json` (`"cache-limit": "80G"`) and `serve.rpc-shredder.json` (`"25G"`); the loader's House configs opt in per-RunConfig with a `"rpc-cache": true` extra-arg line. Upstreamable.
  - **`feat/rpc-socket-lifetime`**, **`feat/rpc-graph-fixes`** — RPC hardening follow-ups (endpoint socket retention + connect-failure logging; clean rejection of bad graph views + iterative graph serialization). The RPC branches form one linear chain: `feat/rpc-robust` → `feat/rpc-socket-lifetime` → `feat/rpc-graph-fixes` → `feat/rpc-cache` → `feat/rpc-cache-preflight`, and `feat/rpc-async` continues it.
  - **`feat/rpc-async`** — async RPC streams/events/caps (`GGML_RPC_ASYNC=1`), non-blocking sched walk, ordered synchronous commands, quiescent-stream sync skip, and the proto-v5 multi-slot graph cache. Extracted from `feat/pipedec` (July 2026) because it is shared infrastructure: pipeline prefill sits on it today and SPD will next. Based on `feat/rpc-cache-preflight`.
  - **`feat/pipeline-prefill`** — pipelined chunked prefill over RPC: async INPUT staging with `n_copies > 1`, client-side GET_ALLOC_SIZE shape cache, runtime `GGML_SCHED_COPIES` depth, submit-trace diagnostics. 306 t/s warm vs 123 t/s sequential on the 6-stage House. Stacked on `feat/rpc-async`; see `docs/pipeline-prefill.md`.
  - **`feat/fit-modes`** — `--fit-whole-layers` (complete layers only, no tensor overrides) and `--fit-fill-rpc-first` (fill non-primary devices with whole layers, confine MoE CPU-spill to the primary device).
  - **`feat/pipedec`** — PipeDec speculative decoding (stage-2 token-body lanes, deferred verify group, sibling probe, streamed draft lanes; `docs/pipedec.md`). Production House decode path (~23.5–25 t/s vs 9.7 stock on Step-3.7). After the July 2026 extraction it is the only branch that still owns decode machinery; the plan is to **drop it from `integration` once SPD (speculative pipeline decoding) replaces it** — the extracted `feat/rpc-async` / `feat/pipeline-prefill` / `feat/fit-modes` branches carry everything worth keeping. Note: `feat/pipedec` history currently interleaves with the `integration` spine (they shared a head before the extraction), so rebuilding a pipedec-free integration means re-merging `master` + the other `feat/*` branches fresh rather than reverting.
  - Model/tool ports, each on its own branch: **`feat/step37`** (StepFun Step-3.7), **`feat/mtp-device`** (pin embedded-MTP layers + output head to the draft device), **`feat/q2_0-gpu`**, **`feat/qwen3-tts`**, **`feat/s2`**, **`feat/voxcpm2`**, **`fix/mmproj-fit-target-index`**.
  - **`feat/rpc-cache-preflight`** - skips local GGUF payload reads on warm RPC-cache hits and loads independent RPC endpoints concurrently. The first opted-in load records each large tensor's full FNV hash in a small `<shard>.rpc-cache-manifest` sidecar, validated by GGUF file size and mtime. Later loads query each endpoint before reading the tensor; hits are loaded from the node's cache while one worker per RPC device runs in parallel with local CUDA loading. Tensors at or below the 10 MiB threshold still upload directly. Logs use aggregate summaries rather than per-tensor hit messages. The Step 3.7 House load measured 8m55s before this change, 3m06s while learning manifests in parallel, and 1m52s manifest-warm.
- **`integration`** — the build/deploy branch: latest `master` + every `feat/*` merged in + this `CLAUDE.md`. **This is the branch we compile and run.** Refresh it by: ff `master` to upstream, then (re)merge each `feat/*` into `integration`, then build.

## Feature: `--cache-disk` (branch `feat/disk-cache`)

**Problem:** on the BC-250 UMA nodes, host RAM is shared with the GPU. The server's prompt cache and per-slot context checkpoints were eating RAM the model + KV cache need — "killing the RAM on the cluster." This is a **runtime-memory** fix and has nothing to do with model-load speed or tensor placement.

**What it does:** streams the server prompt-cache states and context checkpoints to files on disk instead of holding them in host RAM, so peak host memory during save/load stays bounded regardless of state size.

**Server flags:**
- `--cache-disk DIR` — spill prompt-cache states + context checkpoints to files in `DIR` (default: disabled = keep in RAM).
- `--cache-disk-limit N` — cap the on-disk **prompt-cache** footprint in MiB (0 = no limit).

**Key pieces:**
- `include/llama.h`, `src/llama-context.cpp` — new public API `llama_state_seq_save_file_ext` / `llama_state_seq_load_file_ext`: same as the existing `_file` calls but **flag-aware** (so `PARTIAL_ONLY` checkpoints work). The underlying save/load already stream through a bounded IO buffer, so peak host RAM is independent of state size.
- `common/common.{h,cpp}` — `common_state_file`: a RAII handle (path + size) that, on Linux, `fdatasync`s then `posix_fadvise(DONTNEED)`s the spill file so it doesn't linger in the page cache (which competes with VRAM on UMA), and **unlinks it on destruction**. Shared ownership so cache-list copies keep the backing file alive. `common_prompt_checkpoint` gains `file_tgt`/`file_dft` + `update_{tgt,dft}_file`.
- `tools/server/server-task.{h,cpp}` — `server_prompt_data` gains `file_main`/`file_drft`; `server_prompt_cache` gains a disk mode (`disk_dir`, `disk_mode()`); helpers `server_state_file_path` (unique spill name) and `server_state_dir_init` (mkdir + clean stale `cache-*.llstate`).
- `tools/server/server-context.cpp` — wires the flags into init and the state-save / checkpoint paths.

**Known rough edges (from audit, not yet fixed):**
- Context **checkpoint** spill is *not* bounded by `--cache-disk-limit` (only the prompt cache is).
- `--cache-disk-limit` defaults to `0` (no limit) → can fill a small node disk. Set a finite limit in launch configs.
- Per-spill `fdatasync` forces a *durable* flush on the server thread; a recoverable cache only needs page-cache eviction, not durability — a candidate perf knob.
- A `--cache-disk` dir shared by multiple servers can collide (per-process filename counter + startup wipe). Use a **per-node** dir.

## Goals (why this fork exists)

1. **Faster model loading over RPC** *(primary).* Cold RPC load serializes all ~40 GB through the main node's single link. Even upstream's warm cache (`rpc-server -c` — a hash-keyed local tensor cache) still makes the main node read + FNV-hash the *entire* GGUF every launch. Direction: let each node load its own shard directly from its **local GGUF copy** so weights never cross the wire, and/or skip the per-launch re-hash. Zero-code first step: turn on `rpc-server -c` and measure once the ring exists.
2. **Topology-aware MoE splitting via an RPC overhaul** *(exploratory / shaky).* RPC today is **star-only**: `ggml-rpc` has no remote↔remote path (`cpy_tensor` bails when two tensors sit on different RPC sockets), so every inter-node activation hairpins through the main host and the ring's node-to-node links go unused. Zero-code lever: assign **contiguous layer ranges** (weights + their experts) per node with `-ot/--override-tensor` for pipeline-style splitting. Code lever (bigger, unproven): direct remote↔remote activation transfer over the ring. Only worth pursuing after measuring with the 10 GbE ring in place.

## Building on this machine (Windows, CUDA)

**Run `..\build-llama.cmd` — do not hand-roll cmake.** It sets up vcvars64 + Ninja, configures the persistent `build-ninja\` dir (CUDA `sm_86`, `GGML_RPC=ON`, shared libs, Web UI off, `GGML_SCHED_MAX_COPIES=8`), builds `llama-server`, and deploys the exe + DLLs to `llama_loader\dist\llama-cpp\` (refusing while llama-server.exe is running). **Never use the old Visual Studio generator/`build\` dir** — MSBuild compiles the ~100 CUDA template `.cu` files serially (hours); Ninja + ccache is ~15 min full, seconds incremental. Never set `CL=/MP` (nvcc host-compiler invocations inherit `CL` and every `.cu` fails).

Note the RPC server binary is named **`ggml-rpc-server.exe`**, not `rpc-server.exe` (despite the source living at `tools/rpc/rpc-server.cpp` and most upstream docs calling it `rpc-server`).

## Building on shredder (Ubuntu 26.04, 2× GTX 1080, CUDA-in-docker)

Shredder has no root-usable CUDA toolchain: host GCC 15 is too new for nvcc, and Pascal (`sm_61`) needs CUDA ≤ 12.x anyway. The build therefore runs inside `nvidia/cuda:12.9.1-devel-ubuntu24.04` (mark is in the `docker` group) and the resulting binaries run on the host (driver 580). The whole flow is scripted:

```bash
ssh shredder './llama-scripts/build-llama.sh'   # sync fork/integration → docker build → deploy to ~/llama-scripts/llama-cpp/
```

(The same script runs on fedora, where it does a Vulkan build and auto-disperses binaries to fedora2/3. `build-fork.sh.bak` is the retired predecessor. It never restarts running daemons — live rpc-servers keep their old binaries until restarted.)

Key flags: `-DCMAKE_CUDA_ARCHITECTURES=61 -DGGML_CUDA_NO_VMM=ON -DLLAMA_CURL=OFF`. `NO_VMM` is required — the container only has the libcuda *stub*, and the VMM pool needs driver-API symbols at executable link time. Deploy = copy `build/bin/.` plus the container's CUDA runtime libs (`libcudart`, `libcublas`, `libcublasLt`, **`libnccl.so.2`** — the devel image links it in) into `~/llama-scripts/llama-cpp/`; `serve_llama.py` sets `LD_LIBRARY_PATH` to that dir. The old upstream build is parked at `~/llama-scripts/llama-cpp.bak-upstream-b0151`.
