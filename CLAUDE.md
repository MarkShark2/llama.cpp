IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work on upstream llama.cpp code.

---

# Fork notes — BC-250 cluster build of llama.cpp

This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) maintained to run large MoE models across a small heterogeneous cluster. This section documents the fork; everything above/around it is upstream.

## Hardware & topology (current, pre-networking)

- **Desktop (this machine, Windows 11):** RTX 3090 + RTX 3080 Ti — both `sm_86` (Ampere). This is the build host and will be the main `llama-server` / RPC client. Source lives at `llama_loader\llamacpp-src`.
- **Cluster:** 3× Fedora nodes on BC-250 AMD APUs. These are **UMA** — system RAM is shared with the GPU, so host memory is scarce and precious. SSH aliases `fedora` / `fedora2` / `fedora3`; scripts under `/home/mark/llama-scripts`; Vulkan backend. Reached today over ordinary **1 GbE**.
- **Coming (not yet wired):** a **10 GbE ring network** — every NIC has two DAC ports, so the desktop + 3 nodes form a ring. The desktop's ring NIC is **not installed yet**, so **RPC cannot be exercised from this machine yet.** All RPC load-speed and topology numbers below are pending that hardware.

**Target workload:** a ~75B MoE model with ~9B active experts (~40 GB) split across the 3090 + the three nodes over RPC.

## Git branch strategy

Goal: stay easy to merge against fast-moving upstream, keep each feature isolated, have one branch we actually build.

- **`master`** — a clean mirror of `origin/master` (ggml-org). Never commit fork work here. Only fast-forward it:
  `git fetch origin master && git checkout master && git merge --ff-only origin/master`
- **`feat/*`** — one branch per feature, based off `master`, **code only** (no fork docs/config), so each stays rebasable and potentially upstreamable. Current:
  - **`feat/disk-cache`** — the disk cache-streaming feature (see below).
  - **`feat/ssm-dstate-96`** — mamba2 `SSM_SCAN` with `d_state == 96` in the CUDA and Vulkan backends. The Puzzle NAS shrank Nemotron-H's SSM state from 128 to 96 (`nemotron_h_moe.ssm.state_size = 96` in the GGUF); upstream kernels only ship d_state 128/256, so the op fell back to CPU on CUDA and hard-crashed Vulkan RPC workers because the RPC client's `supports_op` is a `return true` stub (upstream TODO in `ggml-rpc.cpp`) that never consults the remote backend. That stub is a standing footgun: *any* op a node's Vulkan backend rejects will abort that node's rpc-server instead of falling back. Upstreamable.
  - **`feat/vk-uma-mem`** — clamp Vulkan reported free memory to host `MemAvailable` on Linux UMA devices. The BC-250 advertises a 16.5 GiB GTT budget on a 15.2 GiB board (the budget double-counts RAM the OS is using), so `--fit`, proportional tensor splits, and the rpc-server memory advertisement all overcommit and the node's rpc-server gets OOM-killed mid-load. Upstreamable.
  - **`feat/puzzle-port`** — Nemotron-3-Puzzle-75B-A9B support (per-layer heterogeneous MoE + mamba2 hybrid + MTP head), tracked from [YanissAmz/llama.cpp `puzzle-port`](https://github.com/YanissAmz/llama.cpp/tree/puzzle-port) rather than authored here; refresh by fetching that branch, not by rebasing onto our `master`. Runs the `Puzzle-75B-A9B-UD-IQ4-XL.gguf` on `G:\lm_studio_models`. MTP speculative decoding (`--spec-type draft-mtp`) loads but is currently *slower* than plain decoding (mamba2 states can't roll back yet) — leave it off.
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

Toolchain is already installed **system-wide** — nothing to download: VS2022 Build Tools (MSVC 14.4x), CUDA 12.6 / 12.9, CMake 4.x. Both local GPUs are `sm_86`, so we compile a single CUDA arch to keep the build short.

```powershell
# from llama_loader\llamacpp-src, on the `integration` branch
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T cuda=12.9 `
      -DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release `
      -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9"
cmake --build build --config Release -j
```

The VS generator lets MSBuild drive the CUDA integration, so no `vcvars` shell is needed; `-T cuda=12.9` pins the CUDA toolset. This machine also has CUDA 13.3 installed, and CMake's `FindCUDAToolkit` will happily pick up its headers even though `-T` pinned the compiler to 12.9 — that mismatch is a hard compiler error (`C1189`), so **both** `CUDA_PATH` and `-DCUDAToolkit_ROOT` must explicitly point at the same 12.9 install. `-DGGML_RPC=ON` is required for cluster work: without it, neither the `--rpc` client flag in `llama-server` nor the RPC server binary get built.

Binaries land in `build\bin\Release\`. Note the RPC server binary is named **`ggml-rpc-server.exe`**, not `rpc-server.exe` (despite the source living at `tools/rpc/rpc-server.cpp` and most upstream docs calling it `rpc-server`).
