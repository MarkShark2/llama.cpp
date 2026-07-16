#!/usr/bin/env python3
"""Convert a PEFT LoRA adapter for Qwen3-TTS into a GGUF adapter for qwen3-tts.

PEFT stores lora_A as (r, in_features) and lora_B as (out_features, r), and
applies the update as:

    y += (x @ A.T @ B.T) * (alpha / r)

torch Linear weights are (out, in), which is exactly the ggml ne (in, out) that
convert_tts_to_gguf.py writes without transposing, so the LoRA factors need no
transpose either: A -> ne (in, r) and B -> ne (r, out) feed ggml_mul_mat as-is.

The adapter names line up with the base GGUF names through a fixed module map,
so an adapter tensor is just "<base stem>.lora_a" / ".lora_b" -- the same layout
convert_s2_lora_to_gguf.py uses.

Speaker embedding: a Qwen3-TTS LoRA is trained against the Base model, which has
no speaker presets and otherwise invents a new speaker on every request. The
voice therefore lives in a separate target_speaker_embedding tensor next to the
adapter, and is folded into the same GGUF here so that one file carries both the
weights and the voice they were trained with.

Usage:
    convert_qwen3tts_lora_to_gguf.py path/to/adapter_dir -o out.gguf --voice-name moka
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

# Add gguf-py to path (if available). Walk up from this script so the converter
# finds the tree's gguf-py from tools/qwen3tts/scripts, and still works if copied
# elsewhere.
try:
    for _parent in Path(__file__).resolve().parents:
        _candidate = _parent / "gguf-py"
        if _candidate.exists():
            sys.path.insert(0, str(_candidate))
            break
except (PermissionError, OSError):
    pass

import gguf  # noqa: E402

ARCH = "qwen3-tts"

# PEFT module name -> base GGUF stem suffix. Only these seven are trainable
# targets in practice; anything else is reported and skipped rather than
# silently dropped.
MODULE_MAP = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}

# The code predictor pattern must be tried first: "talker.code_predictor.model."
# would otherwise be mistaken for a talker layer by a looser match.
BLOCK_PATTERNS = (
    (re.compile(r"^talker\.code_predictor\.model\.layers\.(\d+)\.(.+)$"), "code_pred.blk.{}."),
    (re.compile(r"^talker\.model\.layers\.(\d+)\.(.+)$"), "talker.blk.{}."),
)


def map_name(key: str) -> tuple[str, str] | None:
    """Map a PEFT key to (base stem, "lora_a"|"lora_b"), or None if not LoRA."""
    for suffix, which in ((".lora_A.weight", "lora_a"), (".lora_B.weight", "lora_b")):
        if not key.endswith(suffix):
            continue
        stem = key[: -len(suffix)]
        if stem.startswith("base_model.model."):
            stem = stem[len("base_model.model.") :]
        for pattern, prefix in BLOCK_PATTERNS:
            m = pattern.match(stem)
            if not m:
                continue
            module = MODULE_MAP.get(m.group(2))
            if module is None:
                return None
            return prefix.format(m.group(1)) + module, which
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter", type=Path, help="PEFT adapter dir (or adapter_model.safetensors)")
    ap.add_argument("-o", "--outfile", type=Path, help="output .gguf (default: alongside the adapter)")
    ap.add_argument("--alpha", type=float, default=0.0, help="lora_alpha (default: read from adapter_config.json)")
    ap.add_argument("--rank", type=int, default=0, help="LoRA rank (default: read from config, else inferred)")
    ap.add_argument("--voice-name", type=str, default="", help="name for the bundled speaker embedding (default: adapter dir name)")
    ap.add_argument("--outtype", choices=("f32", "f16"), default="f32", help="adapter tensor type (default: f32)")
    args = ap.parse_args()

    adapter_dir = args.adapter if args.adapter.is_dir() else args.adapter.parent
    weights_path = args.adapter if args.adapter.is_file() else adapter_dir / "adapter_model.safetensors"
    if not weights_path.exists():
        print(f"error: no adapter weights at {weights_path}", file=sys.stderr)
        return 1

    outfile = args.outfile or adapter_dir / (adapter_dir.name + ".gguf")
    voice_name = args.voice_name or adapter_dir.name

    rank = args.rank
    alpha = args.alpha
    cfg_path = adapter_dir / "adapter_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        rank = rank or int(cfg.get("r", 0))
        alpha = alpha or float(cfg.get("lora_alpha", 0.0))
        if cfg.get("use_rslora"):
            print("error: use_rslora adapters scale by alpha/sqrt(r); not supported", file=sys.stderr)
            return 1
        if cfg.get("use_dora"):
            print("error: DoRA adapters need the magnitude vector; not supported", file=sys.stderr)
            return 1

    state = load_file(str(weights_path))

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    skipped: list[str] = []
    for key, val in state.items():
        mapped = map_name(key)
        if mapped is None:
            skipped.append(key)
            continue
        stem, which = mapped
        pairs.setdefault(stem, {})[which] = val

    if skipped:
        print(f"skipped {len(skipped)} non-LoRA/unmapped tensors, e.g. {skipped[0]}")
    if not pairs:
        print("error: no lora_A/lora_B tensors found in the adapter", file=sys.stderr)
        return 1

    incomplete = [s for s, p in pairs.items() if len(p) != 2]
    if incomplete:
        print(f"error: unpaired LoRA tensors for: {', '.join(sorted(incomplete))}", file=sys.stderr)
        return 1

    ranks = {int(p["lora_a"].shape[0]) for p in pairs.values()}
    if len(ranks) != 1:
        print(f"error: inconsistent LoRA ranks across tensors: {sorted(ranks)}", file=sys.stderr)
        return 1
    inferred_rank = ranks.pop()
    if rank and rank != inferred_rank:
        print(f"error: config rank {rank} disagrees with tensor rank {inferred_rank}", file=sys.stderr)
        return 1
    rank = inferred_rank
    if alpha <= 0.0:
        print("error: no lora_alpha in adapter_config.json; pass --alpha", file=sys.stderr)
        return 1

    dtype = np.float32 if args.outtype == "f32" else np.float16
    writer = gguf.GGUFWriter(outfile, ARCH)
    writer.add_string(f"{ARCH}.lora.kind", "peft")
    writer.add_uint32(f"{ARCH}.lora.rank", rank)
    writer.add_float32(f"{ARCH}.lora.alpha", alpha)

    n_talker = n_code_pred = n_params = 0
    for stem in sorted(pairs):
        pair = pairs[stem]
        a = pair["lora_a"].to(torch.float32)
        b = pair["lora_b"].to(torch.float32)
        writer.add_tensor(f"{stem}.lora_a", a.numpy().astype(dtype))
        writer.add_tensor(f"{stem}.lora_b", b.numpy().astype(dtype))
        n_params += a.numel() + b.numel()
        if stem.startswith("code_pred."):
            n_code_pred += 1
        else:
            n_talker += 1

    spk_path = adapter_dir / "speaker_embedding.safetensors"
    if spk_path.exists():
        spk = load_file(str(spk_path))
        key = "target_speaker_embedding"
        if key not in spk:
            print(f"error: {spk_path} has no {key} (found: {', '.join(spk)})", file=sys.stderr)
            return 1
        emb = spk[key].to(torch.float32)
        if emb.ndim != 1:
            print(f"error: {key} must be 1-D, got shape {tuple(emb.shape)}", file=sys.stderr)
            return 1
        # Always f32: this is one row and it is the voice, so it is not worth
        # rounding to save 4 KB.
        writer.add_tensor("speaker_embedding", emb.numpy().astype(np.float32))
        writer.add_string(f"{ARCH}.lora.voice_name", voice_name)
        print(f"speaker embedding: {emb.numel()} dims -> voice '{voice_name}'")
    else:
        print("no speaker_embedding.safetensors found; adapter will carry weights only")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"\nwrote {outfile}")
    print(f"targets: {len(pairs)} ({n_talker} talker, {n_code_pred} code_pred)")
    print(f"rank: {rank}  alpha: {alpha}  scale: {alpha / rank}")
    print(f"params: {n_params / 1e6:.2f}M  type: {args.outtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
