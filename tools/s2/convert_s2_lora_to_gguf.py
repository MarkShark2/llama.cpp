#!/usr/bin/env python3
"""Convert a fish-speech LoRA checkpoint into a GGUF adapter for s2-cli.

fish-speech trains LoRA with loralib (microsoft/LoRA) under PyTorch Lightning,
so the checkpoint is a Lightning dict whose "state_dict" holds only the LoRA
parameters (see TextToSemantic.on_save_checkpoint upstream). Most of the file
size is Adam optimizer state, which is dropped here.

loralib stores lora_A as (r, in_features) and lora_B as (out_features, r), and
applies the update as:

    Linear:    y += (x @ A.T @ B.T) * (alpha / r)
    Embedding: y += (embedding(ids, A.T) @ B.T) * (alpha / r)

The tensor names line up 1:1 with the base model's GGUF names once the "model."
prefix is stripped, so an adapter tensor is just "<base stem>.lora_a" / ".lora_b".

Only the alpha/r scale needs care: Lightning does not store the LoRA config in
the checkpoint, so --alpha must be passed to match the training run (the
fast-AR-only recipe in fishaudio/fish-speech#1234 uses r=32, alpha=16).

Usage:
    convert_s2_lora_to_gguf.py in.ckpt -o out.gguf --alpha 16
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "gguf-py"))
import gguf  # noqa: E402

# Base tensors that are nn.Embedding rather than nn.Linear. loralib feeds ids
# through A.T for these, so lora_a is stored transposed to (vocab, r): that maps
# to ggml ne = (r, vocab), which is what ggml_get_rows needs to return an
# r-length row per id.
EMBEDDING_STEMS = {"fast_embeddings", "embeddings", "codebook_embeddings"}


def map_name(key: str) -> tuple[str, str] | None:
    """Map a checkpoint key to (base stem, "lora_a"|"lora_b")."""
    for suffix, out in ((".lora_A", "lora_a"), (".lora_B", "lora_b")):
        if key.endswith(suffix):
            stem = key[: -len(suffix)]
            if stem.startswith("model."):
                stem = stem[len("model.") :]
            return stem, out
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path, help="fish-speech LoRA .ckpt")
    ap.add_argument("-o", "--outfile", type=Path, help="output .gguf (default: alongside the ckpt)")
    ap.add_argument("--rank", type=int, default=0, help="LoRA rank (default: inferred from tensor shapes)")
    ap.add_argument("--alpha", type=float, required=True, help="lora_alpha used at training time")
    ap.add_argument("--outtype", choices=("f32", "f16"), default="f32", help="adapter tensor type (default: f32)")
    args = ap.parse_args()

    outfile = args.outfile or args.ckpt.with_suffix(".gguf")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, val in state.items():
        mapped = map_name(key)
        if mapped is None:
            print(f"skipping non-LoRA tensor: {key}")
            continue
        stem, which = mapped
        pairs.setdefault(stem, {})[which] = val

    if not pairs:
        print("error: no lora_A/lora_B tensors found in the checkpoint", file=sys.stderr)
        return 1

    incomplete = [s for s, p in pairs.items() if len(p) != 2]
    if incomplete:
        print(f"error: unpaired LoRA tensors for: {', '.join(sorted(incomplete))}", file=sys.stderr)
        return 1

    ranks = {int(p["lora_a"].shape[0]) for p in pairs.values()}
    if len(ranks) != 1:
        print(f"error: inconsistent LoRA ranks across tensors: {sorted(ranks)}", file=sys.stderr)
        return 1
    rank = args.rank or ranks.pop()

    dtype = np.float32 if args.outtype == "f32" else np.float16
    writer = gguf.GGUFWriter(outfile, "fish-speech")
    writer.add_string("s2.lora.kind", "fish-speech-loralib")
    writer.add_uint32("s2.lora.rank", rank)
    writer.add_float32("s2.lora.alpha", args.alpha)

    n_params = 0
    for stem in sorted(pairs):
        pair = pairs[stem]
        a = pair["lora_a"].to(torch.float32)
        b = pair["lora_b"].to(torch.float32)
        if stem in EMBEDDING_STEMS:
            a = a.transpose(0, 1).contiguous()
        writer.add_tensor(f"{stem}.lora_a", a.numpy().astype(dtype))
        writer.add_tensor(f"{stem}.lora_b", b.numpy().astype(dtype))
        n_params += a.numel() + b.numel()
        print(f"{stem}: a={tuple(a.shape)} b={tuple(b.shape)}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"\nwrote {outfile}")
    print(f"targets: {len(pairs)}  rank: {rank}  alpha: {args.alpha}  scale: {args.alpha / rank}")
    print(f"params: {n_params / 1e6:.2f}M  type: {args.outtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
