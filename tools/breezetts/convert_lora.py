#!/usr/bin/env python3
"""Convert a breeze-tts2-finetuning LoRA adapter.safetensors into a GGUF the breeze-tts engine merges at load.

Tensors are stored as `<hf weight name minus .weight>.lora_a` / `.lora_b` in F32, with the
`breeze-tts.lora.rank` / `breeze-tts.lora.alpha` metadata keys.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))

import gguf
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("adapter", type=Path, help="adapter.safetensors")
ap.add_argument("out", type=Path)
ap.add_argument("--alpha", type=float, required=True, help="LoRA alpha used in training")
a = ap.parse_args()

w = gguf.GGUFWriter(str(a.out), "breeze-tts-lora")
rank = None
with safe_open(str(a.adapter), "pt") as f:
    for key in sorted(f.keys()):
        base, kind = key.rsplit(".", 1)  # lora_A / lora_B
        t = f.get_tensor(key).float().numpy()
        if kind == "lora_A":
            rank = t.shape[0]
        w.add_tensor(base + "." + kind.lower(), t)
w.add_uint32("breeze-tts.lora.rank", rank)
w.add_float32("breeze-tts.lora.alpha", a.alpha)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"wrote {a.out} (rank {rank}, alpha {a.alpha})")
