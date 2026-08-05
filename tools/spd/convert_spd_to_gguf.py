#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

import gguf  # noqa: E402


LOGGER = logging.getLogger("convert_spd_to_gguf")
SPD_ARCH = gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.SPD]
SUPPORTED_CHECKPOINT_VERSION = 11


def field_value(reader: gguf.GGUFReader, key: str) -> Any:
    field = reader.get_field(key)
    if field is None:
        raise ValueError(f"target GGUF is missing required metadata: {key}")
    return field.contents()


def copy_tokenizer_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for key, field in reader.fields.items():
        if not key.startswith("tokenizer."):
            continue
        subtype = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(key, field.contents(), field.types[0], subtype)


def checkpoint_tensor_name(name: str) -> str:
    if re.fullmatch(r"aggr_projs\.\d+\.weight", name):
        raise ValueError("aggregation projections are packed into aggr.weight")
    if name == "lm_head.weight":
        return "output.weight"

    match = re.fullmatch(r"spec_layers\.(\d+)\.(.+)", name)
    if match is None:
        raise ValueError(f"unrecognized checkpoint tensor: {name}")

    block = match.group(1)
    suffix = match.group(2)
    suffix_map = {
        "input_layernorm.weight":              "attn_norm.weight",
        "post_attention_layernorm.weight":     "ffn_norm.weight",
        "self_attn.q_proj.weight":             "attn_q.weight",
        "self_attn.k_proj.weight":             "attn_k.weight",
        "self_attn.v_proj.weight":             "attn_v.weight",
        "self_attn.o_proj.weight":             "attn_output.weight",
        "self_attn.q_norm.weight":             "attn_q_norm.weight",
        "self_attn.k_norm.weight":             "attn_k_norm.weight",
        "mlp.gate_proj.weight":                "ffn_gate.weight",
        "mlp.up_proj.weight":                  "ffn_up.weight",
        "mlp.down_proj.weight":                "ffn_down.weight",
    }
    if suffix not in suffix_map:
        raise ValueError(f"unrecognized checkpoint tensor: {name}")
    return f"blk.{block}.{suffix_map[suffix]}"


def to_bf16_bytes(tensor: torch.Tensor) -> np.ndarray[Any, Any]:
    data = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
    return gguf.quantize(data, gguf.GGMLQuantizationType.BF16)


def add_checkpoint_tensor(writer: gguf.GGUFWriter, name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim == 1:
        writer.add_tensor(name, tensor.detach().to(device="cpu", dtype=torch.float32).numpy())
        return
    writer.add_tensor(
        name,
        to_bf16_bytes(tensor),
        raw_dtype=gguf.GGMLQuantizationType.BF16,
    )


SUPPORTED_TARGET_ARCHS = ("qwen35", "gemma4", "deepseek4")


def spd_stage_layers(trunk_blocks: int, num_stages: int) -> list[int]:
    """Layer count per SPD stage.

    Stages are NOT required to be uniform. DeepSeek-V4 has 43 trunk blocks -- a
    prime -- so no stage count between 2 and 42 divides it evenly, and demanding
    divisibility excluded the architecture outright. Every stage takes
    ceil(trunk / stages) layers and the last takes the remainder, matching both
    llama_context's slicing and the offline trainer's STAGE_PRESETS (43 over 9
    stages -> 5x8 + 3). Reduces to an even split whenever the count divides.
    """
    per = -(-trunk_blocks // num_stages)
    return [per]*(num_stages - 1) + [trunk_blocks - per*(num_stages - 1)]


def resolve_train_span(config: dict[str, Any], override: int | None) -> int:
    """The window length the speculation head was trained on, in tokens.

    The trainer cuts fixed-length windows out of the corpus (train_offline.py
    --chunk), so the head has never attended across more positions than that.
    The decode path bounds the sidecar's attention to this span; without it the
    head runs outside its training distribution on any request longer than the
    chunk and acceptance collapses. 0 means "unknown", which the fork loads as
    unbounded attention and warns about.
    """
    if override is not None:
        if override < 0:
            raise ValueError("--train-span must not be negative")
        return int(override)

    span = int(config.get("train_span") or 0)
    if span < 0:
        raise ValueError(f"checkpoint records a negative train_span: {span}")
    return span


def validate_checkpoint(
    config: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    target: gguf.GGUFReader,
) -> dict[str, int | float | list[int] | bool | str]:
    target_arch = str(field_value(target, "general.architecture"))
    if target_arch not in SUPPORTED_TARGET_ARCHS:
        raise ValueError(
            f"SPD checkpoint requires a target GGUF with one of {SUPPORTED_TARGET_ARCHS}, got {target_arch!r}")

    version = int(config["version"])
    if version != SUPPORTED_CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported SPD checkpoint version {version}; expected {SUPPORTED_CHECKPOINT_VERSION}"
        )

    hidden_size = int(config["hidden_size"])
    target_hidden_size = int(field_value(target, f"{target_arch}.embedding_length"))
    if hidden_size != target_hidden_size:
        raise ValueError(
            f"checkpoint hidden size {hidden_size} does not match target hidden size {target_hidden_size}"
        )

    target_vocab_size = len(field_value(target, "tokenizer.ggml.tokens"))
    if int(config["vocab_size"]) != target_vocab_size:
        raise ValueError(
            f"checkpoint target vocabulary {config['vocab_size']} does not match target GGUF {target_vocab_size}"
        )

    draft_token_ids = np.asarray(config["draft_token_ids"], dtype=np.int64)
    draft_vocab_size = int(config["draft_vocab_size"])
    if draft_token_ids.ndim != 1 or draft_token_ids.size != draft_vocab_size:
        raise ValueError("draft_token_ids must be a one-dimensional draft-vocabulary mapping")
    if np.any((draft_token_ids < 0) | (draft_token_ids >= target_vocab_size)):
        raise ValueError("draft_token_ids contains a target token outside the target vocabulary")
    if np.unique(draft_token_ids).size != draft_token_ids.size:
        raise ValueError("draft_token_ids contains duplicates")

    num_stages = int(config["num_stages"])
    num_spec_layers = int(config["num_spec_layers"])
    num_aggr_types = int(config["num_aggr_types"])
    anchors = [int(value) for value in config["aggr_feature_bound"]]
    use_deepest = bool(config["trained_with_use_deepest"])
    if not use_deepest:
        raise ValueError("only checkpoints trained with deepest available snapshots are supported")
    if len(anchors) != num_aggr_types:
        raise ValueError("aggr_feature_bound length does not match num_aggr_types")

    target_blocks = int(field_value(target, f"{target_arch}.block_count"))
    nextn_field = target.get_field(f"{target_arch}.nextn_predict_layers")
    nextn_blocks = 0 if nextn_field is None else int(nextn_field.contents())
    trunk_blocks = target_blocks - nextn_blocks
    if trunk_blocks <= 0 or num_stages <= 0:
        raise ValueError(
            f"target trunk block count {trunk_blocks} cannot be divided into {num_stages} SPD stages"
        )

    expected_stage_layers = spd_stage_layers(trunk_blocks, num_stages)
    if expected_stage_layers[-1] <= 0:
        raise ValueError(
            f"{num_stages} SPD stages over {trunk_blocks} trunk blocks leaves an empty trailing stage"
        )
    if anchors[0] != 0 or anchors[-1] >= trunk_blocks:
        raise ValueError(f"invalid SPD anchors for {trunk_blocks} target trunk blocks: {anchors}")

    stage_layers = config.get("stage_layers")
    if stage_layers is not None:
        stage_layers = [int(x) for x in stage_layers]
        if sum(stage_layers) != trunk_blocks:
            raise ValueError(
                f"checkpoint stage_layers {stage_layers} do not sum to the target trunk ({trunk_blocks})")
        if stage_layers != expected_stage_layers:
            raise ValueError(
                f"checkpoint stage_layers {stage_layers} do not match the layout the decode path "
                f"slices ({expected_stage_layers}); the head must be trained against "
                f"ceil(trunk/stages) layers per stage with the remainder last")
        bounds = [0]
        for count in stage_layers[:-1]:
            bounds.append(bounds[-1] + count)
        if anchors != bounds[: len(anchors)]:
            raise ValueError(
                f"aggr_feature_bound {anchors} does not match stage boundaries {bounds}")

    for index in range(num_aggr_types):
        key = f"aggr_projs.{index}.weight"
        expected = (hidden_size, hidden_size * (index + 1))
        if key not in state_dict or tuple(state_dict[key].shape) != expected:
            actual = None if key not in state_dict else tuple(state_dict[key].shape)
            raise ValueError(f"{key} has shape {actual}, expected {expected}")

    output_shape = (draft_vocab_size, hidden_size)
    if "lm_head.weight" not in state_dict or tuple(state_dict["lm_head.weight"].shape) != output_shape:
        raise ValueError(f"lm_head.weight must have shape {output_shape}")

    layer_ids = {
        int(match.group(1))
        for name in state_dict
        if (match := re.match(r"spec_layers\.(\d+)\.", name)) is not None
    }
    if layer_ids != set(range(num_spec_layers)):
        raise ValueError(
            f"checkpoint speculative layers are {sorted(layer_ids)}, expected 0..{num_spec_layers - 1}"
        )

    mapped_names = [
        checkpoint_tensor_name(name)
        for name in state_dict
        if not name.startswith("aggr_projs.")
    ]
    if len(mapped_names) != len(set(mapped_names)):
        raise ValueError("multiple checkpoint tensors map to the same GGUF tensor name")

    return {
        "target_arch": target_arch,
        "target_vocab_size": target_vocab_size,
        "hidden_size": hidden_size,
        "draft_vocab_size": draft_vocab_size,
        "num_stages": num_stages,
        "num_spec_layers": num_spec_layers,
        "num_aggr_types": num_aggr_types,
        "anchors": anchors,
        "use_deepest": use_deepest,
        "trunk_blocks": trunk_blocks,
        "stage_blocks": expected_stage_layers[0],
        "stage_layers": expected_stage_layers,
        "version": version,
    }


def convert(checkpoint_path: Path, target_path: Path, output_path: Path,
            assets_path: Path | None = None, train_span: int | None = None) -> None:
    LOGGER.info("Loading SPD checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"config", "state_dict"}:
        raise ValueError("SPD checkpoint must contain exactly config and state_dict")
    config = checkpoint["config"]
    state_dict = checkpoint["state_dict"]
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise ValueError("invalid SPD checkpoint config or state_dict")

    LOGGER.info("Reading target GGUF metadata: %s", target_path)
    target = gguf.GGUFReader(target_path)
    meta = validate_checkpoint(config, state_dict, target)
    span = resolve_train_span(config, train_span)

    if assets_path is not None:
        # Training computes spec logits as lm_head(target_final_norm(g0)); the
        # fork sidecar graph has no norm before its output head. RMS norm's
        # per-row 1/rms scalar cannot change the argmax, but the elementwise
        # norm weight can - fold it into the output weight so greedy drafts
        # match training exactly.
        assets = torch.load(assets_path, map_location="cpu", weights_only=False)
        norm_w = assets["final_norm.weight"].to(torch.float32)
        head_w = state_dict["lm_head.weight"]
        if norm_w.ndim != 1 or norm_w.shape[0] != head_w.shape[1]:
            raise ValueError(
                f"assets final_norm.weight shape {tuple(norm_w.shape)} does not match "
                f"lm_head.weight {tuple(head_w.shape)}")
        state_dict["lm_head.weight"] = head_w.to(torch.float32) * norm_w.unsqueeze(0)
        LOGGER.info("folded the target final-norm weight into output.weight (argmax-preserving)")

    LOGGER.info(
        "Validated SPD v%s: %s stages x %s target blocks, %s speculative layers, %s aggregation types",
        meta["version"],
        meta["num_stages"],
        meta["stage_blocks"],
        meta["num_spec_layers"],
        meta["num_aggr_types"],
    )

    target_arch = str(meta["target_arch"])
    spec_attn = config.get("spec_attn")

    writer = gguf.GGUFWriter(output_path, SPD_ARCH, use_temp_file=True)
    writer.add_name(f"{target_arch} SPD s{meta['num_stages']} l{meta['num_spec_layers']}")
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_vocab_size(int(meta["target_vocab_size"]))
    writer.add_context_length(int(field_value(target, f"{target_arch}.context_length")))
    writer.add_embedding_length(int(meta["hidden_size"]))
    writer.add_block_count(int(meta["num_spec_layers"]))
    writer.add_expert_count(int(meta["num_aggr_types"]))
    writer.add_expert_used_count(1)
    if spec_attn is not None:
        # spd-train checkpoints carry the speculation module's own attention
        # geometry (independent of the target's); standard NEOX rope, no
        # dimension sections
        writer.add_feed_forward_length(int(spec_attn["intermediate_size"]))
        writer.add_head_count(int(spec_attn["num_heads"]))
        writer.add_head_count_kv(int(spec_attn["num_kv_heads"]))
        writer.add_key_length(int(spec_attn["head_dim"]))
        writer.add_value_length(int(spec_attn["head_dim"]))
        writer.add_layer_norm_rms_eps(float(spec_attn["rms_norm_eps"]))
        writer.add_rope_dimension_count(int(spec_attn["head_dim"]))
        writer.add_rope_freq_base(float(spec_attn["rope_theta"]))
    else:
        # reference qwen35 checkpoints mirror the target's attention geometry
        writer.add_feed_forward_length(int(field_value(target, f"{target_arch}.feed_forward_length")))
        writer.add_head_count(int(field_value(target, f"{target_arch}.attention.head_count")))
        writer.add_head_count_kv(int(field_value(target, f"{target_arch}.attention.head_count_kv")))
        writer.add_key_length(int(field_value(target, f"{target_arch}.attention.key_length")))
        writer.add_value_length(int(field_value(target, f"{target_arch}.attention.value_length")))
        writer.add_layer_norm_rms_eps(float(field_value(target, f"{target_arch}.attention.layer_norm_rms_epsilon")))
        writer.add_rope_dimension_count(int(field_value(target, f"{target_arch}.rope.dimension_count")))
        writer.add_rope_dimension_sections(
            [int(value) for value in field_value(target, f"{target_arch}.rope.dimension_sections")]
        )
        writer.add_rope_freq_base(float(field_value(target, f"{target_arch}.rope.freq_base")))
    writer.add_target_layers(meta["anchors"])  # type: ignore[arg-type]
    writer.add_target_hidden_size(int(meta["hidden_size"]))
    writer.add_spd_checkpoint_version(int(meta["version"]))
    writer.add_spd_stage_count(int(meta["num_stages"]))
    writer.add_spd_use_deepest(bool(meta["use_deepest"]))
    if span > 0:
        writer.add_spd_train_span(span)
        LOGGER.info("trained attention span: %d tokens (the decode path windows the sidecar to this)", span)
    else:
        LOGGER.warning(
            "this checkpoint does not record the window it was trained on, so the GGUF carries no "
            "%s.train_span and the sidecar will attend over the whole context -- acceptance "
            "collapses on requests longer than the trainer's --chunk. Re-run the conversion with "
            "--train-span N, or set LLAMA_SPD_SPAN=N at serve time.", SPD_ARCH)
    copy_tokenizer_metadata(target, writer)

    num_aggr_types = int(meta["num_aggr_types"])
    hidden_size = int(meta["hidden_size"])
    aggr = torch.zeros(
        (num_aggr_types, hidden_size, num_aggr_types * hidden_size),
        dtype=torch.float32,
    )
    for index in range(num_aggr_types):
        source = state_dict[f"aggr_projs.{index}.weight"]
        aggr[index, :, : source.shape[1]] = source.to(dtype=torch.float32)
    LOGGER.info("%-64s -> %s", "aggr_projs.*.weight", "aggr.weight")
    writer.add_tensor(
        "aggr.weight",
        gguf.quantize(aggr.numpy(), gguf.GGMLQuantizationType.BF16),
        raw_dtype=gguf.GGMLQuantizationType.BF16,
    )
    del aggr

    for source_name, tensor in state_dict.items():
        if source_name.startswith("aggr_projs."):
            continue
        output_name = checkpoint_tensor_name(source_name)
        LOGGER.info("%-64s -> %s", source_name, output_name)
        add_checkpoint_tensor(writer, output_name, tensor)

    draft_token_ids = np.asarray(config["draft_token_ids"], dtype=np.int64)
    writer.add_tensor("d2t", draft_token_ids, raw_dtype=gguf.GGMLQuantizationType.I64)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    LOGGER.info("Wrote %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an SPD v11 PyTorch sidecar checkpoint to BF16 GGUF")
    parser.add_argument("checkpoint", type=Path, help="SPD speculation-head .pt checkpoint")
    parser.add_argument("target_gguf", type=Path, help="matching target GGUF")
    parser.add_argument("--outfile", type=Path, help="output GGUF path")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    parser.add_argument("--assets", type=Path, default=None,
                        help="spd-train assets .pt; folds the target final-norm weight "
                             "into output.weight (required for spd-train checkpoints)")
    parser.add_argument("--train-span", type=int, default=None,
                        help="window length the head was trained on, in tokens (the trainer's "
                             "--chunk). Overrides the value recorded in the checkpoint; needed "
                             "for checkpoints written before the trainer recorded it. 0 serves "
                             "the sidecar with unbounded attention")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    target = args.target_gguf.resolve()
    output = (
        args.outfile.resolve()
        if args.outfile is not None
        else checkpoint.with_name(f"{checkpoint.stem}-BF16.gguf")
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not target.is_file():
        raise FileNotFoundError(target)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}; pass --overwrite to replace it")
    if args.assets is not None and not args.assets.is_file():
        raise FileNotFoundError(args.assets)

    convert(checkpoint, target, output, assets_path=args.assets, train_span=args.train_span)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
