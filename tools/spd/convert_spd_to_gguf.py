#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import logging
import math
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
FINAL_RAW_ARCH = "glm_final_raw_anchor_reader_v1"
BANK_FFN_ARCH = "glm_bank_ffn_norm_v1"
READER_ARCHS = (FINAL_RAW_ARCH, BANK_FFN_ARCH)


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


AGGR_PREFIXES = ("aggr_projs.", "aggr_blocks.", "aggr_scale", "aggr_bias")
PACKED_READER_PREFIXES = ("raw_anchor_norm.", "bank_anchor_norm.", "bank_correction_a.", "bank_correction_u.")


def packed_reader_tensor(name: str) -> bool:
    return name.startswith(PACKED_READER_PREFIXES) or ".mlp.addition_" in name


def checkpoint_tensor_name(name: str) -> str:
    if name.startswith(AGGR_PREFIXES):
        raise ValueError("aggregation tensors are packed into aggr.weight / aggr_blk.weight")
    if name == "lm_head.weight":
        return "output.weight"
    if name in ("target_mem_norm.weight", "target_k.weight", "target_v.weight", "target_k_norm.weight",
                "raw_anchor_k.weight", "raw_anchor_v.weight", "raw_anchor_k_norm.weight"):
        return name
    if name in ("raw_anchor_stage_k.weight", "raw_anchor_stage_v.weight"):
        return name.removesuffix(".weight")

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
        "reader_norm.weight":                 "reader_norm.weight",
        "reader_q.weight":                    "reader_q.weight",
        "reader_q_norm.weight":               "reader_q_norm.weight",
        "reader_o.weight":                    "reader_o.weight",
        "reader_stage_gate":                  "reader_stage_gate",
        "raw_reader_norm.weight":             "raw_reader_norm.weight",
        "raw_reader_q.weight":                "raw_reader_q.weight",
        "raw_reader_q_norm.weight":           "raw_reader_q_norm.weight",
        "raw_reader_o.weight":                "raw_reader_o.weight",
    }
    if suffix not in suffix_map:
        raise ValueError(f"unrecognized checkpoint tensor: {name}")
    return f"blk.{block}.{suffix_map[suffix]}"


def to_bf16_bytes(tensor: torch.Tensor) -> np.ndarray[Any, Any]:
    data = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
    return gguf.quantize(data, gguf.GGMLQuantizationType.BF16)


def add_checkpoint_tensor(writer: gguf.GGUFWriter, name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim == 1 or not name.endswith(".weight"):
        writer.add_tensor(name, tensor.detach().to(device="cpu", dtype=torch.float32).numpy())
        return
    writer.add_tensor(
        name,
        to_bf16_bytes(tensor),
        raw_dtype=gguf.GGMLQuantizationType.BF16,
    )


SUPPORTED_TARGET_ARCHS = ("qwen35", "deepseek4", "glm5-next")


def load_hashed_source(path: Path, *, weights_only: bool) -> tuple[Any, str]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        value = torch.load(source, map_location="cpu", weights_only=weights_only)
        source.seek(0)
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return value, digest.hexdigest()


def checkpoint_uint(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not minimum <= value <= 0xFFFFFFFF:
        raise ValueError(f"{name} must be an integer in [{minimum}, 4294967295], got {value!r}")
    return int(value)


def spd_stage_layers(trunk_blocks: int, num_stages: int) -> list[int]:
    """Legacy ceil-sized stages with the remainder in the last stage."""
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


def reader_checkpoint_shapes(config: dict[str, Any], hidden: int, anchors: int,
                             layers: int) -> dict[str, tuple[int, ...]]:
    eps = float(config["spec_attn"]["rms_norm_eps"])
    contracts = {
        "reader": {
            "width": 512, "heads": 8, "head_dim": 64, "shared_kv": True,
            "insertion": "after_self_attention_before_mlp", "rope_theta": 10000.0,
            "rms_norm_eps": eps, "dropout": 0.0, "sum_policy": "ascending_anchor_model_dtype_v1",
        },
        "layer_views": {
            "gates": "channelwise_tanh_zero_start", "accumulation": "float32_ascending_anchor_v1",
            "shared_kv_weights": True, "shared_kv_values": False,
        },
        "raw_reader": {
            "input": "raw_target_anchor", "normalization": "per_anchor_learned_rms_fp32",
            "width": 128, "heads": 4, "head_dim": 32, "stage_identity": "learned_key_and_value",
            "shared_kv": True, "insertion": "parallel_to_inherited_reader_before_mlp",
            "zero_start": "output_projection", "availability": "all_anchors_in_legal_aggregate_prefix",
            "history": "unchanged_legal_window", "query_rows": 128, "reader_layers": "final_only",
        },
    }
    for name, expected in contracts.items():
        if config.get(name) != expected:
            raise ValueError(f"unsupported GLM {name} contract")

    shapes = {
        "target_mem_norm.weight": (hidden,), "target_k.weight": (512, hidden),
        "target_v.weight": (512, hidden), "target_k_norm.weight": (64,),
        "raw_anchor_k.weight": (128, hidden), "raw_anchor_v.weight": (128, hidden),
        "raw_anchor_stage_k.weight": (anchors, 128), "raw_anchor_stage_v.weight": (anchors, 128),
        "raw_anchor_k_norm.weight": (32,),
    }
    for index in range(anchors):
        shapes[f"raw_anchor_norm.{index}.weight"] = (hidden,)
    for index in range(layers):
        shapes.update({f"spec_layers.{index}.{name}": shape for name, shape in {
            "reader_norm.weight": (hidden,), "reader_q.weight": (512, hidden),
            "reader_q_norm.weight": (64,), "reader_o.weight": (hidden, 512),
            "reader_stage_gate": (anchors, hidden),
        }.items()})
    shapes.update({f"spec_layers.{layers - 1}.{name}": shape for name, shape in {
        "raw_reader_norm.weight": (hidden,), "raw_reader_q.weight": (128, hidden),
        "raw_reader_q_norm.weight": (32,), "raw_reader_o.weight": (hidden, 128),
    }.items()})

    if config["architecture"] == BANK_FFN_ARCH:
        bank = config.get("bank_ffn_norm")
        expected_bank = {
            "normalization": "per_anchor_learned_rms_fp32_scalar_identity_gate_v1",
            "rank": 64, "correction": "depth_specific_u_after_inherited_bank_v1",
            "correction_sum": "ascending_anchor_model_dtype_v1", "ffn_layer": "final_only",
            "ffn_extension": "half_inherited_intermediate_size",
            "ffn_production_inherited": 8192, "ffn_production_extended": 12288,
            "ffn_execution": "separate_added_term_preserve_inherited_matmuls_v1",
            "bypass": "additions_off_all_three",
        }
        if not isinstance(bank, dict) or any(bank.get(key) != value for key, value in expected_bank.items()):
            raise ValueError("unsupported GLM bank_ffn_norm contract")
        width = int(config["spec_attn"]["intermediate_size"])
        if width < 2 or width % 2:
            raise ValueError("bank FFN extension requires an even inherited intermediate size")
        for index in range(anchors):
            shapes[f"bank_anchor_norm.{index}.weight"] = (hidden,)
            shapes[f"bank_anchor_norm.{index}.gate"] = ()
            shapes[f"bank_correction_a.{index}.weight"] = (64, hidden)
            shapes[f"bank_correction_u.{index}.weight"] = (hidden, 64)
        shapes.update({f"spec_layers.{layers - 1}.mlp.addition_{name}_proj.weight": shape
                       for name, shape in {"gate": (width // 2, hidden), "up": (width // 2, hidden),
                                           "down": (hidden, width // 2)}.items()})
    elif "bank_ffn_norm" in config:
        raise ValueError("bank_ffn_norm requires its own checkpoint architecture")
    return shapes


def validate_checkpoint(
    config: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    target: gguf.GGUFReader,
) -> dict[str, int | float | list[int] | bool | str]:
    target_arch = str(field_value(target, "general.architecture"))
    if target_arch not in SUPPORTED_TARGET_ARCHS:
        raise ValueError(
            f"SPD checkpoint requires a target GGUF with one of {SUPPORTED_TARGET_ARCHS}, got {target_arch!r}")

    architecture = config.get("architecture", "v11")
    has_readers = architecture in READER_ARCHS
    version = config["version"] if has_readers else int(config["version"])
    if (has_readers and (version != architecture or target_arch != "glm5-next")) or (
            not has_readers and (architecture != "v11" or version != SUPPORTED_CHECKPOINT_VERSION)):
        raise ValueError(
            f"unsupported SPD checkpoint architecture/version: {architecture!r}/{version!r}"
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

    draft_token_ids = np.asarray(config["draft_token_ids"])
    draft_vocab_size = checkpoint_uint(config["draft_vocab_size"], "draft_vocab_size", 1)
    if draft_token_ids.ndim != 1 or draft_token_ids.size != draft_vocab_size:
        raise ValueError("draft_token_ids must be a one-dimensional draft-vocabulary mapping")
    if not np.issubdtype(draft_token_ids.dtype, np.integer):
        raise ValueError("draft_token_ids must contain integer target token ids")
    if np.any((draft_token_ids < 0) | (draft_token_ids >= target_vocab_size)):
        raise ValueError("draft_token_ids contains a target token outside the target vocabulary")
    if np.unique(draft_token_ids).size != draft_token_ids.size:
        raise ValueError("draft_token_ids contains duplicates")

    num_stages = checkpoint_uint(config["num_stages"], "num_stages", 1)
    num_spec_layers = checkpoint_uint(config["num_spec_layers"], "num_spec_layers", 1)
    num_aggr_types = checkpoint_uint(config["num_aggr_types"], "num_aggr_types", 1)
    anchors = [checkpoint_uint(value, "aggr_feature_bound") for value in config["aggr_feature_bound"]]
    use_deepest = bool(config["trained_with_use_deepest"])
    if not use_deepest:
        raise ValueError("only checkpoints trained with deepest available snapshots are supported")
    if len(anchors) != num_aggr_types:
        raise ValueError("aggr_feature_bound length does not match num_aggr_types")

    target_blocks = int(field_value(target, f"{target_arch}.block_count"))
    nextn_field = target.get_field(f"{target_arch}.nextn_predict_layers")
    if target_arch == "glm5-next" and nextn_field is None:
        raise ValueError("glm5-next target GGUF must declare nextn_predict_layers to identify its trunk")
    nextn_blocks = 0 if nextn_field is None else int(nextn_field.contents())
    trunk_blocks = target_blocks - nextn_blocks
    if nextn_blocks < 0 or trunk_blocks <= 0 or num_stages > trunk_blocks:
        raise ValueError(
            f"target trunk block count {trunk_blocks} cannot be divided into {num_stages} SPD stages"
        )

    if (anchors[0] != 0 or anchors[-1] >= trunk_blocks
            or any(a >= b for a, b in zip(anchors, anchors[1:]))):
        raise ValueError(f"invalid SPD anchors for {trunk_blocks} target trunk blocks: {anchors}")

    stage_layers = config.get("stage_layers")
    if stage_layers is None:
        if target_arch == "glm5-next":
            raise ValueError("glm5-next checkpoints must declare explicit stage_layers")
        stage_layers = spd_stage_layers(trunk_blocks, num_stages)
        if stage_layers[-1] <= 0:
            raise ValueError(
                f"{num_stages} SPD stages over {trunk_blocks} trunk blocks leaves an empty trailing stage")
    else:
        if not isinstance(stage_layers, (list, tuple)) or len(stage_layers) != num_stages:
            raise ValueError("stage_layers must contain exactly num_stages layer counts")
        stage_layers = [checkpoint_uint(value, "stage_layers", 1) for value in stage_layers]
        if sum(stage_layers) != trunk_blocks:
            raise ValueError(f"checkpoint stage_layers {stage_layers} do not sum to the target trunk ({trunk_blocks})")
    bounds = [0]
    for count in stage_layers[:-1]:
        bounds.append(bounds[-1] + count)
    if config.get("stage_layers") is not None and any(anchor not in bounds for anchor in anchors):
        raise ValueError(f"aggr_feature_bound {anchors} must be stage-input boundaries from {bounds}")

    spec_attn = config.get("spec_attn")
    if target_arch == "glm5-next":
        if config.get("model_type") != "glm5-next-offline":
            raise ValueError("glm5-next checkpoints must declare model_type 'glm5-next-offline'")
        if spec_attn is None:
            raise ValueError("glm5-next checkpoints must declare the sidecar's spec_attn geometry")

    aggr_shared = bool(config.get("aggr_shared", False))
    if has_readers:
        if not aggr_shared or num_spec_layers != 2:
            raise ValueError("GLM reader checkpoints require the shared bank and two speculative layers")
        if config.get("mask_semantics") != "pre_step_snapshot_v1":
            raise ValueError("GLM readers require pre_step_snapshot_v1 mask semantics")
    expected_shapes: dict[str, tuple[int, ...]]
    if aggr_shared:
        expected_shapes = {f"aggr_blocks.{index}.weight": (hidden_size, hidden_size)
                           for index in range(num_aggr_types)}
        expected_shapes["aggr_scale"] = (num_aggr_types, hidden_size)
        expected_shapes["aggr_bias"] = (num_aggr_types, hidden_size)
    else:
        expected_shapes = {f"aggr_projs.{index}.weight": (hidden_size, hidden_size * (index + 1))
                           for index in range(num_aggr_types)}

    unexpected_aggr = {name for name in state_dict if name.startswith(AGGR_PREFIXES)} - set(expected_shapes)
    if unexpected_aggr:
        raise ValueError(f"unexpected aggregation tensors for aggr_shared={aggr_shared}: {sorted(unexpected_aggr)}")

    if spec_attn is not None:
        if not isinstance(spec_attn, dict):
            raise ValueError("spec_attn must contain the sidecar's attention geometry")
        dims = {}
        for key in ("num_heads", "num_kv_heads", "head_dim", "intermediate_size"):
            dims[key] = checkpoint_uint(spec_attn.get(key), f"spec_attn.{key}", 1)
        if dims["num_heads"] % dims["num_kv_heads"] != 0 or dims["head_dim"] % 2 != 0:
            raise ValueError("spec_attn requires an even head_dim and num_heads divisible by num_kv_heads")
        for key in ("rms_norm_eps", "rope_theta"):
            value = spec_attn.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"spec_attn.{key} must be finite and positive")
        q_width = dims["num_heads"] * dims["head_dim"]
        kv_width = dims["num_kv_heads"] * dims["head_dim"]
        intermediate_size = dims["intermediate_size"]
        layer_shapes = {
            "input_layernorm.weight": (hidden_size,),
            "post_attention_layernorm.weight": (hidden_size,),
            "self_attn.q_proj.weight": (q_width, hidden_size),
            "self_attn.k_proj.weight": (kv_width, hidden_size),
            "self_attn.v_proj.weight": (kv_width, hidden_size),
            "self_attn.o_proj.weight": (hidden_size, q_width),
            "self_attn.q_norm.weight": (dims["head_dim"],),
            "self_attn.k_norm.weight": (dims["head_dim"],),
            "mlp.gate_proj.weight": (intermediate_size, hidden_size),
            "mlp.up_proj.weight": (intermediate_size, hidden_size),
            "mlp.down_proj.weight": (hidden_size, intermediate_size),
        }
        for index in range(num_spec_layers):
            expected_shapes.update({f"spec_layers.{index}.{name}": shape for name, shape in layer_shapes.items()})

    if has_readers:
        expected_shapes.update(reader_checkpoint_shapes(config, hidden_size, num_aggr_types, num_spec_layers))
        unexpected = set(state_dict) - set(expected_shapes) - {"lm_head.weight"}
        if unexpected:
            raise ValueError(f"unexpected GLM reader checkpoint tensors: {sorted(unexpected)}")
    elif any(packed_reader_tensor(name) or name.startswith(("target_", "raw_anchor_"))
             or ".reader_" in name or ".raw_reader_" in name for name in state_dict):
        raise ValueError("reader tensors require an explicit supported GLM reader architecture")

    expected_shapes["lm_head.weight"] = (draft_vocab_size, hidden_size)
    for key, expected in expected_shapes.items():
        tensor = state_dict.get(key)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected:
            actual = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else None
            raise ValueError(f"{key} has shape {actual}, expected {expected}")
        if not tensor.is_floating_point():
            raise ValueError(f"{key} must be a floating-point tensor")

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
        if not name.startswith(AGGR_PREFIXES) and not packed_reader_tensor(name)
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
        "aggr_shared": aggr_shared,
        "anchors": anchors,
        "use_deepest": use_deepest,
        "trunk_blocks": trunk_blocks,
        "stage_layers": stage_layers,
        "version": version,
        "architecture": architecture,
    }


def convert(checkpoint_path: Path, target_path: Path, output_path: Path,
            assets_path: Path | None = None, train_span: int | None = None) -> None:
    LOGGER.info("Loading SPD checkpoint: %s", checkpoint_path)
    checkpoint, checkpoint_sha256 = load_hashed_source(checkpoint_path, weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise ValueError("SPD checkpoint must contain a config dictionary")
    config = checkpoint["config"]
    has_readers = config.get("architecture") in READER_ARCHS
    tensor_key = "target_memory_state_dict" if has_readers else "state_dict"
    if tensor_key not in checkpoint or set(checkpoint) - {"config", tensor_key, "training_step"}:
        raise ValueError(f"SPD checkpoint must contain config and {tensor_key}, with optional training_step")
    state_dict = checkpoint[tensor_key]
    if not isinstance(state_dict, dict):
        raise ValueError(f"invalid SPD checkpoint {tensor_key}")

    LOGGER.info("Reading target GGUF metadata: %s", target_path)
    target = gguf.GGUFReader(target_path)
    meta = validate_checkpoint(config, state_dict, target)
    span = resolve_train_span(config, train_span)

    if meta["target_arch"] == "glm5-next" and assets_path is None:
        raise ValueError("glm5-next conversion requires --assets to fold the trained target final norm into the output head")

    assets_sha256 = None
    if assets_path is not None:
        # Training computes spec logits as lm_head(target_final_norm(g0)); the
        # fork sidecar graph has no norm before its output head. RMS norm's
        # per-row 1/rms scalar cannot change the argmax, but the elementwise
        # norm weight can - fold it into the output weight for greedy drafts.
        assets, assets_sha256 = load_hashed_source(assets_path, weights_only=False)
        if meta["target_arch"] == "glm5-next":
            if not isinstance(assets, dict) or not isinstance(assets.get("meta"), dict):
                raise ValueError("glm5-next assets must include target metadata")
            asset_meta = assets["meta"]
            for key, expected in (("arch", "glm5-next"), ("n_embd", meta["hidden_size"]),
                                  ("vocab_size", meta["target_vocab_size"]), ("n_layer", meta["trunk_blocks"])):
                if asset_meta.get(key) != expected:
                    raise ValueError(f"assets {key} is {asset_meta.get(key)!r}, expected {expected!r}")
            norm = assets.get("final_norm.weight")
            if not isinstance(norm, torch.Tensor) or not norm.is_floating_point() or not torch.isfinite(norm).all():
                raise ValueError("assets final_norm.weight must be a finite floating-point tensor")
        norm_w = assets["final_norm.weight"].to(torch.float32)
        head_w = state_dict["lm_head.weight"]
        if norm_w.ndim != 1 or norm_w.shape[0] != head_w.shape[1]:
            raise ValueError(
                f"assets final_norm.weight shape {tuple(norm_w.shape)} does not match "
                f"lm_head.weight {tuple(head_w.shape)}")
        state_dict["lm_head.weight"] = head_w.to(torch.float32) * norm_w.unsqueeze(0)
        LOGGER.info("folded the target final-norm weight into output.weight (argmax-preserving)")
        LOGGER.warning("folding final-norm weights preserves greedy argmax, not normalized logits or probabilities")

    LOGGER.info(
        "Validated SPD v%s: %s stages with layer counts %s, %s speculative layers, %s aggregation types",
        meta["version"],
        meta["num_stages"],
        meta["stage_layers"],
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
        intermediate_size = int(spec_attn["intermediate_size"])
        if meta["architecture"] == BANK_FFN_ARCH:
            widths = [intermediate_size] * int(meta["num_spec_layers"])
            widths[-1] += intermediate_size // 2
            writer.add_feed_forward_length(widths)
        else:
            writer.add_feed_forward_length(intermediate_size)
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
    writer.add_string(f"{SPD_ARCH}.target_architecture", target_arch)
    writer.add_string(f"{SPD_ARCH}.checkpoint_sha256", checkpoint_sha256)
    if assets_sha256 is not None:
        writer.add_string(f"{SPD_ARCH}.assets_sha256", assets_sha256)
    if "training_step" in checkpoint:
        writer.add_uint32(f"{SPD_ARCH}.training_step", checkpoint_uint(checkpoint["training_step"], "training_step"))
    writer.add_spd_checkpoint_version(SUPPORTED_CHECKPOINT_VERSION)
    if has_readers:
        writer.add_string(f"{SPD_ARCH}.head_architecture", str(meta["architecture"]))
        writer.add_string(f"{SPD_ARCH}.mask_semantics", config["mask_semantics"])
        reader = config["reader"]
        writer.add_uint32(f"{SPD_ARCH}.reader.head_count", int(reader["heads"]))
        writer.add_uint32(f"{SPD_ARCH}.reader.head_dim", int(reader["head_dim"]))
        writer.add_float32(f"{SPD_ARCH}.reader.rope_freq_base", float(reader["rope_theta"]))
        writer.add_float32(f"{SPD_ARCH}.reader.rms_norm_eps", float(reader["rms_norm_eps"]))
        writer.add_uint32(f"{SPD_ARCH}.raw_reader.head_count", int(config["raw_reader"]["heads"]))
        writer.add_uint32(f"{SPD_ARCH}.raw_reader.head_dim", int(config["raw_reader"]["head_dim"]))
        if meta["architecture"] == BANK_FFN_ARCH:
            writer.add_uint32(f"{SPD_ARCH}.bank_correction_rank", int(config["bank_ffn_norm"]["rank"]))
    writer.add_spd_stage_count(int(meta["num_stages"]))
    writer.add_key_value(f"{SPD_ARCH}.stage_layers", meta["stage_layers"], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32)
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
    if meta["aggr_shared"]:
        # one block per anchor (expert k = anchor k) plus the per-type affine;
        # the sidecar keeps per-position block sums on the device. scale/bias
        # carry no ".weight" suffix so llama-quantize leaves them f32.
        blocks = torch.stack([state_dict[f"aggr_blocks.{index}.weight"].to(torch.float32)
                              for index in range(num_aggr_types)])
        LOGGER.info("%-64s -> %s", "aggr_blocks.*.weight", "aggr_blk.weight")
        writer.add_tensor(
            "aggr_blk.weight",
            gguf.quantize(blocks.numpy(), gguf.GGMLQuantizationType.BF16),
            raw_dtype=gguf.GGMLQuantizationType.BF16,
        )
        del blocks
        for name in ("aggr_scale", "aggr_bias"):
            LOGGER.info("%-64s -> %s", name, name)
            writer.add_tensor(name, state_dict[name].detach().to(device="cpu", dtype=torch.float32).numpy())
    else:
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

    if has_readers:
        for source, destination in (("raw_anchor_norm", "raw_anchor_norm"),
                                    ("bank_anchor_norm", "bank_anchor_norm"),
                                    ("bank_correction_a", "bank_correction_a.weight"),
                                    ("bank_correction_u", "bank_correction_u.weight")):
            if source != "raw_anchor_norm" and meta["architecture"] != BANK_FFN_ARCH:
                continue
            packed = torch.stack([state_dict[f"{source}.{index}.weight"] for index in range(num_aggr_types)])
            LOGGER.info("%-64s -> %s", f"{source}.*.weight", destination)
            add_checkpoint_tensor(writer, destination, packed)
        if meta["architecture"] == BANK_FFN_ARCH:
            gates = torch.stack([state_dict[f"bank_anchor_norm.{index}.gate"] for index in range(num_aggr_types)])
            add_checkpoint_tensor(writer, "bank_anchor_gate", gates.reshape(num_aggr_types, 1))
            LOGGER.info("folding the final FFN extension into gate/up rows and down columns")

    final_mlp = f"spec_layers.{int(meta['num_spec_layers']) - 1}.mlp."
    for source_name, tensor in state_dict.items():
        if source_name.startswith(AGGR_PREFIXES) or packed_reader_tensor(source_name):
            continue
        if has_readers and source_name.endswith(".reader_stage_gate"):
            # The graph consumes these constant tanh values in its F32 view sum.
            tensor = tensor.to(torch.float32).tanh()
        if meta["architecture"] == BANK_FFN_ARCH and source_name.startswith(final_mlp):
            suffix = source_name.removeprefix(final_mlp)
            addition = state_dict[final_mlp + "addition_" + suffix]
            tensor = torch.cat((tensor, addition), dim=1 if suffix == "down_proj.weight" else 0)
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
    parser = argparse.ArgumentParser(description="Convert an SPD v11 or GLM reader sidecar checkpoint to BF16 GGUF")
    parser.add_argument("checkpoint", type=Path, help="SPD speculation-head .pt checkpoint")
    parser.add_argument("target_gguf", type=Path, help="matching target GGUF")
    parser.add_argument("--outfile", type=Path, help="output GGUF path")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    parser.add_argument("--assets", type=Path, default=None,
                        help="spd-train assets .pt; folds the target final-norm weight "
                             "into output.weight (required for glm5-next; needed for offline-trained greedy parity)")
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
