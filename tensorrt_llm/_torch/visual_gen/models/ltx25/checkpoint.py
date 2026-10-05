# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 packed NVFP4 checkpoint layout and activation-consumer compatibility."""

import json
import re

import torch
from safetensors import safe_open

from tensorrt_llm._torch.modules.linear import NVFP4LinearMethod
from tensorrt_llm._torch.visual_gen.modules.attention import QKVMode
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def _nvfp4_quant_config(path: str, num_layers: int) -> QuantConfig:
    """Build TRT-LLM's mixed-precision config from LTX's embedded metadata."""
    with safe_open(path, framework="pt") as checkpoint:
        metadata = checkpoint.metadata() or {}
    try:
        quantization = json.loads(metadata["_quantization_metadata"])
        layers = quantization["layers"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Missing NVFP4 quantization metadata in {path}") from exc

    formats = {value.get("format") for value in layers.values()}
    if formats != {"nvfp4"}:
        raise ValueError(f"Unsupported quantization formats in {path}: {sorted(formats)}")

    quantized_blocks = {
        int(match.group(1))
        for name in layers
        if (match := re.match(r"model\.diffusion_model\.transformer_blocks\.(\d+)\.", name))
    }
    if not quantized_blocks:
        raise ValueError(f"No quantized transformer blocks declared in {path}")

    # The official checkpoint leaves projections outside the block stack,
    # attention gates, and its final blocks in BF16.
    exclude_modules = [
        "*adaln_single.*",
        "audio_patchify_proj",
        "audio_proj_out",
        "patchify_proj",
        "proj_out",
        "transformer_blocks.*.*.to_gate_logits",
        *(
            f"transformer_blocks.{index}.*"
            for index in range(num_layers)
            if index not in quantized_blocks
        ),
    ]
    return QuantConfig(
        quant_algo=QuantAlgo.NVFP4,
        group_size=16,
        exclude_modules=exclude_modules,
    )


def _separate_nvfp4_self_attention_qkv(model: torch.nn.Module) -> int:
    """Preserve each prequantized self-attention shard's calibration scales."""
    converted = 0
    for block in model.transformer_blocks:
        for attention in (block.attn1, block.audio_attn1):
            if attention.qkv_mode != QKVMode.FUSE_QKV:
                continue
            if attention._modules.pop("qkv_proj", None) is None:
                raise RuntimeError("Fused self-attention is missing qkv_proj")
            attention.qkv_mode = QKVMode.SEPARATE_QKV
            attention._init_qkv_proj()
            converted += 1
    model.__post_init__()
    return converted


def _convert_ltx_nvfp4_layout(weights: dict[str, torch.Tensor], device: torch.device) -> int:
    """Convert LTX packed nibbles and tiled scales to TRT loader input layout."""
    converted = 0
    for scale_2_key in [key for key in weights if key.endswith(".weight_scale_2")]:
        prefix = scale_2_key.removesuffix(".weight_scale_2")
        weight_key = f"{prefix}.weight"
        scale_key = f"{prefix}.weight_scale"
        if weight_key not in weights or scale_key not in weights:
            continue

        packed = weights[weight_key]
        if packed.dtype != torch.uint8:
            continue
        scale = weights[scale_key]
        if scale.dtype != torch.float8_e4m3fn:
            raise ValueError(f"Expected E4M3 NVFP4 scale for {scale_key}, got {scale.dtype}")

        weights[weight_key] = ((packed & 0x0F) << 4) | (packed >> 4)
        scale_device = scale.device
        linear_scale = torch.ops.trtllm.block_scale_interleave_reverse(
            scale.view(torch.uint8).to(device=device)
        )
        weights[scale_key] = linear_scale.to(device=scale_device).view(torch.float8_e4m3fn)
        converted += 1
    return converted


def _force_local_nvfp4_inputs(model: torch.nn.Module) -> int:
    """Keep fused producers BF16 wherever another consumer also needs BF16."""
    projections: list[torch.nn.Module] = []
    for module in model.modules():
        if getattr(module, "to_gate_logits", None) is None:
            continue
        projection = getattr(module, "qkv_proj", None)
        if projection is None:
            projection = getattr(module, "to_q", None)
        if projection is not None:
            projections.append(projection)

    for block in model.transformer_blocks:
        for attention in (block.audio_to_video_attn, block.video_to_audio_attn):
            # The AV dual AdaLN pairs a2v.Q/v2a.K and a2v.K/v2a.Q.
            # Since gated Q inputs must stay BF16, their paired K inputs must
            # do the same; each downstream projection quantizes locally.
            projections.append(attention.to_k)

    forced = 0
    for projection in projections:
        if not isinstance(getattr(projection, "quant_method", None), NVFP4LinearMethod):
            continue
        if getattr(projection, "force_dynamic_quantization", False):
            continue
        projection.force_dynamic_quantization = True
        forced += 1
    return forced
