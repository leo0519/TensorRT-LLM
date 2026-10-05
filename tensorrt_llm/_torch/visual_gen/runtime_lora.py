# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Startup-preloaded LoRA adapters for VisualGen transformers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import safetensors.torch
import torch
import torch.nn as nn

from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.args import RuntimeLoRAConfig

_QKV_SUFFIXES = (".to_q", ".to_k", ".to_v")
_DEFAULT_STRIP_PREFIXES = (
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "model.",
    "dit.",
)


@dataclass(frozen=True)
class RuntimeLoRAApplication:
    """Summary returned after applying one LoRA adapter."""

    path: str
    applied_modules: Tuple[str, ...]
    skipped_non_targets: int
    skipped_incomplete: int


@dataclass(frozen=True)
class _LoRAPair:
    name: str
    a: torch.Tensor
    b: torch.Tensor
    scale: float


@dataclass(frozen=True)
class _RuntimeLoRAAdapter:
    a: torch.Tensor
    b: torch.Tensor
    scale: float
    output_start: Optional[int] = None

    @property
    def output_end(self) -> Optional[int]:
        if self.output_start is None:
            return None
        return self.output_start + self.b.shape[0]


@dataclass(frozen=True)
class _PreparedRuntimeLoRADelta:
    adapter: _RuntimeLoRAAdapter
    delta: Optional[torch.Tensor]


@dataclass(frozen=True)
class _PreparedRuntimeLoRATarget:
    module_name: str
    base_layer: nn.Module
    deltas: Tuple[_PreparedRuntimeLoRADelta, ...]


@dataclass(frozen=True)
class _RuntimeLoRAPlan:
    application: RuntimeLoRAApplication
    targets: Tuple[_PreparedRuntimeLoRATarget, ...]


class RuntimeLoRABranch(nn.Module):
    """Resident low-rank branch added to a linear without changing its base weight."""

    def __init__(
        self,
        adapters: Iterable[_RuntimeLoRAAdapter],
        output_features: int,
    ) -> None:
        super().__init__()
        adapters = tuple(adapters)
        if not adapters:
            raise ValueError("Runtime LoRA branch requires at least one adapter")
        self.enabled = False
        self.output_features = output_features
        self._output_spans: list[tuple[Optional[int], Optional[int]]] = []
        for index, adapter in enumerate(adapters):
            self.register_buffer(f"a_{index}", adapter.a.contiguous(), persistent=False)
            self.register_buffer(
                f"b_{index}",
                adapter.b.mul(float(adapter.scale)).contiguous(),
                persistent=False,
            )
            self._output_spans.append((adapter.output_start, adapter.output_end))
        self._native_op_available = hasattr(torch.ops.trtllm, "resident_lora_gemm_add_")

    @staticmethod
    def _floating_input(x: torch.Tensor | Fp4QuantizedTensor) -> torch.Tensor:
        if isinstance(x, Fp4QuantizedTensor):
            x = x.unquantized_hidden_states
            if x is None:
                raise ValueError(
                    "Runtime LoRA requires an Fp4QuantizedTensor with an unquantized view"
                )
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "Runtime LoRA branches require the original floating-point linear input"
            )
        return x

    def _delta(self, x: torch.Tensor) -> torch.Tensor:
        if len(self._output_spans) == 1 and self._output_spans[0][0] is None:
            return torch.nn.functional.linear(torch.nn.functional.linear(x, self.a_0), self.b_0)

        result = x.new_zeros((*x.shape[:-1], self.output_features))
        for index, (output_start, output_end) in enumerate(self._output_spans):
            delta = torch.nn.functional.linear(
                torch.nn.functional.linear(x, getattr(self, f"a_{index}")),
                getattr(self, f"b_{index}"),
            )
            if output_start is None:
                result.add_(delta)
            else:
                result[..., output_start:output_end].add_(delta)
        return result

    def _can_use_native(self, output: torch.Tensor, x: torch.Tensor) -> bool:
        return (
            self._native_op_available
            and len(self._output_spans) == 1
            and self._output_spans[0][0] is None
            and not torch.is_grad_enabled()
            and x.is_cuda
            and output.is_cuda
            and x.is_contiguous()
            and output.is_contiguous()
            and x.dtype in (torch.float16, torch.bfloat16)
            and output.dtype == x.dtype == self.a_0.dtype == self.b_0.dtype
            and x.device == output.device == self.a_0.device == self.b_0.device
            and x.shape[:-1] == output.shape[:-1]
            and x.shape[-1] == self.a_0.shape[1]
            and output.shape[-1] == self.b_0.shape[0]
        )

    def add_to(
        self,
        output: torch.Tensor,
        x: torch.Tensor | Fp4QuantizedTensor,
    ) -> torch.Tensor:
        x = self._floating_input(x)
        if self._can_use_native(output, x):
            torch.ops.trtllm.resident_lora_gemm_add_(output, x, self.a_0, self.b_0)
            return output
        return output.add_(self._delta(x).to(dtype=output.dtype))

    def forward(self, x: torch.Tensor | Fp4QuantizedTensor) -> torch.Tensor:
        return self._delta(self._floating_input(x))


def apply_runtime_lora(
    transformer: nn.Module,
    config: RuntimeLoRAConfig,
    *,
    default_strip_prefixes: Iterable[str] = (),
    raise_on_no_matches: bool = True,
) -> RuntimeLoRAApplication:
    """Fuse startup LoRA deltas into matching transformer module weights."""

    plan = _prepare_runtime_lora(
        transformer,
        config,
        default_strip_prefixes=default_strip_prefixes,
        raise_on_no_matches=raise_on_no_matches,
    )
    application = _commit_runtime_lora_plan(plan)
    _log_runtime_lora_application(application)
    return application


def attach_runtime_lora(
    transformer: nn.Module,
    config: RuntimeLoRAConfig,
    *,
    default_strip_prefixes: Iterable[str] = (),
    raise_on_no_matches: bool = True,
) -> RuntimeLoRAApplication:
    """Attach toggleable A/B branches while preserving quantized base weights."""
    plan = _prepare_runtime_lora(
        transformer,
        config,
        default_strip_prefixes=default_strip_prefixes,
        raise_on_no_matches=raise_on_no_matches,
        materialize_deltas=False,
    )
    for target in plan.targets:
        branch = RuntimeLoRABranch(
            (prepared.adapter for prepared in target.deltas),
            int(target.base_layer.out_features),
        )
        target.base_layer.add_module("_trtllm_runtime_lora_branch", branch)
        # TensorRT-LLM Linear calls branch.add_to() from apply_linear so it can
        # use the native in-place path. Keep a hook fallback for torch.nn.Linear.
        if not getattr(target.base_layer, "_supports_runtime_lora_branch", False):
            target.base_layer.register_forward_hook(_runtime_lora_forward_hook)
    logger.info(
        f"Runtime LoRA branches attached from {plan.application.path}: "
        f"applied={len(plan.application.applied_modules)}"
    )
    return plan.application


def set_runtime_lora_enabled(transformer: nn.Module, enabled: bool) -> int:
    """Toggle every resident low-rank branch below a transformer."""
    branches = [module for module in transformer.modules() if isinstance(module, RuntimeLoRABranch)]
    for branch in branches:
        branch.enabled = enabled
    return len(branches)


def _runtime_lora_forward_hook(module: nn.Module, args: tuple, output: torch.Tensor):
    branch = module._trtllm_runtime_lora_branch
    if not branch.enabled:
        return None
    return branch.add_to(output, args[0])


def _prepare_runtime_lora(
    transformer: nn.Module,
    config: RuntimeLoRAConfig,
    *,
    default_strip_prefixes: Iterable[str] = (),
    raise_on_no_matches: bool = True,
    materialize_deltas: bool = True,
) -> _RuntimeLoRAPlan:
    """Validate and stage startup LoRA deltas without mutating transformer weights."""

    if not isinstance(transformer, nn.Module):
        raise ValueError(
            f"Runtime LoRA requires an nn.Module target, got {type(transformer).__name__}"
        )

    raw_tensors = _load_lora_tensors(config.path)
    peft_alpha = _load_peft_lora_alpha(config.path)
    pairs, skipped_incomplete = _extract_lora_pairs(raw_tensors, config, peft_alpha)
    modules = dict(transformer.named_modules())
    targets: Dict[str, List[_RuntimeLoRAAdapter]] = {}
    skipped_non_targets = 0

    strip_prefixes = _dedupe_prefixes(
        tuple(config.strip_prefixes) + tuple(default_strip_prefixes) + _DEFAULT_STRIP_PREFIXES
    )

    qkv_groups: Dict[str, Dict[str, _LoRAPair]] = {}
    incomplete_qkv_names: List[str] = []
    skipped_target_names: List[str] = []
    for pair in pairs:
        target_name, fused_suffix = _resolve_target_name(
            pair.name,
            modules,
            strip_prefixes=strip_prefixes,
            key_map=config.key_map,
            fuse_qkv=config.fuse_qkv,
        )
        if target_name is None:
            skipped_non_targets += 1
            skipped_target_names.append(pair.name)
            continue

        if fused_suffix is None:
            targets.setdefault(target_name, []).append(
                _RuntimeLoRAAdapter(pair.a, pair.b, pair.scale)
            )
        else:
            qkv_groups.setdefault(target_name, {})[fused_suffix] = pair

    for target_name, parts in qkv_groups.items():
        if not all(suffix in parts for suffix in _QKV_SUFFIXES):
            skipped_incomplete += len(parts)
            incomplete_qkv_names.append(target_name)
            continue
        base_module = modules[target_name]
        fused_out = int(getattr(base_module, "out_features", 0))
        total_out = sum(parts[suffix].b.shape[0] for suffix in _QKV_SUFFIXES)
        if total_out != fused_out:
            raise ValueError(
                f"Runtime LoRA fused-QKV spans for '{target_name}' sum to {total_out} "
                f"but the layer output is {fused_out}. The adapter q/k/v split does "
                "not match the fused layer; disable fuse_qkv or use a matching adapter."
            )
        output_start = 0
        for suffix in _QKV_SUFFIXES:
            pair = parts[suffix]
            targets.setdefault(target_name, []).append(
                _RuntimeLoRAAdapter(pair.a, pair.b, pair.scale, output_start=output_start)
            )
            output_start += pair.b.shape[0]

    has_resolved_targets = bool(targets)
    if config.strict and skipped_non_targets and (raise_on_no_matches or has_resolved_targets):
        sample_names = ", ".join(skipped_target_names[:5])
        if len(skipped_target_names) > 5:
            sample_names += ", ..."
        raise ValueError(
            f"Runtime LoRA skipped {skipped_non_targets} adapter target(s) from "
            f"{config.path!r}: {sample_names}. Every LoRA target must match a "
            "transformer module in strict mode. Check target_components, "
            "strip_prefixes, and key_map."
        )

    if config.strict and incomplete_qkv_names:
        raise ValueError(
            "Runtime LoRA found incomplete fused-QKV adapter groups for "
            f"{incomplete_qkv_names}. Provide all to_q/to_k/to_v tensors, "
            "disable fuse_qkv, or set strict=False."
        )

    prepared_targets: List[_PreparedRuntimeLoRATarget] = []
    for module_name, adapters in targets.items():
        base_module = modules[module_name]
        if getattr(base_module, "_trtllm_runtime_lora_fused", False):
            raise ValueError(f"Runtime LoRA target '{module_name}' is already fused")
        if hasattr(base_module, "_trtllm_runtime_lora_branch"):
            raise ValueError(f"Runtime LoRA target '{module_name}' already has a branch")
        if not _is_linear_like(base_module):
            if config.strict:
                raise ValueError(
                    f"Runtime LoRA target '{module_name}' is not linear-like: "
                    f"{type(base_module).__name__}"
                )
            skipped_non_targets += len(adapters)
            continue

        try:
            prepared_targets.append(
                _prepare_lora_target_for_fusion(
                    base_module,
                    adapters,
                    module_name,
                    materialize_deltas=materialize_deltas,
                )
            )
        except ValueError:
            if config.strict:
                raise
            skipped_non_targets += len(adapters)
            continue

    applied_modules = tuple(target.module_name for target in prepared_targets)
    if not applied_modules and raise_on_no_matches and config.strict:
        raise ValueError(
            f"No Runtime LoRA modules from {config.path!r} matched the transformer. "
            "Check target_components, strip_prefixes, and key_map."
        )

    application = RuntimeLoRAApplication(
        path=config.path,
        applied_modules=applied_modules,
        skipped_non_targets=skipped_non_targets,
        skipped_incomplete=skipped_incomplete,
    )
    return _RuntimeLoRAPlan(application, tuple(prepared_targets))


def _commit_runtime_lora_plan(plan: _RuntimeLoRAPlan) -> RuntimeLoRAApplication:
    for target in plan.targets:
        _fuse_lora_into_linear(target.base_layer, target.deltas, target.module_name)
    return plan.application


def _log_runtime_lora_application(application: RuntimeLoRAApplication) -> None:
    logger.info(
        f"Runtime LoRA fused from {application.path}: "
        f"applied={len(application.applied_modules)}, "
        f"skipped_non_targets={application.skipped_non_targets}, "
        f"skipped_incomplete={application.skipped_incomplete}"
    )


def _prepare_lora_target_for_fusion(
    base_layer: nn.Module,
    adapters: Iterable[_RuntimeLoRAAdapter],
    module_name: str,
    *,
    materialize_deltas: bool = True,
) -> _PreparedRuntimeLoRATarget:
    weight = getattr(base_layer, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.is_meta:
        raise ValueError(f"Runtime LoRA target '{module_name}' has no loaded weight tensor")
    if materialize_deltas and weight.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise ValueError(
            f"Runtime LoRA target '{module_name}' cannot fuse into weight dtype {weight.dtype}"
        )

    in_features = int(getattr(base_layer, "in_features", 0))
    out_features = int(getattr(base_layer, "out_features", 0))
    device = weight.device
    prepared_deltas: List[_PreparedRuntimeLoRADelta] = []
    with torch.no_grad():
        for adapter in adapters:
            local_adapter = _shard_adapter(base_layer, adapter, module_name, device)
            if not materialize_deltas:
                # nn.Linear has no dtype attribute, and a later .to(dtype=...)
                # can leave a custom Linear's declared dtype stale. For floating
                # weights, their actual dtype determines the linear input dtype.
                activation_dtype = (
                    weight.dtype
                    if weight.is_floating_point()
                    else getattr(base_layer, "dtype", None)
                )
                if activation_dtype not in (
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                    torch.float64,
                ):
                    activation_dtype = adapter.a.dtype
                local_adapter = _cast_adapter(local_adapter, activation_dtype)
            _validate_adapter_for_fusion(local_adapter, module_name, in_features, out_features)
            delta = None
            if materialize_deltas:
                _validate_lora_delta_shape_for_adapter(weight, local_adapter, module_name)
                delta = _compute_lora_weight_delta(local_adapter, device, weight.dtype)
                delta = _prepare_lora_delta_for_weight(weight, delta, local_adapter, module_name)
                target = _lora_weight_target(weight, local_adapter, module_name)
                _validate_lora_delta_shape(
                    tuple(delta.shape),
                    tuple(target.shape),
                    local_adapter,
                    module_name,
                )
            prepared_deltas.append(_PreparedRuntimeLoRADelta(local_adapter, delta))
    return _PreparedRuntimeLoRATarget(module_name, base_layer, tuple(prepared_deltas))


def _fuse_lora_into_linear(
    base_layer: nn.Module,
    prepared_deltas: Iterable[_PreparedRuntimeLoRADelta],
    module_name: str,
) -> None:
    weight = getattr(base_layer, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.is_meta:
        raise ValueError(f"Runtime LoRA target '{module_name}' has no loaded weight tensor")
    with torch.no_grad():
        for prepared_delta in prepared_deltas:
            if prepared_delta.delta is None:
                raise ValueError(f"Runtime LoRA target '{module_name}' has no materialized delta")
            _add_lora_delta_to_weight(
                weight,
                prepared_delta.delta,
                prepared_delta.adapter,
                module_name,
            )
    setattr(base_layer, "_trtllm_runtime_lora_fused", True)


def _validate_adapter_for_fusion(
    adapter: _RuntimeLoRAAdapter,
    module_name: str,
    in_features: int,
    out_features: int,
) -> None:
    if adapter.a.ndim != 2 or adapter.b.ndim != 2:
        raise ValueError(
            f"Runtime LoRA target '{module_name}' requires 2D A/B tensors, "
            f"got A={tuple(adapter.a.shape)}, B={tuple(adapter.b.shape)}"
        )
    if adapter.a.shape[0] != adapter.b.shape[1]:
        raise ValueError(
            f"Runtime LoRA rank mismatch for '{module_name}': "
            f"A rank {adapter.a.shape[0]} != B rank {adapter.b.shape[1]}"
        )
    if adapter.a.shape[1] != in_features:
        raise ValueError(
            f"Runtime LoRA input mismatch for '{module_name}': "
            f"A has {adapter.a.shape[1]} columns, layer expects {in_features}"
        )

    if adapter.output_start is None:
        if adapter.b.shape[0] != out_features:
            raise ValueError(
                f"Runtime LoRA output mismatch for '{module_name}': "
                f"B has {adapter.b.shape[0]} rows, expected {out_features}"
            )
        return

    if adapter.output_start < 0 or adapter.output_end is None:
        raise ValueError(f"Invalid Runtime LoRA output span for '{module_name}'")
    if adapter.output_end > out_features:
        raise ValueError(
            f"Runtime LoRA output span for '{module_name}' exceeds layer output: "
            f"end={adapter.output_end}, out_features={out_features}"
        )


def _compute_lora_weight_delta(
    adapter: _RuntimeLoRAAdapter,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    a = adapter.a.to(device=device, dtype=dtype)
    b = adapter.b.to(device=device, dtype=dtype)
    return torch.matmul(b, a).mul_(float(adapter.scale))


def _add_lora_delta_to_weight(
    weight: torch.Tensor,
    delta: torch.Tensor,
    adapter: _RuntimeLoRAAdapter,
    module_name: str,
) -> None:
    target = _lora_weight_target(weight, adapter, module_name)
    _validate_lora_delta_shape(tuple(delta.shape), tuple(target.shape), adapter, module_name)
    target.add_(delta)


def _prepare_lora_delta_for_weight(
    weight: torch.Tensor,
    delta: torch.Tensor,
    adapter: _RuntimeLoRAAdapter,
    module_name: str,
) -> torch.Tensor:
    if weight.ndim == 2:
        return delta
    if weight.ndim == 3 and weight.shape[0] == 1:
        return delta.t().unsqueeze(0)

    raise ValueError(
        f"Runtime LoRA target '{module_name}' has unsupported weight shape {tuple(weight.shape)}"
    )


def _validate_lora_delta_shape_for_adapter(
    weight: torch.Tensor,
    adapter: _RuntimeLoRAAdapter,
    module_name: str,
) -> None:
    target = _lora_weight_target(weight, adapter, module_name)
    if weight.ndim == 2:
        delta_shape = (adapter.b.shape[0], adapter.a.shape[1])
    elif weight.ndim == 3 and weight.shape[0] == 1:
        delta_shape = (1, adapter.a.shape[1], adapter.b.shape[0])
    else:
        raise ValueError(
            f"Runtime LoRA target '{module_name}' has unsupported weight shape "
            f"{tuple(weight.shape)}"
        )
    _validate_lora_delta_shape(delta_shape, tuple(target.shape), adapter, module_name)


def _lora_weight_target(
    weight: torch.Tensor,
    adapter: _RuntimeLoRAAdapter,
    module_name: str,
) -> torch.Tensor:
    output_start = adapter.output_start
    output_end = adapter.output_end
    if weight.ndim == 2:
        if output_start is None:
            return weight
        return weight[output_start:output_end, :]

    if weight.ndim == 3 and weight.shape[0] == 1:
        if output_start is None:
            return weight
        return weight[:, :, output_start:output_end]

    raise ValueError(
        f"Runtime LoRA target '{module_name}' has unsupported weight shape {tuple(weight.shape)}"
    )


def _validate_lora_delta_shape(
    delta_shape: Tuple[int, ...],
    target_shape: Tuple[int, ...],
    adapter: _RuntimeLoRAAdapter,
    module_name: str,
) -> None:
    if delta_shape == target_shape:
        return
    if adapter.output_start is None:
        raise ValueError(
            f"Runtime LoRA delta shape mismatch for '{module_name}': "
            f"delta={delta_shape}, weight={target_shape}"
        )
    raise ValueError(
        f"Runtime LoRA fused span mismatch for '{module_name}': "
        f"delta={delta_shape}, span={target_shape}"
    )


def _find_safetensors_files(path: str) -> List[Path]:
    lora_path = Path(path)
    if lora_path.is_file() and lora_path.suffix == ".safetensors":
        return [lora_path]
    if lora_path.is_dir():
        return sorted(lora_path.glob("*.safetensors"))
    return []


def _load_lora_tensors(path: str) -> Dict[str, torch.Tensor]:
    sft_paths = _find_safetensors_files(path)
    if not sft_paths:
        raise ValueError(f"No safetensors files found at {path}")

    raw: Dict[str, torch.Tensor] = {}
    key_sources: Dict[str, Path] = {}
    for sft_path in sft_paths:
        with safetensors.torch.safe_open(sft_path, framework="pt") as f:
            for key in f.keys():
                if key in raw:
                    raise ValueError(
                        f"Runtime LoRA duplicate tensor key {key!r} in {sft_path} "
                        f"and {key_sources[key]}"
                    )
                raw[key] = f.get_tensor(key)
                key_sources[key] = sft_path
    return raw


def _load_peft_lora_alpha(path: str) -> Optional[float]:
    lora_path = Path(path)
    config_path = (
        lora_path.parent / "adapter_config.json"
        if lora_path.is_file()
        else lora_path / "adapter_config.json"
    )
    if not config_path.is_file():
        return None

    with config_path.open("r", encoding="utf-8") as config_file:
        adapter_config = json.load(config_file)
    lora_alpha = adapter_config.get("lora_alpha")
    if lora_alpha is None:
        return None
    if not isinstance(lora_alpha, (int, float)):
        raise ValueError(
            f"Runtime LoRA adapter_config.json lora_alpha must be a number, got {type(lora_alpha).__name__}"
        )
    return float(lora_alpha)


def _extract_lora_pairs(
    raw: Dict[str, torch.Tensor],
    config: RuntimeLoRAConfig,
    peft_alpha: Optional[float] = None,
) -> Tuple[List[_LoRAPair], int]:
    down_suffixes = (".lora_A.weight", ".lora_down.weight")
    up_suffixes = (".lora_B.weight", ".lora_up.weight")

    down_tensors: Dict[str, torch.Tensor] = {}
    up_tensors: Dict[str, torch.Tensor] = {}
    alpha: Dict[str, float] = {}
    for key, tensor in raw.items():
        base = _strip_suffix(key, down_suffixes)
        if base is not None:
            down_tensors[base] = tensor
            continue
        base = _strip_suffix(key, up_suffixes)
        if base is not None:
            up_tensors[base] = tensor
            continue
        if key.endswith(".alpha"):
            alpha[key[: -len(".alpha")]] = float(tensor.item())

    pairs: List[_LoRAPair] = []
    skipped_incomplete = 0
    for base_name, a in down_tensors.items():
        b = up_tensors.get(base_name)
        if b is None:
            skipped_incomplete += 1
            continue
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(
                f"Runtime LoRA tensors for '{base_name}' must be 2D, "
                f"got A={tuple(a.shape)}, B={tuple(b.shape)}"
            )
        rank = a.shape[0]
        if rank <= 0 or b.shape[1] != rank:
            raise ValueError(
                f"Runtime LoRA rank mismatch for '{base_name}': "
                f"A={tuple(a.shape)}, B={tuple(b.shape)}"
            )
        pair_alpha = alpha.get(base_name, peft_alpha if peft_alpha is not None else float(rank))
        pair_scale = float(config.scale) * pair_alpha / float(rank)
        pairs.append(_LoRAPair(base_name, a, b, pair_scale))

    skipped_incomplete += len(set(up_tensors) - set(down_tensors))
    return pairs, skipped_incomplete


def _resolve_target_name(
    base_name: str,
    modules: Dict[str, nn.Module],
    *,
    strip_prefixes: Tuple[str, ...],
    key_map: Dict[str, str],
    fuse_qkv: bool,
) -> Tuple[Optional[str], Optional[str]]:
    for candidate in _candidate_names(base_name, strip_prefixes, key_map):
        if candidate in modules:
            return candidate, None
        if not fuse_qkv:
            continue
        for suffix in _QKV_SUFFIXES:
            if candidate.endswith(suffix):
                attn_prefix = candidate[: -len(suffix)]
                fused_name = f"{attn_prefix}.qkv_proj"
                if fused_name in modules:
                    return fused_name, suffix
    return None, None


def _candidate_names(
    base_name: str,
    strip_prefixes: Tuple[str, ...],
    key_map: Dict[str, str],
) -> Iterable[str]:
    names = [base_name, _strip_known_prefix(base_name, strip_prefixes)]
    seen = set()
    for name in names:
        for normalized in (_normalize_common_name(name), _apply_key_map(name, key_map)):
            normalized = _normalize_common_name(normalized)
            if normalized not in seen:
                seen.add(normalized)
                yield normalized


def _normalize_common_name(name: str) -> str:
    normalized = name
    for ff_prefix in (".ff.", ".audio_ff.", ".ffn.", ".img_mlp.", ".txt_mlp."):
        if ff_prefix + "net.0.proj" in normalized:
            normalized = normalized.replace(ff_prefix + "net.0.proj", ff_prefix + "up_proj")
        elif ff_prefix + "net.2" in normalized:
            normalized = normalized.replace(ff_prefix + "net.2", ff_prefix + "down_proj")
    normalized = normalized.replace(".q_norm.", ".norm_q.")
    normalized = normalized.replace(".k_norm.", ".norm_k.")
    return normalized


def _strip_known_prefix(name: str, strip_prefixes: Tuple[str, ...]) -> str:
    for prefix in strip_prefixes:
        if prefix and name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _apply_key_map(name: str, key_map: Dict[str, str]) -> str:
    for source, target in key_map.items():
        if name == source:
            return target
        source_prefix = f"{source}."
        if name.startswith(source_prefix):
            return f"{target}{name[len(source) :]}"
    return name


def _strip_suffix(key: str, suffixes: Tuple[str, ...]) -> Optional[str]:
    for suffix in suffixes:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def _dedupe_prefixes(prefixes: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted({prefix for prefix in prefixes if prefix}, key=len, reverse=True))


def _is_linear_like(module: nn.Module) -> bool:
    return (
        callable(getattr(module, "forward", None))
        and hasattr(module, "in_features")
        and hasattr(module, "out_features")
    )


def _tp_mode_value(tp_mode) -> Optional[str]:
    if tp_mode is None:
        return None
    return getattr(tp_mode, "value", str(tp_mode))


def _cast_adapter(
    adapter: _RuntimeLoRAAdapter,
    dtype: torch.dtype,
) -> _RuntimeLoRAAdapter:
    return _RuntimeLoRAAdapter(
        adapter.a.to(dtype=dtype),
        adapter.b.to(dtype=dtype),
        adapter.scale,
        adapter.output_start,
    )


def _shard_adapter(
    base_layer: nn.Module,
    adapter: _RuntimeLoRAAdapter,
    module_name: str,
    device: torch.device,
) -> _RuntimeLoRAAdapter:
    tp_size = int(getattr(base_layer, "tp_size", 1) or 1)
    tp_mode = _tp_mode_value(getattr(base_layer, "tp_mode", None))
    if tp_size <= 1 or tp_mode is None:
        return _RuntimeLoRAAdapter(
            adapter.a.to(device=device),
            adapter.b.to(device=device),
            adapter.scale,
            adapter.output_start,
        )

    if isinstance(getattr(base_layer, "tp_sharding", None), dict):
        raise ValueError(
            f"Runtime LoRA target '{module_name}' uses fused tensor-parallel sharding; "
            "fused TP runtime LoRA is not supported yet."
        )
    if adapter.output_start is not None:
        raise ValueError(
            f"Runtime LoRA target '{module_name}' uses fused QKV output spans with TP; "
            "fused QKV TP runtime LoRA is not supported yet."
        )
    if not hasattr(base_layer, "load_shard"):
        raise ValueError(
            f"Runtime LoRA target '{module_name}' is tensor-parallel but has no load_shard()"
        )

    a = adapter.a
    b = adapter.b
    if tp_mode == "row":
        a = base_layer.load_shard(adapter.a, device=device)
        b = adapter.b.to(device=device)
    elif tp_mode == "column":
        a = adapter.a.to(device=device)
        b = base_layer.load_shard(adapter.b, device=device)
    else:
        raise ValueError(f"Unsupported tensor-parallel mode for Runtime LoRA: {tp_mode}")

    return _RuntimeLoRAAdapter(a, b, adapter.scale, adapter.output_start)
