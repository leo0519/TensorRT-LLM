# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in shared-input QKV LoRA contraction for resident LTX25 adapters."""

import torch
from torch import nn

from tensorrt_llm._torch.modules.linear import NVFP4LinearMethod, UnquantizedLinearMethod
from tensorrt_llm._torch.visual_gen.modules.attention import QKVMode
from tensorrt_llm._torch.visual_gen.runtime_lora import RuntimeLoRABranch


class GroupedSelfAttentionLoRA(nn.Module):
    """One packed A GEMM plus three B/accumulate GEMMs, instead of six GEMMs.

    Base quantized projections and scaled B buffers remain unchanged. Only
    explicitly installed, single-GPU self-attention groups can use this path.
    """

    def __init__(self, branches: tuple[RuntimeLoRABranch, ...]) -> None:
        super().__init__()
        # Original branches remain owned/registered by their linear layers.
        self.branches = branches
        self.register_buffer("packed_a", torch.cat([b.a_0 for b in branches]), persistent=False)
        self.ranks = tuple(b.a_0.shape[0] for b in branches)
        self.enabled = True

    def can_apply(self, hidden_states: torch.Tensor) -> bool:
        x = RuntimeLoRABranch._floating_input(hidden_states)
        return (
            self.enabled
            and all(branch.enabled for branch in self.branches)
            and not torch.is_grad_enabled()
            and x.is_cuda
            and x.ndim in (2, 3)
            and x.is_contiguous()
            and x.dtype == self.packed_a.dtype
            and x.device == self.packed_a.device
        )

    def project(
        self, attention: nn.Module, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        x = RuntimeLoRABranch._floating_input(hidden_states)
        projections = (attention.to_q, attention.to_k, attention.to_v)
        # Bypass only the already-accounted-for resident branch, not quantization.
        outputs = [p.quant_method.apply(p, hidden_states, p.bias) for p in projections]
        reduced = torch.ops.trtllm.cublas_mm(
            x.view(-1, x.shape[-1]), self.packed_a.t(), None, x.dtype
        )
        for output, intermediate, branch in zip(
            outputs, reduced.split(self.ranks, dim=-1), self.branches
        ):
            matrix = output.view(-1, output.shape[-1])
            torch.addmm(matrix, intermediate, branch.b_0.t(), out=matrix)
        return tuple(outputs)


def install_grouped_self_attention_lora(transformer: nn.Module) -> int:
    """Install compatible groups once; unsupported targets keep resident fallback."""
    installed = 0
    for block in transformer.transformer_blocks:
        for attention in (block.attn1, block.audio_attn1):
            if attention.qkv_mode != QKVMode.SEPARATE_QKV:
                continue
            if getattr(attention, "_grouped_runtime_lora", None) is not None:
                continue
            projections = (attention.to_q, attention.to_k, attention.to_v)
            if any(
                p.tp_size != 1
                or p.lora is not None
                or p.gather_output
                or not isinstance(p.quant_method, (NVFP4LinearMethod, UnquantizedLinearMethod))
                for p in projections
            ):
                continue
            branches = tuple(getattr(p, "_trtllm_runtime_lora_branch", None) for p in projections)
            if any(
                not isinstance(b, RuntimeLoRABranch)
                or b._output_spans != [(None, None)]
                or b.a_0.ndim != 2
                or b.b_0.ndim != 2
                for b in branches
            ):
                continue
            first = branches[0].a_0
            if first.dtype not in (torch.bfloat16, torch.float16) or not first.is_cuda:
                continue
            if any(
                b.a_0.shape[1] != first.shape[1]
                or b.a_0.dtype != first.dtype
                or b.b_0.dtype != first.dtype
                or b.a_0.device != first.device
                or b.b_0.device != first.device
                for b in branches
            ):
                continue
            attention.add_module("_grouped_runtime_lora", GroupedSelfAttentionLoRA(branches))
            installed += 1
    return installed
