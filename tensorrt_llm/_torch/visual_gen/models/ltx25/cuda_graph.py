# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""LTX25 V2V graph inputs, including stage-specific modulation and strides."""

from dataclasses import replace

import torch

from ..ltx23.ltx23_core.modality import LTX23Modality
from .pipeline_ltx25 import _LTX25CUDAGraphRunner


class LTX25V2VCUDAGraphRunner(_LTX25CUDAGraphRunner):
    """Keep V2V graph contracts separate from the LTX23/T2V runners."""

    def _key_parts_for(self, prefix, value):
        yield from super()._key_parts_for(prefix, value)
        if isinstance(value, LTX23Modality):
            yield (
                f"{prefix}.layout",
                (value.latent.stride(), value.latent.dtype, value.latent.device),
            )
            cross_sigma = value.cross_modality_sigma
            yield (
                f"{prefix}.cross_sigma",
                None if cross_sigma is None else tuple(cross_sigma.shape),
            )

    @staticmethod
    def _clone_value(value):
        cloned = _LTX25CUDAGraphRunner._clone_value(value)
        if isinstance(value, LTX23Modality):
            return replace(
                cloned,
                cross_modality_sigma=(
                    value.cross_modality_sigma.clone()
                    if value.cross_modality_sigma is not None
                    else None
                ),
            )
        return cloned

    @staticmethod
    def _copy_value(dst, src):
        result = _LTX25CUDAGraphRunner._copy_value(dst, src)
        if isinstance(src, LTX23Modality) and isinstance(dst, LTX23Modality):
            _LTX25CUDAGraphRunner._copy_optional_tensor(
                dst.cross_modality_sigma, src.cross_modality_sigma
            )
        return result

    def capture(self, key, fn, args, kwargs):
        static_args = [self._clone_value(arg) for arg in args]
        static_kwargs = {name: self._clone_value(value) for name, value in kwargs.items()}
        for _ in range(self.WARMUP_STEPS):
            fn(*static_args, **static_kwargs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._get_pool()):
            output = fn(*static_args, **static_kwargs)
        self.graphs[key] = graph
        self.static_inputs[key] = (static_args, static_kwargs)
        # Keep outputs alive even when another stage captures into the shared pool.
        self._graph_output_refs[key] = output
        self.graph_outputs[key] = output
        self.memory_pool = graph.pool()
