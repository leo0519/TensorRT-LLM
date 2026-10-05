# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""Native single-file LTX-2.5 VisualGen pipeline."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn.attention import SDPBackend, sdpa_kernel

from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm.logger import logger

from ..ltx2.ltx2_core.rope import LTXRopeType
from ..ltx2.pipeline_ltx2 import _find_safetensors_files
from ..ltx2.transformer_ltx2 import LTXModelType
from ..ltx23.ltx23_core.connector import (
    LTX23AudioConnectorConfigurator,
    LTX23GemmaFeaturesExtractor,
    LTX23VideoConnectorConfigurator,
)
from ..ltx23.ltx23_core.modality import LTX23Modality
from ..ltx23.pipeline_ltx23 import LTX23Pipeline, _LTX23CUDAGraphRunner
from ..ltx23.pipeline_ltx23 import detect_native_ltx_pipeline as _detect_legacy_ltx_pipeline
from .text_encoder import (
    load_gemma4_config,
    load_gemma4_model,
    load_gemma4_tokenizer,
    pack_gemma4_hidden_states,
    tokenize_gemma4,
)
from .transformer_ltx25 import LTX25Model


class _LTX25CUDAGraphRunner(_LTX23CUDAGraphRunner):
    """Track LTX-2.5 keyframe masks without changing the LTX-2.3 runner."""

    def _key_parts_for(self, prefix, value):
        yield from super()._key_parts_for(prefix, value)
        if isinstance(value, LTX23Modality):
            yield (
                f"{prefix}.keyframes_mask",
                tuple(value.keyframes_mask.shape) if value.keyframes_mask is not None else None,
            )

    @staticmethod
    def _clone_value(value):
        cloned = _LTX23CUDAGraphRunner._clone_value(value)
        if isinstance(value, LTX23Modality):
            return replace(
                cloned,
                keyframes_mask=(
                    value.keyframes_mask.clone() if value.keyframes_mask is not None else None
                ),
            )
        return cloned

    @staticmethod
    def _copy_value(dst, src):
        result = _LTX23CUDAGraphRunner._copy_value(dst, src)
        if isinstance(src, LTX23Modality) and isinstance(dst, LTX23Modality):
            _LTX23CUDAGraphRunner._copy_optional_tensor(dst.keyframes_mask, src.keyframes_mask)
        return result


def detect_native_ltx_pipeline(config: dict) -> str:
    """Extend legacy checkpoint detection with the LTX-2.5 architecture marker."""
    detected = _detect_legacy_ltx_pipeline(config)
    if detected != "LTX23Pipeline":
        return detected
    keyframe_embedding = config["transformer"].get("use_keyframes_abs_pos_embedding", False)
    if not isinstance(keyframe_embedding, bool):
        raise ValueError("use_keyframes_abs_pos_embedding must be a boolean when present.")
    return "LTX25Pipeline" if keyframe_embedding else detected


def _build_ltx25_transformer(pipeline_config) -> LTX25Model:
    model_config = pipeline_config.model_configs["transformer"]
    cfg = model_config.pretrained_config
    rope_type = LTXRopeType(getattr(cfg, "rope_type", "interleaved"))
    double_precision_rope = getattr(cfg, "frequencies_precision", False) == "float64"
    apply_gated_attention = getattr(cfg, "apply_gated_attention", False)

    logger.info(
        f"LTX-2.5 transformer config: rope_type={rope_type.value}, "
        f"double_precision_rope={double_precision_rope}, "
        f"apply_gated_attention={apply_gated_attention} (AudioVideo)"
    )
    transformer = LTX25Model(
        model_type=LTXModelType.AudioVideo,
        use_keyframes_abs_pos_embedding=getattr(cfg, "use_keyframes_abs_pos_embedding", True),
        num_attention_heads=getattr(cfg, "num_attention_heads", 32),
        attention_head_dim=getattr(cfg, "attention_head_dim", 128),
        in_channels=getattr(cfg, "in_channels", 128),
        out_channels=getattr(cfg, "out_channels", 128),
        num_layers=getattr(cfg, "num_layers", 48),
        cross_attention_dim=getattr(cfg, "cross_attention_dim", 4096),
        audio_num_attention_heads=getattr(cfg, "audio_num_attention_heads", 32),
        audio_attention_head_dim=getattr(cfg, "audio_attention_head_dim", 64),
        audio_in_channels=getattr(cfg, "audio_in_channels", 128),
        audio_out_channels=getattr(cfg, "audio_out_channels", 128),
        audio_cross_attention_dim=getattr(cfg, "audio_cross_attention_dim", 2048),
        audio_positional_embedding_max_pos=getattr(cfg, "audio_positional_embedding_max_pos", [20]),
        norm_eps=float(getattr(cfg, "norm_eps", 1e-6)),
        caption_channels=getattr(cfg, "caption_channels", 3840),
        positional_embedding_theta=float(getattr(cfg, "positional_embedding_theta", 10000.0)),
        positional_embedding_max_pos=getattr(cfg, "positional_embedding_max_pos", [20, 2048, 2048]),
        timestep_scale_multiplier=getattr(cfg, "timestep_scale_multiplier", 1000),
        av_ca_timestep_scale_multiplier=getattr(cfg, "av_ca_timestep_scale_multiplier", 1),
        use_middle_indices_grid=getattr(cfg, "use_middle_indices_grid", True),
        rope_type=rope_type,
        double_precision_rope=double_precision_rope,
        apply_gated_attention=apply_gated_attention,
        model_config=model_config,
    )
    transformer._transformer_config = vars(cfg)
    return transformer


def _checkpoint_gemma_version(paths: list[str]) -> str:
    versions = set()
    for path in paths:
        with safe_open(path, framework="pt") as checkpoint:
            source = (checkpoint.metadata() or {}).get("gemma_source_checkpoint")
        if source is not None:
            versions.add(json.loads(source).get("gemma_version"))
    if len(versions) != 1 or not all(isinstance(version, str) and version for version in versions):
        raise ValueError(
            "LTX-2.5 checkpoint must declare one gemma_source_checkpoint.gemma_version."
        )
    return versions.pop()


def _load_component_strict(paths: list[str], module: torch.nn.Module, prefix: str) -> None:
    weights = {}
    for path in paths:
        with safe_open(path, framework="pt") as checkpoint:
            for key in checkpoint.keys():
                if key.startswith(prefix):
                    target = key.removeprefix(prefix)
                    if target in weights:
                        raise ValueError(f"Duplicate LTX-2.5 component weight: {key}")
                    weights[target] = checkpoint.get_tensor(key)
    # In particular, do not silently use randomly initialized readout projections.
    module.load_state_dict(weights, strict=True)


def _has_component(paths: list[str], prefix: str) -> bool:
    for path in paths:
        with safe_open(path, framework="pt") as checkpoint:
            if any(key.startswith(prefix) for key in checkpoint.keys()):
                return True
    return False


@register_pipeline(
    "LTX25Pipeline",
    hf_ids=["Lightricks/LTX-2.5"],
    defaults={
        # Require an explicit LTX-tuned Gemma4 checkpoint, not a generic Gemma model.
        "text_encoder_path": "",
        "workflow": "generation",
        "video_vae_path": None,
        "audio_vae_path": None,
        "spatial_upsampler_path": None,
        "iclora_path": None,
        "iclora_scale": 1.0,
        "iclora_grouped_qkv": False,
    },
    doc="Lightricks LTX-2.5 text-to-video generation with audio.",
)
class LTX25Pipeline(LTX23Pipeline):
    """LTX-2.3 generation flow with LTX-2.5 architecture and Gemma4 conditioning."""

    _cuda_graph_runner_cls = _LTX25CUDAGraphRunner

    @classmethod
    def resolve_variant(cls, config):
        workflow = config.extra_attrs.get("workflow", "generation")
        if workflow == "v2v":
            from .pipeline_ltx25_v2v import LTX25V2VPipeline

            return LTX25V2VPipeline
        if workflow != "generation":
            raise ValueError(f"Unsupported LTX-2.5 workflow: {workflow!r}")
        return cls

    def _init_transformer(self) -> None:
        cfg = self.pipeline_config.model_configs["transformer"].pretrained_config
        self.transformer_in_channels = getattr(cfg, "in_channels", 128)
        self.audio_in_channels = getattr(cfg, "audio_in_channels", 128)
        self.audio_out_channels = getattr(cfg, "audio_out_channels", 128)
        self.transformer = _build_ltx25_transformer(self.pipeline_config)

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: list[str] | None = None,
        *,
        text_encoder_path: str = "",
        **kwargs,
    ) -> None:
        skip_components = list(skip_components or [])
        needs_text = any(
            component not in skip_components
            for component in (
                PipelineComponent.TOKENIZER,
                PipelineComponent.TEXT_ENCODER,
                "connectors",
            )
        )
        if needs_text:
            paths = _find_safetensors_files(checkpoint_dir)
            self._gemma_config = load_gemma4_config(
                text_encoder_path, _checkpoint_gemma_version(paths)
            )
            self._gemma_weight_paths = (
                _find_safetensors_files(text_encoder_path)
                if Path(text_encoder_path).exists()
                else []
            )
            if PipelineComponent.TOKENIZER not in skip_components:
                self.tokenizer = load_gemma4_tokenizer(text_encoder_path)
            if PipelineComponent.TEXT_ENCODER not in skip_components:
                self.text_encoder = load_gemma4_model(
                    text_encoder_path, self._gemma_config, self.dtype, device
                )
        # Reuse LTX-2.3's VAE, scheduler and patchifier loading, but never its
        # Gemma3 classes. The connector override below uses the Gemma4 config.
        super().load_standard_components(
            checkpoint_dir,
            device,
            skip_components=[
                *skip_components,
                PipelineComponent.TOKENIZER,
                PipelineComponent.TEXT_ENCODER,
            ],
            **kwargs,
        )

    def _load_native_components(
        self,
        config: dict,
        sft_paths: list[str],
        device: torch.device,
        dtype: torch.dtype,
        skip_components: list[str] | None = None,
    ) -> None:
        skip_components = list(skip_components or [])
        super()._load_native_components(
            config, sft_paths, device, dtype, [*skip_components, "connectors"]
        )
        if "connectors" in skip_components:
            return
        cfg = config["transformer"]
        text_config = self._gemma_config.text_config
        if text_config.hidden_size != cfg.get("caption_channels", 3840):
            raise ValueError("Gemma4 hidden size does not match LTX-2.5 caption_channels.")
        self.feature_extractor = LTX23GemmaFeaturesExtractor(
            caption_channels=text_config.hidden_size,
            video_dim=cfg.get("cross_attention_dim", 4096),
            audio_dim=cfg.get("audio_cross_attention_dim", 2048),
            num_hidden_states=text_config.num_hidden_layers + 1,
        )
        prefix = "text_embedding_projection."
        projection_paths = (
            self._gemma_weight_paths
            if _has_component(self._gemma_weight_paths, prefix)
            else sft_paths
        )
        _load_component_strict(projection_paths, self.feature_extractor, prefix)
        self.feature_extractor.to(device=device, dtype=dtype).eval()
        self.video_connector = LTX23VideoConnectorConfigurator.from_config(config)
        self.audio_connector = LTX23AudioConnectorConfigurator.from_config(config)
        for name, connector in (("video", self.video_connector), ("audio", self.audio_connector)):
            _load_component_strict(
                sft_paths, connector, f"model.diffusion_model.{name}_embeddings_connector."
            )
            connector.to(device=device, dtype=dtype).eval()

    def _process_connectors(
        self, prompt_embeds: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Match LTX-2.5's masked-attention dispatch. PyTorch's default priority
        # can select a different kernel, changing BF16 context rounding before
        # NVFP4 denoising. Keep this policy local to LTX25 text connectors.
        with sdpa_kernel(
            [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ],
            set_priority=True,
        ):
            return super()._process_connectors(prompt_embeds, attention_mask)

    def _encode_prompt(
        self,
        prompt: str | list[str],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 1024,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompts = [prompt] if isinstance(prompt, str) else prompt
        input_ids, attention_mask = tokenize_gemma4(
            self.tokenizer, prompts, max_sequence_length, self.device
        )
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        embeddings = pack_gemma4_hidden_states(outputs.hidden_states, attention_mask).to(
            dtype=self.dtype
        )
        return (
            embeddings.repeat_interleave(num_videos_per_prompt, dim=0),
            attention_mask.repeat_interleave(num_videos_per_prompt, dim=0),
        )
