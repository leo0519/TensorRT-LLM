# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""Native, resident two-stage IC-LoRA V2V, adapted from LTX-2 ICLoraPipeline.

Like LTX23 retake, this is a separate VisualGen workflow. Scheduling, reference
conditioning, LoRA stage switching and media components are all owned here;
neither ltx_core nor ltx_pipelines is a runtime dependency.
"""

from __future__ import annotations

import json
import math
from itertools import islice
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from safetensors import safe_open

from tensorrt_llm._torch.modules.linear import NVFP4LinearMethod
from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm._torch.visual_gen.runtime_lora import (
    attach_runtime_lora,
    set_runtime_lora_enabled,
)
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.args import RuntimeLoRAConfig

from ..ltx2.ltx2_core.audio_vae import AudioDecoderConfigurator, decode_audio
from ..ltx2.ltx2_core.types import AudioLatentShape, VideoLatentShape, VideoPixelShape
from ..ltx2.ltx2_core.upsampler import LatentUpsamplerConfigurator, upsample_video
from ..ltx2.ltx2_core.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ..ltx23.ltx23_core.audio_vae.vocoder import LTX23VocoderConfigurator
from ..ltx23.ltx23_core.modality import LTX23Modality
from ..ltx23.ltx23_core.video_vae_ltx23 import (
    LTX23VideoDecoderConfigurator,
    LTX23VideoEncoderConfigurator,
)
from ..ltx23.media_io import decode_video_by_frame
from .checkpoint import (
    _convert_ltx_nvfp4_layout,
    _force_local_nvfp4_inputs,
    _nvfp4_quant_config,
    _separate_nvfp4_self_attention_qkv,
)
from .cuda_graph import LTX25V2VCUDAGraphRunner
from .grouped_lora import install_grouped_self_attention_lora
from .iclora import (
    STAGE_1_SIGMAS,
    STAGE_2_SIGMAS,
    ICLoraState,
    add_noise,
    append_reference,
    euler_step,
    video_state,
)
from .pipeline_ltx25 import LTX25Pipeline

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
    from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest


def _pad_audio_for_video(
    audio: torch.Tensor, frames: int, fps: float, sample_rate: int
) -> torch.Tensor:
    # The VAE rounds audio duration down to its latent grid. VisualGen's muxer
    # uses -shortest, so preserve all video frames by padding only trailing silence.
    missing = math.ceil(frames * sample_rate / fps) - audio.shape[-1]
    return F.pad(audio, (0, missing)) if missing > 0 else audio


def _read_config(path: str) -> dict:
    with safe_open(path, framework="pt") as checkpoint:
        metadata = checkpoint.metadata() or {}
    if "config" not in metadata:
        raise ValueError(f"Missing embedded config in {path}")
    return json.loads(metadata["config"])


def _load_split_component(path: str, module: torch.nn.Module, prefix: str) -> None:
    """Load only the component's declared state, including shared VAE statistics."""
    weights = {}
    with safe_open(path, framework="pt") as checkpoint:
        keys = set(checkpoint.keys())
        for name in module.state_dict():
            candidates = [prefix + name]
            if name.startswith("per_channel_statistics."):
                candidates.append(name)
                if prefix.startswith("audio_vae."):
                    candidates.append("audio_vae." + name)
            candidates += ["vae." + key for key in candidates]
            matches = [key for key in candidates if key in keys]
            if len(matches) != 1:
                raise ValueError(f"Expected one checkpoint tensor for {prefix + name}: {matches}")
            weights[name] = checkpoint.get_tensor(matches[0])
    module.load_state_dict(weights, strict=True)


@register_pipeline("LTX25V2VPipeline", doc="Native LTX-2.5 two-stage IC-LoRA V2V.")
class LTX25V2VPipeline(LTX25Pipeline):
    """Single-GPU SDR reference-video conditioning with resident runtime LoRA."""

    _cuda_graph_runner_cls = LTX25V2VCUDAGraphRunner

    def __init__(self, pipeline_config: DiffusionPipelineConfig) -> None:
        if pipeline_config.runtime_lora is not None:
            raise ValueError(
                "LTX25 V2V owns stage-specific LoRA; use pipeline_config.iclora_path "
                "instead of runtime_lora_config."
            )
        parallel = pipeline_config.parallel
        if any(
            size != 1
            for size in (
                parallel.cfg_size,
                parallel.tp_size,
                parallel.seq_parallel_size,
                parallel.parallel_vae_size,
            )
        ):
            raise NotImplementedError("LTX-2.5 IC-LoRA V2V currently supports one GPU only.")
        if pipeline_config.cache_backend is not None:
            raise NotImplementedError("LTX-2.5 IC-LoRA V2V requires cache acceleration disabled.")
        for name in ("video_vae_path", "audio_vae_path", "spatial_upsampler_path", "iclora_path"):
            if not pipeline_config.extra_attrs.get(name):
                raise ValueError(f"LTX-2.5 V2V requires pipeline_config.{name}.")
        self._stage1_dynamic_modules: list[torch.nn.Module] = []
        self._lora_ready = False
        self._packed_nvfp4 = False
        self._stage_one = False
        self._graph_geometry: tuple | None = None
        super().__init__(pipeline_config)

    def _setup_cuda_graphs(self) -> None:
        super()._setup_cuda_graphs()
        runner = self._cuda_graph_runners.get("transformer")
        if runner is not None:
            runner.register_extra_key_fn(
                "iclora_stage_one", lambda *args, **kwargs: self._stage_one
            )

    def _init_transformer(self) -> None:
        model_config = self.pipeline_config.model_configs["transformer"]
        path = model_config.pretrained_config._name_or_path
        with safe_open(path, framework="pt") as checkpoint:
            metadata = checkpoint.metadata() or {}
        quant_metadata = json.loads(metadata.get("_quantization_metadata", "{}"))
        formats = {entry.get("format") for entry in quant_metadata.get("layers", {}).values()}
        if formats:
            if formats != {"nvfp4"}:
                raise ValueError(f"Unsupported LTX25 V2V packed quantization: {formats}")
            self._packed_nvfp4 = True
            NVFP4LinearMethod.use_tunable_quantize = True
            model_config.quant_config = _nvfp4_quant_config(
                path, model_config.pretrained_config.num_layers
            )
            model_config.skip_create_weights_in_init = True
        super()._init_transformer()
        if self._packed_nvfp4:
            _separate_nvfp4_self_attention_qkv(self.transformer)

    def load_transformer_weights(self, checkpoint_dir: str) -> dict[str, torch.Tensor]:
        weights = super().load_transformer_weights(checkpoint_dir)
        if self._packed_nvfp4:
            _convert_ltx_nvfp4_layout(weights, self.device)
        return weights

    def post_load_weights(self) -> None:
        super().post_load_weights()
        if self._packed_nvfp4:
            _force_local_nvfp4_inputs(self.transformer)
        # PipelineLoader compiles blocks after this hook; bind against original names.
        self._set_stage(False)
        grouped = self.pipeline_config.extra_attrs.get("iclora_grouped_qkv", False)
        if not isinstance(grouped, bool):
            raise ValueError("iclora_grouped_qkv must be a bool.")
        if grouped:
            logger.warning(
                "Experimental grouped LoRA changes BF16 rounding and has not passed "
                "the V2V output-quality gate. Keep iclora_grouped_qkv=False for validated output."
            )
            count = install_grouped_self_attention_lora(self.transformer)
            if not count:
                raise ValueError("No compatible self-attention runtime LoRA groups found.")
            logger.info(f"LTX25 grouped runtime LoRA: {count} self-attention groups")

    def _load_native_components(
        self,
        config: dict,
        sft_paths: list[str],
        device: torch.device,
        dtype: torch.dtype,
        skip_components: list[str] | None = None,
    ) -> None:
        skip_components = list(skip_components or [])
        # LTX25Pipeline already loaded its own tokenizer/encoder before this hook.
        if set(skip_components) - {PipelineComponent.TOKENIZER, PipelineComponent.TEXT_ENCODER}:
            raise NotImplementedError("LTX25 V2V requires all native media components.")
        # The parent owns native Gemma4 connectors and patchification, not split VAEs.
        super()._load_native_components(
            config, sft_paths, device, dtype, [*skip_components, PipelineComponent.VAE]
        )
        attrs = self.pipeline_config.extra_attrs
        video_path, audio_path = attrs["video_vae_path"], attrs["audio_vae_path"]
        video_config, audio_config = _read_config(video_path), _read_config(audio_path)
        if video_config["vae"].get("_class_name") != "CausalVideoAutoencoder":
            raise ValueError("LTX25 V2V currently requires the convolutional video VAE.")
        self.video_encoder = LTX23VideoEncoderConfigurator.from_config(video_config)
        _load_split_component(video_path, self.video_encoder, "encoder.")
        self.video_decoder = LTX23VideoDecoderConfigurator.from_config(video_config)
        # Only these two statistics are present in the LTX-2.5 conv-VAE export.
        self.video_decoder.per_channel_statistics = self.video_encoder.per_channel_statistics
        _load_split_component(video_path, self.video_decoder, "decoder.")
        self.audio_decoder = AudioDecoderConfigurator.from_config(audio_config)
        _load_split_component(audio_path, self.audio_decoder, "audio_vae.decoder.")
        self.vocoder = LTX23VocoderConfigurator.from_config(audio_config)
        _load_split_component(audio_path, self.vocoder, "vocoder.")
        upsampler_path = attrs["spatial_upsampler_path"]
        self.spatial_upsampler = LatentUpsamplerConfigurator.from_config(
            _read_config(upsampler_path)
        )
        _load_split_component(upsampler_path, self.spatial_upsampler, "")
        for module in (
            self.video_encoder,
            self.video_decoder,
            self.audio_decoder,
            self.spatial_upsampler,
        ):
            module.to(device=device, dtype=dtype).eval()
        self.vocoder.to(device=device, dtype=torch.float32).eval()
        self.audio_patchifier = self.audio_decoder.patchifier
        self.audio_sampling_rate = self.audio_decoder.sample_rate
        self.audio_hop_length = self.audio_decoder.mel_hop_length
        self.audio_mel_bins = self.audio_decoder.mel_bins
        with safe_open(attrs["iclora_path"], framework="pt") as checkpoint:
            metadata = checkpoint.metadata() or {}
        self.reference_downscale_factor = int(metadata.get("reference_downscale_factor", 1))
        self.reference_temporal_scale_factor = int(
            metadata.get("reference_temporal_scale_factor", 1)
        )
        if min(self.reference_downscale_factor, self.reference_temporal_scale_factor) < 1:
            raise ValueError("IC-LoRA reference scales must be positive integers.")

    def _set_stage(self, stage_one: bool) -> None:
        self._stage_one = stage_one
        if not self._lora_ready:
            attrs = self.pipeline_config.extra_attrs
            application = attach_runtime_lora(
                self.transformer,
                RuntimeLoRAConfig(path=attrs["iclora_path"], scale=attrs.get("iclora_scale", 1.0)),
            )
            if not application.applied_modules:
                raise ValueError("IC-LoRA matched no transformer projections.")
            self._runtime_lora_applications.append(application)
            self._stage1_dynamic_modules = [
                module
                for module in self.transformer.modules()
                if isinstance(getattr(module, "quant_method", None), NVFP4LinearMethod)
                and hasattr(module, "_trtllm_runtime_lora_branch")
                and not getattr(module, "force_dynamic_quantization", False)
            ]
            self._lora_ready = True
        for module in self._stage1_dynamic_modules:
            module.force_dynamic_quantization = stage_one
        set_runtime_lora_enabled(self.transformer, stage_one)

    @property
    def default_generation_params(self) -> dict:
        return {
            "height": 768,
            "width": 1280,
            "num_frames": 65,
            "frame_rate": 24.0,
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "seed": 42,
            "max_sequence_length": 1024,
        }

    @property
    def extra_param_specs(self) -> dict[str, ExtraParamSchema]:
        return {
            "v2v_video_path": ExtraParamSchema(
                type="str", description="SDR reference/control video path."
            ),
            "v2v_strength": ExtraParamSchema(
                type="float",
                default=1.0,
                range=(0.0, 1.0),
                description="Reference latent conditioning strength in [0, 1].",
            ),
        }

    def warmup(self) -> None:
        logger.info("Skipping LTX25 V2V warmup; a reference-video request is required.")

    def _read_reference(self, path: str, height: int, width: int, frames: int) -> torch.Tensor:
        scale = self.reference_downscale_factor
        if height % (32 * scale) or width % (32 * scale):
            raise ValueError("Stage-1 dimensions must align with the scaled reference VAE grid.")
        height, width = height // scale, width // scale
        decoded = []
        reader = decode_video_by_frame(path)
        try:
            for frame in islice(reader, frames):
                frame = frame.to(device=self.device, dtype=torch.float32).permute(0, 3, 1, 2)
                ratio = max(height / frame.shape[-2], width / frame.shape[-1])
                new_h, new_w = (
                    math.ceil(frame.shape[-2] * ratio),
                    math.ceil(frame.shape[-1] * ratio),
                )
                frame = F.interpolate(
                    frame, size=(new_h, new_w), mode="bilinear", align_corners=False
                )
                top, left = (new_h - height) // 2, (new_w - width) // 2
                decoded.append(
                    (frame[:, :, top : top + height, left : left + width] / 127.5 - 1).to(
                        self.dtype
                    )
                )
        finally:
            reader.close()
        if len(decoded) != frames:
            raise ValueError(
                f"Reference video has {len(decoded)} frames; expected at least {frames}."
            )
        video = torch.stack(decoded, dim=2)
        if self.reference_temporal_scale_factor > 1:
            video = video[:, :, [0, *range(1, frames, self.reference_temporal_scale_factor)]]
        return self.video_encoder(video)

    def _audio_state(self, latent: torch.Tensor) -> ICLoraState:
        tokens = self.audio_patchifier.patchify(latent)
        positions = self.audio_patchifier.get_patch_grid_bounds(
            AudioLatentShape.from_torch_shape(latent.shape), device=latent.device
        )
        mask = torch.ones((*tokens.shape[:2], 1), device=latent.device, dtype=torch.float32)
        return ICLoraState(tokens, tokens.clone(), mask, positions)

    def _denoise(
        self,
        video: ICLoraState,
        audio: ICLoraState,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> tuple[ICLoraState, ICLoraState]:
        # Reference ICLoraPipeline consumes post-connector contexts without a mask.
        cache = self.transformer.prepare_text_cache(
            video_context=video_context,
            video_context_mask=None,
            video_positions=video.positions,
            audio_context=audio_context,
            audio_context_mask=None,
            audio_positions=audio.positions,
            dtype=self.dtype,
        )
        # Resolve scalar divisors once, avoiding a device sync at every Euler step.
        sigma_values = sigmas.detach().cpu().tolist()
        for index, sigma in self._profile_denoise_steps(sigmas[:-1]):

            def modality(state: ICLoraState, context: torch.Tensor) -> LTX23Modality:
                return LTX23Modality(
                    latent=state.latent,
                    timesteps=state.mask * sigma,
                    sigma=sigma.reshape(1),
                    cross_modality_sigma=sigma.reshape(1),
                    positions=state.positions,
                    context=context,
                    context_mask=None,
                    keyframes_mask=state.keyframes_mask,
                )

            video_velocity, audio_velocity = self.transformer(
                video=modality(video, video_context),
                audio=modality(audio, audio_context),
                text_cache=cache,
                step_index=index,
            )
            video = euler_step(
                video, video_velocity, sigma, sigmas[index + 1], sigma_value=sigma_values[index]
            )
            audio = euler_step(
                audio, audio_velocity, sigma, sigmas[index + 1], sigma_value=sigma_values[index]
            )
        return video, audio

    @torch.inference_mode()
    def infer(self, req: DiffusionRequest) -> PipelineOutput:
        params = req.params
        if getattr(params, "image", None) is not None:
            raise ValueError("LTX25 V2V does not support image conditioning; use v2v_video_path.")
        if getattr(params, "negative_prompt", None):
            raise ValueError("LTX25 V2V does not support negative_prompt without CFG.")
        if getattr(params, "num_images_per_prompt", 1) != 1:
            raise ValueError("LTX25 V2V requires num_images_per_prompt=1.")
        extra = params.extra_params or {}
        path = extra.get("v2v_video_path")
        if not isinstance(path, str) or not path:
            raise ValueError("v2v_video_path is required.")
        prompt = req.prompt
        if isinstance(prompt, list) and len(prompt) == 1:
            prompt = prompt[0]
        if not isinstance(prompt, str):
            raise ValueError("LTX25 V2V requires a single text prompt.")
        height, width, frames, fps = (
            params.height,
            params.width,
            params.num_frames,
            params.frame_rate,
        )
        if height <= 0 or width <= 0 or height % 64 or width % 64:
            raise ValueError("Two-stage V2V dimensions must be positive multiples of 64.")
        if frames <= 0 or (frames - 1) % 8 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("V2V requires 8n+1 frames and a positive finite frame rate.")
        if params.num_inference_steps != 8 or params.guidance_scale != 1.0:
            raise ValueError("Distilled IC-LoRA uses a fixed 8+3 step schedule without CFG.")
        strength = extra.get("v2v_strength", 1.0)
        if (
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not 0 <= strength <= 1
        ):
            raise ValueError("v2v_strength must be in [0, 1].")
        scale = self.reference_downscale_factor
        if height % (64 * scale) or width % (64 * scale):
            raise ValueError("V2V dimensions must align with the scaled reference VAE grid.")
        # Bound graph memory to one requested geometry. FPS/content changes use
        # replay copies, while new dimensions/context capacity require recapture.
        geometry = (height, width, frames, params.max_sequence_length)
        if self._graph_geometry != geometry:
            for runner in self._cuda_graph_runners.values():
                runner.clear()
            self._graph_geometry = geometry
        timer = CudaPhaseTimer()
        timer.mark_pre_start()
        generator = torch.Generator(device=self.device).manual_seed(params.seed)
        embeddings, mask = self._encode_prompt(
            prompt, max_sequence_length=params.max_sequence_length
        )
        video_context, audio_context, _ = self._process_connectors(embeddings, mask)
        reference = self._read_reference(path, height // 2, width // 2, frames)
        pixel_shape = VideoPixelShape(1, frames, height // 2, width // 2, fps)
        vshape = VideoLatentShape.from_pixel_shape(pixel_shape)
        ashape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        self.transformer.configure_audio_ulysses(ashape.frames)
        video = video_state(
            torch.zeros(vshape.to_torch_shape(), device=self.device, dtype=self.dtype), fps
        )
        target_tokens = video.latent.shape[1]
        video = append_reference(
            video,
            reference,
            fps,
            self.reference_downscale_factor,
            self.reference_temporal_scale_factor,
            strength,
        )
        video = add_noise(video, generator, 1.0)
        audio = add_noise(
            self._audio_state(
                torch.zeros(ashape.to_torch_shape(), device=self.device, dtype=self.dtype)
            ),
            generator,
            1.0,
        )
        sigmas = torch.tensor(STAGE_1_SIGMAS, device=self.device, dtype=torch.float32)
        self._set_stage(True)
        timer.mark_denoise_start()
        try:
            video, audio = self._denoise(video, audio, video_context, audio_context, sigmas)
        finally:
            self._set_stage(False)
        latent = self.video_patchifier.unpatchify(video.latent[:, :target_tokens], vshape)
        latent = upsample_video(
            latent[:1], self.video_encoder.per_channel_statistics, self.spatial_upsampler
        )
        video = add_noise(video_state(latent, fps), generator, STAGE_2_SIGMAS[0])
        audio_latent = self.audio_patchifier.unpatchify(audio.latent, ashape)
        audio = add_noise(self._audio_state(audio_latent), generator, STAGE_2_SIGMAS[0])
        sigmas = torch.tensor(STAGE_2_SIGMAS, device=self.device, dtype=torch.float32)
        video, audio = self._denoise(video, audio, video_context, audio_context, sigmas)
        timer.mark_post_start()
        output_shape = VideoLatentShape.from_pixel_shape(
            VideoPixelShape(1, frames, height, width, fps)
        )
        latent = self.video_patchifier.unpatchify(video.latent, output_shape)
        # Match ICLoraPipeline's conv-VAE AUTO_TILING, without changing LTX23 defaults.
        tiling = TilingConfig(
            spatial_config=SpatialTilingConfig(768, 64),
            temporal_config=TemporalTilingConfig(80, 24),
        )
        chunks = list(self.video_decoder.tiled_decode(latent, tiling, generator=generator))
        output_video = postprocess_video_tensor(torch.cat(chunks, dim=2))
        output_audio = decode_audio(
            self.audio_patchifier.unpatchify(audio.latent, ashape), self.audio_decoder, self.vocoder
        )
        sample_rate = int(self.vocoder.output_sampling_rate)
        output_audio = _pad_audio_for_video(output_audio, frames, fps, sample_rate)
        timer.mark_end()
        return timer.fill(
            PipelineOutput(
                video=output_video,
                audio=output_audio,
                frame_rate=float(fps),
                audio_sample_rate=sample_rate,
            )
        )
