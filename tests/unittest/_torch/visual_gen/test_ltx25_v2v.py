# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""IC-LoRA numerical contracts and isolation from the LTX23 retake workflow."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23 import LTX23Pipeline
from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23_retake import LTX23RetakePipeline
from tensorrt_llm._torch.visual_gen.models.ltx25.iclora import (
    STAGE_1_SIGMAS,
    STAGE_2_SIGMAS,
    add_noise,
    append_reference,
    euler_step,
    video_state,
)
from tensorrt_llm._torch.visual_gen.models.ltx25.pipeline_ltx25 import LTX25Pipeline
from tensorrt_llm._torch.visual_gen.models.ltx25.pipeline_ltx25_v2v import (
    LTX25V2VPipeline,
    _pad_audio_for_video,
)


@pytest.mark.parametrize(
    "frames,fps,samples", [(9, 24.0, 18000), (65, 24.0, 130000), (9, 29.97, 14415)]
)
def test_audio_tail_padding_preserves_video_duration(frames, fps, samples) -> None:
    audio = torch.ones(1, 2, 100)
    result = _pad_audio_for_video(audio, frames, fps, 48000)
    assert result.shape == (1, 2, samples)
    assert torch.equal(result[..., :100], audio)
    assert result[..., 100:].count_nonzero() == 0
    assert _pad_audio_for_video(result, frames, fps, 48000) is result


def test_v2v_rejects_conflicting_generic_lora() -> None:
    with pytest.raises(ValueError, match="owns stage-specific LoRA"):
        LTX25V2VPipeline(SimpleNamespace(runtime_lora=object()))


def test_v2v_requires_native_media_components() -> None:
    pipeline = LTX25V2VPipeline.__new__(LTX25V2VPipeline)
    with pytest.raises(NotImplementedError, match="all native media components"):
        pipeline._load_native_components({}, [], torch.device("cpu"), torch.bfloat16, ["vae"])


def test_v2v_workflow_is_separate_from_retake() -> None:
    assert (
        LTX25Pipeline.resolve_variant(SimpleNamespace(extra_attrs={"workflow": "v2v"}))
        is LTX25V2VPipeline
    )
    assert (
        LTX23Pipeline.resolve_variant(SimpleNamespace(extra_attrs={"workflow": "retake"}))
        is LTX23RetakePipeline
    )
    with pytest.raises(ValueError, match="Unsupported"):
        LTX23Pipeline.resolve_variant(SimpleNamespace(extra_attrs={"workflow": "v2v"}))
    with pytest.raises(ValueError, match="Unsupported"):
        LTX25Pipeline.resolve_variant(SimpleNamespace(extra_attrs={"workflow": "retake"}))


@pytest.mark.parametrize("spatial_scale", [1, 2])
@pytest.mark.parametrize("temporal_scale", [1, 2, 4])
def test_reference_coordinates_and_keyframe_marker(spatial_scale, temporal_scale) -> None:
    state = video_state(torch.zeros(1, 8, 3, 4, 4, dtype=torch.bfloat16), 24)
    reference = torch.ones(1, 8, 2, 2, 2, dtype=torch.bfloat16)
    combined = append_reference(state, reference, 24, spatial_scale, temporal_scale)
    assert combined.latent.shape == (1, 56, 8)
    assert combined.mask[:, 48:].count_nonzero() == 0
    assert combined.keyframes_mask[:, :16].all()
    assert combined.keyframes_mask[:, 16:].count_nonzero() == 0
    torch.testing.assert_close(combined.positions[:, :, :48], state.positions)
    expected = video_state(reference, 24).positions
    expected[:, 0] = (expected[:, 0] * temporal_scale - (temporal_scale - 1) / 24).clamp(min=0)
    expected[:, 1:] *= spatial_scale
    torch.testing.assert_close(combined.positions[:, :, 48:], expected)


def test_reference_tokens_participate_in_random_draws() -> None:
    state = video_state(torch.zeros(1, 8, 2, 4, 4, dtype=torch.bfloat16), 24)
    state = append_reference(state, torch.ones(1, 8, 2, 2, 2, dtype=torch.bfloat16), 24)
    generator = torch.Generator().manual_seed(42)
    result = add_noise(state, generator, 1.0)
    reference_generator = torch.Generator().manual_seed(42)
    noise = torch.randn(state.latent.shape, generator=reference_generator, dtype=torch.bfloat16)
    torch.testing.assert_close(result.latent[:, :32], noise[:, :32], rtol=0, atol=0)
    assert result.latent[:, 32:].eq(1).all()
    assert torch.equal(generator.get_state(), reference_generator.get_state())


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("schedule", [STAGE_1_SIGMAS, STAGE_2_SIGMAS])
def test_euler_matches_reference_rounding(strength, schedule) -> None:
    generator = torch.Generator().manual_seed(42)
    state = video_state(torch.zeros(1, 8, 2, 2, 2, dtype=torch.bfloat16), 24)
    reference = torch.randn(1, 8, 2, 1, 1, dtype=torch.bfloat16, generator=generator)
    state = append_reference(state, reference, 24, strength=strength)
    state = add_noise(state, generator, schedule[0])
    sigmas = torch.tensor(schedule)
    for index, sigma in enumerate(sigmas[:-1]):
        velocity = torch.randn(state.latent.shape, dtype=state.latent.dtype, generator=generator)
        # LTX-2 X0Model, post_process_latent and EulerDiffusionStep cast separately.
        x0 = (state.latent.float() - velocity.float() * (state.mask * sigma)).to(state.latent.dtype)
        x0 = (x0 * state.mask + state.clean.float() * (1 - state.mask)).to(x0.dtype)
        derivative = ((state.latent.float() - x0.float()) / sigma.item()).to(state.latent.dtype)
        expected = (state.latent.float() + derivative.float() * (sigmas[index + 1] - sigma)).to(
            state.latent.dtype
        )
        state = euler_step(state, velocity, sigma, sigmas[index + 1])
        torch.testing.assert_close(state.latent, expected, rtol=0, atol=0)


@pytest.mark.parametrize("strength", [-1, 2, float("nan")])
def test_invalid_reference_strength(strength) -> None:
    state = video_state(torch.zeros(1, 8, 1, 2, 2), 24)
    with pytest.raises(ValueError, match="strength"):
        append_reference(state, torch.zeros(1, 8, 1, 1, 1), 24, strength=strength)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA rounding regression")
@pytest.mark.parametrize("sigma_value,next_value", [(0.975, 0.909375), (0.725, 0.421875)])
def test_cuda_euler_uses_reference_scalar_division(sigma_value, next_value) -> None:
    generator = torch.Generator(device="cuda").manual_seed(42)
    latent = torch.randn(
        1, 128, 2, 64, 64, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    state = video_state(latent, 24)
    velocity = torch.randn(
        state.latent.shape, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    sigma, next_sigma = torch.tensor([sigma_value, next_value], device="cuda")
    x0 = (state.latent.float() - velocity.float() * (state.mask * sigma)).to(state.latent.dtype)
    x0 = (x0 * state.mask + state.clean.float() * (1 - state.mask)).to(x0.dtype)
    derivative = ((state.latent.float() - x0.float()) / sigma.item()).to(state.latent.dtype)
    expected = (state.latent.float() + derivative.float() * (next_sigma - sigma)).to(
        state.latent.dtype
    )
    result = euler_step(state, velocity, sigma, next_sigma, sigma_value=sigma.item())
    torch.testing.assert_close(result.latent, expected, rtol=0, atol=0)


def test_connector_priority_is_local_to_ltx25(monkeypatch) -> None:
    from torch.nn.attention import SDPBackend

    before = torch._C._get_sdp_priority_order()

    def process(self, embeddings, mask):
        assert torch._C._get_sdp_priority_order()[0] == SDPBackend.CUDNN_ATTENTION.value
        return embeddings, embeddings, mask

    monkeypatch.setattr(LTX23Pipeline, "_process_connectors", process)
    pipeline = LTX25Pipeline.__new__(LTX25Pipeline)
    embeddings, mask = torch.zeros(1, 1, 1), torch.ones(1, 1)
    pipeline._process_connectors(embeddings, mask)
    assert torch._C._get_sdp_priority_order() == before


def test_patch_projection_preserves_refinement_latent_layout() -> None:
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.modality import Modality
    from tensorrt_llm._torch.visual_gen.models.ltx25.transformer_ltx25 import _LTX25ArgsPreprocessor

    latent = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8).transpose(1, 2)
    assert not latent.is_contiguous()
    seen = []
    projection = torch.nn.Identity()
    projection.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    preprocessor = _LTX25ArgsPreprocessor.__new__(_LTX25ArgsPreprocessor)
    preprocessor.patchify_proj = projection
    preprocessor.keyframes_embedding_provider = None
    preprocessor._prepare_timestep = lambda ts, batch, dtype: (ts, ts)
    context = torch.zeros_like(latent)
    modality = Modality(
        latent=latent, timesteps=torch.ones(1), positions=torch.zeros(1, 3, 8, 2), context=context
    )
    result = preprocessor.prepare(modality, context, None, (torch.ones(1), torch.zeros(1)))
    assert seen == [latent]
    assert seen[0] is latent
    assert result.x.stride() == latent.stride()


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("image", "image.png", "image conditioning"),
        ("negative_prompt", "blur", "negative_prompt"),
        ("num_images_per_prompt", 2, "num_images_per_prompt"),
    ],
)
def test_v2v_rejects_ignored_conditioning_before_gpu_work(field, value, message) -> None:
    pipeline = LTX25V2VPipeline.__new__(LTX25V2VPipeline)
    params = SimpleNamespace(**{field: value})
    with pytest.raises(ValueError, match=message):
        pipeline.infer(SimpleNamespace(params=params, prompt="test"))


@pytest.mark.parametrize("strength", [-1, 2, float("nan"), float("inf"), None, "0.5", True])
def test_v2v_rejects_invalid_request_strength_before_gpu_work(strength) -> None:
    pipeline = LTX25V2VPipeline.__new__(LTX25V2VPipeline)
    params = SimpleNamespace(**pipeline.default_generation_params)
    params.extra_params = {"v2v_video_path": "control.mp4", "v2v_strength": strength}
    with pytest.raises(ValueError, match="v2v_strength"):
        pipeline.infer(SimpleNamespace(params=params, prompt="test"))


def test_v2v_validates_reference_grid_before_text_encoding() -> None:
    pipeline = LTX25V2VPipeline.__new__(LTX25V2VPipeline)
    pipeline.reference_downscale_factor = 2
    params = SimpleNamespace(**pipeline.default_generation_params)
    params.height = params.width = 192  # Valid output grid, invalid half-resolution reference.
    params.extra_params = {"v2v_video_path": "control.mp4"}
    with pytest.raises(ValueError, match="scaled reference VAE grid"):
        pipeline.infer(SimpleNamespace(params=params, prompt="test"))


@pytest.mark.parametrize("strength", [-1, 2, float("nan")])
def test_v2v_strength_range_is_validated_at_api_entry(strength) -> None:
    from tensorrt_llm.visual_gen.params import VisualGenParams, validate_visual_gen_params

    pipeline = LTX25V2VPipeline.__new__(LTX25V2VPipeline)
    with pytest.raises(ValueError, match="v2v_strength"):
        validate_visual_gen_params(
            VisualGenParams(extra_params={"v2v_strength": strength}),
            declared_defaults=pipeline.default_generation_params,
            extra_param_specs=pipeline.extra_param_specs,
        )
