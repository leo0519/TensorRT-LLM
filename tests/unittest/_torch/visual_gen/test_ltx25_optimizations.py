# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""P1 scoped encoding, P2 V2V graph contracts, P3 grouped LoRA regressions."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunnerConfig
from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.modality import LTX23Modality
from tensorrt_llm._torch.visual_gen.models.ltx25.cuda_graph import LTX25V2VCUDAGraphRunner
from tensorrt_llm._torch.visual_gen.models.ltx25.grouped_lora import GroupedSelfAttentionLoRA
from tensorrt_llm._torch.visual_gen.runtime_lora import RuntimeLoRABranch, _RuntimeLoRAAdapter
from tensorrt_llm.media.encoding import _VIDEO_ENCODING_ARGS, video_encoding_options


def test_video_encoding_scope_restores_and_is_thread_local() -> None:
    assert _VIDEO_ENCODING_ARGS.get() is None
    with video_encoding_options(preset="veryfast"):
        original = _VIDEO_ENCODING_ARGS.get()
        assert "veryfast" in original
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(_VIDEO_ENCODING_ARGS.get).result() is None
        with pytest.raises(RuntimeError), video_encoding_options(codec="h264_nvenc"):
            assert "-cq" in _VIDEO_ENCODING_ARGS.get()
            assert "-crf" not in _VIDEO_ENCODING_ARGS.get()
            raise RuntimeError("restore")
        assert _VIDEO_ENCODING_ARGS.get() == original
    assert _VIDEO_ENCODING_ARGS.get() is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"codec": "bad"},
        {"preset": "p4"},
        {"codec": "h264_nvenc", "preset": "medium"},
        {"quality": -1},
        {"quality": 52},
        {"quality": True},
    ],
)
def test_video_encoding_options_reject_invalid_settings(kwargs) -> None:
    with pytest.raises(ValueError), video_encoding_options(**kwargs):
        pass


def test_video_encoding_options_reject_avi_fallback(tmp_path, monkeypatch) -> None:
    from tensorrt_llm.media import encoding

    monkeypatch.setattr(encoding, "_get_video_encoder", lambda: encoding._PurePythonEncoder())
    video = torch.zeros(2, 16, 16, 3, dtype=torch.uint8)
    with video_encoding_options(codec="h264_nvenc"):
        with pytest.raises(RuntimeError, match="require.*FFmpeg"):
            encoding.save_video(video, tmp_path / "out.avi", frame_rate=24, format="avi")
        with pytest.raises(RuntimeError, match="require.*FFmpeg"):
            encoding._PurePythonEncoder().encode_video(video, str(tmp_path / "out.avi"), 24)


def _modality(device: str = "cpu") -> LTX23Modality:
    latent = torch.arange(24, device=device, dtype=torch.float32).view(1, 4, 6).transpose(1, 2)
    return LTX23Modality(
        latent=latent,
        timesteps=torch.ones(1, 6, 1, device=device),
        sigma=torch.ones(1, device=device),
        cross_modality_sigma=torch.full((1,), 0.5, device=device),
        positions=torch.zeros(1, 3, 6, 2, device=device),
        context=torch.ones(1, 8, 4, device=device),
        keyframes_mask=torch.ones(1, 6, 1, device=device),
    )


def test_v2v_graph_preserves_layout_and_copies_cross_sigma() -> None:
    source = _modality()
    cloned = LTX25V2VCUDAGraphRunner._clone_value(source)
    assert cloned.latent.stride() == source.latent.stride()
    assert cloned.cross_modality_sigma.data_ptr() != source.cross_modality_sigma.data_ptr()
    changed = replace(source, cross_modality_sigma=source.cross_modality_sigma + 2)
    LTX25V2VCUDAGraphRunner._copy_value(cloned, changed)
    torch.testing.assert_close(cloned.cross_modality_sigma, changed.cross_modality_sigma)
    runner = LTX25V2VCUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    state = [False]
    runner.register_extra_key_fn("stage", lambda **kwargs: state[0])
    key = runner.get_graph_key(video=source)
    state[0] = True
    assert runner.get_graph_key(video=source) != key
    assert runner.get_graph_key(
        video=replace(source, latent=source.latent.contiguous())
    ) != runner.get_graph_key(video=source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_v2v_graph_replays_new_conditioning_and_separates_stages() -> None:
    runner = LTX25V2VCUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    stage = [False]
    runner.register_extra_key_fn("stage", lambda **kwargs: stage[0])

    def forward(video):
        return (video.latent + video.cross_modality_sigma + video.context.mean()) * (
            2 if stage[0] else 1
        )

    wrapped = runner.wrap(forward)
    first = _modality("cuda")
    with torch.inference_mode():
        torch.testing.assert_close(wrapped(video=first), forward(first))
        changed = replace(
            first, context=first.context + 5, cross_modality_sigma=first.cross_modality_sigma + 3
        )
        torch.testing.assert_close(wrapped(video=changed), forward(changed))
        stage[0] = True
        torch.testing.assert_close(wrapped(video=changed), forward(changed))
        stage[0] = False
        torch.testing.assert_close(wrapped(video=first), forward(first))
    assert len(runner.graphs) == 2
    assert len(runner._graph_output_refs) == 2
    runner.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("zero_base", [False, True])
def test_grouped_lora_matches_resident_branch_and_graph(zero_base: bool) -> None:
    generator = torch.Generator(device="cuda").manual_seed(17)

    def tensor(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.02

    branches = tuple(
        RuntimeLoRABranch([_RuntimeLoRAAdapter(tensor(16, 128), tensor(128, 16), 1.0)], 128)
        for _ in range(3)
    )
    for branch in branches:
        branch.enabled = True

    class QuantMethod:
        def apply(self, module, x, bias):
            return torch.nn.functional.linear(x, module.weight, bias)

    projections = [
        SimpleNamespace(quant_method=QuantMethod(), weight=tensor(128, 128), bias=None)
        for _ in range(3)
    ]
    if zero_base:
        for projection in projections:
            projection.weight.zero_()
    attention = SimpleNamespace(to_q=projections[0], to_k=projections[1], to_v=projections[2])
    group = GroupedSelfAttentionLoRA(branches)
    x = tensor(1, 256, 128)
    with torch.inference_mode():
        expected = tuple(
            b.add_to(p.quant_method.apply(p, x, None), x) for p, b in zip(projections, branches)
        )
        assert group.can_apply(x)
        actual = group.project(attention, x)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, atol=1e-6 if zero_base else 3e-4, rtol=0.03)
            if zero_base:
                assert a.abs().max() > 1e-5
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = group.project(attention, x)
        x.add_(0.01)
        graph.replay()
        reference = group.project(attention, x)
        for a, e in zip(captured, reference):
            torch.testing.assert_close(a, e, atol=0, rtol=0)
        branches[0].enabled = False
        assert not group.can_apply(x)
