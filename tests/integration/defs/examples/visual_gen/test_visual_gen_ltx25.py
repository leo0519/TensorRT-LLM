# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Native BF16/NVFP4 V2V smoke tests and a shared BF16-reference quality gate.

Create the golden with the external BF16 ICLoraPipeline using the same adapter,
prompt, seed, frame rate and geometry. Missing golden assets do not skip smoke.
"""

import os
from pathlib import Path

import pytest
import torch
from defs.common import venv_check_call
from defs.examples.visual_gen.visual_gen_test_utils import (
    _assert_lpips_below_threshold,
    _lpips_model_path,
    _run_reusable_video_lpips_eval,
    _skip_if_missing,
)


@pytest.fixture(params=["bf16", "nvfp4"])
def ltx25_v2v_output(request, llm_root, llm_venv, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    precision = request.param

    root = Path(os.environ.get("LTX25_MODEL_ROOT", _lpips_model_path("LTX-2.5")))
    source = os.environ.get("LTX25_V2V_SOURCE", str(root / "control-canny-65f.mp4"))
    paths = {
        "model": root / f"diffusion_models/ltx-2.5-22b-distilled-transformer-{precision}.safetensors",
        "text_encoder_path": root / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "video_vae_path": root / "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "audio_vae_path": root / "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "spatial_upsampler_path": root
        / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "iclora_path": root / "loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    }
    for name, path in {**paths, "source": source}.items():
        _skip_if_missing(str(path), f"LTX25 V2V {name}")
    request.getfixturevalue("_visual_gen_deps")
    output = tmp_path / f"ltx25_v2v_{precision}.mp4"
    command = [str(Path(llm_root) / "examples/visual_gen/models/ltx25_v2v.py")]
    for name, path in paths.items():
        command += [f"--{name}", str(path)]
    command += [
        "--source",
        source,
        "--prompt",
        "A cinematic video following the structure and motion of the reference.",
        "--width",
        "256",
        "--height",
        "256",
        "--num_frames",
        "9",
        "--frame_rate",
        "24",
        "--seed",
        "42",
        "--output_path",
        str(output),
    ]
    venv_check_call(llm_venv, command)
    return output


def test_ltx25_native_iclora_smoke(ltx25_v2v_output):
    import av

    output = ltx25_v2v_output
    with av.open(str(output)) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == 9
        assert all((frame.width, frame.height) == (256, 256) for frame in frames)
        assert frames[4].to_ndarray(format="rgb24").std() > 1
    with av.open(str(output)) as container:
        assert len(container.streams.audio) == 1
        assert container.streams.audio[0].rate == 48000
        assert len(list(container.decode(audio=0))) > 0


@pytest.fixture
def ltx25_bf16_golden():
    root = Path(os.environ.get("LTX25_MODEL_ROOT", _lpips_model_path("LTX-2.5")))
    golden = os.environ.get(
        "LTX25_V2V_BF16_GOLDEN", str(root / "iclora-bf16-256x256-9f-golden.mp4")
    )
    _skip_if_missing(golden, "LTX25 V2V BF16 reference golden")
    return golden


def test_ltx25_native_iclora_lpips(
    ltx25_bf16_golden, _visual_gen_lpips_scorer, ltx25_v2v_output
):
    score = _run_reusable_video_lpips_eval(
        "ltx25_v2v", ltx25_bf16_golden, str(ltx25_v2v_output), _visual_gen_lpips_scorer
    )
    _assert_lpips_below_threshold(score, 0.05)
