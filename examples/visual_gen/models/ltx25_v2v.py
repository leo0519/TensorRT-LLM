#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Run native single-GPU LTX-2.5 IC-LoRA V2V through the public VisualGen API."""

import argparse
from pathlib import Path

from tensorrt_llm import VisualGen, VisualGenArgs
from tensorrt_llm.media.encoding import video_encoding_options


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Transformer safetensors checkpoint.")
    parser.add_argument("--source", required=True, help="SDR reference/control video.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--text_encoder_path", required=True)
    parser.add_argument("--video_vae_path", required=True)
    parser.add_argument("--audio_vae_path", required=True)
    parser.add_argument("--spatial_upsampler_path", required=True)
    parser.add_argument("--iclora_path", required=True)
    parser.add_argument("--output_path", default="ltx25_v2v.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--num_frames", type=int, default=65)
    parser.add_argument("--frame_rate", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda_graph", action="store_true")
    parser.add_argument(
        "--grouped_lora",
        action="store_true",
        help="Experimental grouped LoRA; currently fails the V2V quality gate. Not recommended.",
    )
    parser.add_argument("--video_codec", choices=("libx264", "h264_nvenc"), default="libx264")
    parser.add_argument("--video_preset", default=None)
    parser.add_argument("--video_quality", type=int, default=23)
    args = parser.parse_args()
    config_path = Path(__file__).resolve().parents[1] / "configs/ltx25-v2v-1gpu.yaml"
    visual_gen_args = VisualGenArgs.from_yaml(str(config_path))
    pipeline_config = dict(visual_gen_args.pipeline_config)
    pipeline_config["iclora_grouped_qkv"] = args.grouped_lora
    for name in (
        "text_encoder_path",
        "video_vae_path",
        "audio_vae_path",
        "spatial_upsampler_path",
        "iclora_path",
    ):
        pipeline_config[name] = getattr(args, name)
    visual_gen_args = visual_gen_args.model_copy(update={"pipeline_config": pipeline_config})
    visual_gen_args.cuda_graph_config.enable = args.cuda_graph
    with VisualGen(model=args.model, args=visual_gen_args) as visual_gen:
        params = visual_gen.default_params
        params.width, params.height = args.width, args.height
        params.num_frames, params.frame_rate = args.num_frames, args.frame_rate
        params.seed = args.seed
        params.extra_params["v2v_video_path"] = args.source
        output = visual_gen.generate(inputs=args.prompt, params=params)
        with video_encoding_options(
            codec=args.video_codec, preset=args.video_preset, quality=args.video_quality
        ):
            output.save(args.output_path)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
