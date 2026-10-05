# Visual Generation Examples

See [the VisualGen doc](https://nvidia.github.io/TensorRT-LLM/models/visual-generation.html)
for feature details.

## Layout

| Path | Purpose |
|---|---|
| [`quickstart_example.py`](quickstart_example.py) | Minimal VisualGen API example |
| [`models/`](models/) | Per-model example scripts |
| [`configs/`](configs/) | Shared `VisualGenArgs` YAMLs (used by `--visual_gen_args` and `trtllm-serve`) |
| [`serve/`](serve/) | `trtllm-serve` usage, benchmarking, and clients |

## Usage

```bash
# Defaults
python quickstart_example.py
python models/wan_t2v.py
python models/ltx2.py
python models/ltx2.py --model_type ltx23
python models/flux1.py
python models/flux2.py
python models/cosmos3_ti2v.py --prompt "A robot arm picks fruit in a grocery store"
python models/qwen_image.py
python models/qwen_image_layered.py --image /path/to/image.png
python models/qwen_image_edit.py --image /path/to/source.png --prompt "Make the image look like a watercolor painting"
python models/glm_image.py
python models/hunyuan_t2v.py

# With engine config (quant, parallelism, etc.)
python models/wan_t2v.py --visual_gen_args configs/wan2.2-t2v-fp4-1gpu.yaml
python models/wan_i2v.py --visual_gen_args configs/wan2.2-i2v-fp4-1gpu.yaml --image /path/to/image.png
python models/ltx2.py --visual_gen_args configs/ltx2-1gpu.yaml
python models/ltx2.py --model_type ltx23 --visual_gen_args configs/ltx23-t2v-bf16-1gpu.yaml
python models/ltx2.py --model_type ltx23 --visual_gen_args configs/ltx23-t2v-fp8-1gpu.yaml
python models/ltx23_retake.py --model /path/to/ltx-2.3 --source /path/to/retake_input.mp4 --text_encoder_path /path/to/gemma-3-12b-it
python models/ltx23_retake.py --visual_gen_args configs/ltx23-retake-fp8-1gpu.yaml --model /path/to/ltx-2.3 --source /path/to/retake_input.mp4 --text_encoder_path /path/to/gemma-3-12b-it
python models/ltx23_retake.py --visual_gen_args configs/ltx23-retake-fp4-1gpu.yaml --model /path/to/ltx-2.3 --source /path/to/retake_input.mp4 --text_encoder_path /path/to/gemma-3-12b-it
python models/flux1.py --visual_gen_args configs/flux1-dev-fp4-1gpu.yaml
python models/flux2.py --visual_gen_args configs/flux2-dev-fp4-1gpu.yaml
python models/cosmos3_ti2v.py --visual_gen_args configs/cosmos3-nano-1gpu.yaml --prompt "A robot arm picks fruit in a grocery store"
python models/qwen_image.py --visual_gen_args configs/qwen-image-fp8-1gpu.yaml
python models/qwen_image_layered.py --visual_gen_args configs/qwen-image-layered-1gpu.yaml --image /path/to/image.png
python models/qwen_image_edit.py --visual_gen_args configs/qwen-image-edit-2511-fp4-1gpu.yaml --image /path/to/source.png --prompt "Make the image look like a watercolor painting"
python models/hunyuan_t2v.py --visual_gen_args configs/hunyuan-t2v-fp8-1gpu.yaml
```

Install deps from the repo root: `pip install -r requirements-dev.txt`.

Output: `.png` for image models; `.mp4` for video models when FFmpeg is installed (otherwise `.avi`).

LTX-2.5 IC-LoRA V2V uses the standalone [`models/ltx25_v2v.py`](models/ltx25_v2v.py)
example and [`configs/ltx25-v2v-1gpu.yaml`](configs/ltx25-v2v-1gpu.yaml). It supports
split BF16 or packed NVFP4 transformer checkpoints with BF16 attention, one GPU,
and a fixed 8+3-step schedule. See the [native LTX-2.5 guide](../../tensorrt_llm/_torch/visual_gen/models/ltx25/README.md)
for the complete command, component paths, and validation limits. Gemma4 requires
a compatible isolated environment (validated with Transformers 5.14.1); the
repository's default Transformers pin does not supply this dependency.
