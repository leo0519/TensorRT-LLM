# Native LTX-2.5 implementation

`LTX25Pipeline` follows the LTX-2.3 native pipeline structure. It reuses
generation, scheduling, VAE loading, and connector processing, and overrides
the transformer and text-conditioning loaders. LTX-2.5 owns its transformer
builder, block construction, and keyframe-aware CUDA Graph runner. LTX-2.3
source and tests remain unchanged; inherited functionality is reused read-only.
Generation and an independent `v2v` workflow are registered; LTX25 retake is
not enabled by this implementation.

## Native IC-LoRA V2V

`pipeline_ltx25_v2v.py` follows the separate workflow/registry pattern of
LTX23 retake, but implements LTX-2 `ICLoraPipeline` semantics rather than
masked temporal retake. Stage 1 appends reference tokens and enables resident
runtime LoRA; stage 2 removes reference tokens, disables LoRA, spatially
upsamples the latent and refines it. All weights stay on the GPU. Neither
`ltx_core` nor `ltx_pipelines` is imported by the runtime implementation.

The standalone example is `examples/visual_gen/models/ltx25_v2v.py`, with
`examples/visual_gen/configs/ltx25-v2v-1gpu.yaml`. It calls the existing public
VisualGen API directly. Split transformer, Gemma4, convolutional video VAE,
audio VAE/vocoder, upsampler and IC-LoRA paths are explicit. Model-specific
options remain in registry defaults and `extra_param_specs`; no public API
class is changed.

For the supported split-checkpoint layout, run in a compatible TRTLLM environment:

```bash
LTX25_ROOT=/path/to/ltx-2.5
python examples/visual_gen/models/ltx25_v2v.py \
  --model "$LTX25_ROOT/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors" \
  --text_encoder_path "$LTX25_ROOT/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --video_vae_path "$LTX25_ROOT/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
  --audio_vae_path "$LTX25_ROOT/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --spatial_upsampler_path "$LTX25_ROOT/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
  --iclora_path /path/to/union-control.safetensors \
  --source /path/to/control.mp4 --prompt "A cinematic video." \
  --width 1280 --height 768 --num_frames 65 --output_path output.mp4
```

Use the adapter appropriate for the checkpoint; the development parity run
explicitly used the LTX-2.3 Union-Control ref0.5 adapter with LTX-2.5 weights.
Output audio retains the generated waveform and pads only trailing silence
to the video duration, preventing the public muxer's `-shortest` option from
discarding final video frames.

Initial scope is one SDR reference/control video, one prompt and one GPU,
with distilled 8+3 steps and audio generation. Nonempty negative prompts,
image conditioning and `num_images_per_prompt != 1` are rejected explicitly.
Output dimensions must be divisible by `64 * reference_downscale_factor`.
Reference frames are consumed in source order (no FPS resampling), matching
the external IC-LoRA workflow. Cache acceleration,
HDR, additional image/keyframe conditioning and spatial attention masks are
not enabled. This is native orchestration, not complete feature parity with
every optional argument of the external ICLoraPipeline. Quality/performance
validation is a separate gate; do not treat registration as E2E certification.

### Opt-in V2V optimizations

The standalone example accepts these independent options (all default off):

- `--video_preset veryfast`: CPU x264 at the existing CRF23, with a different
  speed/compression tradeoff. `--video_codec h264_nvenc --video_preset p4`
  instead uses GPU encoding and CQ23, which is not equivalent to CRF23.
  `--video_quality` controls CRF/CQ. Explicit encoder requests fail if unavailable.
- `--cuda_graph`: separate captures for LoRA-enabled stage 1 and disabled
  stage 2, preserving latent strides and refreshing conditioning buffers.
  Geometry changes release old captures; first-use capture is warmup overhead.
- `--grouped_lora`: sets model-specific `iclora_grouped_qkv` and packs the
  three self-attention LoRA A projections into one GEMM. Base NVFP4 weights
  stay unchanged. This uses existing CUDA operators and adds packed-A storage;
  changed BF16 rounding requires output-quality validation. **Experimental:
  the initial full-model small-geometry check failed the quality gate.**
  Keep this disabled for validated output; passing isolated GEMM tests does
  not establish end-to-end numerical parity.

Library callers can wrap `output.save(...)` in
`tensorrt_llm.media.encoding.video_encoding_options(...)`. Settings are local
to the current thread/task and restored on exit; unrelated saves keep their
existing behavior. No LTX23 pipeline or public VisualGen argument class changes
are needed. Validate individual and combined modes before deployment; fewer
GEMMs or launches alone do not establish an E2E speedup.

## Text conditioning and isolation

Set `pipeline_config.text_encoder_path` explicitly to the LTX-tuned Gemma4
encoder, either a packed safetensors file or an HF model directory/repository.
The encoder's `gemma_version` must match the native checkpoint's
`gemma_source_checkpoint.gemma_version` metadata. A generic Gemma model is not
a substitute for the LTX-tuned checkpoint.

The packed loader supports embedded tokenizer assets and remaps supported
Comfy/HF weight names. Loading is strict: missing model weights, readout
projections, or connector weights are errors. Readout projections are loaded
from the encoder when present, otherwise from the native pipeline checkpoint.
For generation, the native checkpoint must also contain the components expected
by the parent LTX-2.3 loader. The V2V workflow instead loads explicit split
component paths; a transformer-only file alone is still not a complete pipeline.

Gemma4 classes are imported lazily. Use an isolated environment with
Transformers supporting `gemma4_unified` (development validation uses 5.14.1).
The repository-wide Transformers pin is intentionally unchanged, and LTX-2
and LTX-2.3 retain their existing text encoder paths. The native implementation
does not import the external LTX repository.

Tokenization ensures a leading BOS and left padding. Feature packing preserves
the encoder computation dtype and masks padding before readout. Parameter
loading uses the requested dtype while retaining FP32 RoPE buffers; casting
the entire encoder to BF16 would change its hidden states.

## Validation and remaining integration gates

Focused tests live in `tests/unittest/_torch/visual_gen/`:

- `test_ltx25_pipeline.py`: tokenizer, metadata, strict loading, BF16/FP32
  hidden states and buffer preservation, padding, batching, and LTX-2.3 isolation.
- `test_ltx25_transformer.py`: architecture and weight-mapping regressions.
- `test_ltx25_v2v.py`: workflow isolation, reference coordinates/keyframes,
  RNG consumption, Euler rounding, connector backend priority, refinement
  latent layout and unsupported-configuration guards.
- `test_runtime_lora.py`: adapter dtype and native CUDA Graph lifetime regressions.

Run these alongside `test_ltx23_pipeline.py` using the normal repository test
environment. Native CUDA Graph regressions require a wheel rebuilt from the
modified C++ sources; source-only Python overlays cannot validate that fix.

The native V2V source path was exercised on RTX PRO 6000 with packed NVFP4,
BF16 attention and resident weights at 256x256x9, 1280x768x65 and 1920x1152x65.
At the small geometry, both stages' video and audio output latents match the
external ICLoraPipeline reference exactly. Encoded-video SSIM against the
existing same-input reference is 0.9847 / 0.9878 at the two larger geometries;
decoded audio correlations exceed 0.9997. These are one-prompt/seed checks,
not a claim of bit-identical encoded media or a broad quality study.

`tests/integration/defs/examples/visual_gen/test_visual_gen_ltx25.py` exercises
the standalone public API example and specifies an LPIPS < 0.05 gate. It
parametrizes BF16 and NVFP4 smoke independently of the LPIPS gate. Both precisions
are compared against the same external **BF16** golden (matching PR #18169's
quantization-quality policy), with identical prompt, seed 42, 24 fps and
256x256x9 geometry. Configure `LTX25_MODEL_ROOT`, `LTX25_V2V_SOURCE` and
`LTX25_V2V_BF16_GOLDEN`; a missing golden skips only LPIPS. Do not reuse the
old NVFP4-reference golden as the BF16 baseline.

Unit suites are registered in `l0_cpu.yml` and `l0_b200.yml`; both precision
smoke/LPIPS cases are registered in `l0_b200.yml`. CI asset provisioning, a
Gemma4-compatible test environment, and execution of the LPIPS gate remain
required before merging; source-overlay checks do not certify a
newly packaged wheel. No native T2V E2E result is implied by V2V validation.
The standalone public example was also run through its spawned worker with
imports of external `ltx_core` and `ltx_pipelines` explicitly blocked. Its
256x256 output decodes to all 9 requested frames with 48 kHz stereo audio.

The experimental SM120 NVFP4 attention backend is separate and opt-in. This
pipeline does not enable approximate attention by default or make a new
performance claim.
