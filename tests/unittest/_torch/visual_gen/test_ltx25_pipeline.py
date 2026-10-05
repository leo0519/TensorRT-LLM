# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Native LTX-2.5 conditioning and LTX-2.3 isolation regressions."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, AutoModelForImageTextToText, PreTrainedTokenizerFast

from tensorrt_llm._torch.visual_gen.models.ltx23.ltx23_core.modality import LTX23Modality
from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23 import LTX23Pipeline
from tensorrt_llm._torch.visual_gen.models.ltx25.pipeline_ltx25 import (
    LTX25Pipeline,
    _checkpoint_gemma_version,
    _load_component_strict,
    _LTX25CUDAGraphRunner,
    detect_native_ltx_pipeline,
)
from tensorrt_llm._torch.visual_gen.models.ltx25.text_encoder import (
    _gemma4_weight_key,
    load_gemma4_config,
    load_gemma4_model,
    load_gemma4_tokenizer,
    pack_gemma4_hidden_states,
    tokenize_gemma4,
)
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, PipelineComponent


def _tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        WordLevel(
            {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "a": 4, "b": 5, "c": 6},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.padding_side = "left"
    return tokenizer


def _gemma_config():
    try:
        return AutoConfig.for_model(
            "gemma4_unified",
            gemma_version="gemma4-test",
            text_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_global_key_value_heads": 1,
                "head_dim": 16,
                "global_head_dim": 16,
                "vocab_size": 32,
                "max_position_embeddings": 32,
                "sliding_window": 16,
                "layer_types": ["sliding_attention", "full_attention"],
                "rope_parameters": {
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                    "full_attention": {
                        "rope_type": "proportional",
                        "rope_theta": 10000.0,
                        "partial_rotary_factor": 0.5,
                    },
                },
            },
            vision_config={
                "mm_embed_dim": 16,
                "mm_posemb_size": 8,
                "patch_size": 2,
                "pooling_kernel_size": 2,
                "num_soft_tokens": 4,
            },
            audio_config={"audio_embed_dim": 8},
        )
    except ValueError:
        pytest.skip("This Transformers installation does not include Gemma4 unified")


@pytest.mark.parametrize(
    "prompts,length,expected",
    [
        ([" a ", "b c"], 4, [[0, 0, 2, 4], [0, 2, 5, 6]]),
        (["a b c"], 3, [[2, 4, 5]]),
        ([""], 2, [[0, 2]]),
        (["<bos> a"], 3, [[0, 2, 4]]),
    ],
)
def test_gemma4_bos_padding_and_truncation(prompts, length, expected) -> None:
    ids, mask = tokenize_gemma4(_tokenizer(), prompts, length, torch.device("cpu"))
    torch.testing.assert_close(ids, torch.tensor(expected))
    torch.testing.assert_close(mask, ids.ne(0).long())


@pytest.mark.parametrize(
    "source,target",
    [
        ("model.layers.0.weight", "model.language_model.layers.0.weight"),
        ("model.language_model.norm.weight", "model.language_model.norm.weight"),
        ("vision_model.pos_embedding", "model.embed_vision.pos_embedding"),
        (
            "multi_modal_projector.embedding_projection.weight",
            "model.embed_vision.multimodal_embedder.embedding_projection.weight",
        ),
        (
            "audio_projector.embedding_projection.weight",
            "model.embed_audio.embedding_projection.weight",
        ),
        ("text_embedding_projection.video_aggregate_embed.weight", None),
        ("hf_asset__tokenizer_config.json", None),
    ],
)
def test_packed_gemma4_weight_names(source, target) -> None:
    assert _gemma4_weight_key(source) == target


def test_gemma4_rejects_wrong_family_and_version(tmp_path) -> None:
    path = tmp_path / "encoder.safetensors"
    save_file({}, str(path), metadata={"gemma_config": json.dumps({"model_type": "gemma3"})})
    with pytest.raises(ValueError, match="requires gemma4_unified"):
        load_gemma4_config(str(path), "gemma4-test")
    config = _gemma_config()
    save_file({}, str(path), metadata={"gemma_config": config.to_json_string()})
    with pytest.raises(ValueError, match="version mismatch"):
        load_gemma4_config(str(path), "different-version")
    assert load_gemma4_config(str(path), "gemma4-test").gemma_version == "gemma4-test"


def test_checkpoint_gemma_version_is_required(tmp_path) -> None:
    path = tmp_path / "transformer.safetensors"
    save_file({}, str(path))
    with pytest.raises(ValueError, match="must declare"):
        _checkpoint_gemma_version([str(path)])
    save_file(
        {},
        str(path),
        metadata={"gemma_source_checkpoint": json.dumps({"gemma_version": "gemma4-test"})},
    )
    assert _checkpoint_gemma_version([str(path)]) == "gemma4-test"


def test_missing_projection_weights_are_fatal(tmp_path) -> None:
    path = tmp_path / "missing.safetensors"
    save_file({"unrelated": torch.ones(1)}, str(path))
    with pytest.raises(RuntimeError, match="Missing key"):
        _load_component_strict([str(path)], torch.nn.Linear(2, 3), "text_embedding_projection.")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gemma4_feature_packing_masks_padding_and_keeps_dtype(dtype) -> None:
    hidden = torch.tensor([[[3.0, 4.0], [1.0, 2.0]]], dtype=dtype)
    mask = torch.tensor([[0, 1]])
    actual = pack_gemma4_hidden_states((hidden, hidden * 2), mask)
    stacked = torch.stack((hidden, hidden * 2), dim=-1)
    expected = (stacked * torch.rsqrt(stacked.square().mean(2, keepdim=True) + 1e-6)).flatten(2)
    expected[:, 0] = 0
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ltx25_inherits_generation_but_not_gemma3_loading() -> None:
    assert LTX25Pipeline.forward is LTX23Pipeline.forward
    assert LTX25Pipeline._process_connectors is not LTX23Pipeline._process_connectors
    assert LTX25Pipeline._per_token_rms_pack is LTX23Pipeline._per_token_rms_pack
    assert LTX25Pipeline._encode_prompt is not LTX23Pipeline._encode_prompt
    assert LTX25Pipeline.load_standard_components is not LTX23Pipeline.load_standard_components
    assert (
        PIPELINE_REGISTRY["LTX23Pipeline"].defaults["text_encoder_path"] == "google/gemma-3-12b-it"
    )
    assert PIPELINE_REGISTRY["LTX25Pipeline"].defaults["text_encoder_path"] == ""
    assert LTX25Pipeline._cuda_graph_runner_cls is _LTX25CUDAGraphRunner
    assert LTX25Pipeline._cuda_graph_runner_cls is not LTX23Pipeline._cuda_graph_runner_cls


@pytest.mark.parametrize("has_mask", [False, True])
def test_ltx25_graph_runner_preserves_keyframe_masks(has_mask) -> None:
    def modality(fill):
        return LTX23Modality(
            latent=torch.full((1, 4, 2), fill),
            timesteps=torch.full((1, 4), fill),
            sigma=torch.full((1,), fill),
            positions=torch.full((1, 3, 4, 2), fill),
            context=torch.full((1, 2, 8), fill),
            context_mask=torch.full((1, 2), fill),
            keyframes_mask=torch.full((1, 4, 1), fill) if has_mask else None,
        )

    source = modality(1.0)
    static = _LTX25CUDAGraphRunner._clone_value(source)
    updated = modality(2.0)
    runner = object.__new__(_LTX25CUDAGraphRunner)
    key = dict(runner._key_parts_for("video", source))
    assert key["video.keyframes_mask"] == ((1, 4, 1) if has_mask else None)
    if has_mask:
        assert static.keyframes_mask.data_ptr() != source.keyframes_mask.data_ptr()
    assert _LTX25CUDAGraphRunner._copy_value(static, updated) is static
    for field in ("latent", "timesteps", "sigma", "positions", "context", "context_mask"):
        torch.testing.assert_close(getattr(static, field), getattr(updated, field))
    if has_mask:
        torch.testing.assert_close(static.keyframes_mask, updated.keyframes_mask)
        torch.testing.assert_close(source.keyframes_mask, torch.ones_like(source.keyframes_mask))
    else:
        assert static.keyframes_mask is None
    legacy = object.__new__(LTX23Pipeline._cuda_graph_runner_cls)
    assert "video.keyframes_mask" not in dict(legacy._key_parts_for("video", source))


@pytest.mark.parametrize("marker,expected", [(False, "LTX23Pipeline"), (True, "LTX25Pipeline")])
def test_native_detection_keeps_legacy_selection(marker, expected) -> None:
    from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23 import (
        detect_native_ltx_pipeline as legacy_detect,
    )

    config = {
        "transformer": {
            "caption_proj_before_connector": True,
            "caption_projection_first_linear": False,
            "caption_projection_second_linear": False,
            "caption_proj_input_norm": False,
            "cross_attention_adaln": True,
            "use_keyframes_abs_pos_embedding": marker,
        }
    }
    assert detect_native_ltx_pipeline(config) == expected
    assert legacy_detect(config) == "LTX23Pipeline"
    assert detect_native_ltx_pipeline({"transformer": {}}) == "LTX2Pipeline"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("packed", [True, False])
def test_gemma4_matches_hf_hidden_states(tmp_path, dtype, packed) -> None:
    config = _gemma_config()
    torch.manual_seed(8)
    reference = AutoModelForImageTextToText.from_config(config).eval()
    # Cast checkpoint state, including persistent layer_scalar buffers, while
    # preserving the non-persistent FP32 RoPE and embedding-scale buffers.
    for parameter in reference.parameters():
        parameter.data = parameter.data.to(dtype)
    state_keys = reference.state_dict().keys()
    for name, buffer in reference.named_buffers():
        if name in state_keys and buffer.is_floating_point():
            buffer.data = buffer.data.to(dtype)
    weights = {}
    for name, tensor in reference.state_dict().items():
        # Exercise Comfy-flat language weights plus canonical HF tower names.
        key = name.replace("model.language_model.", "model.")
        weights[key] = tensor.clone()
    tokenizer = _tokenizer()
    weights["tokenizer_json"] = torch.tensor(
        list(tokenizer.backend_tokenizer.to_str().encode()), dtype=torch.uint8
    )
    tokenizer_config = {"pad_token": "<pad>", "bos_token": "<bos>", "eos_token": "<eos>"}
    weights["hf_asset__tokenizer_config.json"] = torch.tensor(
        list(json.dumps(tokenizer_config).encode()), dtype=torch.uint8
    )
    path = tmp_path / "packed.safetensors"
    save_file(weights, str(path), metadata={"gemma_config": config.to_json_string()})
    if not packed:
        path = tmp_path / "hf_encoder"
        reference.save_pretrained(path)
        tokenizer.save_pretrained(path)
    loaded_config = load_gemma4_config(str(path), "gemma4-test")
    loaded = load_gemma4_model(str(path), loaded_config, dtype, torch.device("cpu"))
    expected_buffers = dict(reference.model.named_buffers())
    for name, buffer in loaded.named_buffers():
        assert buffer.dtype == expected_buffers[name].dtype, name
        torch.testing.assert_close(buffer, expected_buffers[name], rtol=0, atol=0)
    loaded_tokenizer = load_gemma4_tokenizer(str(path))
    ids, mask = tokenize_gemma4(loaded_tokenizer, ["a", "b c"], 8, torch.device("cpu"))
    with torch.inference_mode():
        expected = reference.model(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        actual = loaded(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    for left, right in zip(actual.hidden_states, expected.hidden_states, strict=True):
        torch.testing.assert_close(left, right)
    assert not any(t.is_meta for t in loaded.buffers())


def test_incomplete_gemma4_weights_are_fatal(tmp_path) -> None:
    config = _gemma_config()
    reference = AutoModelForImageTextToText.from_config(config)
    weights = {name: value.clone() for name, value in reference.state_dict().items()}
    del weights["model.language_model.layers.0.self_attn.q_proj.weight"]
    path = tmp_path / "incomplete.safetensors"
    save_file(weights, str(path))
    with pytest.raises(ValueError, match="Incomplete or incompatible"):
        load_gemma4_model(str(path), config, torch.float32, torch.device("cpu"))


def test_skipped_text_components_never_enter_gemma3_loader(monkeypatch) -> None:
    pipeline = object.__new__(LTX25Pipeline)
    torch.nn.Module.__init__(pipeline)
    skipped = [PipelineComponent.TOKENIZER, PipelineComponent.TEXT_ENCODER, "connectors"]
    original = list(skipped)
    calls = []

    def parent_loader(self, checkpoint_dir, device, skip_components, **kwargs):
        calls.append(list(skip_components))
        skip_components.append("parent_mutation")

    monkeypatch.setattr(LTX23Pipeline, "load_standard_components", parent_loader)
    pipeline.load_standard_components("unused", torch.device("cpu"), skipped)
    assert skipped == original
    assert len(calls) == 1
    assert PipelineComponent.TOKENIZER in calls[0]
    assert PipelineComponent.TEXT_ENCODER in calls[0]


def test_prompt_batch_repetition_keeps_masks_aligned(monkeypatch) -> None:
    pipeline = object.__new__(LTX25Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.pipeline_config = SimpleNamespace(torch_dtype=torch.float32)
    pipeline.tokenizer = _tokenizer()
    monkeypatch.setattr(LTX25Pipeline, "device", property(lambda self: torch.device("cpu")))

    class Encoder(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            hidden = torch.stack((input_ids.float(), input_ids.float() + 1), dim=-1)
            return SimpleNamespace(hidden_states=(hidden, hidden * 2))

    pipeline.text_encoder = Encoder()
    embeds, masks = pipeline._encode_prompt(
        ["a", "b c"], num_videos_per_prompt=2, max_sequence_length=4
    )
    assert embeds.shape == (4, 4, 4)
    torch.testing.assert_close(embeds[0], embeds[1])
    torch.testing.assert_close(embeds[2], embeds[3])
    torch.testing.assert_close(
        masks, torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]])
    )
