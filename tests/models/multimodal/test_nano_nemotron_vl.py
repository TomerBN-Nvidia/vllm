# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import vllm.model_executor.models.nano_nemotron_vl as nano_nemotron_vl
from vllm.model_executor.models.nano_nemotron_vl import NemotronH_Nano_VL_V2
from vllm.transformers_utils.processors.nano_nemotron_vl import (
    BaseNanoNemotronVLProcessor,
)


class _TextOnlyMultiModalConfig:
    def get_limit_per_prompt(self, modality: str) -> int:
        return 0


class _ImageOnlyMultiModalConfig:
    def get_limit_per_prompt(self, modality: str) -> int:
        return 1 if modality == "image" else 0


class _ModelConfig:
    multimodal_config = _TextOnlyMultiModalConfig()


class _ImageOnlyModelConfig:
    multimodal_config = _ImageOnlyMultiModalConfig()


class _LanguageModel:
    def __init__(self) -> None:
        self.loaded_weights: list[tuple[str, object]] = []

    def load_weights(self, weights):
        self.loaded_weights = list(weights)


class _MissingMultiModalModule:
    def named_parameters(self):
        raise AssertionError("multimodal weights should not be inspected")

    def load_weights(self, weights):
        raise AssertionError("multimodal weights should not be loaded")


class _AdapterModule:
    def named_parameters(self):
        return []


class _WeightModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


def _make_native_adapter() -> nn.Module:
    adapter = nn.Module()
    adapter.add_module("0", _WeightModule())
    adapter.add_module("1", _WeightModule())
    adapter.add_module("3", _WeightModule())
    return adapter


class _VisionModel:
    def __init__(self) -> None:
        self.loaded_weights: list[tuple[str, object]] = []

    def load_weights(self, weights):
        self.loaded_weights = list(weights)


class _FakeTensor:
    """Sentinel stand-in for torch.Tensor in load_weights tests. Supports the
    .detach().clone() chain used by load_weights for buffered mm weights;
    both methods return self so identity (and the existing equality
    assertions) are preserved through cloning."""

    def detach(self):
        return self

    def clone(self):
        return self


def test_nano_nemotron_vl_skips_multimodal_weights_in_text_only_mode():
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    object.__setattr__(model, "model_config", _ModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", _MissingMultiModalModule())
    object.__setattr__(model, "sound_encoder", None)

    language_weight = object()
    model.load_weights(
        [
            ("language_model.layers.0.weight", language_weight),
            ("mlp1.0.weight", object()),
            ("vision_model.radio_model.encoder.weight", object()),
            ("sound_encoder.encoder.weight", object()),
        ]
    )

    assert language_model.loaded_weights == [("layers.0.weight", language_weight)]


def test_nano_nemotron_vl_loads_vision_weights_without_sound_encoder():
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    vision_model = _VisionModel()
    object.__setattr__(model, "model_config", _ImageOnlyModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "sound_encoder", None)

    language_weight = object()
    vision_weight = _FakeTensor()
    model.load_weights(
        [
            ("language_model.layers.0.weight", language_weight),
            ("vision_model.radio_model.encoder.weight", vision_weight),
        ]
    )

    assert language_model.loaded_weights == [("layers.0.weight", language_weight)]
    assert vision_model.loaded_weights == [
        ("radio_model.encoder.weight", vision_weight)
    ]


def test_nano_nemotron_vl_loads_native_flat_vision_weights():
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    vision_model = _VisionModel()
    adapter = _make_native_adapter()
    object.__setattr__(model, "model_config", _ImageOnlyModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", adapter)
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "sound_encoder", None)

    vision_weight = _FakeTensor()
    model.load_weights(
        [
            ("vision_projector.mlp1.norm.weight", torch.ones(1)),
            ("vision_projector.mlp1.linear1.weight", torch.full((1,), 2.0)),
            ("vision_projector.mlp1.linear2.weight", torch.full((1,), 3.0)),
            ("vision_model.embeddings.position_embedding", vision_weight),
        ]
    )

    assert torch.equal(adapter.get_submodule("0").weight, torch.ones(1))
    assert torch.equal(adapter.get_submodule("1").weight, torch.full((1,), 2.0))
    assert torch.equal(adapter.get_submodule("3").weight, torch.full((1,), 3.0))
    assert vision_model.loaded_weights == [
        ("embeddings.position_embedding", vision_weight)
    ]


def test_nano_nemotron_vl_requires_sound_encoder_for_sound_weights():
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    vision_model = _VisionModel()
    object.__setattr__(model, "model_config", _ImageOnlyModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "sound_encoder", None)

    with pytest.raises(AssertionError):
        model.load_weights([("sound_encoder.encoder.weight", object())])


def test_nano_nemotron_vl_builds_radio_config_from_flat_config(monkeypatch):
    monkeypatch.setattr(nano_nemotron_vl, "RadioModel", lambda config: config)
    vision_config = SimpleNamespace(
        hidden_size=1280,
        num_hidden_layers=32,
        num_attention_heads=16,
        mlp_ratio=4.0,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        qkv_bias=True,
        layerscale_value=1.0,
        image_size=224,
        patch_size=16,
        max_img_size=2048,
        num_cls_tokens=3,
        num_registers=7,
        summary_idxs=[0, 1],
        norm_mean=[0.1, 0.2, 0.3],
        norm_std=[0.4, 0.5, 0.6],
        video_temporal_patch_size=2,
        use_swiglu_ffn=False,
    )
    hf_config = SimpleNamespace(vision_config=vision_config)

    radio_config = NemotronH_Nano_VL_V2.get_vit_model_from_radio_config(
        object(), hf_config
    )

    assert radio_config.model_name == "vit_huge_patch16_224"
    assert radio_config.hidden_size == 1280
    assert radio_config.num_hidden_layers == 32
    assert radio_config.num_attention_heads == 16
    assert radio_config.intermediate_size == 5120
    assert radio_config.cpe_max_size == 2048
    assert radio_config.num_cls_tokens == 3
    assert radio_config.num_registers == 7
    assert radio_config.summary_idxs == [0, 1]
    assert radio_config.video_temporal_patch_size == 2


def test_nano_nemotron_vl_builds_radio_config_from_legacy_args(monkeypatch):
    monkeypatch.setattr(nano_nemotron_vl, "RadioModel", lambda config: config)
    args = {
        "model": "vit_huge_patch16_224",
        "image_size": 224,
        "patch_size": 16,
        "teachers": [{"name": "teacher"}],
    }
    vision_config = SimpleNamespace(args=args, preferred_resolution=(432, 432))
    hf_config = SimpleNamespace(
        vision_config=vision_config,
        norm_mean=[0.1, 0.2, 0.3],
        norm_std=[0.4, 0.5, 0.6],
    )

    radio_config = NemotronH_Nano_VL_V2.get_vit_model_from_radio_config(
        object(), hf_config
    )

    assert args["model"] == "vit_huge_patch16_224"
    assert radio_config.model_name == "vit_huge_patch16_224"
    assert radio_config.image_size == 432
    assert radio_config.hidden_size == 1280
    assert radio_config.num_hidden_layers == 32


def test_nano_nemotron_vl_flat_radio_config_is_not_dynamic_resolution():
    config = SimpleNamespace(vision_config=SimpleNamespace(model_type="radio"))

    assert not BaseNanoNemotronVLProcessor.use_dynamic_resolution(config)


def test_nano_nemotron_vl_legacy_dynamic_resolution_config():
    config = SimpleNamespace(
        vision_config=SimpleNamespace(
            args={"min_num_patches": 1, "max_num_patches": 12}
        )
    )

    assert BaseNanoNemotronVLProcessor.use_dynamic_resolution(config)


def test_nano_nemotron_vl_rejects_partial_dynamic_resolution_config():
    config = SimpleNamespace(vision_config=SimpleNamespace(min_num_patches=1))

    with pytest.raises(ValueError, match="both min_num_patches"):
        BaseNanoNemotronVLProcessor.use_dynamic_resolution(config)
