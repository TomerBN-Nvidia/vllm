# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.nano_nemotron_vl import NemotronH_Nano_VL_V2


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


@pytest.mark.parametrize(
    "prefix",
    ["vision_final_layernorm.", "vision_projector.vision_final_layernorm."],
)
def test_nano_nemotron_vl_loads_and_applies_radio_final_layernorm(prefix: str):
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    vision_model = _VisionModel()
    final_layernorm = nn.LayerNorm(4, eps=1.0e-6).float()
    object.__setattr__(model, "model_config", _ImageOnlyModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "vision_final_layernorm", final_layernorm)
    object.__setattr__(model, "_loaded_vision_final_layernorm_params", set())
    object.__setattr__(model, "_vision_final_layernorm_enabled", False)
    object.__setattr__(model, "sound_encoder", None)

    weight = torch.tensor([1.0, 1.5, 2.0, 2.5], dtype=torch.bfloat16)
    bias = torch.tensor([-0.5, 0.0, 0.5, 1.0], dtype=torch.bfloat16)
    model.load_weights(
        [
            (f"{prefix}weight", weight),
            (f"{prefix}bias", bias),
        ]
    )

    assert model._vision_final_layernorm_enabled
    assert torch.equal(final_layernorm.weight, weight.float())
    assert torch.equal(final_layernorm.bias, bias.float())
    assert vision_model.loaded_weights == []

    inputs = torch.tensor(
        [[[1.0, 2.0, 4.0, 8.0], [2.0, 3.0, 5.0, 9.0]]],
        dtype=torch.bfloat16,
    )
    expected = final_layernorm(inputs.float()).to(inputs.dtype)
    actual = model._apply_vision_final_layernorm(inputs)
    torch.testing.assert_close(actual, expected)


def test_nano_nemotron_vl_does_not_apply_unloaded_radio_final_layernorm():
    model = object.__new__(NemotronH_Nano_VL_V2)
    object.__setattr__(model, "vision_final_layernorm", nn.LayerNorm(4))
    object.__setattr__(model, "_vision_final_layernorm_enabled", False)
    inputs = torch.randn(2, 3, 4)

    assert model._apply_vision_final_layernorm(inputs) is inputs


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
