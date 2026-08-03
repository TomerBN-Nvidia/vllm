# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models import nemotron_h_mtp
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize(
    "architecture",
    [
        "NemotronH_Omni_Reasoning_V3",
        "NemotronH_Super_Omni_Reasoning_V3",
    ],
)
def test_nemotron_h_omni_mtp_hf_config_override(architecture: str):
    text_config = PretrainedConfig(
        architectures=["NemotronHForCausalLM"],
        num_nextn_predict_layers=1,
        mtp_layers_block_type=["attention", "moe"],
    )
    text_config.model_type = "nemotron_h"
    config = PretrainedConfig(architectures=[architecture])
    config.model_type = "nemotron_h_omni"
    config.llm_config = text_config
    config.text_config = text_config

    overridden = SpeculativeConfig.hf_config_override(config)

    assert overridden is config.llm_config
    assert overridden.model_type == "nemotron_h_mtp"
    assert overridden.architectures == ["NemotronHMTPModel"]
    assert overridden.n_predict == 1


def test_nemotron_h_mtp_uses_draft_model_config(monkeypatch: pytest.MonkeyPatch):
    class StubPredictor(nn.Module):
        def __init__(self, *, vllm_config, prefix: str = ""):
            super().__init__()

        def make_empty_intermediate_tensors(self):
            raise NotImplementedError

    class StubHead(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class StubLogitsProcessor:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(nemotron_h_mtp, "NemotronHMultiTokenPredictor", StubPredictor)
    monkeypatch.setattr(nemotron_h_mtp, "ParallelLMHead", StubHead)
    monkeypatch.setattr(nemotron_h_mtp, "LogitsProcessor", StubLogitsProcessor)

    target_model_config = type("TargetModelConfig", (), {"hf_config": object()})()
    draft_hf_config = PretrainedConfig(
        num_hidden_layers=52,
        hidden_size=1024,
        vocab_size=131072,
    )
    draft_model_config = type("DraftModelConfig", (), {"hf_config": draft_hf_config})()
    speculative_config = type(
        "TestSpeculativeConfig", (), {"draft_model_config": draft_model_config}
    )()
    vllm_config = type(
        "TestVllmConfig",
        (),
        {
            "model_config": target_model_config,
            "speculative_config": speculative_config,
            "quant_config": None,
            "parallel_config": None,
        },
    )()

    model = nemotron_h_mtp.NemotronHMTP(vllm_config=vllm_config)

    assert model.config is draft_hf_config
    assert model.mtp_start_layer_idx == 52


@pytest.mark.parametrize("checkpoint_prefix", ["", "language_model."])
def test_nemotron_h_mtp_loads_normalization_weights(checkpoint_prefix: str):
    parameter_names = [
        "model.layers.0.enorm.weight",
        "model.layers.0.hnorm.weight",
        "model.layers.0.norm.weight",
        "model.layers.1.norm.weight",
        "model.layers.1.final_layernorm.weight",
    ]
    parameters = {name: nn.Parameter(torch.zeros(2)) for name in parameter_names}
    model = type(
        "TestNemotronHMTP",
        (),
        {
            "config": PretrainedConfig(n_routed_experts=None),
            "named_parameters": lambda self: parameters.items(),
        },
    )()
    weights = [
        (
            f"{checkpoint_prefix}{name.replace('model.layers.', 'mtp.layers.')}",
            torch.ones(2),
        )
        for name in parameter_names
    ]

    loaded = nemotron_h_mtp.NemotronHMTP.load_weights(model, weights)

    assert loaded == set(parameter_names)
    assert all(torch.equal(param, torch.ones(2)) for param in parameters.values())


def test_nemotron_h_omni_mtp_uses_img_context_token_id():
    config = PretrainedConfig(img_context_token_id=18)

    image_token_index = SpecDecodeBaseProposer._get_image_token_index(
        "NemotronH_Nano_VL_V2", config
    )

    assert image_token_index == 18
