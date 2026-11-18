# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.configs.nemotron_shortcut import (
    NemotronShortcutConfig,
    NemotronShortcutTextConfig,
)


@pytest.mark.parametrize("quantization_location", ["outer", "text"])
def test_mtp_preserves_target_config(quantization_location):
    text = NemotronShortcutTextConfig(num_nextn_predict_layers=1)
    target = NemotronShortcutConfig(text_config=text)
    quantization = {"quant_method": "modelopt", "quant_algo": "MXFP8"}
    owner = target if quantization_location == "outer" else text
    owner.quantization_config = quantization

    draft = SpeculativeConfig.hf_config_override(target)

    assert draft.architectures == ["NemotronShortcutMTPModel"]
    assert text.model_type == "nemotron4_text"
    assert getattr(text, "n_predict", None) is None
    assert draft.quantization_config == quantization
    draft.quantization_config["quant_algo"] = "changed"
    assert quantization["quant_algo"] == "MXFP8"
