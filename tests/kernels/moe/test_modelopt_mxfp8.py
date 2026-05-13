# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from enum import Enum
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
from vllm.model_executor.layers.quantization import modelopt


class FakeActivationType(Enum):
    Swiglu = 0
    Relu2 = 1


class FakeFp8QuantizationType(Enum):
    MxFp8 = 0


def test_modelopt_mxfp8_trtllm_forwards_relu2_activation(monkeypatch):
    flashinfer = types.ModuleType("flashinfer")
    fused_moe = types.ModuleType("flashinfer.fused_moe")
    core = types.ModuleType("flashinfer.fused_moe.core")
    core.ActivationType = FakeActivationType
    core.Fp8QuantizationType = FakeFp8QuantizationType
    fused_moe.core = core
    flashinfer.fused_moe = fused_moe
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe", fused_moe)
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe.core", core)

    captured_kwargs = {}

    def fake_mxfp8_quantize(x, is_sf_swizzled_layout):
        return torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty(
            (1,), dtype=torch.uint8
        )

    def fake_trtllm_moe(**kwargs):
        captured_kwargs.update(kwargs)
        return torch.empty((4, 32), dtype=torch.bfloat16)

    monkeypatch.setattr(modelopt, "mxfp8_e4m3_quantize", fake_mxfp8_quantize)
    monkeypatch.setattr(
        modelopt, "flashinfer_trtllm_fp8_block_scale_moe", fake_trtllm_moe
    )

    layer = SimpleNamespace(
        eplb_state=None,
        activation=MoEActivation.RELU2_NO_MUL,
        routing_method_type=RoutingMethodType.Renormalize,
        e_score_correction_bias=None,
        num_expert_group=0,
        topk_group=0,
        w13_weight=torch.empty((2, 32, 32), dtype=torch.float8_e4m3fn),
        w13_weight_scale=torch.empty((2, 32, 1), dtype=torch.uint8),
        w2_weight=torch.empty((2, 32, 32), dtype=torch.float8_e4m3fn),
        w2_weight_scale=torch.empty((2, 32, 1), dtype=torch.uint8),
        global_num_experts=2,
        top_k=1,
        intermediate_size_per_partition=32,
        ep_rank=0,
        local_num_experts=2,
        routed_scaling_factor=None,
    )

    method = object.__new__(modelopt.ModelOptMxFp8FusedMoE)
    method.mxfp8_backend = Fp8MoeBackend.FLASHINFER_TRTLLM

    method.apply_monolithic(
        layer,
        torch.randn((4, 32), dtype=torch.bfloat16),
        torch.randn((4, 2), dtype=torch.bfloat16),
    )

    assert captured_kwargs["activation_type"] == FakeActivationType.Relu2
