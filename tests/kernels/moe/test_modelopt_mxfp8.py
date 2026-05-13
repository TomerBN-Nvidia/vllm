# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from enum import Enum
from types import SimpleNamespace

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutlassMxfp8LinearKernel,
)
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


def test_flashinfer_mxfp8_linear_keeps_checkpoint_scale_for_refit():
    layer = torch.nn.Module()
    layer.weight = Parameter(
        torch.empty((128, 128), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = Parameter(
        torch.ones((128, 4), dtype=torch.uint8), requires_grad=False
    )

    kernel = object.__new__(FlashInferCutlassMxfp8LinearKernel)
    kernel.process_weights_after_loading(layer)

    checkpoint_scale = layer.weight_scale
    apply_scale = layer.weight_scale_for_apply
    layer.weight_scale.data.fill_(2)
    kernel.process_weights_after_loading(layer)

    assert layer.weight_scale is checkpoint_scale
    assert layer.weight_scale_for_apply is apply_scale
    assert torch.all(layer.weight_scale_for_apply == 2)


def test_modelopt_mxfp8_moe_reuses_apply_scales_for_refit(monkeypatch):
    fake_flashinfer = types.ModuleType("flashinfer")
    fake_flashinfer.reorder_rows_for_gated_act_gemm = lambda x: x
    fake_flashinfer.shuffle_matrix_a = lambda x, epilogue_tile_m: x
    fake_flashinfer.shuffle_matrix_sf_a = lambda x, epilogue_tile_m: x
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)

    layer = torch.nn.Module()
    layer.intermediate_size_per_partition = 32
    layer.hidden_size = 32
    layer.w13_weight = Parameter(
        torch.empty((2, 32, 32), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.w2_weight = Parameter(
        torch.empty((2, 32, 32), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.w13_weight_scale = Parameter(
        torch.ones((2, 32, 1), dtype=torch.uint8), requires_grad=False
    )
    layer.w2_weight_scale = Parameter(
        torch.ones((2, 32, 1), dtype=torch.uint8), requires_grad=False
    )

    method = object.__new__(modelopt.ModelOptMxFp8FusedMoE)
    method.moe = SimpleNamespace(is_act_and_mul=False)

    method.process_weights_after_loading(layer)

    checkpoint_w13_scale = layer.w13_weight_scale
    checkpoint_w2_scale = layer.w2_weight_scale
    apply_w13_scale = layer.w13_scale_for_apply
    apply_w2_scale = layer.w2_scale_for_apply
    layer.w13_weight_scale.data.fill_(2)
    layer.w2_weight_scale.data.fill_(2)
    method.process_weights_after_loading(layer)

    assert layer.w13_weight_scale is checkpoint_w13_scale
    assert layer.w2_weight_scale is checkpoint_w2_scale
    assert layer.w13_scale_for_apply is apply_w13_scale
    assert layer.w2_scale_for_apply is apply_w2_scale
    assert torch.all(layer.w13_scale_for_apply == 2)
    assert torch.all(layer.w2_scale_for_apply == 2)
