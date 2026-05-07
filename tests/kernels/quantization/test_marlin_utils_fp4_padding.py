# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the _pad_tensor_dim and _pad_w13_for_marlin_tile
helpers used by the FP4 Marlin n-dim padding fix on H100 (NVFP4 MoE).

Run `pytest tests/kernels/quantization/test_marlin_utils_fp4_padding.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    FP4_MARLIN_TILE_N_SIZE,
    _pad_tensor_dim,
    _pad_w13_for_marlin_tile,
)


def test_constant_is_64():
    assert FP4_MARLIN_TILE_N_SIZE == 64


def test_pad_no_op_returns_same_tensor():
    t = torch.randn(2, 64)
    out = _pad_tensor_dim(t, dim=1, padded_size=64)
    assert out is t


def test_pad_dim1_zeros_appended():
    t = torch.ones(2, 50)
    out = _pad_tensor_dim(t, dim=1, padded_size=64)
    assert out.shape == (2, 64)
    assert torch.equal(out[:, :50], torch.ones(2, 50))
    assert torch.equal(out[:, 50:], torch.zeros(2, 14))


def test_pad_dim0_zeros_appended():
    t = torch.ones(50, 4)
    out = _pad_tensor_dim(t, dim=0, padded_size=64)
    assert out.shape == (64, 4)
    assert torch.equal(out[:50], torch.ones(50, 4))
    assert torch.equal(out[50:], torch.zeros(14, 4))


def test_pad_3d_dim1_for_moe_layout():
    """MoE FP4 weights are (E, size_n, size_k // 2); padding goes on dim 1."""
    e, unpadded_n, half_k = 8, 50, 32
    t = torch.ones(e, unpadded_n, half_k)
    out = _pad_tensor_dim(t, dim=1, padded_size=64)
    assert out.shape == (e, 64, half_k)
    assert torch.equal(out[:, :unpadded_n, :], torch.ones(e, unpadded_n, half_k))
    assert torch.equal(out[:, unpadded_n:, :], torch.zeros(e, 14, half_k))


def test_pad_negative_size_raises():
    t = torch.zeros(2, 100)
    with pytest.raises(ValueError, match="Cannot pad"):
        _pad_tensor_dim(t, dim=1, padded_size=50)


def test_pad_preserves_dtype_and_device():
    t = torch.ones(2, 50, dtype=torch.bfloat16)
    out = _pad_tensor_dim(t, dim=1, padded_size=64)
    assert out.dtype == torch.bfloat16
    assert out.device == t.device


@pytest.mark.parametrize("intermediate", [232, 464, 928])
def test_pad_to_tile_alignment_matches_design_table(intermediate: int):
    """For Nemotron-H 3.5 nano (non-gated MoE, w13_num_shards=1):
    intermediate=464 (TP=4) needs +48 -> 512; 232 (TP=8) needs +24 -> 256;
    928 (TP=2) is already 64-aligned.
    """
    e = 4
    half_k = 16
    t = torch.ones(e, intermediate, half_k)
    padded = (
        (intermediate + FP4_MARLIN_TILE_N_SIZE - 1) // FP4_MARLIN_TILE_N_SIZE
    ) * FP4_MARLIN_TILE_N_SIZE
    out = _pad_tensor_dim(t, dim=1, padded_size=padded)
    assert out.shape[1] == padded
    assert out.shape[1] % FP4_MARLIN_TILE_N_SIZE == 0


def test_pad_w13_helper_aligned_returns_input_identity():
    """Aligned input must return the same tensor objects unchanged.
    The in-place ``prepare_moe_fp4_layer_for_marlin`` caller relies on
    ``is`` identity to skip the nn.Parameter rewrap when no padding occurred.
    """
    w13 = torch.ones(4, 128, 16)
    w13_scale = torch.ones(4, 128, 4)
    out_w13, out_scale, padded_n = _pad_w13_for_marlin_tile(
        w13, w13_scale, unpadded_w13_size_n=128
    )
    assert out_w13 is w13
    assert out_scale is w13_scale
    assert padded_n == 128


def test_pad_w13_helper_misaligned_pads_both_to_tile():
    """Misaligned input must pad both w13 and w13_scale to the same
    tile-aligned size_n on dim 1."""
    e, unpadded, half_k = 4, 50, 16
    w13 = torch.ones(e, unpadded, half_k)
    w13_scale = torch.ones(e, unpadded, 4)
    out_w13, out_scale, padded_n = _pad_w13_for_marlin_tile(
        w13, w13_scale, unpadded_w13_size_n=unpadded
    )
    assert padded_n == 64
    assert out_w13.shape == (e, 64, half_k)
    assert out_scale.shape == (e, 64, 4)
    # Padded rows are zero.
    assert torch.equal(out_w13[:, unpadded:, :], torch.zeros(e, 14, half_k))
    assert torch.equal(out_scale[:, unpadded:, :], torch.zeros(e, 14, 4))
