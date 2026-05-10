# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for FP4 Marlin MoE n-dim padding helpers.

Run `pytest tests/kernels/quantization/test_marlin_utils_fp4_padding.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    FP4_MARLIN_TILE_N_SIZE,
    _pad_w2_for_marlin_tile,
    _pad_w13_for_marlin_tile,
)

# (unpadded, expected_padded) covering the Nemotron-H 3.5 nano NVFP4 MoE-only
# validation matrix from PR #41947: TP=1 (1856) aligned; TP=2 (928), TP=4 (464),
# TP=8 (232) require padding to the next 64-multiple.
TILE_CASES = [(1856, 1856), (928, 960), (464, 512), (232, 256)]


@pytest.mark.parametrize("unpadded,expected", TILE_CASES)
def test_pad_w13_for_marlin_tile_matches_design_table(unpadded, expected):
    e, half_k, scale_k = 4, 16, 4
    w13 = torch.ones(e, unpadded, half_k)
    scale = torch.ones(e, unpadded, scale_k)
    out_w13, out_scale, padded_n = _pad_w13_for_marlin_tile(
        w13, scale, unpadded_w13_size_n=unpadded
    )
    assert padded_n == expected
    assert padded_n % FP4_MARLIN_TILE_N_SIZE == 0
    assert out_w13.shape == (e, expected, half_k)
    assert out_scale.shape == (e, expected, scale_k)
    if expected == unpadded:
        # In-place caller relies on identity to skip nn.Parameter rewrap.
        assert out_w13 is w13 and out_scale is scale


@pytest.mark.parametrize("unpadded,expected", TILE_CASES)
def test_pad_w2_for_marlin_tile_matches_design_table(unpadded, expected):
    e, hidden, group_size = 4, 32, 16
    w2 = torch.ones(e, hidden, unpadded // 2)
    scale = torch.ones(e, hidden, unpadded // group_size)
    out_w2, out_scale, padded_k = _pad_w2_for_marlin_tile(
        w2, scale, unpadded_w2_size_k=unpadded, group_size=group_size
    )
    assert padded_k == expected
    assert padded_k % FP4_MARLIN_TILE_N_SIZE == 0
    assert out_w2.shape == (e, hidden, expected // 2)
    assert out_scale.shape == (e, hidden, expected // group_size)
    if expected == unpadded:
        assert out_w2 is w2 and out_scale is scale
