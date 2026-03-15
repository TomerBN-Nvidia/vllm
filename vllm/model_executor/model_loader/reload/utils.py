# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from .types import LayerTensors

__all__ = [
    "get_layer_tensors",
    "get_layer_params_buffers",
    "get_layer_size",
]


def get_layer_tensors(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Get all parameters and buffers from a module as a dict."""
    params, buffers = get_layer_params_buffers(layer)
    return params | buffers


def get_layer_params_buffers(layer: torch.nn.Module) -> LayerTensors:
    """Get all parameters and buffers of a module as a tuple of dicts."""
    params, buffers = (
        {name: param for name, param in layer._parameters.items() if param is not None},
        {name: buffer for name, buffer in layer._buffers.items() if buffer is not None},
    )
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    if isinstance(layer, FusedMoE) and "e_score_correction_bias" in params:
        del params["e_score_correction_bias"]
    if isinstance(layer, FusedMoE) and "_expert_map" in buffers:
        del buffers["_expert_map"]
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
    if isinstance(layer, MambaMixer2) and "conv_weights" in buffers:
        del buffers["conv_weights"]
    return params, buffers


def get_layer_size(layer: torch.nn.Module) -> int:
    """Calculate total number of elements across all tensors in a layer."""
    return sum(tensor.numel() for tensor in get_layer_tensors(layer).values())
