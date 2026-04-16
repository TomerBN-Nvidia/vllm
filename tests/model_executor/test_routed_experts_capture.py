# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

pytestmark = pytest.mark.cpu_test

_REC_MODULE = "vllm.model_executor.layers.fused_moe.routed_experts_capturer"


def _capturer_with_buffer(
    *,
    max_tokens: int = 8,
    num_layers: int = 4,
    num_experts_per_tok: int = 2,
    dp_rank: int = 0,
) -> RoutedExpertsCapturer:
    c = RoutedExpertsCapturer()
    c.dp_rank = dp_rank
    c._device_buffer = torch.full(
        (max_tokens, num_layers, num_experts_per_tok),
        -1,
        dtype=torch.int32,
    )
    return c


class DummyRouter(BaseRouter):
    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.FUSED_TOPK

    def _compute_routing(self, hidden_states, router_logits, indices_type):
        topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        return topk_weights, topk_ids

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # Make mapping observable without requiring CUDA EPLB path.
        return topk_ids + 10


def _make_router() -> DummyRouter:
    return DummyRouter(
        top_k=2,
        global_num_experts=16,
        eplb_state=EplbLayerState(),
        enable_eplb=False,
        indices_type_getter=None,
    )


def test_base_router_capture_pre_eplb_mapping():
    router = _make_router()
    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert topk_weights.shape == topk_ids.shape
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_base_router_capture_with_eplb_enabled():
    router = _make_router()
    router.enable_eplb = True
    router.eplb_state.expert_load_view = torch.zeros(32, dtype=torch.int64)
    router.eplb_state.logical_to_physical_map = torch.arange(32).view(32, 1)
    router.eplb_state.logical_replica_count = torch.ones(32, dtype=torch.int64)
    router.eplb_state.should_record_tensor = torch.ones((), dtype=torch.bool)

    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    _, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert len(captured) == 1
    # Capture should see logical ids pre-EPLB mapping.
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    # Our DummyRouter mapping adds +10.
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_gpu_model_runner_binds_router_capture(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 7
            self.router = _make_router()

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids):
            self.calls.append((layer_id, topk_ids))

    dummy_module = DummyFusedMoE()

    # Patch the runtime import inside _bind_routed_experts_capturer.
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "FusedMoE", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            static_forward_context={"dummy": dummy_module}
        )
    )

    capturer = DummyCapturer()
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    assert dummy_module.router.capture_fn is not None
    dummy_module.router.capture_fn(torch.tensor([[5, 6]]))

    assert len(capturer.calls) == 1
    layer_id, topk_ids = capturer.calls[0]
    assert layer_id == 7
    assert torch.equal(topk_ids, torch.tensor([[5, 6]]))


def test_gpu_model_runner_binding_stage(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 11
            self.router = _make_router()

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids):
            self.calls.append((layer_id, topk_ids))

    dummy_module = DummyFusedMoE()

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "FusedMoE", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            static_forward_context={"dummy": dummy_module}
        )
    )

    # Before binding, no capture hook.
    assert dummy_module.router.capture_fn is None

    capturer = DummyCapturer()
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    # After binding, hook should exist and be callable.
    assert callable(dummy_module.router.capture_fn)
    dummy_module.router.capture_fn(torch.tensor([[9, 10]]))
    assert len(capturer.calls) == 1


# =========================================================================
# Tests for device-cache routing replay architecture
# =========================================================================


class TestRoutedExpertsDeviceCache:
    """Tests for _RoutedExpertsDeviceCache (GPU buffer for routing data)."""

    def test_allocation_shape_and_dtype(self):
        """Device cache allocates (L, N, K) int16 buffer."""
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsDeviceCache,
        )

        cache = _RoutedExpertsDeviceCache(
            num_hidden_layers=40,
            max_num_batched_tokens=8192,
            num_experts_per_tok=8,
        )
        assert cache.buffer.shape == (40, 8192, 8)
        assert cache.buffer.dtype == torch.int16

    def test_per_layer_view_is_contiguous(self):
        """buffer[layer_id] gives contiguous (N, K) view for FlashInfer."""
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsDeviceCache,
        )

        cache = _RoutedExpertsDeviceCache(
            num_hidden_layers=40,
            max_num_batched_tokens=8192,
            num_experts_per_tok=8,
        )
        layer_view = cache.buffer[0]
        assert layer_view.is_contiguous()
        assert layer_view.shape == (8192, 8)


class TestRoutedExpertsHostCache:
    """Tests for _RoutedExpertsHostCache (per-request numpy buffer)."""

    def test_sentinel_initialization(self):
        """Host cache initializes with zeros by default."""
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsHostCache,
        )
        import numpy as np

        cache = _RoutedExpertsHostCache(
            num_hidden_layers=40,
            num_experts_per_tok=8,
        )
        buf = cache.get_or_grow_buffer("req1", max_pos=100)
        assert buf.dtype == np.int16
        assert (buf == 0).all(), "Host cache must initialize with zeros"

    def test_grow_preserves_existing_data(self):
        """Growing the buffer preserves previously written data."""
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsHostCache,
        )

        cache = _RoutedExpertsHostCache(
            num_hidden_layers=40,
            num_experts_per_tok=8,
        )
        buf = cache.get_or_grow_buffer("req1", max_pos=50)
        buf[0, 0, 0] = 42
        buf2 = cache.get_or_grow_buffer("req1", max_pos=200)
        assert buf2[0, 0, 0] == 42, "Data lost during buffer grow"

    def test_free_request_removes_buffer(self):
        """Freeing a request removes its buffer."""
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsHostCache,
        )

        cache = _RoutedExpertsHostCache(
            num_hidden_layers=40,
            num_experts_per_tok=8,
        )
        cache.get_or_grow_buffer("req1", max_pos=50)
        cache.free_request("req1")
        assert cache.get_buffer("req1") is None

