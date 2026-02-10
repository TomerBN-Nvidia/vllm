import logging
from abc import ABC
from typing import Optional

import numpy as np
import torch
import torch.distributed
from vllm.config.model import ModelConfig


logger = logging.getLogger(__name__)

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024


def get_tensor_size_bytes(t: torch.Tensor):
    return np.prod(t.shape) * t.dtype.itemsize


class _RoutedExpertsDeviceCache:
    """Per-device (GPU) cache for capturing routed expert IDs during forward pass."""

    # Use int16 to save memory - sufficient for up to 32767 experts
    DTYPE = torch.int16

    def __init__(
        self,
        num_batched_tokens: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        num_fused_shared_experts: int,
        device: str,
    ) -> None:
        self.buffer = torch.zeros(
            (num_batched_tokens, num_hidden_layers, num_experts_per_tok),
            dtype=self.DTYPE,
            device=device,
        )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        return get_tensor_size_bytes(self.buffer)

    def capture_fwd_routed_experts(self, layer_id: int, topk_ids: torch.Tensor):
        assert layer_id is not None, (
            "capturing routing experts but get layer_id None"
        )
        batch, _ = topk_ids.shape
        # copy_() handles the dtype cast (e.g. int64 → int16) in a single
        # fused kernel, avoiding a temporary tensor allocation from .to().
        self.buffer[:batch, layer_id, :].copy_(topk_ids, non_blocking=True)

    def _finalize_allocation_log(self):
        buf_mb = self.get_buffer_size_bytes() / _MB
        logger.info(
            f"Routing experts device buffer allocated. "
            f"shape={tuple(self.buffer.shape)}, size={buf_mb:.2f} MB"
        )


class _RoutedExpertsHostCache:
    """Host (CPU) cache using numpy arrays for per-request routing data.

    Numpy arrays avoid torch dispatcher overhead for scatter operations.
    Lazy per-request allocation avoids a massive up-front buffer.
    """

    DTYPE = np.int16

    def __init__(
        self,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        max_running_requests: int,
        max_model_len: int,
        use_shared_memory: bool = True,
    ) -> None:
        self.max_model_len = max_model_len
        self.max_running_requests = max_running_requests
        self.num_hidden_layers = num_hidden_layers
        self.num_experts_per_tok = num_experts_per_tok
        self._use_shared_memory = use_shared_memory

        self._req_buffers: dict[str, np.ndarray] = {}
        self._filled_len: dict[str, int] = {}
        self._total_allocated_bytes = 0

        self._finalize_allocation_log()

    def get_buffer_size_bytes(self) -> int:
        return self._total_allocated_bytes

    def get_or_grow_buffer(self, req_id: str, max_pos: int) -> np.ndarray:
        required_len = max_pos + 1

        if req_id not in self._req_buffers:
            buf = np.zeros(
                (required_len, self.num_hidden_layers, self.num_experts_per_tok),
                dtype=self.DTYPE,
            )
            self._req_buffers[req_id] = buf
            self._total_allocated_bytes += buf.nbytes
            return buf

        buf = self._req_buffers[req_id]
        if buf.shape[0] >= required_len:
            return buf

        new_len = min(max(required_len, buf.shape[0] * 2), self.max_model_len)
        new_buf = np.zeros(
            (new_len, self.num_hidden_layers, self.num_experts_per_tok),
            dtype=self.DTYPE,
        )
        new_buf[: buf.shape[0]] = buf
        self._total_allocated_bytes += new_buf.nbytes - buf.nbytes
        self._req_buffers[req_id] = new_buf
        return new_buf

    def get_buffer(self, req_id: str) -> np.ndarray | None:
        return self._req_buffers.get(req_id)

    def update_filled_len(self, req_id: str, max_pos: int) -> None:
        new_len = max_pos + 1
        self._filled_len[req_id] = max(self._filled_len.get(req_id, 0), new_len)

    def get_filled_len(self, req_id: str) -> int:
        return self._filled_len.get(req_id, 0)

    def free_request(self, req_id: str) -> None:
        if req_id in self._req_buffers:
            self._total_allocated_bytes -= self._req_buffers.pop(req_id).nbytes
        self._filled_len.pop(req_id, None)

    def _finalize_allocation_log(self):
        logger.info(
            f"Routing experts host cache initialized (lazy allocation). "
            f"max_model_len={self.max_model_len}, "
            f"layers={self.num_hidden_layers}, "
            f"experts_per_tok={self.num_experts_per_tok}"
        )


class RoutedExpertsCapturer(ABC):
    @staticmethod
    def create(
        enable: bool,
        model_config: ModelConfig,
        num_fused_shared_experts: int,
        num_batched_tokens: int,
        max_running_requests: int,
        max_model_len: int,
        device: str,
        shared_host_cache: Optional[_RoutedExpertsHostCache] = None,
        skip_host_cache: bool = False,
    ):
        if enable:
            return _RoutedExpertsCapturerReal(
                model_config,
                num_batched_tokens=num_batched_tokens,
                max_running_requests=max_running_requests,
                num_fused_shared_experts=num_fused_shared_experts,
                max_model_len=max_model_len,
                device=device,
                shared_host_cache=shared_host_cache,
                skip_host_cache=skip_host_cache,
            )
        return _RoutedExpertsCapturerNoop()

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        raise NotImplementedError

    def get_routed_experts(
        self, req_id: str, seqlen: Optional[int] = None, free_slot: bool = True
    ):
        raise NotImplementedError

    def sync_fwd_experts_buffer_DtoH(
        self, positions: torch.Tensor, num_scheduled_tokes: dict[str, int],
    ):
        raise NotImplementedError

    def get_host_cache(self):
        raise NotImplementedError

    def get_device_cache(self):
        raise NotImplementedError


class _RoutedExpertsCapturerReal(RoutedExpertsCapturer):
    """Capturer with GPU device cache and optional CPU host cache."""

    def __init__(
        self,
        model_config: ModelConfig,
        num_batched_tokens: int,
        max_running_requests: int,
        num_fused_shared_experts: int,
        max_model_len: int,
        device: str,
        shared_host_cache: Optional[_RoutedExpertsHostCache] = None,
        skip_host_cache: bool = False,
    ):
        self.forward_batch = None
        self.num_fused_shared_experts = num_fused_shared_experts
        self.num_hidden_layers = model_config.hf_text_config.layers_block_type.count("moe")
        self.num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok
        self.num_batched_tokens = num_batched_tokens
        self.max_model_len = max_model_len
        self._skip_host_cache = skip_host_cache

        if skip_host_cache:
            self.host_cache = None
            logger.info(f"Skipping host cache for device {device} (non-rank-0)")
        elif shared_host_cache is not None:
            self.host_cache = shared_host_cache
        else:
            self.host_cache = _RoutedExpertsHostCache(
                max_running_requests=max_running_requests,
                num_hidden_layers=self.num_hidden_layers,
                num_experts_per_tok=self.num_experts_per_tok,
                max_model_len=self.max_model_len,
                use_shared_memory=False,
            )

        self.device_cache = _RoutedExpertsDeviceCache(
            num_batched_tokens=self.num_batched_tokens,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
            num_fused_shared_experts=self.num_fused_shared_experts,
            device=device,
        )

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        self.device_cache.capture_fwd_routed_experts(layer_id, topk_ids)

    def sync_fwd_experts_buffer_DtoH(
        self,
        positions: torch.Tensor,
        num_scheduled_tokes: dict[str, int],
    ):
        if self.host_cache is None:
            return

        total_tokens = sum(num_scheduled_tokes.values())
        if total_tokens == 0:
            return

        # Synchronous D2H copy.
        host_values = self.device_cache.buffer[:total_tokens].cpu().numpy()

        # Scatter into per-request numpy host cache buffers.
        positions_np = positions.numpy()
        offset = 0
        for req_id, n_tokens in num_scheduled_tokes.items():
            if n_tokens == 0:
                continue

            vals = host_values[offset: offset + n_tokens]
            pos = positions_np[offset: offset + n_tokens]
            offset += n_tokens

            max_pos = int(pos.max())
            buf = self.host_cache.get_or_grow_buffer(req_id, max_pos)

            if n_tokens == 1:
                buf[int(pos[0])] = vals[0]
            else:
                buf[pos] = vals

            self.host_cache.update_filled_len(req_id, max_pos)

    def get_routed_experts(
        self, req_id: str, seqlen: int | None = None, free_slot: bool = True,
    ):
        if self.host_cache is None:
            return None
        buf = self.host_cache.get_buffer(req_id)
        if buf is None:
            return None
        result = buf[:seqlen] if seqlen is not None else buf
        if free_slot:
            result = result.copy()
            self.host_cache.free_request(req_id)
        return result

    def get_host_cache(self):
        return self.host_cache

    def get_device_cache(self):
        return self.device_cache


class _RoutedExpertsCapturerNoop(RoutedExpertsCapturer):
    def __init__(self):
        pass

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        pass

    def get_routed_experts(self, req_id: str, seqlen=None, free_slot=True):
        return None

    def sync_fwd_experts_buffer_DtoH(self, positions, num_scheduled_tokes):
        pass

    def get_host_cache(self):
        return None

    def get_device_cache(self):
        pass


# Global capturer instance (per-process)
_global_expert_capturer: Optional[RoutedExpertsCapturer] = _RoutedExpertsCapturerNoop()
_shared_host_cache: Optional[_RoutedExpertsHostCache] = None


def get_global_experts_capturer():
    return _global_expert_capturer


def set_global_experts_capturer(capturer: RoutedExpertsCapturer):
    global _global_expert_capturer
    _global_expert_capturer = capturer


def get_shared_host_cache() -> Optional[_RoutedExpertsHostCache]:
    return _shared_host_cache


def create_shared_host_cache(
    model_config: ModelConfig,
    max_running_requests: int,
    max_model_len: int,
) -> _RoutedExpertsHostCache:
    global _shared_host_cache
    num_hidden_layers = model_config.hf_text_config.layers_block_type.count("moe")
    num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok
    _shared_host_cache = _RoutedExpertsHostCache(
        max_running_requests=max_running_requests,
        num_hidden_layers=num_hidden_layers,
        num_experts_per_tok=num_experts_per_tok,
        max_model_len=max_model_len,
        use_shared_memory=False,
    )
    return _shared_host_cache


def init_routed_experts_capturer_with_shared_cache(
    enable: bool,
    model_config: ModelConfig,
    num_fused_shared_experts: int,
    num_batched_tokens: int,
    max_running_requests: int,
    max_model_len: int,
    device: str,
    rank: int = 0,
    world_size: int = 1,
) -> RoutedExpertsCapturer:
    """Initialize capturer with rank-aware handling (only rank 0 captures)."""
    if not enable:
        capturer = _RoutedExpertsCapturerNoop()
        set_global_experts_capturer(capturer)
        return capturer

    if world_size > 1 and rank != 0:
        logger.info(f"Skipping routed experts capturer for rank {rank}")
        capturer = _RoutedExpertsCapturerNoop()
        set_global_experts_capturer(capturer)
        return capturer

    capturer = RoutedExpertsCapturer.create(
        enable=True,
        model_config=model_config,
        num_fused_shared_experts=num_fused_shared_experts,
        num_batched_tokens=num_batched_tokens,
        max_running_requests=max_running_requests,
        max_model_len=max_model_len,
        device=device,
        skip_host_cache=False,
    )
    set_global_experts_capturer(capturer)
    return capturer
