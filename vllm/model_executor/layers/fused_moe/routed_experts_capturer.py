import logging
from abc import ABC
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
from vllm.config.model import ModelConfig


logger = logging.getLogger(__name__)

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024


def get_tensor_size_bytes(t: torch.Tensor):
    return np.prod(t.shape) * t.dtype.itemsize


class _RoutedExpertsDeviceCache:
    def __init__(
        self,
        num_batched_tokens: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        num_fused_shared_experts: int, #we don't need this for now
        device: str,
    ) -> None:
        self.buffer = torch.zeros(
            (
                num_batched_tokens,
                num_hidden_layers,
                num_experts_per_tok,
            ),
            dtype=torch.int32,
            device=device,
        )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        return get_tensor_size_bytes(self.buffer)

    def capture_fwd_routed_experts(self, layer_id: int, topk_ids: torch.Tensor):
        assert layer_id is not None, "capturing routing experts but get layer_id None"
        batch, _ = topk_ids.shape
        self.buffer[:batch, layer_id, :].copy_(topk_ids, non_blocking=True)

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers."""
        buffer_size_MB = self.get_buffer_size_bytes() / _MB
        logger.info(
            f"Routing experts device buffer allocated. #shape: {tuple(self.buffer.shape)}, size: {buffer_size_MB:.2f} MB"
        )


class _RoutedExpertsHostCache:
    def __init__(
        self,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        max_running_requests: int,
        max_model_len: int,
    ) -> None:
        self.max_model_len = max_model_len
        
        self.buffer = torch.zeros(
            (
                max_running_requests,
                max_model_len,
                num_hidden_layers,
                num_experts_per_tok,
            ),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        return get_tensor_size_bytes(self.buffer)

    def set_experts_buffer(self, layer_id: int, loc: torch.Tensor, top_k: torch.Tensor):
        self.buffer[layer_id, loc, :] = top_k.to(device="cpu", non_blocking=True)

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers."""
        buffer_size_GB = self.get_buffer_size_bytes() / _GB
        logger.info(
            f"Routing experts host buffer allocated. #tokens: {self.max_model_len}, size: {buffer_size_GB:.2f} GB"
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
    ):
        if enable:
            return _RoutedExpertsCapturerReal(
                model_config,
                num_batched_tokens=num_batched_tokens,
                max_running_requests=max_running_requests,
                num_fused_shared_experts=num_fused_shared_experts,
                max_model_len=max_model_len,
                device=device,
            )
        else:
            return _RoutedExpertsCapturerNoop()

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        raise NotImplementedError
    
    def get_routed_experts(
        self,
        token_indices: torch.Tensor,
        seqlen: Optional[int] = None,
    ):
        raise NotImplementedError

    def sync_fwd_experts_buffer_DtoH(
        self,
        positions: torch.Tensor,
        num_scheduled_tokes: dict[str, int],
    ):
        raise NotImplementedError

    def get_host_cache(self):
        raise NotImplementedError

    def get_device_cache(self):
        raise NotImplementedError


class _RoutedExpertsCapturerReal(RoutedExpertsCapturer):
    """Capturer for routed experts with host buffer (Singleton)"""

    _instance: Optional["_RoutedExpertsCapturerReal"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_config: ModelConfig,
        num_batched_tokens: int,
        max_running_requests: int,
        num_fused_shared_experts: int,
        max_model_len: int,
        device: str,
    ):
        # Skip re-initialization if already initialized
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.forward_batch = None
        self.num_fused_shared_experts = num_fused_shared_experts
        self.num_hidden_layers = model_config.hf_text_config.layers_block_type.count("moe")
        self.num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok
        self.num_batched_tokens = num_batched_tokens
        self.max_model_len = max_model_len
        
        self.host_cache = _RoutedExpertsHostCache(
            max_running_requests=max_running_requests,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
            max_model_len=self.max_model_len,
        )

        self.device_cache = _RoutedExpertsDeviceCache(
            num_batched_tokens=self.num_batched_tokens,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
            num_fused_shared_experts=self.num_fused_shared_experts,
            device=device,
        )

        self._initialized = True

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        self.device_cache.capture_fwd_routed_experts(layer_id, topk_ids)

    def sync_fwd_experts_buffer_DtoH(
        self,
        positions: torch.Tensor,
        num_scheduled_tokes: dict[str, int],
    ):
        acc_size = 0
        for req_id, num_scheduled_tokens in num_scheduled_tokes.items():
            pos_ids = positions[acc_size:acc_size + num_scheduled_tokens]
            self.host_cache.buffer[int(req_id), pos_ids] = self.device_cache.buffer[acc_size:acc_size + num_scheduled_tokens].cpu()
            acc_size += num_scheduled_tokens


    def get_routed_experts(
       self,
       req_pool_idx: int,
       seqlen: int | None = None,
    ):  
       return self.get_host_cache().buffer[req_pool_idx, :seqlen]

    def get_host_cache(self):
        return self.host_cache

    def get_device_cache(self):
        return self.device_cache


class _RoutedExpertsCapturerNoop(RoutedExpertsCapturer):
    def __init__(self):
        pass

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        pass

    def get_routed_experts(
        self,
        token_indices: torch.Tensor,
        seqlen: Optional[int] = None,
    ):
        pass

    def sync_fwd_experts_buffer_DtoH(
        self,
        positions: torch.Tensor,
        num_scheduled_tokes: dict[str, int],
    ):
        pass

    def get_host_cache(self):
        pass

    def get_device_cache(self):
        pass


_global_expert_capturer: Optional[RoutedExpertsCapturer] = _RoutedExpertsCapturerNoop()


def get_global_experts_capturer():
    return _global_expert_capturer


def set_global_experts_capturer(capturer: RoutedExpertsCapturer):
    global _global_expert_capturer
    _global_expert_capturer = capturer
