import logging
from abc import ABC
from contextlib import contextmanager
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
    
    def __init__(
        self,
        num_batched_tokens: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        num_fused_shared_experts: int,  # we don't need this for now
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
    """Shared host (CPU) cache for storing routed expert IDs across all requests.
    
    This cache is designed to be shared across multiple GPU workers/processes
    using torch shared memory.
    """
    
    def __init__(
        self,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        max_running_requests: int,
        max_model_len: int,
        use_shared_memory: bool = True,
    ) -> None:
        self.max_model_len = max_model_len
        self._use_shared_memory = use_shared_memory
        
        self.buffer = torch.zeros(
            (
                max_running_requests,
                max_model_len,
                num_hidden_layers,
                num_experts_per_tok,
            ),
            dtype=torch.int32,
            device="cpu",
            pin_memory=not use_shared_memory,  # Can't pin shared memory
        )
        
        # Note: share_memory_() is disabled as it can cause crashes
        # if use_shared_memory:
        #     self.buffer.share_memory_()
            
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        return get_tensor_size_bytes(self.buffer)

    def set_experts_buffer(self, layer_id: int, loc: torch.Tensor, top_k: torch.Tensor):
        self.buffer[layer_id, loc, :] = top_k.to(device="cpu", non_blocking=True)

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers."""
        buffer_size_GB = self.get_buffer_size_bytes() / _GB
        shared_str = " (shared memory)" if self._use_shared_memory else ""
        logger.info(
            f"Routing experts host buffer allocated{shared_str}. "
            f"#tokens: {self.max_model_len}, size: {buffer_size_GB:.2f} GB"
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
        """Create a routed experts capturer.
        
        Args:
            enable: Whether to enable capturing.
            model_config: The model configuration.
            num_fused_shared_experts: Number of fused shared experts.
            num_batched_tokens: Maximum number of batched tokens.
            max_running_requests: Maximum number of running requests.
            max_model_len: Maximum model length.
            device: The device to use for device cache.
            shared_host_cache: Optional pre-created shared host cache.
                If provided, this cache will be used instead of creating a new one.
            skip_host_cache: If True, don't create host cache (for non-rank-0 workers).
        """
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
    """Capturer for routed experts with per-device GPU cache and optional host cache.
    
    Each GPU device has its own instance with:
    - A per-device _RoutedExpertsDeviceCache (GPU memory)
    - An optional _RoutedExpertsHostCache (CPU memory) - only on rank 0 for multi-GPU
    
    For multi-GPU setups, only rank 0 needs the host cache since all ranks see
    the same routing decisions and only rank 0 needs to return them in outputs.
    """

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
        
        # Host cache handling:
        # - If skip_host_cache=True (non-rank-0 in multi-GPU), don't create host cache
        # - If shared_host_cache provided, use it
        # - Otherwise create a new one (single-GPU or rank 0)
        if skip_host_cache:
            self.host_cache = None
            logger.info(f"Skipping host cache for device {device} (non-rank-0)")
        elif shared_host_cache is not None:
            self.host_cache = shared_host_cache
            logger.info(f"Using provided host cache for device {device}")
        else:
            # Create host cache (for single-GPU or rank 0 in multi-GPU)
            self.host_cache = _RoutedExpertsHostCache(
                max_running_requests=max_running_requests,
                num_hidden_layers=self.num_hidden_layers,
                num_experts_per_tok=self.num_experts_per_tok,
                max_model_len=self.max_model_len,
                use_shared_memory=False,
            )
        # Each device has its own device cache
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
        # Skip if no host cache (non-rank-0 in multi-GPU)
        if self.host_cache is None:
            return
            
        acc_size = 0
        for req_id, num_scheduled_tokens in num_scheduled_tokes.items():
            pos_ids = positions[acc_size:acc_size + num_scheduled_tokens]
            self.host_cache.buffer[int(req_id), pos_ids] = self.device_cache.buffer[acc_size:acc_size + num_scheduled_tokens].cpu()
            acc_size += num_scheduled_tokens

    def get_routed_experts(
       self,
       req_id: int | str,
       seqlen: int | None = None,
    ):  
        if self.host_cache is None:
            return None
        if isinstance(req_id, str):
            req_id = int(req_id)
        return self.host_cache.buffer[req_id, :seqlen]

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


# Global capturer instance (per-process)
_global_expert_capturer: Optional[RoutedExpertsCapturer] = _RoutedExpertsCapturerNoop()

# Shared host cache (created once, shared across processes via shared memory)
_shared_host_cache: Optional[_RoutedExpertsHostCache] = None


def get_global_experts_capturer():
    return _global_expert_capturer


def set_global_experts_capturer(capturer: RoutedExpertsCapturer):
    global _global_expert_capturer
    _global_expert_capturer = capturer


def get_shared_host_cache() -> Optional[_RoutedExpertsHostCache]:
    """Get the shared host cache instance."""
    return _shared_host_cache


def create_shared_host_cache(
    model_config: ModelConfig,
    max_running_requests: int,
    max_model_len: int,
) -> _RoutedExpertsHostCache:
    """Create a shared host cache for routed experts.
    
    This should be called ONCE on rank 0 before workers are initialized.
    The returned cache uses shared memory and can be accessed by all worker processes.
    
    Args:
        model_config: The model configuration.
        max_running_requests: Maximum number of running requests.
        max_model_len: Maximum model length.
    
    Returns:
        A shared _RoutedExpertsHostCache instance.
    """
    global _shared_host_cache
    
    num_hidden_layers = model_config.hf_text_config.layers_block_type.count("moe")
    num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok
    
    _shared_host_cache = _RoutedExpertsHostCache(
        max_running_requests=max_running_requests,
        num_hidden_layers=num_hidden_layers,
        num_experts_per_tok=num_experts_per_tok,
        max_model_len=max_model_len,
        use_shared_memory=False,  # Disabled: share_memory_() can cause crashes
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
    """Initialize routed experts capturer with proper cache handling.
    
    For multi-GPU setups:
    - Only rank 0 creates the host cache (since all ranks see the same routing
      decisions and only rank 0 needs to return them in outputs)
    - Non-rank-0 workers only have device caches and skip host cache operations
    
    Args:
        enable: Whether to enable capturing.
        model_config: The model configuration.
        num_fused_shared_experts: Number of fused shared experts.
        num_batched_tokens: Maximum number of batched tokens.
        max_running_requests: Maximum number of running requests.
        max_model_len: Maximum model length.
        device: The device string (e.g., "cuda:0").
        rank: The current process rank.
        world_size: Total number of processes.
    
    Returns:
        A RoutedExpertsCapturer instance.
    """
    if not enable:
        return _RoutedExpertsCapturerNoop()
    
    # For multi-GPU: only rank 0 needs host cache
    # Non-rank-0 workers skip host cache entirely (they only capture to device cache)
    skip_host_cache = (world_size > 1 and rank != 0)
    
    # Create the capturer
    capturer = RoutedExpertsCapturer.create(
        enable=True,
        model_config=model_config,
        num_fused_shared_experts=num_fused_shared_experts,
        num_batched_tokens=num_batched_tokens,
        max_running_requests=max_running_requests,
        max_model_len=max_model_len,
        device=device,
        skip_host_cache=skip_host_cache,
    )
    
    set_global_experts_capturer(capturer)
    return capturer
