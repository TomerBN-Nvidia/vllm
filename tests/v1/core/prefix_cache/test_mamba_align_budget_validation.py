# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" mode schedules prefill in whole blocks, so a resolved mamba
block that exceeds the effective per-step budget can never make progress. The
scheduler enforces this at init, where the resolved block size first exists:
it raises when the budget (max_num_scheduled_tokens, falling back to
max_num_batched_tokens) cannot hold one block, and clamps a smaller
long_prefill_token_threshold up to the block size with a warning."""

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_request
from tests.v1.core.utils import create_scheduler
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _align_scheduler(
    block_size: int,
    max_num_batched_tokens: int = 8192,
    max_num_scheduled_tokens: int | None = None,
    long_prefill_token_threshold: int = 0,
):
    return create_scheduler(
        block_size=block_size,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_scheduled_tokens=max_num_scheduled_tokens,
        long_prefill_token_threshold=long_prefill_token_threshold,
        max_model_len=40960,
        mamba_cache_mode="align",
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                MambaSpec(
                    block_size=block_size,
                    shapes=((1,),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            )
        ],
    )


def test_raise_when_block_exceeds_max_num_batched_tokens():
    """A mamba block larger than a 2048 prefill budget (120B TP4 shape;
    the resolved block on this branch is 2080)."""
    with pytest.raises(ValueError, match="2112"):
        _align_scheduler(block_size=2112, max_num_batched_tokens=2048)


def test_raise_when_block_exceeds_max_num_scheduled_tokens():
    """The effective budget can sit below max_num_batched_tokens (speculative
    decoding reserves draft slots). The old config-time assert compared only
    max_num_batched_tokens, so this geometry booted and then hung."""
    with pytest.raises(ValueError, match="2112"):
        _align_scheduler(
            block_size=2112,
            max_num_batched_tokens=8192,
            max_num_scheduled_tokens=2048,
        )


def test_config_validator_no_longer_asserts_size():
    """The size check moved out of VllmConfig.validate_block_size, which
    compared only max_num_batched_tokens rather than the effective per-step
    budget, into Scheduler.__init__. The validator must accept this
    geometry."""
    from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig

    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model="facebook/opt-125m",
            trust_remote_code=True,
            dtype="float16",
            seed=42,
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=2048,
            max_model_len=2048,
            is_encoder_decoder=False,
        ),
        cache_config=CacheConfig(
            block_size=2112,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        ),
    )
    vllm_config.validate_block_size()


def test_threshold_clamped_with_warning(caplog_vllm):
    scheduler = _align_scheduler(block_size=512, long_prefill_token_threshold=384)
    assert scheduler.scheduler_config.long_prefill_token_threshold == 512
    assert "clamping" in caplog_vllm.text


def test_threshold_at_block_size_untouched(caplog_vllm):
    scheduler = _align_scheduler(block_size=512, long_prefill_token_threshold=512)
    assert scheduler.scheduler_config.long_prefill_token_threshold == 512
    assert "clamping" not in caplog_vllm.text


@pytest.mark.parametrize("budget", [8192, 4096, 2112])
def test_block_aligned_chunks_unchanged_when_block_fits(budget):
    """Whenever a whole block fits, chunk ends stay on block boundaries."""
    block_size = 2112
    scheduler = _align_scheduler(block_size=block_size, max_num_batched_tokens=budget)
    req = make_request("0", [0] * 30000, 16, sha256)
    chunks = []
    while req.num_computed_tokens < 30000:
        n = scheduler._mamba_block_aligned_split(
            request=req,
            num_new_tokens=min(budget, 30000 - req.num_computed_tokens),
        )
        assert n > 0
        chunks.append(n)
        req.num_computed_tokens += n
    assert all(c % block_size == 0 for c in chunks[:-1]), chunks
    assert sum(chunks) == 30000


def test_transient_subblock_leftover_returns_zero():
    """Mid-step leftover budget below one block floors to zero; the scheduler
    skips the request for the step and retries. That contract is unchanged."""
    scheduler = _align_scheduler(block_size=2112)
    req = make_request("0", [0] * 30000, 16, sha256)
    assert scheduler._mamba_block_aligned_split(request=req, num_new_tokens=300) == 0
