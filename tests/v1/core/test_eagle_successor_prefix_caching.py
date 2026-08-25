# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for successor-aware colocated EAGLE prefix caching."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    get_request_eagle_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import create_scheduler

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

BLOCK_SIZE = 4


@pytest.fixture(autouse=True)
def _init_hash() -> None:
    init_none_hash(sha256)


def _make_request(
    request_id: str,
    token_ids: list[int],
    *,
    eagle: bool = True,
    successor_mm_id: str | None = None,
) -> Request:
    sampling_params = SamplingParams(ignore_eos=True, max_tokens=4)
    mm_features = None
    if successor_mm_id is not None:
        mm_features = [
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                mm_position=PlaceholderRange(offset=BLOCK_SIZE, length=1),
                identifier=successor_mm_id,
                modality="image",
            )
        ]
    hasher_factory = (
        get_request_eagle_block_hasher if eagle else get_request_block_hasher
    )
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=mm_features,
        block_hasher=hasher_factory(BLOCK_SIZE, sha256),
    )


def _full_attention_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )


def _make_manager(*, hybrid_mamba: bool = False) -> KVCacheManager:
    groups = [
        KVCacheGroupSpec(
            ["target_and_draft_attention"],
            _full_attention_spec(),
            is_eagle_group=True,
        )
    ]
    if hybrid_mamba:
        groups.append(
            KVCacheGroupSpec(
                ["target_mamba"],
                MambaSpec(
                    block_size=BLOCK_SIZE,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                ),
            )
        )
    return KVCacheManager(
        KVCacheConfig(num_blocks=64, kv_cache_tensors=[], kv_cache_groups=groups),
        max_model_len=128,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=True,
        use_eagle=True,
        use_eagle_prefix_cache_hashing=True,
    )


def _allocate_and_publish(manager: KVCacheManager, request: Request) -> None:
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(request)
    blocks = manager.allocate_slots(
        request,
        request.num_tokens - num_computed_tokens,
        num_computed_tokens,
        computed_blocks,
    )
    assert blocks is not None
    request.mark_eagle_hashes_publishable(request.num_tokens, BLOCK_SIZE)
    manager.cache_blocks(request, request.num_tokens)


def test_identical_successor_reuses_final_safe_unit() -> None:
    manager = _make_manager()
    first = _make_request("first", [1, 2, 3, 4, 5])
    _allocate_and_publish(manager, first)

    repeated = _make_request("repeated", [1, 2, 3, 4, 5])
    blocks, num_computed_tokens = manager.get_computed_blocks(repeated)

    assert num_computed_tokens == BLOCK_SIZE
    assert len(blocks.blocks[0]) == 1


def test_different_successor_falls_back_conservatively() -> None:
    manager = _make_manager()
    first = _make_request("first", [1, 2, 3, 4, 5])
    _allocate_and_publish(manager, first)

    changed = _make_request("changed", [1, 2, 3, 4, 6])
    blocks, num_computed_tokens = manager.get_computed_blocks(changed)

    assert num_computed_tokens == 0
    assert not blocks.blocks[0]


def test_exact_boundary_without_successor_remains_pending() -> None:
    request = _make_request("boundary", [1, 2, 3, 4])
    assert request.block_hashes == []

    request.append_output_token_ids(5)
    assert len(request.block_hashes) == 1


def test_successor_input_identity_is_part_of_hash() -> None:
    first = _make_request("first", [1, 2, 3, 4, 0], successor_mm_id="image-a")
    changed = _make_request("changed", [1, 2, 3, 4, 0], successor_mm_id="image-b")

    assert len(first.block_hashes) == len(changed.block_hashes) == 1
    assert first.block_hashes != changed.block_hashes


def test_hybrid_mamba_align_uses_successor_hash_without_eagle_drop() -> None:
    manager = _make_manager(hybrid_mamba=True)
    first = _make_request("first", [1, 2, 3, 4, 5])
    _allocate_and_publish(manager, first)

    repeated = _make_request("repeated", [1, 2, 3, 4, 5])
    blocks, num_computed_tokens = manager.get_computed_blocks(repeated)

    assert num_computed_tokens == BLOCK_SIZE
    assert [len(group) for group in blocks.blocks] == [1, 1]


def _run_first_scheduler_step(*, acknowledge_draft_kv: bool) -> Request:
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
        max_model_len=128,
        max_num_batched_tokens=128,
    )
    scheduler.use_eagle = True
    scheduler.use_eagle_prefix_cache_hashing = True
    scheduler.kv_cache_manager.use_eagle_prefix_cache_hashing = True
    scheduler.kv_cache_manager.coordinator.use_eagle_prefix_cache_hashing = True

    request = _make_request("request", [1, 2, 3, 4, 5])
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    runner_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[9]],
        prompt_logprobs_dict={},
        pooler_output=[],
        draft_kv_materialized_req_ids=(
            {request.request_id} if acknowledge_draft_kv else None
        ),
    )
    scheduler.update_from_output(scheduler_output, runner_output)
    return request


def test_publication_waits_for_draft_kv_materialization() -> None:
    unacknowledged = _run_first_scheduler_step(acknowledge_draft_kv=False)
    acknowledged = _run_first_scheduler_step(acknowledge_draft_kv=True)

    assert unacknowledged.num_publishable_block_hashes == 0
    assert unacknowledged.num_materialized_eagle_tokens == 0
    assert acknowledged.num_publishable_block_hashes == 1
    assert acknowledged.num_materialized_eagle_tokens == 5


def test_async_publication_requires_contiguous_acknowledged_steps() -> None:
    scheduler = create_scheduler(
        async_scheduling=True,
        max_num_seqs=1,
        enable_prefix_caching=True,
        block_size=2,
        max_num_batched_tokens=3,
        max_model_len=16,
    )
    scheduler.use_eagle_prefix_cache_hashing = True
    scheduler.kv_cache_manager.use_eagle_prefix_cache_hashing = True
    scheduler.kv_cache_manager.coordinator.use_eagle_prefix_cache_hashing = True

    def make_request(request_id: str) -> Request:
        return Request(
            request_id=request_id,
            prompt_token_ids=list(range(7)),
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_hasher=get_request_eagle_block_hasher(2, sha256),
        )

    request = make_request("request")
    scheduler.add_request(request)
    first_step = scheduler.schedule()
    second_step = scheduler.schedule()

    def output(step: SchedulerOutput, *, acknowledged: bool) -> ModelRunnerOutput:
        req_ids = list(step.num_scheduled_tokens)
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[[]],
            prompt_logprobs_dict={},
            pooler_output=[],
            draft_kv_materialized_req_ids=(
                {request.request_id} if acknowledged else None
            ),
        )

    scheduler.update_from_output(first_step, output(first_step, acknowledged=True))
    assert request.num_materialized_eagle_tokens == 3
    probe = make_request("probe")
    _, num_cached_tokens = scheduler.kv_cache_manager.get_computed_blocks(probe)
    assert num_cached_tokens == 2

    scheduler.update_from_output(second_step, output(second_step, acknowledged=False))
    assert request.num_materialized_eagle_tokens == 3
    _, num_cached_tokens = scheduler.kv_cache_manager.get_computed_blocks(probe)
    assert num_cached_tokens == 2

    third_step = scheduler.schedule()
    scheduler.update_from_output(third_step, output(third_step, acknowledged=True))
    assert request.num_materialized_eagle_tokens == 3
    _, num_cached_tokens = scheduler.kv_cache_manager.get_computed_blocks(probe)
    assert num_cached_tokens == 2


def test_stale_async_output_does_not_restore_materialization() -> None:
    scheduler = create_scheduler(
        async_scheduling=True,
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=32,
    )
    scheduler.use_eagle_prefix_cache_hashing = True
    scheduler.kv_cache_manager.use_eagle_prefix_cache_hashing = True
    scheduler.kv_cache_manager.coordinator.use_eagle_prefix_cache_hashing = True
    request = _make_request("eagle", list(range(9)))
    scheduler.add_request(request)

    first_step = scheduler.schedule()
    first_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[9]],
        prompt_logprobs_dict={},
        pooler_output=[],
        draft_kv_materialized_req_ids={request.request_id},
    )
    scheduler.update_from_output(first_step, first_output)
    assert request.num_materialized_eagle_tokens > 0

    in_flight_step = scheduler.schedule()
    scheduler.reset_prefix_cache(reset_running_requests=True)
    assert request.num_materialized_eagle_tokens == 0
    assert request.async_tokens_to_discard > 0

    stale_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[10]],
        prompt_logprobs_dict={},
        pooler_output=[],
        draft_kv_materialized_req_ids={request.request_id},
    )
    scheduler.update_from_output(in_flight_step, stale_output)
    assert request.async_tokens_to_discard == 0
    assert request.num_publishable_block_hashes == 0
    assert request.num_materialized_eagle_tokens == 0


def test_truncation_and_preemption_invalidate_readiness() -> None:
    request = _make_request("request", list(range(9)))
    request.mark_eagle_hashes_publishable(9, BLOCK_SIZE)
    assert request.num_publishable_block_hashes == 2

    request.truncate_block_hashes(5, BLOCK_SIZE, lookahead_tokens=1)
    assert len(request.block_hashes) == 1
    assert request.num_publishable_block_hashes == 1
    assert request.num_materialized_eagle_tokens == 5

    request.status = RequestStatus.RUNNING
    scheduler = SimpleNamespace(
        _free_request_blocks=Mock(),
        encoder_cache_manager=Mock(),
        _inflight_prefills=set(),
        log_stats=False,
        waiting=Mock(),
        reset_preempted_req_ids=set(),
    )
    Scheduler._preempt_request(scheduler, request, 0.0)
    assert request.num_publishable_block_hashes == 0
    assert request.num_materialized_eagle_tokens == 0


def test_non_eagle_hashing_is_unchanged() -> None:
    exact_boundary = _make_request("boundary", [1, 2, 3, 4], eagle=False)
    first = _make_request("first", [1, 2, 3, 4, 5], eagle=False)
    changed_successor = _make_request("changed", [1, 2, 3, 4, 6], eagle=False)

    assert len(exact_boundary.block_hashes) == 1
    assert first.block_hashes == changed_successor.block_hashes


def test_successor_hashing_removes_legacy_mamba_backoff() -> None:
    request = _make_request("request", list(range(55)))
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        use_eagle=True,
        use_eagle_prefix_cache_hashing=True,
    )

    assert (
        Scheduler._mamba_block_aligned_split(
            scheduler,
            request,
            num_new_tokens=50,
        )
        == 48
    )
