# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Training video prompts and metadata through the real multimodal processor."""

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import TokenizersBackend

from vllm.config import ModelConfig
from vllm.models.nemotron_shortcut.nvidia.vision import (
    NemotronShortcutDummyInputsBuilder,
    get_video_temporal_patch_size,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.hasher import MultiModalHasher
from vllm.transformers_utils.configs.nemotron_shortcut import NemotronShortcutConfig
from vllm.transformers_utils.processors.nemotron_shortcut import (
    NemotronShortcutImageProcessor,
    NemotronShortcutProcessor,
)


@pytest.fixture
def processor():
    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(
            WordLevel(
                {
                    "<unk>": 0,
                    "<img>": 1,
                    "<image_break>": 2,
                    "</img>": 3,
                    "<video>": 4,
                },
                unk_token="<unk>",
            )
        ),
        unk_token="<unk>",
        additional_special_tokens=["<img>", "<image_break>", "</img>", "<video>"],
    )
    return NemotronShortcutProcessor(
        NemotronShortcutImageProcessor(),
        tokenizer,
        patch_size=14,
        spatial_merge_size=2,
        image_token="<img>",
        image_break_token="<image_break>",
        image_end_token="</img>",
    )


def _mm_processor(
    tmp_path,
    processor,
    temporal_patch_size,
    num_frames=None,
    image_token_id=1,
    video_kwargs=None,
):
    processor.tokenizer.save_pretrained(tmp_path)
    extra = (
        {}
        if temporal_patch_size is None
        else {"video_temporal_patch_size": temporal_patch_size}
    )
    config = NemotronShortcutConfig(
        text_config={
            "layer_types": ["mlp"],
            "hidden_size": 8,
            "vocab_size": max(5, len(processor.tokenizer)),
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 8,
        },
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "patch_size": 14,
            "image_size": 56,
        },
        image_token_id=image_token_id,
        spatial_merge_size=2,
        architectures=["Nemotron4ForConditionalGeneration"],
        **extra,
    )
    config.save_pretrained(tmp_path)
    video_io_kwargs = dict(video_kwargs or {})
    if num_frames is not None:
        video_io_kwargs["num_frames"] = num_frames
    model_config = ModelConfig(
        str(tmp_path),
        max_model_len=4096,
        limit_mm_per_prompt={"image": 64, "video": 1},
        media_io_kwargs={"video": video_io_kwargs} if video_io_kwargs else None,
    )
    assert get_video_temporal_patch_size(model_config.hf_config) == (
        temporal_patch_size or 1
    )
    return MULTIMODAL_REGISTRY.create_processor(model_config)


def _video(num_frames):
    return np.stack(
        [
            np.full((56, 56, 3), (255 - i, 0, i), dtype=np.uint8)
            for i in range(num_frames)
        ]
    )


def _metadata(num_frames):
    return {
        "fps": 2.0,
        "frames_indices": list(range(0, 3 * num_frames, 3)),
        "total_num_frames": 3 * num_frames,
    }


@pytest.mark.parametrize("temporal_patch_size", [None, 2])
def test_processor_cache_keeps_video_timestamps(
    tmp_path, processor, temporal_patch_size
):
    """A served request goes through the processor cache (on by default).

    The cache rebuilds the items that miss it from their frames only; the
    loader metadata must survive, or the prompt falls back to "frame i at i
    seconds" (for every T: T = 1 labels each frame too). The first call misses
    the cache and the second one hits it. Both must give the prompt of the
    processor without a cache.
    """
    from vllm.multimodal.cache.base import MultiModalProcessorOnlyCache

    mm_processor = _mm_processor(tmp_path, processor, temporal_patch_size)
    cache = MultiModalProcessorOnlyCache(mm_processor.info.ctx.model_config)
    frames = _video(4)

    def run(cache):
        return mm_processor(
            "<video>",
            mm_items=mm_processor.info.parse_mm_data(
                {"video": [(frames, _metadata(4))]}
            ),
            cache=cache,
        )

    reference = run(None)
    for _ in range(2):
        result = run(cache)
        assert result["prompt_token_ids"] == reference["prompt_token_ids"]
        item = result["mm_kwargs"]["video"][0]
        # _metadata(4): fps 2, source frames 0, 3, 6, 9.
        assert item["video_timestamps"].data.tolist() == [0.0, 1.5, 3.0, 4.5]


@pytest.mark.parametrize("temporal_patch_size", [None, 2])
def test_cache_key_includes_video_metadata(tmp_path, processor, temporal_patch_size):
    """Same raw frames with a different fps must not share cached timestamps."""
    mm_processor = _mm_processor(tmp_path, processor, temporal_patch_size)
    frames = _video(4)
    hashes = []
    for fps in (2.0, 4.0):
        metadata = dict(_metadata(4), fps=fps)
        items = mm_processor.info.parse_mm_data({"video": [(frames, metadata)]})
        [key] = items["video"].get_all_items_for_hash()
        hashes.append(MultiModalHasher.hash_kwargs("sha256", video=key))
    assert hashes[0] != hashes[1]
    # Without metadata the key is the frames alone, as before.
    items = mm_processor.info.parse_mm_data({"video": [frames]})
    [key] = items["video"].get_all_items_for_hash()
    assert key is frames or np.array_equal(key, frames)


@pytest.mark.parametrize("temporal_patch_size", [None, 1])
def test_t1_processor_labels_every_frame(tmp_path, processor, temporal_patch_size):
    """T = 1 builds the training prompt too: one labelled block per frame."""
    mm_processor = _mm_processor(tmp_path, processor, temporal_patch_size)
    result = mm_processor(
        "<video>",
        mm_items=mm_processor.info.parse_mm_data(
            {"video": [(_video(5), _metadata(5))]}
        ),
    )
    item = result["mm_kwargs"]["video"][0]
    # _metadata(5): fps 2, source frames 0, 3, 6, 9, 12.
    assert item["video_timestamps"].data.tolist() == [0.0, 1.5, 3.0, 4.5, 6.0]
    assert len(item["pixel_values_videos"].data) == 5
    tokens = result["prompt_token_ids"]
    assert tokens.count(1) == 4 * 5
    assert tokens.count(3) == 5
    # Without metadata the labels fall back to the frame positions.
    result = mm_processor(
        "<video>",
        mm_items=mm_processor.info.parse_mm_data({"video": [_video(3)]}),
    )
    item = result["mm_kwargs"]["video"][0]
    assert item["video_timestamps"].data.tolist() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize("temporal_patch_size", [1, 2])
def test_dummy_video_fits_the_profile(tmp_path, processor, temporal_patch_size):
    mm_processor = _mm_processor(tmp_path, processor, temporal_patch_size, 7)
    builder = NemotronShortcutDummyInputsBuilder(mm_processor.info)
    # 1036 x 1036 frames: 1406 tokens per frame grid; the frame cap (7) binds.
    data = builder.get_dummy_mm_data(100_000, {"video": 1}, {})
    [video] = data["video"]
    if temporal_patch_size == 1:
        assert len(video) == 7
    else:
        assert len(video) == 6
        # A real video with more frames than the cap keeps 6 frames too, so
        # it fits the encoder cache sized from this profile.
        result = mm_processor(
            "<video>",
            mm_items=mm_processor.info.parse_mm_data({"video": [_video(9)]}),
        )
        pixels = result["mm_kwargs"]["video"][0]["pixel_values_videos"].data
        assert len(pixels) == 6


@pytest.mark.parametrize("temporal_patch_size", [1, 2])
def test_training_loader_cap_and_labels(tmp_path, processor, temporal_patch_size):
    """Loader labels survive processing and its frame cap sizes the profile."""
    mm_processor = _mm_processor(
        tmp_path,
        processor,
        temporal_patch_size,
        video_kwargs={
            "video_backend": "nemotron_shortcut_t2",
            "max_frames": 32,
            "temporal_patch_size": temporal_patch_size,
        },
    )
    assert mm_processor.info.get_max_video_frames() == 32
    metadata = dict(
        _metadata(4),
        frames_indices=[0, 1, 2, 3],
        frame_source_indices=[0, 1, 2, 3],
        frame_timestamps=[0.25, 0.75, 1.25, 1.75],
        frame_temporal_patch_size=temporal_patch_size,
    )
    result = mm_processor(
        "<video>",
        mm_items=mm_processor.info.parse_mm_data({"video": [(_video(4), metadata)]}),
    )
    item = result["mm_kwargs"]["video"][0]
    assert item["video_timestamps"].data.tolist() == metadata["frame_timestamps"]


def test_t1_rejects_t2_loader_metadata(tmp_path, processor):
    """A loader cannot silently change the checkpoint's temporal patch size."""
    mm_processor = _mm_processor(tmp_path, processor, 1)
    metadata = dict(
        _metadata(4),
        frame_source_indices=[0, 3, 6, 9],
        frame_timestamps=[0.25, 0.75, 1.25, 1.75],
        frame_temporal_patch_size=2,
    )
    with pytest.raises(ValueError, match="temporal_patch_size"):
        mm_processor(
            "<video>",
            mm_items=mm_processor.info.parse_mm_data(
                {"video": [(_video(4), metadata)]}
            ),
        )
