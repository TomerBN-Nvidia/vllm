# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import TokenizersBackend

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache.base import MultiModalProcessorOnlyCache
from vllm.transformers_utils.configs.nemotron_shortcut import NemotronShortcutConfig


def processor(tmp_path, model_patch_size=1, **video_options):
    tokens = ["<unk>", "<img>", "<image_break>", "</img>", "<video>"]
    TokenizersBackend(
        tokenizer_object=Tokenizer(
            WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="<unk>")
        ),
        unk_token="<unk>",
        additional_special_tokens=tokens[1:],
    ).save_pretrained(tmp_path)
    NemotronShortcutConfig(
        text_config=dict(
            layer_types=["mlp"],
            hidden_size=8,
            vocab_size=5,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=8,
        ),
        vision_config=dict(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=1,
            patch_size=14,
            image_size=56,
        ),
        image_token_id=1,
        spatial_merge_size=2,
        video_temporal_patch_size=model_patch_size,
    ).save_pretrained(tmp_path)
    return MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(
            str(tmp_path),
            max_model_len=4096,
            limit_mm_per_prompt={"image": 0, "video": 1},
            media_io_kwargs={"video": video_options},
        )
    )


def process(mm_processor, metadata, cache=None):
    frames = np.zeros((4, 56, 56, 3), dtype=np.uint8)
    return mm_processor(
        "<video>",
        mm_items=mm_processor.info.parse_mm_data({"video": [(frames, metadata)]}),
        cache=cache,
    )


def test_t1_timestamps_survive_cache(tmp_path):
    mm_processor = processor(tmp_path)
    cache = MultiModalProcessorOnlyCache(mm_processor.info.ctx.model_config)
    metadata = {"fps": 2.0, "frames_indices": [0, 3, 6, 9], "total_num_frames": 12}
    reference = process(mm_processor, metadata)
    hf_processor = mm_processor.info.get_hf_processor()
    image = hf_processor.replace_image_token({"image_sizes": [(56, 56)]}, 0)
    expected = "This is a video:\n" + "".join(
        f"Frame {i + 1} sampled at {time:.2f} seconds: {image}\n"
        for i, time in enumerate([0.0, 1.5, 3.0, 4.5])
    )
    assert reference["prompt_token_ids"] == hf_processor.tokenizer.encode(
        expected, add_special_tokens=False
    )
    for _ in range(2):
        result = process(mm_processor, metadata, cache)
        assert result["prompt_token_ids"] == reference["prompt_token_ids"]
        item = result["mm_kwargs"]["video"][0]
        assert item["video_timestamps"].data.tolist() == [0.0, 1.5, 3.0, 4.5]
    changed = process(mm_processor, dict(metadata, fps=4.0), cache)
    item = changed["mm_kwargs"]["video"][0]
    assert item["video_timestamps"].data.tolist() == [0.0, 0.75, 1.5, 2.25]


@pytest.mark.parametrize("temporal_patch_size", [1, 2])
def test_loader_labels_and_frame_cap(tmp_path, temporal_patch_size):
    mm_processor = processor(
        tmp_path,
        temporal_patch_size,
        video_backend="nemotron_shortcut_t2",
        temporal_patch_size=temporal_patch_size,
        max_frames=32,
    )
    assert mm_processor.info.get_max_video_frames() == 32
    metadata = dict(
        fps=2.0,
        total_num_frames=4,
        frames_indices=[0, 1, 2, 3],
        frame_source_indices=[0, 1, 2, 3],
        frame_timestamps=[0.25, 0.75, 1.25, 1.75],
        frame_temporal_patch_size=temporal_patch_size,
    )
    result = process(mm_processor, metadata)
    item = result["mm_kwargs"]["video"][0]
    assert item["video_timestamps"].data.tolist() == metadata["frame_timestamps"]
    if temporal_patch_size == 1:
        with pytest.raises(ValueError, match="temporal_patch_size"):
            process(mm_processor, dict(metadata, frame_temporal_patch_size=2))
