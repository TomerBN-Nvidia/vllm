# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm.model_executor.models.radio import RadioModel


class _RadioLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(2, 6)
        self.attn.proj = nn.Linear(2, 2)
        self.norm1 = nn.LayerNorm(2)
        self.norm2 = nn.LayerNorm(2)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(2, 4)
        self.mlp.fc2 = nn.Linear(4, 2)
        self.ls1 = nn.Parameter(torch.zeros(2))
        self.ls2 = nn.Parameter(torch.zeros(2))

        def load_qkv_shard(param, weight, shard_id):
            shard_index = {"q": 0, "k": 1, "v": 2}[shard_id]
            shard_size = param.shape[0] // 3
            param.data[shard_index * shard_size : (shard_index + 1) * shard_size].copy_(
                weight
            )

        self.attn.qkv.weight.weight_loader = load_qkv_shard
        self.attn.qkv.bias.weight_loader = load_qkv_shard


def _make_radio_model() -> RadioModel:
    model = object.__new__(RadioModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(initializer_factor=1.0)
    model.model = nn.Module()
    model.model.patch_generator = nn.Module()
    model.model.patch_generator.embedder = nn.Linear(2, 2, bias=False)
    model.model.patch_generator.video_embedder = nn.Linear(4, 2, bias=False)
    model.model.patch_generator.pos_embed = nn.Parameter(torch.zeros(1, 1, 2))
    model.model.patch_generator.cls_token = nn.Module()
    model.model.patch_generator.cls_token.token = nn.Parameter(torch.zeros(3, 2))
    model.model.encoder = nn.Module()
    model.model.encoder.layers = nn.ModuleList([_RadioLayer()])
    return model


def test_radio_load_weights_initializes_missing_layer_scales_to_identity():
    model = _make_radio_model()

    loaded = model.load_weights([])

    assert loaded == set()
    assert torch.equal(model.model.encoder.layers[0].ls1, torch.ones(2))
    assert torch.equal(model.model.encoder.layers[0].ls2, torch.ones(2))


def test_radio_load_weights_preserves_exported_layer_scales():
    model = _make_radio_model()

    loaded = model.load_weights(
        [("radio_model.model.blocks.0.ls1", torch.full((2,), 0.5))]
    )

    assert loaded == {"model.encoder.layers.0.ls1"}
    assert torch.equal(model.model.encoder.layers[0].ls1, torch.full((2,), 0.5))
    assert torch.equal(model.model.encoder.layers[0].ls2, torch.ones(2))


def test_radio_load_weights_maps_native_flat_radio_weights():
    model = _make_radio_model()
    q_weight = torch.full((2, 2), 1.0)
    k_weight = torch.full((2, 2), 2.0)
    v_weight = torch.full((2, 2), 3.0)

    loaded = model.load_weights(
        [
            ("embeddings.patch_projection.weight", torch.full((2, 2), 4.0)),
            (
                "embeddings.video_patch_projection.weight",
                torch.full((2, 4), 5.0),
            ),
            ("embeddings.position_embedding", torch.full((1, 1, 2), 6.0)),
            ("embeddings.cls_register_token", torch.full((3, 2), 7.0)),
            ("encoder.layer.0.attention.attention.query.weight", q_weight),
            ("encoder.layer.0.attention.attention.key.weight", k_weight),
            ("encoder.layer.0.attention.attention.value.weight", v_weight),
            ("encoder.layer.0.layer_scale1.lambda1", torch.full((2,), 0.5)),
        ]
    )

    assert "model.patch_generator.embedder.weight" in loaded
    assert "model.patch_generator.video_embedder.weight" in loaded
    assert "model.patch_generator.pos_embed" in loaded
    assert "model.patch_generator.cls_token.token" in loaded
    assert "model.encoder.layers.0.attn.qkv.weight" in loaded
    assert torch.equal(
        model.model.encoder.layers[0].attn.qkv.weight,
        torch.cat((q_weight, k_weight, v_weight)),
    )
    assert torch.equal(model.model.encoder.layers[0].ls1, torch.full((2,), 0.5))
    assert torch.equal(model.model.encoder.layers[0].ls2, torch.ones(2))
