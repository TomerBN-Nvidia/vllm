# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Radio vision model configuration"""

from collections.abc import Mapping
from typing import Any

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

VIT_TIMM_DIM_BY_NAME: dict[str, tuple[int, int, int, int]] = {
    "vit_small_patch16_224": (384, 12, 6, 1536),
    "vit_base_patch16_224": (768, 12, 12, 3072),
    "vit_large_patch16_224": (1024, 24, 16, 4096),
    "vit_huge_patch16_224": (1280, 32, 16, 5120),
}

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_radio_config_value(config: PretrainedConfig, name: str, default=None):
    legacy_args = getattr(config, "args", None)
    if isinstance(legacy_args, Mapping) and name in legacy_args:
        return legacy_args[name]
    return getattr(config, name, default)


class RadioConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a Radio
    vision model. It is used to instantiate a Radio model according to the
    specified arguments, defining the model architecture.

    Args:
        model_name: Name of the vision transformer model
            (e.g., "vit_base_patch16_224"). Used to determine architecture
            dimensions from `VIT_TIMM_DIM_BY_NAME`.
        image_size: The size (resolution) of each image.
        patch_size: The size (resolution) of each patch.
        qkv_bias: Whether to add a bias to the queries, keys and values.
        qk_normalization: Whether to apply normalization to queries and keys.
        norm_type: The normalization type to use.
        layer_norm_eps: The epsilon used by the layer normalization layers.
        initializer_factor: A factor for initializing all weight matrices.
        hidden_act: The non-linear activation function in the encoder.
        cpe_max_size: Maximum image size for position embeddings.
        max_img_size: Flat RADIO alias for the maximum image size.
        norm_mean: Mean values for image normalization (RGB channels).
            Defaults to (0.48145466, 0.4578275, 0.40821073)).
        norm_std: Standard deviation values for image normalization
            (RGB channels). Defaults to (0.26862954, 0.26130258, 0.27577711)).
        register_multiple: Number of register tokens to use.
        num_cls_tokens: Explicit number of class tokens for flat RADIO configs.
        num_registers: Explicit number of register tokens for flat RADIO configs.
        summary_idxs: Class-token indices returned as summary embeddings.
        teachers: A list of teacher model configurations. Each teacher configuration is
            a dict with keys like "name" and some may have "use_summary".
        cls_token_per_teacher: Whether to use a separate CLS token for each teacher.
        video_temporal_patch_size: Number of consecutive video frames grouped into
            a single tubelet for temporal compression. Default 1 (no compression).
            When > 1, a dedicated video_embedder (3*T*P*P -> hidden) is created
            alongside the image embedder (3*P*P -> hidden).
        separate_video_embedder: When True and video_temporal_patch_size > 1, use a
            dedicated video patch embedder (3*T*P*P -> hidden) separate from the
            image embedder (3*P*P -> hidden). When False, a single embedder with
            input size 3*T*P*P is used for both (images are duplicated T times).
    """

    model_type = "radio"

    def __init__(
        self,
        model_name: str | None = None,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_size: int | None = None,
        num_hidden_layers: int | None = None,
        num_attention_heads: int | None = None,
        intermediate_size: int | None = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_normalization: bool = False,
        norm_type: str = "layer_norm",
        layer_norm_eps: float = 1e-6,
        initializer_factor: float = 1.0,
        hidden_act: str = "gelu",
        cpe_max_size: int | None = None,
        max_img_size: int | None = None,
        norm_mean: tuple[float, float, float] | list = OPENAI_CLIP_MEAN,
        norm_std: tuple[float, float, float] | list = OPENAI_CLIP_STD,
        register_multiple: int | None = None,
        num_cls_tokens: int | None = None,
        num_registers: int | None = None,
        summary_idxs: list[int] | None = None,
        teachers: list[dict[str, Any]] | None = None,
        cls_token_per_teacher: bool = False,
        video_temporal_patch_size: int = 1,
        separate_video_embedder: bool = True,
        **kwargs,
    ):
        self.model_name = model_name
        model_dims = VIT_TIMM_DIM_BY_NAME.get(model_name) if model_name else None
        if model_dims is None and any(
            value is None
            for value in (hidden_size, num_hidden_layers, num_attention_heads)
        ):
            raise ValueError(
                "RADIO config must provide either a supported model_name or "
                "explicit hidden_size, num_hidden_layers, and num_attention_heads"
            )

        if hidden_size is None:
            assert model_dims is not None
            hidden_size = model_dims[0]
        if num_hidden_layers is None:
            assert model_dims is not None
            num_hidden_layers = model_dims[1]
        if num_attention_heads is None:
            assert model_dims is not None
            num_attention_heads = model_dims[2]
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size or (
            model_dims[3]
            if model_dims is not None
            else int(self.hidden_size * mlp_ratio)
        )
        self.mlp_ratio = mlp_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.qkv_bias = qkv_bias
        self.qk_normalization = qk_normalization
        self.norm_type = norm_type
        self.layer_norm_eps = layer_norm_eps
        self.initializer_factor = initializer_factor
        self.hidden_act = hidden_act
        self.cpe_max_size = cpe_max_size or max_img_size or 2048
        self.max_img_size = self.cpe_max_size
        self.norm_mean = (
            list(norm_mean) if isinstance(norm_mean, (tuple, list)) else norm_mean
        )
        self.norm_std = (
            list(norm_std) if isinstance(norm_std, (tuple, list)) else norm_std
        )
        self.register_multiple = register_multiple
        self.num_cls_tokens = num_cls_tokens
        self.num_registers = num_registers
        self.summary_idxs = summary_idxs
        self.teachers = teachers if teachers is not None else []
        self.cls_token_per_teacher = cls_token_per_teacher
        self.video_temporal_patch_size = video_temporal_patch_size
        self.separate_video_embedder = separate_video_embedder
        super().__init__(**kwargs)

    @classmethod
    def from_hf_config(
        cls,
        config: PretrainedConfig,
        *,
        norm_mean=None,
        norm_std=None,
    ) -> "RadioConfig":
        legacy_args = getattr(config, "args", None)
        if isinstance(legacy_args, Mapping):
            radio_kwargs = dict(legacy_args)
            model_name = radio_kwargs.pop("model", None)
            if model_name is None:
                raise ValueError("Legacy RADIO config is missing args['model']")

            preferred_resolution = getattr(config, "preferred_resolution", None)
            if preferred_resolution:
                radio_kwargs["image_size"] = preferred_resolution[0]
            else:
                radio_kwargs.setdefault(
                    "image_size", getattr(config, "image_size", 224)
                )
            radio_kwargs.setdefault("patch_size", getattr(config, "patch_size", 16))
            if norm_mean is not None:
                radio_kwargs["norm_mean"] = norm_mean
            if norm_std is not None:
                radio_kwargs["norm_std"] = norm_std
            radio_kwargs["video_temporal_patch_size"] = (
                getattr(config, "video_temporal_patch_size", None) or 1
            )
            radio_kwargs["separate_video_embedder"] = getattr(
                config, "separate_video_embedder", True
            )
            return cls(model_name=model_name, **radio_kwargs)

        if getattr(config, "num_channels", 3) != 3:
            raise ValueError("RADIO vision encoders must have three input channels")
        if getattr(config, "use_swiglu_ffn", False):
            raise ValueError("SwiGLU RADIO vision encoders are not supported")

        hidden_size = config.hidden_size
        num_hidden_layers = config.num_hidden_layers
        num_attention_heads = config.num_attention_heads
        intermediate_size = getattr(config, "intermediate_size", None)
        if intermediate_size is None:
            intermediate_size = round(hidden_size * getattr(config, "mlp_ratio", 4.0))
        signature = (
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
        )
        model_names = [
            name
            for name, model_dims in VIT_TIMM_DIM_BY_NAME.items()
            if model_dims == signature
        ]
        if len(model_names) != 1:
            raise ValueError(f"Unsupported RADIO architecture dimensions: {signature}")

        return cls(
            model_name=model_names[0],
            image_size=getattr(config, "image_size", 224),
            patch_size=getattr(config, "patch_size", 16),
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            mlp_ratio=getattr(config, "mlp_ratio", 4.0),
            qkv_bias=getattr(config, "qkv_bias", True),
            qk_normalization=getattr(config, "qk_normalization", False),
            norm_type=getattr(config, "norm_type", "layer_norm"),
            layer_norm_eps=getattr(config, "layer_norm_eps", 1e-6),
            initializer_factor=getattr(config, "layerscale_value", 1.0),
            hidden_act=getattr(config, "hidden_act", "gelu"),
            max_img_size=getattr(config, "max_img_size", 2048),
            norm_mean=getattr(
                config,
                "norm_mean",
                norm_mean if norm_mean is not None else OPENAI_CLIP_MEAN,
            ),
            norm_std=getattr(
                config,
                "norm_std",
                norm_std if norm_std is not None else OPENAI_CLIP_STD,
            ),
            num_cls_tokens=getattr(config, "num_cls_tokens", 1),
            num_registers=getattr(config, "num_registers", 0),
            summary_idxs=getattr(config, "summary_idxs", None),
            video_temporal_patch_size=(
                getattr(config, "video_temporal_patch_size", None) or 1
            ),
            separate_video_embedder=getattr(config, "separate_video_embedder", True),
        )
