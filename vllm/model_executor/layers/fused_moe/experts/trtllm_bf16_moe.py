# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import trtllm_moe_pack_topk_ids_weights
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    has_flashinfer_trtllm_bf16_routed_moe,
    has_flashinfer_trtllm_fused_moe,
)


class TrtLlmBf16ExpertsBase:
    """BF16 unquantized TRTLLM-Gen MoE kernels. Shared base for monolithic and
    modular variants."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        # NOTE: do not call ``super().__init__`` here; ``Base`` does not yet
        # inherit from ``mk.FusedMoEExperts``. Mirrors the
        # ``TrtLlmNvFp4ExpertsBase`` pattern. The MK base ``__init__`` only
        # sets ``moe_config``/``quant_config`` (replicated here) and validates
        # batched-format args (we never pass those).
        self.moe_config = moe_config
        self.quant_config = quant_config
        self.routing_method_type = moe_config.routing_method
        self.topk = moe_config.experts_per_token
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.hidden_dim = moe_config.hidden_dim
        self.local_num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        """Supports only Blackwell-family GPUs."""
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(100)
            and has_flashinfer_trtllm_fused_moe()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Supports only unquantized inputs."""
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL]

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in [
            RoutingMethodType.DeepSeekV3,
            RoutingMethodType.Llama4,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    def supports_chunking(self) -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return False

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True


class TrtLlmBf16ExpertsMonolithic(TrtLlmBf16ExpertsBase, mk.FusedMoEExpertsMonolithic):
    """Monolithic variant (router + experts in one kernel)."""

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        """Monolithic kernel so only use with naive DP/EP and TP."""
        return (
            not moe_parallel_config.use_all2all_kernels
            or moe_parallel_config.use_ag_rs_all2all_kernels
        ) and not moe_parallel_config.enable_eplb

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        import flashinfer

        return flashinfer.fused_moe.trtllm_bf16_moe(
            routing_logits=router_logits,
            routing_bias=e_score_correction_bias,
            hidden_states=hidden_states,
            gemm1_weights=w1,
            gemm2_weights=w2,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=num_expert_group,
            topk_group=topk_group,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routing_method_type=self.routing_method_type,
            activation_type=activation_to_flashinfer_int(activation),
        )


class TrtLlmBf16ExpertsModular(TrtLlmBf16ExpertsBase, mk.FusedMoEExpertsModular):
    """Modular variant (just experts, routing already happened upstream).

    Required for the DP/EP and EPLB cases, which need a prepare/finalize stage
    that dispatches tokens via all2all before invoking the experts kernel.

    Scope: non-gated MoE only. Gated SwiGLU (SILU + act_and_mul) is left to
    the monolithic variant until the modular path is exercised for it.
    """

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            TrtLlmBf16ExpertsBase._supports_current_device()
            and has_flashinfer_trtllm_bf16_routed_moe()
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        """Non-gated activations only."""
        return activation == MoEActivation.RELU2_NO_MUL

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        """The modular implementation supports all parallel configs."""
        return True

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, self.hidden_dim)
        return (workspace1, workspace2, output)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        import flashinfer

        packed_tensor = trtllm_moe_pack_topk_ids_weights(topk_ids, topk_weights)

        result = flashinfer.fused_moe.trtllm_bf16_routed_moe(
            topk_ids=packed_tensor,
            hidden_states=hidden_states,
            gemm1_weights=w1,
            gemm2_weights=w2,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routing_method_type=self.routing_method_type,
            activation_type=activation_to_flashinfer_int(activation),
        )
        # trtllm_bf16_routed_moe returns List[torch.Tensor]; the finalized
        # output is at index 0 when do_finalize=True (the default).
        output.copy_(result[0] if isinstance(result, (list, tuple)) else result)
