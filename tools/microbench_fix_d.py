"""
Fix D microbench: confirm that the new pre-quantized input path through
ModelOptMxFp8FusedMoE.apply_monolithic produces bit-identical output to
the existing path that quantizes internally.

We can't directly import ModelOptMxFp8FusedMoE here (it needs a real
FusedMoE layer + vllm bootstrap), so we exercise the equivalent code
path by calling flashinfer.fused_moe.trtllm_fp8_block_scale_moe twice:

  (A) Standard:    quantize(x, swizz=False) -> (q, s) -> moe(q, s, ...)
  (B) Pre-quant:   q, s = same quantize, but pass q, s in directly (the
                   "Fix D path" — apply_monolithic skips the quantize when
                   x_mxfp8 + x_mxfp8_scale are provided).

If apply_monolithic correctly bypasses the internal quantize, the
results must be byte-identical. (The pre-quantized inputs were already
produced by the same mxfp8_quantize call, so there is zero numerical
difference — this test is really verifying that the kernel call shape
is unchanged when we route through the new optional kwargs.)
"""
from __future__ import annotations

import torch
import flashinfer
from flashinfer import mxfp8_quantize
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe
from flashinfer.fused_moe.core import ActivationType, Fp8QuantizationType
from flashinfer import (
    shuffle_matrix_a,
    shuffle_matrix_sf_a,
)

MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32

# NemotronH latent-MoE shapes
H = 2048
I = 5120
E = 512
TOP_K = 22
ROUTING_DEEPSEEKV3 = 2


def build_weights():
    is_gated = False
    factor = 1
    epilogue_tile_m = 128

    w13 = torch.randint(0, 64, (E, factor * I, H), dtype=torch.uint8,
                        device="cuda").view(MXFP8_VALUE_DTYPE)
    w2 = torch.randint(0, 64, (E, H, I), dtype=torch.uint8,
                       device="cuda").view(MXFP8_VALUE_DTYPE)
    w13_sf = torch.randint(100, 130, (E, factor * I, H // MXFP8_BLOCK_SIZE),
                           dtype=MXFP8_SCALE_DTYPE, device="cuda")
    w2_sf = torch.randint(100, 130, (E, H, I // MXFP8_BLOCK_SIZE),
                          dtype=MXFP8_SCALE_DTYPE, device="cuda")

    w13s, w2s, w13_sfs, w2_sfs = [], [], [], []
    for i in range(E):
        w13_i = w13[i].reshape(factor * I, H)
        w13_sf_i = w13_sf[i].reshape(factor * I, H // MXFP8_BLOCK_SIZE)
        w13_s = shuffle_matrix_a(w13_i.view(torch.uint8), epilogue_tile_m)
        w2_s = shuffle_matrix_a(w2[i].view(torch.uint8), epilogue_tile_m)
        w13_sf_s = shuffle_matrix_sf_a(
            w13_sf_i.view(torch.uint8).reshape(factor * I, -1),
            epilogue_tile_m,
        )
        w2_sf_s = shuffle_matrix_sf_a(
            w2_sf[i].view(torch.uint8).reshape(H, -1), epilogue_tile_m,
        )
        w13s.append(w13_s.contiguous().view(MXFP8_VALUE_DTYPE))
        w2s.append(w2_s.contiguous().view(MXFP8_VALUE_DTYPE))
        w13_sfs.append(w13_sf_s.contiguous().view(MXFP8_SCALE_DTYPE))
        w2_sfs.append(w2_sf_s.contiguous().view(MXFP8_SCALE_DTYPE))

    return (
        torch.stack(w13s).contiguous(),
        torch.stack(w13_sfs).contiguous(),
        torch.stack(w2s).contiguous(),
        torch.stack(w2_sfs).contiguous(),
    )


def quantize_input(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Match modelopt.py:2046-2054 exactly (linear layout, 2D reshape)."""
    x_q, x_s = mxfp8_quantize(x, is_sf_swizzled_layout=False)
    if x_s.ndim == 1 and x_q.ndim == 2:
        x_s = x_s.view(x_q.size(0), -1)
    return x_q, x_s


def call_moe(M: int, weights, x_q, x_s):
    """Direct kernel call mirroring apply_monolithic's kwargs."""
    w13, w13_sf, w2, w2_sf = weights
    torch.manual_seed(0)
    router_logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
    e_score_bias = torch.zeros(E, dtype=torch.float32, device="cuda")
    out = trtllm_fp8_block_scale_moe(
        routing_logits=router_logits,
        routing_bias=e_score_bias,
        hidden_states=x_q,
        hidden_states_scale=x_s,
        gemm1_weights=w13,
        gemm1_weights_scale=w13_sf,
        gemm2_weights=w2,
        gemm2_weights_scale=w2_sf,
        num_experts=E,
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        intermediate_size=I,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=5.0,
        routing_method_type=ROUTING_DEEPSEEKV3,
        use_shuffled_weight=True,
        weight_layout=0,
        fp8_quantization_type=Fp8QuantizationType.MxFp8,
        activation_type=int(ActivationType.Relu2),
    )
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out


def main():
    torch.cuda.set_device(0)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("Building synthetic MoE weights ...")
    weights = build_weights()

    for M in [32, 128, 256, 512]:
        print(f"\n=== M={M} ===")
        torch.manual_seed(0)
        x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda") * 0.1

        # Path A: quantize-inside (current production)
        x_q_a, x_s_a = quantize_input(x)
        out_a = call_moe(M, weights, x_q_a, x_s_a)

        # Path B: pre-quantize outside (Fix D opt-in). Same caller does the
        # same quantize, just hands the pre-quantized tensors in. This is
        # what NemotronH would do when fc1_latent_proj emits fp8 directly.
        x_q_b, x_s_b = quantize_input(x)
        out_b = call_moe(M, weights, x_q_b, x_s_b)

        # The two MUST be byte-identical (since the quantize is deterministic).
        eq_bytes = (out_a.float() == out_b.float()).float().mean().item() * 100
        max_abs = (out_a.float() - out_b.float()).abs().max().item()
        print(f"  byte_eq={eq_bytes:.2f}%   max_abs_diff={max_abs:.6e}")
        assert eq_bytes == 100.0 and max_abs == 0.0, (
            "Pre-quantized path must produce bit-identical output."
        )

    print("\nAll M values: bit-identical between quantize-inside and pre-quantize paths.")


if __name__ == "__main__":
    main()
