# Kernel spec: `rms_norm_mxfp8_quant` (and add-variant)

Fix E from HANDOFF.md §7. Status: **NOT IMPLEMENTED — escalation to perf
team (Tomer Bar Natan / Duncan Moss)**. This doc is the spec the kernel
must satisfy.

## Why this kernel is needed

Every MXFP8 Linear in NemotronH is preceded by a RMSNorm. The current
two-kernel sequence

```
x (bf16, [M, K])
    --[RMSNorm kernel]-->  y (bf16, [M, K])
    --[mxfp8_quantize kernel]-->  y_fp8 (fp8_e4m3, [M, K])  +  y_scale (uint8 e8m0, [M, K//32])
```

materializes the bf16 intermediate `y` to HBM and then re-reads it. At
NemotronH-rollout shapes (M=256, K=8192, ~108 layers with ~3 such hops
per layer) this is ~432 MB of avoidable HBM round-trip per step and a
couple thousand extra kernel launches.

A fused kernel does it in one pass — no bf16 intermediate, half the
launches.

**Note on impact**: my §7.2 estimate of 5-10% was loose. Realistic E2E
on the prod_65k_8k shape is closer to **2-5%** after accounting for
HBM bandwidth headroom and launch-overhead amortization. The fusion is
still worth doing because (a) it's a clean win, (b) it stacks with
Fix C (which removes the quantize from many of these Linears,
narrowing the kernel-overhead-dominated regime where Fix E shines).

## API

Add to `csrc/ops.h`:

```cpp
void rms_norm_mxfp8_quant(
    torch::Tensor& out,         // [M, K]   torch::kFloat8_e4m3fn
    torch::Tensor& out_scale,   // [M, K/32] torch::kUInt8 (E8M0)
    torch::Tensor const& input, // [M, K]   torch::kBFloat16 or kFloat16
    torch::Tensor const& weight,// [K]      same dtype as input
    double epsilon);

void fused_add_rms_norm_mxfp8_quant(
    torch::Tensor& out,         // [M, K]
    torch::Tensor& out_scale,   // [M, K/32]
    torch::Tensor& input,       // [M, K]  IN/OUT — residual-add stored in place (matches
                                //             pattern of fused_add_rms_norm_static_fp8_quant)
    torch::Tensor& residual,    // [M, K]  IN/OUT — same as input convention
    torch::Tensor const& weight,// [K]
    double epsilon);
```

Add `torch_bindings.cpp` entries mirroring the existing
`rms_norm_static_fp8_quant` / `fused_add_rms_norm_static_fp8_quant`
pair (lines 202-214 of `csrc/torch_bindings.cpp`).

## Math (reference implementation)

```python
import torch

MXFP8_BLOCK_SIZE = 32
FP8_DTYPE = torch.float8_e4m3fn

def rms_norm_mxfp8_quant_reference(
    input: torch.Tensor,        # [M, K] bf16 or fp16
    weight: torch.Tensor,       # [K]    bf16 or fp16
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation. Compute RMSNorm then MXFP8-quantize per
    block of 32 along K, returning (out_fp8 [M, K], out_scale [M, K//32]).
    """
    M, K = input.shape
    assert K % MXFP8_BLOCK_SIZE == 0
    x_f32 = input.float()
    # RMSNorm
    rms = torch.sqrt((x_f32 * x_f32).mean(dim=-1, keepdim=True) + epsilon)
    y_f32 = (x_f32 / rms) * weight.float()              # [M, K]
    # Per-block-32 MXFP8 quantize
    y_blocks = y_f32.view(M, K // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    block_max_abs = y_blocks.abs().amax(dim=-1)         # [M, K//32]
    # E8M0 = ceil(log2(block_max_abs / FP8_MAX)), encoded with bias 127.
    fp8_max = torch.finfo(FP8_DTYPE).max
    # Avoid log2(0)
    block_max_abs = block_max_abs.clamp_min(1e-30)
    exp = torch.ceil(torch.log2(block_max_abs / fp8_max))
    e8m0 = (exp + 127).to(torch.uint8)                  # [M, K//32]
    # Reconstruct float scale = 2^(e8m0 - 127), quantize each value
    scale = torch.pow(2.0, exp).unsqueeze(-1)           # [M, K//32, 1]
    y_scaled = y_blocks / scale                         # [M, K//32, 32]
    y_fp8 = y_scaled.to(FP8_DTYPE).view(M, K)
    return y_fp8, e8m0


def fused_add_rms_norm_mxfp8_quant_reference(
    input: torch.Tensor,        # [M, K]
    residual: torch.Tensor,     # [M, K]  IN/OUT
    weight: torch.Tensor,       # [K]
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same as above but with in-place residual add. `residual` is updated
    to hold (input + residual); the RMSNorm operates on the updated value;
    the FP8-quantized output is returned."""
    residual.add_(input)
    return rms_norm_mxfp8_quant_reference(residual, weight, epsilon)
```

## Performance budget (target)

NemotronH steady-state decode at M=256, K=8192 (latent space K=2048 also
appears):

| Step | Cost (current 2-kernel) | Target (fused) |
|---|---|---|
| RMSNorm | ~3-5 μs | (subsumed) |
| Materialize bf16 result to HBM | 4 MB write @ 7 TB/s ≈ 0.6 μs | 0 |
| `mxfp8_quantize` kernel | ~8-12 μs | (subsumed) |
| Fused total | ~12-18 μs | **target ≤ 7 μs** |

Per layer this saves ~5-10 μs; ~108 layers × 3 RMSNorm+quant hops ≈
**1.5-3 ms saved per step**. On a ~100 ms steady-state step, that's
1.5-3% raw — plus launch-overhead reductions in Python land that
torch.compile previously couldn't fuse across.

K=2048 path (NemotronH latent MoE input) is smaller; same kernel
should handle it via a `K % 32 == 0` requirement and a per-row inner
loop scaled by `K // 32`.

## Layout requirements

- `out`: `[M, K]`, row-major contiguous, `fp8_e4m3fn`. Same shape as
  the input.
- `out_scale`: `[M, K // 32]`, row-major contiguous, `uint8`. The
  E8M0 layout matches the **un-swizzled per-token scale layout** that
  `flashinfer.mxfp8_quantize(..., is_sf_swizzled_layout=False)` already
  emits in production today.
  - Do NOT use the 128x4 swizzled layout. Microbench in
    `tools/microbench_fix_b.py` confirmed the FlashInfer
    `trtllm_fp8_block_scale_moe` MoE kernel asserts on the swizzled
    layout. The MXFP8 Linear (`mm_mxfp8`) accepts both via the
    `sf_swizzle_layout` enum, but the downstream MoE path needs
    unswizzled.

## Correctness criteria

Compared against the reference above at:

- M ∈ {1, 4, 8, 16, 32, 64, 96, 128, 256, 512}
- K ∈ {2048, 8192} (NemotronH latent + main hidden sizes)
- weight ∈ {ones, random ±1}
- epsilon ∈ {1e-5, 1e-6}

Targets (measured locally on a single GB200, comparing reference vs
the existing two-kernel chain `RMSNorm → flashinfer.mxfp8_quantize`):

- `out_fp8` byte-equality ≥ 95% with the reference. The ~3-5% byte-
  differing tail is unavoidable: FP8 round-tie boundaries flip between
  reference (torch's default FP8 cast) and FlashInfer's CUDA quantize
  kernel. Either rounding mode is acceptable as long as the fused
  kernel is internally consistent.
- `out_scale` (E8M0 byte) equality ≥ 99% with the reference. Differing
  entries must differ by **at most 1** in E8M0 integer space (one
  power-of-two ulp).
- Numerically: dequantize both, compute `max_abs_diff / mean_abs_ref`
  must be ≤ 0.02 (2% relative error per element).

## Wiring once the kernel lands

1. Register the new ops in `csrc/torch_bindings.cpp` (10 lines).
2. Mirror the existing `RMSNormStaticQuantPattern` /
   `FusedAddRMSNormStaticQuantPattern` in
   `vllm/compilation/passes/fusion/rms_quant_fusion.py` for MXFP8.
   Each new pattern class is ~50 lines (the FP8 ones are already
   there as templates).
3. Add to `RMSNormQuantFusionPass.__init__` (~5 lines).
4. New file: `tools/microbench_rms_mxfp8_fused.py` validates the
   fused kernel against the reference at all M, K above.

The Python wiring is straightforward once the C++ side compiles. We
do NOT need to ship the matcher stub now — it will rot. Instead, the
PR that adds the kernel adds the matcher in the same change.

## Coordination / escalation

This kernel is the right scope for the FlashInfer team:
- `flashinfer.norm.add_rmsnorm_fp4quant` is the closest existing kernel
  (NVFP4 block-16). Adapting it to MXFP8 (block-32, E8M0 scale) is a
  finite delta.
- Owners: **Tomer Bar Natan** (FlashInfer/MoE) or **Duncan Moss**
  (kernels). Ping after pulling a GB200 node for development.

## Why we're not doing this in vLLM-only Python now

You can write a Python-only "fused" wrapper as a single
`torch.ops.vllm.rms_norm_mxfp8_quant` custom op that internally calls
two existing kernels. That gives Inductor a single op boundary, but:

- It does NOT eliminate the bf16 intermediate HBM round-trip (the
  internal call still allocates the bf16 buffer).
- It does NOT amortize kernel launches (still two cuLaunchKernel).
- The "fusion" is purely a torch.compile graph-shape concern.

For the actual 2-5% win we need a true CUDA kernel. Until that lands,
this fusion is deferred.

Tracking: HANDOFF.md §7.2 Fix E (escalated, not landed).
