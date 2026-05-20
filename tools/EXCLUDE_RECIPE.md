# Fix C — keep NemotronH small Linears at BF16 under MXFP8

The MXFP8 path quantizes every `LinearBase` layer not listed in the
checkpoint's `exclude_modules` (or `ignore`, depending on the config
format). For NemotronH this includes a long tail of small Linears
(qkv_proj, o_proj, MoE router gate, latent fc1/fc2, Mamba in/out/conv1d)
where the per-step activation quantize cost is significant relative to
the GEMM size. Per HANDOFF.md §7.1 A, this is the hypothesized
dominant overhead on rollout decode.

We can't always re-emit the checkpoint's config. Instead, set:

```bash
export VLLM_MODELOPT_EXTRA_EXCLUDE_MODULES="*qkv_proj,*o_proj,*gate,*fc1_latent_proj,*fc2_latent_proj,*mixer.in_proj,*mixer.out_proj,*mixer.conv1d"
```

before `vllm serve`. The engine will log once:

```
VLLM_MODELOPT_EXTRA_EXCLUDE_MODULES: appended 8 pattern(s) to ModelOpt exclude_modules: [...]
```

These patterns are appended to whatever `exclude_modules` / `ignore` the
checkpoint config already provides, and go through the same
`is_layer_excluded` matcher used by all four ModelOpt variants (FP8,
NVFP4, MXFP8, MIXED_PRECISION).

## What stays MXFP8 vs what reverts to BF16

For NemotronH (`backbone.layers.N.mixer.<sub>`):

| Sub-Linear | After exclusion | Why |
|---|---|---|
| `mixer.qkv_proj` (Attention) | BF16 | small; per-call quantize hurts |
| `mixer.o_proj` (Attention) | BF16 | small |
| `mixer.gate` (MoE router) | BF16 | tiny (8192×512); pure latency drag |
| `mixer.fc1_latent_proj` | BF16 | NemotronH-specific, adds dequant→requant |
| `mixer.fc2_latent_proj` | BF16 | as above |
| `mixer.in_proj` (Mamba) | BF16 | small |
| `mixer.out_proj` (Mamba) | BF16 | small |
| `mixer.conv1d` (Mamba) | BF16 | tiny |
| `mixer.up_proj`, `mixer.down_proj` (pure MLP layer) | **MXFP8** | bulk weight |
| `mixer.shared_experts.up_proj`, `mixer.shared_experts.down_proj` | **MXFP8** | shared expert is the second-largest weight; keep quantized |
| `mixer.experts` (the 512 routed experts) | **MXFP8** | THE biggest weight; keep quantized |

The intuition: leave the multi-billion-parameter expert weights where
MXFP8 buys real DRAM-bandwidth savings, but stop paying the per-call
activation quantize tax on the small per-token Linears that don't gain
much from MXFP8 weight compression.

## Disk size / memory tradeoff

The excluded Linears get loaded in BF16 instead of FP8 + scale. The
extra HBM footprint per excluded Linear is:
- qkv_proj  8192 × 8448 ≈ 132 MB BF16 vs 66 MB FP8 (per-rank at TP=8)
- o_proj    8192 × 8192 ≈ 128 MB BF16
- fc1_latent / fc2_latent  ≈ 32 MB BF16
- gate         ≈ 4 MB BF16
- Mamba (per layer) ≈ ~20-40 MB BF16

Per-layer extra footprint: ~300-500 MB BF16 total across all the
excluded Linears. Across ~108 layers but most layer types only have one
or two excluded Linears each. Rough total: a few GB extra HBM per rank
under TP=8. At `--gpu-memory-utilization 0.90` this fits comfortably.

## How to verify it took

After boot, grep the engine log:

```bash
grep "VLLM_MODELOPT_EXTRA_EXCLUDE_MODULES" /lustre/.../vllm_ultra_mxfp8.log
```

Should print one line listing the patterns. If it's missing, the env
var wasn't exported in the worker before `vllm serve`.

To verify on a per-layer basis (deeper check), set `VLLM_LOGGING_LEVEL=DEBUG`
on the worker and look for `Excluding layer` lines emitted by
`is_layer_excluded`.

## Rollback

Unset the env var. Default behavior is unchanged.
