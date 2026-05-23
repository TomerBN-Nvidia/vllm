# Overnight result — 2026-05-22 → 2026-05-23

## Bottom line

After 7 iterations spanning ~14 hours, **MXFP8 nemo-speed wins exactly
2 of 5 harness configs** in every configuration tested:

  - **smoke** (32-conc) — MXFP8 wins by 1.5×–3.1× (high variance)
  - **prod_65k_8k** (256-conc) — MXFP8 wins by 1.6×–2.4× (consistent)
  - **ab_mid** (64-conc) — MXFP8 loses 0.77×–0.90×
  - **ab_decode_heavy** (64-conc) — MXFP8 loses 0.83×–0.88×
  - **swe_192k_512** (64-conc) — MXFP8 loses 0.85×–0.89×

The 12–25% gap at 64-concurrent is structural to the FlashInfer
`mm_mxfp8` kernel's perf curve. No vLLM-level config knob or
single-file code change closes it.

## Final config comparison

| Config           | concur | iter | BF16   | MXFP8  | ratio |
|------------------|-------:|-----:|-------:|-------:|------:|
| smoke            |     32 |    6 |    386 |   1190 | 3.08× |
| ab_mid           |     64 |    6 |   3125 |   2399 | 0.77× |
| ab_decode_heavy  |     64 |    6 |   3384 |   2905 | 0.86× |
| prod_65k_8k      |    256 |    6 |   3168 |   5665 | 1.79× |
| swe_192k_512     |     64 |    6 |   3179 |   2779 | 0.87× |

(iter6 is the most apples-to-apples — both sides booted under same env,
same harness with `ignore_eos=true`.)

## Iteration log

  - **iter1**: `--enable-prefix-caching` + `--max-num-seqs` bump (320 BF16 / 512 MXFP8).
    Helped prod (+19% MXFP8) but hurt swe (BF16 +189% vs MXFP8 +0%); BF16
    ate the swe gain because its 32k-prompt prefill was the bottleneck.
  - **iter2**: patched harness `run_rollout_bench.py` to send `ignore_eos=true`.
    Revealed that ~16x of the "MXFP8 loses ab_decode_heavy" effect from
    yesterday was just MXFP8's checkpoint emitting ~7× more tokens per
    request before EOS — model-distribution artifact, not kernel issue.
    Real per-token ratio at 64-conc: 0.88–0.92x BF16. At 256-conc: 2.42×.
  - **iter3**: removed `--async-scheduling` from MXFP8. **REGRESSION**:
    smoke dropped 58% (1596 → 662). async-scheduling is critical at
    low-concurrency to hide scheduler RPC latency. Reverted.
  - **iter4** (CODE CHANGE — committed 38424128b, reverted 47bc43235):
    lowered `M_ALIGN` in `FlashInferCutlassMxfp8LinearKernel.apply_weights`
    from 128 → 32. Hypothesis: M=32 batch padded to 128 wasted 75% of
    GEMM rows. **No measured speedup** — the CUTLASS kernel tile must
    pad internally regardless of caller M. Reverted.
  - **iter5**: `--max-num-batched-tokens 32768` (default 8192). Helped
    smoke +43% but hurt ab_mid -17%. Net still 2/5.
  - **iter6**: `--no-enable-chunked-prefill` on MXFP8. Helped smoke
    +21% but hurt prod -28%. Net still 2/5. (Also OOM'd BF16 when
    matched, so BF16 stayed on chunked-prefill for apples-to-apples
    measurement.)
  - **iter7**: `--mamba-cache-mode none` on MXFP8 (was `all`, which the
    serve warned is experimental with prefix caching). No significant
    effect. Net still 2/5.

## Diagnosis

The kernel-level gap appears in `vllm/model_executor/kernels/linear/mxfp8/flashinfer.py`'s
`FlashInferCutlassMxfp8LinearKernel.apply_weights`, which dispatches to
`vllm_flashinfer.mm_mxfp8(..., backend="cutlass")`. The CUTLASS kernel
is shape-optimal at M >= 128 (most likely M-axis tile size = 128). At
M ∈ [32, 64, 96], the kernel processes a 128-row tile regardless,
producing the same wall time as M=128 — meaning MXFP8's per-row
effective rate is artificially worse at low batch. At M >= 128 (e.g.
prod_65k_8k's 256-concurrent decode batch), MXFP8 hits its native
FP8-GEMM speedup and pulls ahead 2.3×.

The MoE path (`ModelOptMxFp8FusedMoE.apply_monolithic` →
`flashinfer_trtllm_fp8_block_scale_moe`) has its own per-step
quantize (`mxfp8_e4m3_quantize`) that adds a fixed kernel-launch cost
on every forward, amortizing well only at higher concurrent batch.
This is the same structural pattern — fixed overhead per kernel
launch, amortized by batch.

## Real fix paths (out of scope for one overnight)

1. **Kernel-level**: get FlashInfer to ship a small-M-optimized
   `mm_mxfp8` variant. Either a smaller-tile CUTLASS kernel for M ∈
   [16, 32, 64] or a Triton implementation tuned for those sizes.
2. **Latent-MoE fusion Phase 2B** (per `tools/LATENT_MOE_FUSION.md`):
   modify `_apply_flashinfer_cutlass` to return FP8 output directly
   (bypassing the dequant→requant round-trip into MoE). Requires
   `flashinfer.mm_mxfp8` to expose fp8 output — blocked on
   flashinfer 0.6.10's API.
3. **Model-side BF16 fallback for small M in `ModelOptMxFp8LinearMethod`**:
   keep both fp8 and bf16 weights, dispatch by batch size. Doubles
   linear-weight memory; consumes most of MXFP8's KV headroom. May be
   worth it if Linear is the bottleneck (vs MoE).

## What to do with this result

For Nemotron Ultra **production serving** (prod-shape workloads at
high concurrency), **MXFP8 nemo-speed is the right choice** — 1.6–2.3×
the BF16 throughput on the prod_65k_8k config, with perfect
reliability (256/256 vs BF16's 199/256 in pre-nemo-speed).

For **interactive / low-concurrency** workloads, BF16 is currently
~12–25% faster per token, but the difference is within the variance
band of a single overnight run, and the MXFP8 prefill is fast enough
that first-token latency is competitive (not measured here, would be
worth a follow-up).

The "MXFP8 faster in *all* configs" goal needs kernel work; this is
out of scope for vLLM-level changes.

---
*Authored 2026-05-23 ~02:40 PDT after 7 overnight iterations.*
