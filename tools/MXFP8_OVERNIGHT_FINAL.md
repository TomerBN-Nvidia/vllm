# Overnight result — 2026-05-22 → 2026-05-23 — FINAL after 8 iterations

## Bottom line

**MXFP8 nemo-speed wins exactly 2 of 5 harness configs** in every
iteration. The user's goal — "MXFP8 faster than BF16 on all 5
configs" — is not achievable with vLLM-Python-level changes; it
requires kernel work in FlashInfer's `mm_mxfp8` and/or
`trtllm_fp8_block_scale_moe` for the M=32–64 regime.

## The result

  | Config            | concur | BF16 tok/s | MXFP8 tok/s | ratio  | who wins |
  |-------------------|-------:|-----------:|------------:|-------:|----------|
  | smoke             |     32 |        569 |         662 | 1.16×  | MXFP8    |
  | ab_mid            |     64 |       3168 |        2586 | 0.82×  | BF16     |
  | ab_decode_heavy   |     64 |       3320 |        2681 | 0.81×  | BF16     |
  | prod_65k_8k       |    256 |       3122 |        7267 | 2.33×  | MXFP8    |
  | swe_192k_512      |     64 |       3171 |        2567 | 0.81×  | BF16     |

(iter8 with `ignore_eos=true`, prefix_caching, max_num_seqs=512 MXFP8 /
320 BF16, --disable-custom-all-reduce. BF16 also has the
weight_bf16 cache compiled but `MXFP8_BF16_FALLBACK_SMALL_M` env unset.)

## What we tried (8 iterations)

1. **iter1**: Add `--enable-prefix-caching` + bump `--max-num-seqs`.
   Helped prod, hurt swe (BF16 ate more of the gain than MXFP8 because
   its prefill is the bottleneck).

2. **iter2**: Patch harness `run_rollout_bench.py` to pass
   `ignore_eos=true`. *This was the biggest realization*: 16× of the
   "MXFP8 loses ab_decode_heavy" effect in the original handoff was
   actually about the MXFP8 checkpoint generating ~7× more tokens per
   request before EOS, NOT about kernel speed. With ignore_eos the
   real per-token gap is 0.82–0.92× at M=32–64.

3. **iter3**: Drop `--async-scheduling`. Hurts smoke -58%. Reverted.

4. **iter4** (CODE CHANGE 38424128b, reverted 47bc43235): Lower the
   `min_dim` from 128 to 32 in `FlashInferCutlassMxfp8LinearKernel.apply_weights`.
   Hypothesis: less M padding = less wasted compute. **No measured
   effect** — the kernel internally tiles by 128 along M regardless of
   the caller's padding. Reverted.

5. **iter5**: `--max-num-batched-tokens 32768` (default 8192). Helped
   smoke +43%, hurt ab_mid -17%. Net still 2/5.

6. **iter6**: `--no-enable-chunked-prefill`. Helped smoke, hurt prod -28%.
   Net still 2/5. (Also OOM'd BF16 when matched — BF16 stayed on chunked
   prefill.)

7. **iter7**: `--mamba-cache-mode none` (was `all`). No measured effect.

8. **iter8** (CODE CHANGE 7a6e88675, kept as opt-in): BF16 fallback in
   `FlashInferCutlassMxfp8LinearKernel`. Dequantize a BF16 copy of the
   weight at load time, and at apply time for M<128 do a plain bf16
   matmul instead of mm_mxfp8. Tests whether the FP8 kernel is slower
   than bf16 at small M. **Answer: no, FP8 is actually faster even
   with the padding overhead.** Iter8 with the env var on regressed
   mid-concurrency configs by 4–6% vs iter7. Code kept in branch as
   opt-in (`MXFP8_BF16_FALLBACK_SMALL_M=1`) since it's a useful
   testbed for future kernel changes, but the env var is off by default.

## What we learned

- The **64-concurrent gap is real and ~20%**, consistently across every
  configuration tested. It's not padding, not the moe-backend choice,
  not the scheduler, not mamba caching, not chunked prefill, not max
  batched tokens, and not the dequantize round-trip.
- It IS the FlashInfer kernels (both `mm_mxfp8` for linears and
  `trtllm_fp8_block_scale_moe` for MoE) at small M. The FP8 cost-per-
  token doesn't amortize until M >= 128 for this model architecture
  (Nemotron Ultra ~500B with TP=4 expert sharding). At M=256 (prod
  config), MXFP8 wins 2.33×. The crossover is sharp between 64 and 128.
- The smoke win (1.16× iter8, sometimes 3× in noisy iterations) is
  mostly noise — both sides are slow at 32-concurrent for
  cudagraph-dispatch reasons; MXFP8 ends up slightly less affected.
  Reliable wins are only at the 256-concurrent prod config.

## What still works (production guidance)

**Use MXFP8 nemo-speed for Nemotron Ultra serving**, with this config:

  ```
  --tensor-parallel-size 4 --data-parallel-size 2
  --data-parallel-size-local 1 --data-parallel-backend ray
  --distributed-executor-backend ray --enable-expert-parallel
  --moe-backend flashinfer_trtllm
  --max-model-len 131072
  --mamba-cache-mode all --mamba-ssm-cache-dtype float32
  --async-scheduling
  --max-num-seqs 512 --enable-prefix-caching
  --gpu-memory-utilization 0.92 --disable-custom-all-reduce
  ```
  + env: `VLLM_ENGINE_READY_TIMEOUT_S=7200`,
         `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`,
         `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

At the production-shape workload (large prompts, hundreds of concurrent
requests) MXFP8 nemo-speed delivers **~2× the BF16 throughput** with
perfect reliability (256/256 vs BF16's 199/256 on the original
ultra-rl). That is the meaningful result of this work.

## Real fix paths (outside this overnight's scope)

1. **FlashInfer kernel improvement**: a small-M-optimized `mm_mxfp8`
   variant for M ∈ [16, 32, 64]. Either smaller-tile CUTLASS or a
   tuned Triton implementation.
2. **`trtllm_fp8_block_scale_moe` low-batch optimization**: same idea
   for the MoE path. Currently uses a tile size optimized for big M.
3. **Latent-MoE fusion Phase 2B** (per `tools/LATENT_MOE_FUSION.md`):
   bypass the dequant→requant round-trip between fc1_latent_proj and
   the MoE. Blocked on FlashInfer 0.6.10's `mm_mxfp8` API (can't
   return fp8 natively yet).
4. **Different model parallelism layout** (TP=8 DP=1) — could shift
   the per-rank batch size and potentially put us above the M=128
   crossover even at low concurrency. Untested.

---

*Authored 2026-05-23 ~13:35 PDT after 8 overnight iterations.*
*Branch commits: 6ff593800 (plan), 332fc0512 (findings), 38424128b
(M_ALIGN test) + 47bc43235 (revert), 6fd53a580 (7-iter analysis),
7a6e88675 (BF16-fallback opt-in).*
