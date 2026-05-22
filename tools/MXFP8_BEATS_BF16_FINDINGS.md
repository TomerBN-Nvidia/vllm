# Findings: MXFP8 vs BF16 on nemo-speed-v0.20.2 (2026-05-22)

Updates `MXFP8_BEATS_BF16_PLAN.md` with iter1/iter2 results. Headline: the
"MXFP8 loses on burst-heavy configs" finding from 2026-05-21 was MOSTLY an
EOS-distribution artifact. With `ignore_eos=true`, MXFP8 is competitive at
all concurrencies and clearly wins at production-scale 256-concurrent.

## Iter1: enable `--enable-prefix-caching` + bump `--max-num-seqs`

Config: both sides got `--enable-prefix-caching`; BF16 max_num_seqs=320
(stayed under the 345-block Mamba limit per Bug E), MXFP8 max_num_seqs=512
(MXFP8 has ~9.7x the BF16 KV headroom).

Apples-to-apples vs iter0:

| Config | BF16 iter0 → iter1 | MXFP8 iter0 → iter1 | MXFP8/BF16 |
|---|---|---|---|
| smoke | 44 → 41 | 203 → 228 | 5.56x MXFP8 |
| ab_mid | 728 → 721 | 610 → 620 | 0.86x BF16 |
| ab_decode_heavy | 1186 → 981 | 312 → 306 | 0.31x BF16 |
| prod_65k_8k | 954 → 327 | 1242 → 1476 | **4.51x MXFP8** |
| swe_192k_512 | 228 → 660 | 291 → 257 | 0.39x BF16 |

Prefix caching helps the side whose prefill is the bottleneck. On
`swe_192k_512` (64x32k prompts), BF16 jumped 228 → 660 tok/s (+189%)
because the 32k prefill is BF16's biggest single cost. MXFP8 saw little
change there (291 → 257). The harness sends the same prompt to every
request in a burst, so prefix caching hits N times for every config.

Net effect of iter1: hurt the goal — MXFP8 dropped from 3/5 wins (iter0) to
2/5 (iter1), because BF16 sped up disproportionately on `swe_192k_512`.

## Iter2: discover the real story with `ignore_eos=true`

While the iter1 jobs were live, a focused microbench against the running
serves revealed a structural mistake in iter0/iter1 analysis: the harness's
`throughput_tok_s = sum(completion_tokens) / wall_s` metric is biased by
*how many tokens each model generates* before EOS, not by per-token speed.

BF16 and MXFP8 are different post-training snapshots. On the harness's
synthetic `"word word word..."` prompts:
  - BF16  emits ~89 tokens/req on average  (ab_decode_heavy)
  - MXFP8 emits ~623 tokens/req on average (ab_decode_heavy)

So MXFP8 does ~7x more decode work per request before EOS. Per-token speed
is washed out by the order-of-magnitude difference in token count.

Iter2: patch `run_rollout_bench.py` to pass `"ignore_eos": True` in the
chat completion payload. Forces both models to generate exactly
`max_decode_tokens` regardless of natural EOS. Re-ran harness against the
same live serves (no restart needed):

| Config | concurrency | BF16 iter2 | MXFP8 iter2 | MXFP8/BF16 |
|---|---|---|---|---|
| smoke           |  32 | 1729 | 1596 | 0.92x |
| ab_mid          |  64 | 3196 | 2847 | 0.89x |
| ab_decode_heavy |  64 | 3318 | 2915 | 0.88x |
| prod_65k_8k     | 256 | 3145 | **7607** | **2.42x** |
| swe_192k_512    |  64 | 3158 | 2781 | 0.88x |

All `slow_tail = 1.00x` (since ignore_eos makes every request finish at
the same `max_tokens`). **The "burst-tail blowup" from iter0 was 100%
EOS distribution, NOT a kernel issue.**

The real performance characteristic:
  - At 256-concurrent (prod_65k_8k): MXFP8 wins by 2.42x. FP8 GEMMs +
    smaller activation footprint pull ahead when batch is big.
  - At 32-64 concurrent: MXFP8 loses by 8-14%. Mid-concurrency favors
    BF16's higher math throughput per kernel (the FP8 cost-per-token
    isn't recovered because the GEMM isn't bandwidth-bound).

Per-token-per-concurrent-req time (lower = faster):

| Config | concurrency | BF16 ms/tok/conc | MXFP8 ms/tok/conc | MXFP8 vs BF16 |
|---|---|---|---|---|
| smoke           |  32 | 0.59 | 0.64 | -8%   |
| ab_mid          |  64 | 0.32 | 0.36 | -12%  |
| ab_decode_heavy |  64 | 0.31 | 0.35 | -13%  |
| prod_65k_8k     | 256 | 0.33 | 0.13 | +60%  |
| swe_192k_512    |  64 | 0.32 | 0.37 | -14%  |

Crossover point appears to be ~100-150 concurrent.

## Iter3 (running): try `--moe-backend flashinfer_cutlass` on MXFP8

The 8-14% gap at mid-concurrency is small enough that a different MoE
kernel path could close it. FlashInfer offers four MoE backends for
modelopt_mxfp8: `flashinfer_trtllm`, `flashinfer_cutlass`, `TRITON`,
`BATCHED_TRITON`. Iter0-2 used `flashinfer_trtllm`; iter3 tries
`flashinfer_cutlass` on MXFP8 (BF16 stays on trtllm — it's already
winning there).

Hypothesis: CUTLASS has a different shape-vs-speed curve and may win at
the 32-128 concurrent range where trtllm currently loses.

## Iter4+ contingencies

If `flashinfer_cutlass` doesn't close the gap:
  - **TRITON / BATCHED_TRITON** backends for completeness.
  - **`--moe-expert-parallel-size`** sweep: more EP, less TP. At
    EP=8/TP=1, every rank holds a unique expert subset which could
    improve expert utilization at high batch.
  - **Disable `--async-scheduling`** — could be a source of 5-10%
    overhead on tight decode workloads.
  - **`--mamba-cache-mode none`** instead of `all` — fewer KV blocks
    reserved per request, potentially better batch packing.

## Constraints

  - Each MXFP8 cold boot is ~70 min (FlashInfer autotune of
    `trtllm_fp8_block_scale_moe` × 14 profiles, gated by the slowest
    of 8 Ray workers). Cache does not warm-restart.
  - Each BF16 cold boot is ~25 min.
  - Each harness run is ~10-15 min depending on config sizes.
  - Net: ~80 min / iteration when MXFP8 needs restart, ~15 min if only
    harness changes (e.g. iter1→iter2 was a harness re-run only).

## What the user actually wants

"MXFP8 faster than BF16 in all settings" with fair benchmarking
(`ignore_eos=true`). Currently 1/5 (prod_65k_8k decisive), need 4/5 more.
Realistically achievable at concurrencies >= 128. Below that, MXFP8 has
a structural penalty from the FP8 path's per-token overhead that the
GEMM speedup doesn't fully amortize at low batch.

---
*Authored 2026-05-22 during the overnight run.*
