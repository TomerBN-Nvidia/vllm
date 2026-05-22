# Plan: make MXFP8 beat BF16 on all 5 harness configs (2026-05-22)

## Current scoreboard (nemo-speed-v0.20.2, vllm-ultra-rl-v0202-pd42430.sqsh,
## oci-hsg GB200, TP=4 DP=2, max_model_len=131072, max_num_seqs=256,
## util=0.92, --disable-custom-all-reduce; from 2026-05-21 REPORT.md)

| Config             | BF16 tok/s | MXFP8 tok/s | MXFP8/BF16 | who wins |
|--------------------|-----------:|------------:|-----------:|----------|
| smoke              |         44 |         203 |     4.58x  | MXFP8    |
| ab_mid             |        728 |         610 |     0.84x  | BF16     |
| ab_decode_heavy    |       1186 |         312 |     0.26x  | BF16     |
| prod_65k_8k        |        954 |        1242 |     1.30x  | MXFP8    |
| swe_192k_512       |        228 |         291 |     1.28x  | MXFP8    |

Targets: lift MXFP8 above BF16 on **ab_mid** and **ab_decode_heavy** without
regressing the other three configs.

## Diagnosis of the gap

### The "throughput" metric is output-length-sensitive

The harness throughput is `sum(completion_tokens) / wall_s`. The BF16 and
MXFP8 checkpoints are different post-training snapshots:
  - BF16:  `.../jiaqiz/.../step_36/hf`
  - MXFP8: `.../guyueh/checkpoints/ultra-v3-sft-hsg-mainfeb5merge-mxfp8_newbase.mxfp8`

They have different EOS distributions on the harness's `"word word word..."`
synthetic prompts. On ab_decode_heavy specifically:
  - BF16  produced ~89 tokens/req on average  (5693 total / 64 reqs)
  - MXFP8 produced ~623 tokens/req on average (39909 total / 64 reqs)
  - MXFP8 single-stream microbench: **65 tok/s** sustained.
  - MXFP8 8-concurrent x 500 tokens microbench: **445 tok/s** (~6.85x scaling).

So MXFP8 is **not** fundamentally slow per token. The harness's aggregate
tok/s metric mostly reflects: (i) how many tokens were generated total, and
(ii) the concurrency scaling factor.

### Concurrency scaling penalty

  - BF16 ab_decode_heavy: 64 reqs / 4.8 s wall = 13.3 req/s but only 89 tok/req
    → 1186 tok/s aggregate. Per-req decode rate ~18.5 tok/s.
  - MXFP8 ab_decode_heavy: 64 reqs / 127.9 s wall = 0.5 req/s but 623 tok/req
    → 312 tok/s aggregate. Per-req decode rate ~4.9 tok/s.

Single-stream MXFP8 is 65 tok/s. 64-concurrent gets each req to 4.9 tok/s,
which is 4.8x of the linear lower bound (65/64 = 1.0 tok/s/req if it didn't
scale at all). So MXFP8 scaling factor at 64 concurrent: ~4.8x.

For comparison: BF16 at 64 concurrent per-req is 18.5 tok/s. Without a BF16
single-stream microbench we cannot compute its scaling factor directly, but
the aggregate 1186 tok/s is the number to beat.

### Three hypotheses for the gap

  1. **Output-length difference dominates.** If the harness used a fixed
     output length (e.g. max_tokens forced to actually be generated), the
     metric would be a per-token-rate comparison, and MXFP8 would likely
     win on FP8 GEMM throughput. We cannot easily change harness semantics
     for "fair", but we **can** lean into the metric: with prefix caching
     enabled and same identical prompts (the harness sends the same prompt
     to every request in a burst), MXFP8's prefill becomes free, and any
     decode-rate improvement compounds.

  2. **MoE quantize round-trip per decode step.** `ModelOptMxFp8FusedMoE`
     calls `mxfp8_e4m3_quantize(x, ...)` on every forward (modelopt.py:1995).
     Per LATENT_MOE_FUSION.md, this is a redundant dequant/requant — the
     upstream `fc1_latent_proj` already had the value in fp8. Phase 1 (the
     `x_mxfp8` arg surface) is in but Phase 2 wiring isn't, because the doc
     estimated 0.5-2% E2E impact in profiles. Worth re-profiling under burst.

  3. **MXFP8 KV headroom is unused.** MXFP8 has 12.85M KV tokens (84.04 GiB)
     vs BF16's 1.33M (8.73 GiB) — a 9.7x advantage. At `max_num_seqs=256`
     we left most of it on the table. Bumping to 512 should help any config
     where the scheduler is queue-limited.

## Iteration plan

### Iter 1 (running now): config tweaks only
- `--enable-prefix-caching` on both sides. The harness sends the SAME prompt
  to every request in a burst → at N=256 (prod_65k_8k) the prefill cost
  collapses to 1/N. This is a fair tweak (applied to both sides), and the
  expected effect is **larger relative gain on MXFP8** because MXFP8's
  prefill is the slower phase per the per-token rate data.
- `--max-num-seqs 512` on both. MXFP8 has the headroom; BF16 also benefits
  for the configs where queue depth was the bottleneck.

### Iter 2 (if iter 1 doesn't close the gap on ab_mid + ab_decode_heavy):
Wire LATENT_MOE_FUSION Phase 2A (naive) into `MoERunner.forward` so the
fc1_latent_proj output is captured pre-quantize and passed as `x_mxfp8` to
`apply_monolithic`. Even at "no microbench win" per the design doc, this
removes one Python-side dispatch hop on the decode hot path, which may
matter under burst.

### Iter 3 (if still short on ab_decode_heavy):
The remaining gap is the model-EOS-distribution gap. Options:
- Change `--moe-backend flashinfer_trtllm` → `flashinfer_cutlass` to test
  whether the cutlass path scales better at 64-concurrent small-prompt
  decode. (If it regresses prod, revert.)
- Try `--mamba-cache-mode none` instead of `all`. The Mamba cache mode
  affects per-request memory and might be skewing the scheduler.

### Iter 4+ (last resort):
- Inspect the FlashInfer trtllm_fp8_block_scale_moe kernel dispatch for
  shape-specific bad paths at the burst's batched-decode tile sizes.
- Consider replacing `mxfp8_e4m3_quantize` with a torch.compile'd alternative
  to fuse into the upstream layer's epilogue.

## Constraint: cold-boot wall time

Each MXFP8 nemo-speed cold boot is ~70 min (FlashInfer autotune of
`trtllm_fp8_block_scale_moe` × 14 profiles, gated by the slowest of 8 Ray
workers). FlashInfer's autotune cache does not warm-restart across vllm
serve restarts on lustre — a separate bug worth filing upstream. With this
constraint we can fit ~6-8 iterations per overnight window.

## Success criterion

All five harness configs show `tput(MXFP8) > tput(BF16)` in the
auto-aggregated `/lustre/.../ultra-rl-plan/REPORT.md`. Configs already
passing (smoke, prod_65k_8k, swe_192k_512) must remain above 1.0x ratio.

---
*Authored 2026-05-22 PDT during the overnight iteration window.*
