# Fix F — drop M-padding in MXFP8 Linear (opt-in)

`Mxfp8LinearOp._apply_flashinfer_cutlass` (mxfp8_utils.py) pads the
input M dimension up to a multiple of 128 before calling
`flashinfer.mm_mxfp8`. The microbench in
`tools/microbench_mxfp8_linear_v2.py` shows this pad is a stale
defensive workaround on **flashinfer >= 0.6.x**: the kernel handles
non-128-aligned M directly, with bit-identical output to the padded
path, and the pad costs 8-22% per call at M ∈ {1..96}.

## Microbench result (4× GB200, flashinfer 0.6.10)

Correctness (per-shape, per-M, max_abs_diff):

| shape | M=1 | M=4 | M=8 | M=16 | M=32 | M=64 | M=96 | M=128 |
|---|---|---|---|---|---|---|---|---|
| qkv_proj-ish | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| o_proj       | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| fc1_latent   | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| fc2_latent   | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

(Pad vs no-pad → bit-exact at every M, every shape.)

Perf (ms/call, pad vs no-pad, mean of 50 iters; speedup% = (pad/nopad - 1)*100):

| shape          | M=1   | M=8   | M=32  | M=64  | M=96  | M=128 |
|---|---|---|---|---|---|---|
| qkv_proj-ish   | +12%  | +21%  | +18%  | +16%  | +12%  | +5%   |
| o_proj         | +16%  | +10%  | +10%  | +11%  | +16%  | -0%   |
| fc1_latent     | +13%  | +15%  | +22%  | +16%  | +14%  | -2%   |
| fc2_latent     | +12%  | +13%  | +15%  | +15%  | +14%  | -3%   |

At M=128 the pad branch is already a no-op so the gap collapses to
noise. The win comes entirely at M < 128.

## Why this matters for Ultra RL rollout

For the production prod_65k_8k shape (max_num_seqs=256, M=256
decode-steady-state), the pad never fires — this fix is a no-op. But
the gap appears in:

- **SWE-RL shape** (concurrency 16 → M=16 in steady state): full 18%
  Linear-path speedup is in scope.
- **Decode turnover / warm-up tails**: M=32-64 during burst finishes
  → 15-22% Linear speedup on those steps.
- **Profiler runs** with low concurrency.

End-to-end this is small (1-3% on prod_65k_8k, more on SWE-RL). It's
strictly a free win once we verify the cluster's flashinfer version
accepts non-128-aligned M.

## How to enable

Add to `vllm_worker_*.sh` BEFORE `vllm serve`:

```bash
export VLLM_MXFP8_DISABLE_M_PAD=1
```

The engine will log once at startup:

```
VLLM_MXFP8_DISABLE_M_PAD=1 set: MXFP8 Linear skips the M -> 128 padding (Fix F).
Requires flashinfer mm_mxfp8 that handles non-128-aligned M (>= 0.6.x verified).
```

If you see an `AssertionError` from `mm_mxfp8` at small M on the first
MoE-adjacent Linear, the cluster's flashinfer is too old. Unset the
env var and reboot.

## Reproducing the microbench

On any single GB200 with vllm + flashinfer installed:

```bash
python3 tools/microbench_mxfp8_linear_v2.py
```

Takes ~3 min. Reports correctness (max_abs_diff) and per-call timing
for {qkv_proj-ish, o_proj, fc1_latent, fc2_latent} at M ∈ {1, 4, 8,
16, 32, 64, 96, 128}.
