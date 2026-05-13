# Per-layer-type profiling (Fix A from the perf plan)

`tools/profile_one_burst.py` + `tools/summarize_profile.py` give a
~5 min answer to "which NemotronH layer type dominates a step?" — the
prerequisite for prioritising the other MXFP8 fixes (C, D, E).

## On the cluster — launch flags

Add to `vllm_worker_<tag>.sh`, inside the `vllm serve` invocation:

```bash
vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $PORT \
    --tensor-parallel-size 8 \
    --distributed-executor-backend ray \
    --disable-custom-all-reduce \
    --gpu-memory-utilization 0.90 \
    --max-model-len 131072 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --profiler torch \
    --torch-profiler-dir /lustre/fsw/portfolios/llmservice/users/riship/profile_traces/${SLURM_JOB_NAME}_${SLURM_JOB_ID} \
    --delay-iterations 20 \
    --max-iterations 5 \
    --no-torch-profiler-with-stack \
    --no-torch-profiler-record-shapes \
    > ...
```

Notes:
- `--profiler torch` switches on the harness; profiling stays *armed*
  but inactive until /start_profile is hit.
- `--delay-iterations 20` lets the engine reach decode steady state.
- `--max-iterations 5` keeps the trace small (one trace per worker
  process; with stack frames off, ~30-100 MB per worker).
- `--no-torch-profiler-with-stack` is critical: with-stack adds 5-10×
  overhead and bloats traces past the chrome-tracing limit.
- `--no-torch-profiler-record-shapes` for the same reason; we can
  re-enable later if shape inspection is needed.
- The directory MUST be absolute and must exist (or the worker will
  create it). Per-worker traces land as
  `<pid>_<hostname>_<timestamp>.pt.trace.json.gz`.

## Trigger one capture

Once `/health` is green, on a host that can reach the vLLM tunnel
(your laptop or this container):

```bash
python3 tools/profile_one_burst.py \
    --base-url $V17RL_URL \
    --model "$V17_MODEL" \
    --num-requests 64 --concurrency 64 --max-tokens 256
```

Behaviour:
1. Warm-up: 8 small requests (defaults).
2. POST `/start_profile`.
3. Burst of `--num-requests` requests at `--concurrency`. With 256-token
   max_tokens and 64 concurrent, this is ~8 steady-state decode steps —
   long enough for the 5-iteration profile window inside.
4. POST `/stop_profile` (safety; profiler auto-stops at max-iterations).

The trace files are written by each worker into the
`--torch-profiler-dir` you set.

## Pull the trace

```bash
ssh -i ~/.ssh/id_ed25519 riship@oci-hsg-cs-001-login-01.nvidia.com \
  'ls -la /lustre/fsw/portfolios/llmservice/users/riship/profile_traces/vllm-ultra-v17rl_*/'

# Pull the rank-0 trace (the engine head — most informative)
scp -i ~/.ssh/id_ed25519 \
  'riship@oci-hsg-cs-001-login-01.nvidia.com:/lustre/.../profile_traces/<jobname_jobid>/<rank0>.pt.trace.json.gz' \
  /tmp/
```

## Summarize

```bash
python3 tools/summarize_profile.py /tmp/<rank0>.pt.trace.json.gz --top 30
```

Output sections:
- **Totals**: wall time and total kernel/CPU op time.
- **Layer-type bucket — GPU kernel time**: % of step in {moe, mamba,
  attention, rmsnorm, quantize, allreduce, linear, embedding, misc,
  other}. **This is the actionable signal.** Whichever bucket is
  largest dictates which proposed fix (C/D/E) to do first.
- **Top 30 GPU kernels by total time**: confirms the bucket attribution
  for the highest hitters.

## What to look for, by hypothesis

| If bucket dominates | Then the highest-value fix is |
|---|---|
| `quantize` (>10% of kernel time) | **Fix C** (exclude small Linears from MXFP8) — quantize is paying per-Linear cost. Optionally Fix E (RMSNorm+quantize fusion). |
| `moe` (>50% of kernel time) | **Fix D** (latent-MoE fusion to skip dequant→requant) — confirms MoE is the bottleneck. Also escalate **Fix G** (top_k=22 kernel tuning) to Tomer. |
| `mamba` (>30% of kernel time) | Move §8.7 (Mamba kernel optimization) up to week 1. |
| `linear` (>30% of kernel time) | Cross-check against `quantize`: if quant is also high, Fix E is the win. If quant is low, the small-Linear hot path needs a different fix (TP/EP rethink, §8.6). |
| `allreduce` (>20% of kernel time) | TP=8 communication is dominating. §8.6 (DP=2 TP=4) becomes the win. |

If no single bucket exceeds ~20%, the workload is fragmented across
many small ops; this is the case where Python overhead (§8.5) becomes
worth profiling separately.

## Cluster-side hygiene

- Profile traces can be 50 MB - 1 GB per worker. Trim
  `--torch-profiler-dir` after each session.
- Captures run inside the worker process — *do not* enable profiling
  on a node still serving production traffic.
- The `--profiler torch` flag adds ~1-3% overhead even when /start_profile
  hasn't been triggered (idle harness setup). Acceptable for the brief
  diagnostic; remove from the worker after we collect what we need.
