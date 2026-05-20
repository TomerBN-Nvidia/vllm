# Round 9 findings — Fix C is incompatible with NemotronH out-of-the-box, plus a concurrent-burst engine deadlock

Investigation date: 2026-05-19. Cluster: oci-hsg GB200 NVL72 2-node TP=8.
Container: `vllm-hsg-03-16.sqsh`. vLLM branch: `ultra-rl-v0.17` @
`d229daa09` with `nemo-speed` Fix C (`0ad4301a2`) and Fix F
(`a8fbf96df`) cherry-picked.

## Fix C bug 1 — `TypeError: unhashable type: 'list'` in info_once call

**Symptom:**

```
File "vllm/model_executor/layers/quantization/modelopt.py", line 163, in __init__
    logger.info_once(
        "VLLM_MODELOPT_EXTRA_EXCLUDE_MODULES: appended %d "
        "pattern(s) to ModelOpt exclude_modules: %s",
        len(extra_patterns), extra_patterns,
    )
File "vllm/logger.py", line 137, in info_once
    _print_info_once(self, msg, *args)
TypeError: unhashable type: 'list'
```

**Root cause:** `vllm/logger.py:137` types `info_once`'s args as
`Hashable` because the dedupe LRU hashes them to detect repeats. The
Fix C call site passes `extra_patterns` (a `list[str]`) directly,
which is not hashable. This is a real bug in Fix C, not a
port-only issue — it would trip on any vLLM that uses the same
`info_once` signature.

**Fix (applied as commit `3571d7095` on local ultra-rl-v0.17-with-fixes
branch; should be applied to `nemo-speed` too):**

```python
# Replace:
len(extra_patterns), extra_patterns,
# With:
len(extra_patterns), ", ".join(extra_patterns),
```

Or use `tuple(extra_patterns)` if you want the list-y display.

## Fix C bug 2 — `KeyError: weight_scale` when latent_proj layers are excluded

**Symptom:**

```
File "vllm/model_executor/models/nemotron_h.py", line 781, in load_weights
    param = params_dict[name]
KeyError: 'layers.33.mixer.fc1_latent_proj.weight_scale'
```

**Root cause:** `tools/EXCLUDE_RECIPE.md` recommends excluding
`*fc1_latent_proj` and `*fc2_latent_proj` from MXFP8 (load as BF16).
But the MXFP8 **checkpoint** (`guyueh/...mxfp8_newbase.mxfp8` and
similar) emits a `weight_scale` parameter for every quantized layer,
including those latent_projs. When Fix C excludes them, the model
builds them as BF16 `Linear` layers with no `weight_scale` parameter.
NemotronH's loader iterates checkpoint weights and does
`params_dict[name]` without a fallback — `KeyError` on the
unexpected `weight_scale`.

**Two ways to fix:**

### Fix C.1 (model-side) — gracefully skip unknown `weight_scale`

Patch `vllm/model_executor/models/nemotron_h.py` around the
`params_dict[name]` lookup (line ~781). Sketch:

```python
if name not in params_dict:
    # Tolerate dangling weight_scale params from the MXFP8 checkpoint
    # when the corresponding Linear was excluded from MXFP8 by Fix C
    # (loaded as BF16, no weight_scale).
    if name.endswith(".weight_scale"):
        continue
    raise KeyError(name)
param = params_dict[name]
```

This is the cleanest. Make sure to skip the `loaded_params.add(name)`
bookkeeping below too if the loader tracks loaded weights.

### Fix C.2 (config-side) — narrow the exclude pattern

If the model patch isn't available, narrow
`VLLM_MODELOPT_EXTRA_EXCLUDE_MODULES` to omit the latent_proj entries:

```bash
export VLLM_MODELOPT_EXTRA_EXCLUDE_MODULES="*qkv_proj,*o_proj,*gate,*mixer.in_proj,*mixer.out_proj,*mixer.conv1d"
```

Loses the latent_proj exclusion (per `EXCLUDE_RECIPE.md` worth a
small share of the expected 3-8% gain) but works without model
changes.

## Concurrent-burst engine deadlock (NEW runtime bug)

Round 9 baseline booted to `Application startup complete` and served a
single sequential request in 22 s (first inference includes JIT
compile of FlashInfer cubins at runtime shapes). Then under T0.1
smoke harness (32 concurrent OpenAI-protocol requests, 1024 prompt /
2048 decode each), the engine:

```
16:17:00   9 Running, 23 Waiting, 831 prompt-tok/s
16:17:10   32 Running, 2495 prompt-tok/s + 192 gen-tok/s   ← productive
16:17:20   32 Running, 0.0 / 0.0 tok/s                    ← STUCK
16:18+    Engine log silent. /health returns 200 OK. KV cache 2%.
```

Engine processed the burst productively for ~10 s, then deadlocked.
`RAY_CGRAPH_get_timeout=3600` prevents the Ray DAG auto-kill but
doesn't fix the underlying hang. 32 requests sit in-flight forever.

**Probable causes (need py-spy or gdb on the compute node to confirm
— not currently accessible):**
- CUDA graph mismatch under the runtime shape distribution from 32
  concurrent reqs (captured shapes don't cover one or more
  variants). cudagraph_capture_sizes covers up to 512 token batch.
  Might be a partial shape that escapes the captured set and falls
  back to eager, which deadlocks.
- NCCL collective deadlock under high concurrency.
- FlashInfer kernel hang at this batch+shape combo.

**Diagnostic next steps:**
1. Restart baseline. Repeat warmup. Run harness with concurrency=1,
   2, 4, 8, 16 in order. Find the inflection where it breaks.
2. Try `--max-num-seqs 8` to see if upper-bounded concurrency stays
   alive.
3. Get key trust to the compute node (`scontrol show job ...` shows
   `nvl72NNN-TXX` head — SSH directly from login node may or may not
   work) and run `py-spy dump --pid <head Python PID>` on a hung
   worker.

**For tomorrow:** this is a hard runtime bug that's separate from
the Fix C work above. Diagnose it independently; once boot works at
the target concurrency, the kernel-perf experiments from
`POST_BOOT_EXPERIMENTS.md` become unblocked.

## Recipe summary — running Fix C + Fix F on `ultra-rl-v0.17`

This is the working recipe (with the bugs above accounted for):

```bash
# One-time setup on the cluster:
cd /lustre/.../riship
git clone --branch ultra-rl-v0.17 https://github.com/TomerBN-Nvidia/vllm.git vllm-ultra-rl-with-fixes
cd vllm-ultra-rl-with-fixes

# nemo-speed must be a full clone (not --depth shallow) — unshallow if needed:
#   cd ../vllm-nemo-speed && git fetch --unshallow

git remote add nemo-speed /lustre/.../riship/vllm-nemo-speed
git fetch nemo-speed
git checkout -b ultra-rl-v0.17-with-fixes
git cherry-pick 0ad4301   # Fix C → 86bdd5351
git cherry-pick a8fbf96   # Fix F → e49fe3955

# Apply the info_once Hashable fix on top of cherry-picked Fix C:
# sed -i 's/len(extra_patterns), extra_patterns,/len(extra_patterns), ", ".join(extra_patterns),/' \
#   vllm/model_executor/layers/quantization/modelopt.py
# git add vllm/model_executor/layers/quantization/modelopt.py
# git commit -m "modelopt: render extra_patterns list as string for info_once"

# In the worker script, set:
#   BRANCH_SRC="$WORK_DIR/vllm-ultra-rl-with-fixes"
#   export VLLM_MODELOPT_EXTRA_EXCLUDE_MODULES="*qkv_proj,*o_proj,*gate,*mixer.in_proj,*mixer.out_proj,*mixer.conv1d"
#                                              # ← drop *fc1_latent_proj,*fc2_latent_proj for now
#   export VLLM_MXFP8_DISABLE_M_PAD=1
#   export RAY_CGRAPH_get_timeout=3600
```

Once the model-side `weight_scale` skip patch lands (Fix C.1), the
latent_proj patterns can be added back to the exclude list.
