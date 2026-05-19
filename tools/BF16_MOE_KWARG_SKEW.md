# BF16 MoE `activation_type` kwarg skew (branch ↔ container)

On the `vllm-hsg-03-16.sqsh` container, the `ultra-rl-v0.17` branch
(BF16 MoE path) crashes during KV cache profiling:

```
TypeError: trtllm_bf16_moe() got an unexpected keyword argument 'activation_type'
```

The crash sits behind the MNNVL boot fix (see `MNNVL_BOOT_FIX.md`) —
prior rounds never got far enough to surface it. With
`fuse_allreduce_rms=false`, NemotronH on `ultra-rl-v0.17`:

- ✅ Loads all 225 BF16 shards
- ✅ Logs `Using FlashInfer TRTLLM backend for Unquantized MoE`
- ✅ Logs `Using HND KV cache layout for FLASHINFER backend`
- ❌ Crashes inside `determine_available_memory` on the first forward
   pass through a MoE layer.

## Where the skew is

**Branch** (`vllm-ultra-rl/vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py`,
`flashinfer_fused_moe_bf16`):

```python
return flashinfer_trtllm_bf16_moe(
    ...
    activation_type=activation_type,    # ← required for NemotronH (RELU2_NO_MUL)
    routing_replay_out=routing_replay_out,
)
```

Added by branch commit `14fa276fb8` (Tomer Barnatan, 2026-04-05),
"Fix BF16 trtllm-gen MoE weight corruption for non-gated".

**Container's FlashInfer**
(`/opt/vllm-build/flashinfer/flashinfer/fused_moe/core.py:2248`):

```python
def trtllm_bf16_moe(
    routing_logits, routing_bias, hidden_states,
    gemm1_weights, gemm2_weights,
    num_experts, top_k, n_group, topk_group,
    intermediate_size, local_expert_offset, local_num_experts,
    routed_scaling_factor=None,
    routing_method_type=0,
    use_shuffled_weight=True,
    weight_layout=WeightLayout.BlockMajorK,
    do_finalize=True,
    enable_pdl=True,
    tune_max_num_tokens=8192,
):
```

No `activation_type` parameter. The kernel hardcodes some default
activation, likely SwiGLU (gated). NemotronH is non-gated ReLU² so it
needs the explicit `RELU2_NO_MUL` enum.

**Container build date is 2026-03-16. Branch fix landed 2026-04-05** —
the container predates the FlashInfer side of the patch by ~3 weeks.

## MXFP8 path is NOT affected

`trtllm_fp8_block_scale_moe` in the same container *does* have:

```python
activation_type: int = ActivationType.Swiglu.value,
```

(at `core.py:~2400`, added by branch commit `c791141b4e` 2026-03-08,
predates container build). MXFP8 boot is structurally immune to this
skew. Only the BF16 path is broken.

## Fix options (ranked by correctness)

1. **Newer container.** Ask `dmosallanezh` (path:
   `/lustre/.../dmosallanezh/containers/`) for a FlashInfer ≥
   2026-04-05 build. Likely `vllm-hsg-04-XX.sqsh` exists. **Correct
   solution.**
2. **Sniff and strip with `inspect.signature`.** Patch the branch's
   `flashinfer_fused_moe_bf16` to introspect
   `flashinfer.fused_moe.trtllm_bf16_moe`'s signature and skip
   `activation_type` if absent. **Boots but applies wrong activation
   (kernel default, not RELU²) — useless for any real benchmark, fine
   as a smoke test of the rest of the stack.**
3. **Disable TRTLLM BF16 MoE.** Unset
   `VLLM_USE_FLASHINFER_MOE_FP16=1` and `VLLM_FLASHINFER_MOE_BACKEND`
   from the worker. Falls back to Triton MoE which receives the
   activation function as a Python callable and applies it correctly.
   **Correct output but loses AC#1 (TRTLLM MoE).**

## Smoke-test patch (option 2 — wrong output, debugging only)

If you need to boot v17rl just to verify the rest of the engine works,
add to `flashinfer_trtllm_moe.py` near top:

```python
import inspect as _inspect
from vllm.utils.flashinfer import flashinfer_trtllm_bf16_moe as _bf16_impl

try:
    _BF16_SIG = _inspect.signature(_bf16_impl)
    _BF16_SUPPORTS_ACTIVATION = "activation_type" in _BF16_SIG.parameters
except Exception:
    _BF16_SUPPORTS_ACTIVATION = True  # assume new path
```

then in `flashinfer_fused_moe_bf16` build kwargs and conditionally
include:

```python
kw = dict(routing_logits=..., gemm1_weights=..., ...)
if _BF16_SUPPORTS_ACTIVATION:
    kw["activation_type"] = activation_type
    kw["routing_replay_out"] = routing_replay_out
return flashinfer_trtllm_bf16_moe(**kw)
```

DO NOT MERGE THIS — it silently produces wrong NemotronH output. Land
the container upgrade (option 1) for real perf numbers.

## How this maps to AC#1

AC#1 is "TRTLLM-Gen MoE re-enabled on 0.17-ultra-rl (BF16 + MXFP8)".
MXFP8 side: confirmed via `Using 'FLASHINFER_TRTLLM' MxFp8 MoE backend`
and `ModelOptMxFp8FusedMoE` construction (once mxfp8 boot completes).
BF16 side: dispatch confirmed (`Using FlashInfer TRTLLM backend for
Unquantized MoE` in logs) but the kernel call itself fails until we
get a newer container. Half-green until then.
