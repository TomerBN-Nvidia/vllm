# MNNVL boot hang — root cause and fix

vLLM 0.17 on the `vllm-hsg-03-16.sqsh` container hangs after weight
load on 2-node GB200 TP=8 setups, with repeated:

```
File "/opt/vllm-build/flashinfer/flashinfer/comm/mnnvl.py", line 993, in __del__
    if self.signal_pads_dev:
AttributeError: 'SymmDeviceMemory' object has no attribute 'signal_pads_dev'
```

Six rounds of launch-config knobs (`VLLM_ALLREDUCE_USE_SYMM_MEM=0`,
`VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`, `--disable-custom-all-reduce`,
`NCCL_MNNVL_ENABLE=0`, removing `VLLM_FLASHINFER_ALLREDUCE_BACKEND`) did
not fix it. The trigger is a vLLM compile pass, not an allreduce
config.

## Fix

Add to `vllm serve`:

```
--compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'
```

This matches Konuk's production launch and disables the
`AllReduceFusionPass` registration. With the pass disabled,
`flashinfer.comm.create_allreduce_fusion_workspace()` is never called,
and FlashInfer's MNNVL `SymmDeviceMemory` bootstrap never runs.

Verify in the engine config dump on startup:

```
'pass_config': {..., 'fuse_allreduce_rms': False}
```

## Trigger trace (read FlashInfer + vLLM source from the container)

```
vLLM compile pass `allreduce_rms_fusion` (registered iff
  pass_config.fuse_allreduce_rms == True)
  → _init_workspace() in
      vllm/distributed/device_communicators/flashinfer_all_reduce.py:67
  → flashinfer_comm.create_allreduce_fusion_workspace(...)
  → backend heuristic picks "mnnvl"
  → MNNVLAllReduceFusionWorkspace(...)   [flashinfer/comm/allreduce.py:433]
  → McastGPUBuffer(...)                  [flashinfer/comm/trtllm_mnnvl_ar.py:135]
  → SymmDeviceMemory(...)                [flashinfer/comm/mnnvl.py:1323]
  → CUDA P2P probe across NVL72 blades — hang
```

Gates investigated and found NOT to gate this path:

| Knob | Effect on MNNVL path |
|---|---|
| `VLLM_ALLREDUCE_USE_SYMM_MEM=0` | no effect (gates a different SymmMem) |
| `--disable-custom-all-reduce` | no effect (gates vLLM's own all-reduce, not the compile-pass init) |
| `NCCL_MNNVL_ENABLE=0` | no effect (FlashInfer probes CUDA P2P directly, doesn't consult NCCL env) |
| Removing `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm` | no effect (the compile pass still calls `create_allreduce_fusion_workspace` and the heuristic still picks mnnvl) |
| `--compilation-config pass_config.fuse_allreduce_rms=false` | **WORKS** — pass never registers |

## Why this trips on 2-node GB200 TP=8 specifically

`is_mnnvl_fabric_supported(device_idx)` at `mnnvl.py:849`:

1. Queries `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED` — true on
   GB200 (always).
2. Queries `nvmlDeviceGetGpuFabricInfoV` for `state >=
   NVML_GPU_FABRIC_STATE_COMPLETED` and `clusterUuid[0] != 0` — true on
   GB200 NVL72.

So the heuristic in `create_allreduce_fusion_workspace` always picks
`"mnnvl"` on GB200, then `SymmDeviceMemory.__init__` tries to
peer-connect via fabric handles. If the two TP nodes are on separate
NVL72 compute blades (no cross-blade NVLink), peer connect hangs.
Single-blade allocations (consecutive `T[N,N+1]` within one rack) may
or may not be safe depending on SLURM placement — fuse_allreduce_rms=false
makes it irrelevant either way.

## How to investigate similar hangs

The squashfs container can be inspected from the login node without a
SLURM allocation:

```bash
# Locate the subtree
unsquashfs -ll /lustre/.../containers/vllm-hsg-03-16.sqsh | grep mnnvl

# Extract only what you need
unsquashfs -d /tmp/sqfs /lustre/.../containers/vllm-hsg-03-16.sqsh \
  opt/vllm-build/flashinfer/flashinfer/comm \
  opt/vllm-build/vllm/vllm/distributed \
  opt/vllm-build/vllm/vllm/compilation
```

Then trace callers from the leaf (e.g. `SymmDeviceMemory`) upward:

```bash
grep -rn "SymmDeviceMemory(" /tmp/sqfs/  # call sites
grep -rn "McastGPUBuffer(" /tmp/sqfs/    # one level up
# ...
```

Login-node `srun` is blocked by `QOSMinGRES` so don't try to launch
a container just to grep — extract via unsquashfs.

## Cost of the fix

`AllReduceFusionPass` fuses inter-GPU allreduce with the following
RMSNorm into a single kernel, saving one round trip through HBM per
layer. Disabling it reverts to two separate kernels (NCCL allreduce
then RMSNorm). For 2-node GB200 TP=8 with NemotronH (~50% MoE layers,
~12-14 attention layers, ~50% Mamba), the perf cost is small relative
to the rest of the step time. Konuk's production launch runs with
this disabled, so we're inheriting the same baseline.

A proper future fix would expose a FlashInfer-level flag to force the
backend to `"trtllm"` (intra-blade IPC only) and skip the MNNVL probe.
The vLLM-side call could then re-enable `fuse_allreduce_rms` safely.
Not blocking for the GA window.
