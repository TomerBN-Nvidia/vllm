"""
Summarize a torch.profiler chrome trace (.json or .json.gz) by op-name
prefix and by raw kernel, for spotting which NemotronH layer types
dominate.

Goal: answer the "is it MoE, Mamba, MLP, or per-Linear quantize?"
question without having to load the trace into Perfetto / chrome://tracing.

Buckets reported:
  - Total kernel time (GPU-side)
  - Total CPU op time
  - Top-N raw GPU kernels
  - Aggregated by op-name prefix (vllm::*, flashinfer::*, aten::mm,
    aten::linear, etc.)
  - Heuristic NemotronH layer-type buckets:
        moe:        flashinfer_trtllm_*_moe, *MxFp8FusedMoE*, fused_moe
        mamba:      mamba, ssm, selective_scan
        attention:  flash_attn, paged_attn, attention
        rmsnorm:    rms_norm
        quantize:   mxfp8_quantize, scaled_fp4_quant, _custom_ops.mxfp8
        linear:     aten::linear, aten::mm, aten::addmm, mm_mxfp8
        other:      everything else

Usage:
    python3 summarize_profile.py <trace.json[.gz]>  [--top 30] [--bucket-only]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import io
import json
import re
from pathlib import Path


_LAYER_BUCKETS = [
    # (bucket_name, list_of_substring_patterns_case_insensitive)
    ("moe",       ["fused_moe", "flashinfer_trtllm", "mxfp8fusedmoe",
                   "MxFp8FusedMoE", "fp8_block_scale_moe", "moe_align",
                   "trtllm_bf16_moe", "mm_fp4", "scaled_fp4"]),
    ("mamba",     ["mamba", "ssm_state", "selective_scan", "causal_conv1d"]),
    ("attention", ["flash_attn", "paged_attn", "scaled_dot_product",
                   "varlen", "flashinfer_attn"]),
    ("rmsnorm",   ["rms_norm", "rmsnorm"]),
    ("quantize",  ["mxfp8_quantize", "scaled_fp4_quant", "fp8_quantize",
                   "nvfp4_quantize", "_custom_ops::mxfp8", "block_scale"]),
    ("allreduce", ["all_reduce", "allreduce", "nccl"]),
    ("linear",    ["aten::linear", "aten::mm", "aten::addmm", "aten::bmm",
                   "mm_mxfp8", "linear_method", "trtllm_fp8_per_tensor"]),
    ("embedding", ["embedding", "vocab", "lm_head"]),
    ("misc",      ["copy_", "to_", "view", "reshape", "contiguous", "cat",
                   "stack", "split"]),
]


def _open_trace(p: Path) -> dict:
    raw = p.read_bytes()
    if p.suffix == ".gz" or raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8", errors="replace"))


def _events(trace: dict) -> list[dict]:
    """Return the trace events list, handling both shapes vLLM emits."""
    if isinstance(trace, list):
        return trace
    return trace.get("traceEvents", [])


def _strip_args(name: str) -> str:
    """Normalize op-name: drop tensor-shape tail and tail of '< at::OperatorName>'."""
    name = re.sub(r"\[.*?\]", "", name)  # drop bracketed shape suffix
    name = re.sub(r"\s+<.*?>$", "", name)
    return name.strip()


def _bucket_for(name: str) -> str:
    lower = name.lower()
    for bucket_name, patterns in _LAYER_BUCKETS:
        for pat in patterns:
            if pat.lower() in lower:
                return bucket_name
    return "other"


def _is_kernel(ev: dict) -> bool:
    """Heuristic: events with cat 'kernel' OR with args.device='GPU'.
    Different trace versions use different conventions."""
    cat = (ev.get("cat") or "").lower()
    if "kernel" in cat or "gpu" in cat:
        return True
    args = ev.get("args") or {}
    return "device" in args and "gpu" in str(args["device"]).lower()


def _is_cpu_op(ev: dict) -> bool:
    cat = (ev.get("cat") or "").lower()
    return cat in {"cpu_op", "operator", "python_function"}


def summarize(trace_path: Path, top_n: int, bucket_only: bool) -> None:
    print(f"# Loading {trace_path} ...")
    trace = _open_trace(trace_path)
    events = _events(trace)
    print(f"# {len(events)} events parsed")

    kernel_by_name: dict[str, list[int]] = collections.defaultdict(list)
    cpu_by_name: dict[str, list[int]] = collections.defaultdict(list)
    kernel_bucket_us: dict[str, int] = collections.defaultdict(int)
    cpu_bucket_us: dict[str, int] = collections.defaultdict(int)
    total_kernel_us = 0
    total_cpu_us = 0
    trace_start: int | None = None
    trace_end: int | None = None

    for ev in events:
        if ev.get("ph") not in ("X", None):
            continue
        dur = int(ev.get("dur") or 0)
        if dur <= 0:
            continue
        ts = ev.get("ts")
        if isinstance(ts, (int, float)):
            ts_i = int(ts)
            if trace_start is None or ts_i < trace_start:
                trace_start = ts_i
            end = ts_i + dur
            if trace_end is None or end > trace_end:
                trace_end = end

        name = _strip_args(str(ev.get("name") or "<noname>"))
        if _is_kernel(ev):
            kernel_by_name[name].append(dur)
            kernel_bucket_us[_bucket_for(name)] += dur
            total_kernel_us += dur
        elif _is_cpu_op(ev):
            cpu_by_name[name].append(dur)
            cpu_bucket_us[_bucket_for(name)] += dur
            total_cpu_us += dur

    wall_us = (trace_end - trace_start) if (trace_start and trace_end) else 0

    print()
    print(f"## Totals  (wall ~ {wall_us / 1e3:.2f} ms)")
    print(f"  total_kernel_time = {total_kernel_us / 1e3:.2f} ms  "
          f"({100*total_kernel_us/wall_us:.1f}% of wall)"
          if wall_us else f"  total_kernel_time = {total_kernel_us / 1e3:.2f} ms")
    print(f"  total_cpu_op_time = {total_cpu_us / 1e3:.2f} ms")

    print()
    print("## Layer-type bucket — GPU kernel time")
    print(f"  {'bucket':<12} {'time_ms':>10} {'% kernel':>10}")
    for bucket, us in sorted(kernel_bucket_us.items(), key=lambda kv: -kv[1]):
        pct = 100 * us / max(total_kernel_us, 1)
        print(f"  {bucket:<12} {us/1e3:>10.2f} {pct:>9.1f}%")

    print()
    print("## Layer-type bucket — CPU op time")
    print(f"  {'bucket':<12} {'time_ms':>10} {'% cpu':>10}")
    for bucket, us in sorted(cpu_bucket_us.items(), key=lambda kv: -kv[1]):
        pct = 100 * us / max(total_cpu_us, 1)
        print(f"  {bucket:<12} {us/1e3:>10.2f} {pct:>9.1f}%")

    if bucket_only:
        return

    def _table(rows: dict[str, list[int]], label: str, total_us: int) -> None:
        print()
        print(f"## Top {top_n} {label} by total time")
        print(f"  {'count':>7} {'total_ms':>10} {'avg_us':>9}  name")
        items = [(n, sum(d), len(d)) for n, d in rows.items()]
        items.sort(key=lambda t: -t[1])
        for name, total_us_n, count in items[:top_n]:
            avg = total_us_n / max(count, 1)
            print(f"  {count:>7d} {total_us_n/1e3:>10.2f} {avg:>9.1f}  {name}")

    _table(kernel_by_name, "GPU kernels",  total_kernel_us)
    _table(cpu_by_name,    "CPU ops",      total_cpu_us)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path,
                    help="Path to chrome trace .json or .json.gz")
    ap.add_argument("--top", type=int, default=30,
                    help="Number of rows in the per-op tables (default 30)")
    ap.add_argument("--bucket-only", action="store_true",
                    help="Skip the per-op tables; print only bucket summary")
    args = ap.parse_args()
    summarize(args.trace, args.top, args.bucket_only)


if __name__ == "__main__":
    main()
