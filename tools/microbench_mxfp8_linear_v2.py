"""
Cross-check Fix F: does flashinfer.mm_mxfp8 actually handle M not aligned
to 128 correctly, with the same numerical output as the padded path?

Microbench v1 timed pad vs no-pad and showed 15-42% speedup at M ∈ {32,
64, 96} when the pad is skipped. v2 adds:
  - Correctness comparison (max_abs / max_rel diff between padded and
    unpadded outputs at each M).
  - Smaller M (1, 4, 8, 16) where CUTLASS may have a hard minimum tile.
"""
from __future__ import annotations

import math
import time

import torch
from flashinfer import mxfp8_quantize, mm_mxfp8

MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32


def swizzle_mxfp8_scale(sf: torch.Tensor, M: int, K: int) -> torch.Tensor:
    scaling_vector_size = MXFP8_BLOCK_SIZE
    factor = scaling_vector_size * 4
    num_m_tiles = (M + 127) // 128
    num_k_tiles = (K + factor - 1) // factor
    m_padded = num_m_tiles * 128
    k_scale_padded = num_k_tiles * 4
    scale_cols = K // scaling_vector_size
    sf_padded = torch.zeros((m_padded, k_scale_padded), dtype=sf.dtype, device=sf.device)
    sf_padded[:M, :scale_cols] = sf
    sf_reshaped = sf_padded.view(num_m_tiles, 4, 32, num_k_tiles, 4)
    sf_swizzled = sf_reshaped.transpose(1, 3)
    return sf_swizzled.contiguous().view(-1)


def build_weight(N: int, K: int):
    weight = torch.randint(
        0, 64, (N, K), dtype=torch.uint8, device="cuda"
    ).view(MXFP8_VALUE_DTYPE)
    weight_scale_2d = torch.randint(
        110, 130, (N, K // MXFP8_BLOCK_SIZE),
        dtype=MXFP8_SCALE_DTYPE, device="cuda",
    )
    weight_scale_swizzled = swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)
    return weight, weight_scale_swizzled


def call(M: int, N: int, K: int, weight, weight_scale, pad: bool):
    torch.manual_seed(M * 31 + K)  # deterministic per M, K
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    input_2d = x
    M_orig = M
    if pad:
        M_padded = ((M_orig + 127) // 128) * 128
        if M_padded != M_orig:
            input_2d = torch.nn.functional.pad(input_2d, (0, 0, 0, M_padded - M_orig))
    inp_q, inp_s = mxfp8_quantize(input_2d, is_sf_swizzled_layout=True)
    out = mm_mxfp8(inp_q, weight.t(), inp_s, weight_scale,
                   out_dtype=torch.bfloat16, backend="cutlass")
    if pad and out.shape[0] != M_orig:
        out = out[:M_orig, :]
    return out


def time_one(M: int, N: int, K: int, pad: bool, n_warm: int = 5, n_iter: int = 50):
    weight, weight_scale = build_weight(N, K)
    try:
        for _ in range(n_warm):
            _ = call(M, N, K, weight, weight_scale, pad)
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"
    torch.cuda.synchronize()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(n_iter):
        _ = call(M, N, K, weight, weight_scale, pad)
    ender.record()
    torch.cuda.synchronize()
    return starter.elapsed_time(ender) / n_iter, None


def correctness(M: int, N: int, K: int):
    weight, weight_scale = build_weight(N, K)
    try:
        out_pad = call(M, N, K, weight, weight_scale, pad=True)
        out_nop = call(M, N, K, weight, weight_scale, pad=False)
    except Exception as e:
        return None, f"call err: {type(e).__name__}: {str(e)[:120]}"
    diff = (out_pad.float() - out_nop.float()).abs()
    ref = out_pad.float().abs().clamp_min(1e-6)
    stats = {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float((diff / ref).max()),
        "mean_abs_pad": float(out_pad.float().abs().mean()),
    }
    return stats, None


def main():
    torch.cuda.set_device(0)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    shapes = [
        ("qkv_proj-ish",  8448,  8192),
        ("o_proj",        8192,  8192),
        ("fc1_latent",    2048,  8192),
        ("fc2_latent",    8192,  2048),
    ]
    Ms = [1, 4, 8, 16, 32, 64, 96, 128]

    print()
    print("=== Correctness (pad vs no-pad output diff) ===")
    print(f"{'shape':<14} {'M':>4} {'max_abs':>10} {'mean_abs':>10} "
          f"{'max_rel':>10} {'ref_avg':>10}  status")
    for label, N, K in shapes:
        for M in Ms:
            stats, err = correctness(M, N, K)
            if stats is None:
                print(f"{label:<14} {M:>4} {'':>10} {'':>10} {'':>10} {'':>10}  {err}")
            else:
                ok = stats["max_abs"] < 5e-3 and stats["max_rel"] < 5e-2
                tag = "PASS" if ok else "DIFFER"
                print(f"{label:<14} {M:>4} {stats['max_abs']:>10.4e} "
                      f"{stats['mean_abs']:>10.4e} {stats['max_rel']:>10.4e} "
                      f"{stats['mean_abs_pad']:>10.4e}  {tag}")

    print()
    print("=== Perf: ms/call (pad vs no-pad) ===")
    print(f"{'shape':<14} {'M':>4} {'pad_ms':>10} {'nopad_ms':>10} "
          f"{'speedup_%':>11}  notes")
    for label, N, K in shapes:
        for M in Ms:
            pad_ms, pad_err = time_one(M, N, K, pad=True)
            nop_ms, nop_err = time_one(M, N, K, pad=False)
            notes = []
            if pad_err: notes.append(f"pad_err: {pad_err}")
            if nop_err: notes.append(f"nopad_err: {nop_err}")
            if pad_ms is None or nop_ms is None:
                print(f"{label:<14} {M:>4} {'':>10} {'':>10} {'':>11}  {'; '.join(notes)}")
                continue
            su = (pad_ms / nop_ms - 1) * 100
            print(f"{label:<14} {M:>4} {pad_ms:>10.4f} {nop_ms:>10.4f} "
                  f"{su:>10.1f}%  {'; '.join(notes)}")


if __name__ == "__main__":
    main()
