#!/usr/bin/env python3
"""
Benchmark script comparing mxfp8_e4m3_quantize (flashinfer) vs mxfp8_e4m3_quantize_python.

This script measures only the quantization time, excluding kernel compilation
and launch overhead by using warmup iterations.
"""

import argparse

import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
    mxfp8_e4m3_quantize_python,
)

# Global compiled function - will be initialized in main() based on args
mxfp8_e4m3_quantize_compiled = None


def get_compiled_quantize(compile_mode: str = "reduce-overhead"):
    """
    Get a torch.compiled version of the Python implementation.
    
    Args:
        compile_mode: One of "default", "reduce-overhead", "max-autotune"
            - "default": Good balance of compile time and runtime
            - "reduce-overhead": Best for latency, uses CUDA graphs
            - "max-autotune": Best performance but longer compile time
    """
    return torch.compile(
        mxfp8_e4m3_quantize_python,
        mode=compile_mode,
        fullgraph=True,
    )


def benchmark_quantize(
    func,
    x: torch.Tensor,
    is_sf_swizzled_layout: bool,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
) -> tuple[float, float]:
    """
    Benchmark a quantization function using CUDA events for accurate GPU timing.
    
    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    assert x.is_cuda, f"Input tensor must be on CUDA, got {x.device}"
    
    # Warmup to exclude compilation and kernel launch overhead
    for _ in range(warmup_iters):
        _ = func(x, is_sf_swizzled_layout=is_sf_swizzled_layout)
    
    torch.cuda.synchronize()
    
    # Benchmark using CUDA events for accurate GPU timing
    times = []
    for _ in range(benchmark_iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        _ = func(x, is_sf_swizzled_layout=is_sf_swizzled_layout)
        end_event.record()
        
        # Wait for the events to complete
        end_event.synchronize()
        
        # Get elapsed time in milliseconds
        elapsed_ms = start_event.elapsed_time(end_event)
        times.append(elapsed_ms)
    
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return mean_time, std_time


def verify_correctness(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> bool:
    """Verify that both implementations produce similar results and outputs are on GPU."""
    q_flashinfer, scales_flashinfer = mxfp8_e4m3_quantize(
        x, is_sf_swizzled_layout=is_sf_swizzled_layout
    )
    q_python, scales_python = mxfp8_e4m3_quantize_python(
        x, is_sf_swizzled_layout=is_sf_swizzled_layout
    )
    
    # Verify outputs are on GPU
    assert q_flashinfer.is_cuda, "Flashinfer output not on CUDA!"
    assert q_python.is_cuda, "Python output not on CUDA!"
    assert scales_flashinfer.is_cuda, "Flashinfer scales not on CUDA!"
    assert scales_python.is_cuda, "Python scales not on CUDA!"
    
    # Compare quantized values (convert to float for comparison)
    q_match = torch.allclose(
        q_flashinfer.float(), q_python.float(), rtol=rtol, atol=atol
    )
    
    # Compare scales
    scales_match = torch.allclose(
        scales_flashinfer.float(), scales_python.float(), rtol=rtol, atol=atol
    )
    
    return q_match and scales_match


def run_benchmark(
    shapes: list[tuple[int, int]],
    dtype: torch.dtype = torch.bfloat16,
    is_sf_swizzled_layout: bool = False,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
    verify: bool = True,
):
    """Run benchmarks for multiple input shapes."""
    
    # Get GPU info
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    
    print(f"\n{'='*100}")
    print(f"MXFP8 Quantization Benchmark (GPU)")
    print(f"{'='*100}")
    print(f"Device: {gpu_name} (cuda:{device})")
    print(f"Config: dtype={dtype}, swizzled={is_sf_swizzled_layout}")
    print(f"        warmup_iters={warmup_iters}, benchmark_iters={benchmark_iters}")
    print(f"{'='*100}\n")
    
    header = (
        f"{'Shape':>16} | {'Flashinfer (ms)':>16} | {'Python (ms)':>16} | "
        f"{'Compiled (ms)':>16} | {'FI Speedup':>10} | {'Compile Speedup':>15}"
    )
    print(header)
    print("-" * len(header))
    
    results = []
    
    for M, N in shapes:
        # Clear CUDA cache to avoid memory fragmentation
        torch.cuda.empty_cache()
        
        # Generate random input (ensure N is divisible by 32 for both implementations)
        N_padded = ((N + 31) // 32) * 32
        x = torch.randn(M, N_padded, dtype=dtype, device="cuda")
        
        # Verify correctness first
        if verify:
            try:
                is_correct = verify_correctness(x, is_sf_swizzled_layout)
                if not is_correct:
                    print(f"WARNING: Results don't match for shape ({M}, {N_padded})!")
            except Exception as e:
                print(f"WARNING: Verification failed for shape ({M}, {N_padded}): {e}")
        
        # Benchmark flashinfer implementation
        try:
            fi_mean, fi_std = benchmark_quantize(
                mxfp8_e4m3_quantize,
                x,
                is_sf_swizzled_layout,
                warmup_iters,
                benchmark_iters,
            )
        except Exception as e:
            print(f"Flashinfer failed for shape ({M}, {N_padded}): {e}")
            fi_mean, fi_std = float('nan'), float('nan')
        
        # Benchmark python implementation (eager)
        try:
            py_mean, py_std = benchmark_quantize(
                mxfp8_e4m3_quantize_python,
                x,
                is_sf_swizzled_layout,
                warmup_iters,
                benchmark_iters,
            )
        except Exception as e:
            print(f"Python failed for shape ({M}, {N_padded}): {e}")
            py_mean, py_std = float('nan'), float('nan')
        
        # Benchmark compiled python implementation
        try:
            compiled_mean, compiled_std = benchmark_quantize(
                mxfp8_e4m3_quantize_compiled,
                x,
                is_sf_swizzled_layout,
                warmup_iters,
                benchmark_iters,
            )
        except Exception as e:
            print(f"Compiled failed for shape ({M}, {N_padded}): {e}")
            compiled_mean, compiled_std = float('nan'), float('nan')
        
        # Calculate speedups
        # FI Speedup: how much faster is flashinfer vs eager python
        if fi_mean > 0 and py_mean > 0:
            fi_speedup = py_mean / fi_mean
        else:
            fi_speedup = float('nan')
        
        # Compile Speedup: how much faster is compiled vs eager python
        if compiled_mean > 0 and py_mean > 0:
            compile_speedup = py_mean / compiled_mean
        else:
            compile_speedup = float('nan')
        
        shape_str = f"({M}, {N_padded})"
        fi_str = f"{fi_mean:.3f} ± {fi_std:.3f}"
        py_str = f"{py_mean:.3f} ± {py_std:.3f}"
        compiled_str = f"{compiled_mean:.3f} ± {compiled_std:.3f}"
        fi_speedup_str = f"{fi_speedup:.2f}x"
        compile_speedup_str = f"{compile_speedup:.2f}x"
        
        print(
            f"{shape_str:>16} | {fi_str:>16} | {py_str:>16} | "
            f"{compiled_str:>16} | {fi_speedup_str:>10} | {compile_speedup_str:>15}"
        )
        
        results.append({
            "shape": (M, N_padded),
            "flashinfer_ms": fi_mean,
            "flashinfer_std": fi_std,
            "python_ms": py_mean,
            "python_std": py_std,
            "compiled_ms": compiled_mean,
            "compiled_std": compiled_std,
            "fi_speedup": fi_speedup,
            "compile_speedup": compile_speedup,
        })
    
    print("\n" + "=" * 100)
    print("Summary:")
    print("  - FI Speedup > 1.0 means Flashinfer is faster than eager Python")
    print("  - Compile Speedup > 1.0 means torch.compile is faster than eager Python")
    print("=" * 100)
    
    return results


def run_3d_benchmark(
    shapes: list[tuple[int, int, int]],
    dtype: torch.dtype = torch.bfloat16,
    is_sf_swizzled_layout: bool = False,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
):
    """Run benchmarks for 3D input shapes (batched)."""
    
    # Get GPU info
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    
    print(f"\n{'='*100}")
    print(f"MXFP8 Quantization Benchmark (3D/Batched, GPU)")
    print(f"{'='*100}")
    print(f"Device: {gpu_name} (cuda:{device})")
    print(f"Config: dtype={dtype}, swizzled={is_sf_swizzled_layout}")
    print(f"        warmup_iters={warmup_iters}, benchmark_iters={benchmark_iters}")
    print(f"{'='*100}\n")
    
    header = f"{'Shape':>25} | {'Python (ms)':>18} | {'Compiled (ms)':>18} | {'Compile Speedup':>15}"
    print(header)
    print("-" * len(header))
    print("Note: Flashinfer doesn't support 3D inputs directly")
    print()
    
    for B, M, N in shapes:
        # Clear CUDA cache to avoid memory fragmentation
        torch.cuda.empty_cache()
        
        # Generate random input
        N_padded = ((N + 31) // 32) * 32
        x = torch.randn(B, M, N_padded, dtype=dtype, device="cuda")
        
        # Benchmark python implementation (eager)
        try:
            py_mean, py_std = benchmark_quantize(
                mxfp8_e4m3_quantize_python,
                x,
                is_sf_swizzled_layout,
                warmup_iters,
                benchmark_iters,
            )
        except Exception as e:
            print(f"Python failed for shape ({B}, {M}, {N_padded}): {e}")
            py_mean, py_std = float('nan'), float('nan')
        
        # Benchmark compiled python implementation
        try:
            compiled_mean, compiled_std = benchmark_quantize(
                mxfp8_e4m3_quantize_compiled,
                x,
                is_sf_swizzled_layout,
                warmup_iters,
                benchmark_iters,
            )
        except Exception as e:
            print(f"Compiled failed for shape ({B}, {M}, {N_padded}): {e}")
            compiled_mean, compiled_std = float('nan'), float('nan')
        
        # Calculate speedup
        if compiled_mean > 0 and py_mean > 0:
            compile_speedup = py_mean / compiled_mean
        else:
            compile_speedup = float('nan')
        
        shape_str = f"({B}, {M}, {N_padded})"
        py_str = f"{py_mean:.4f} ± {py_std:.4f}"
        compiled_str = f"{compiled_mean:.4f} ± {compiled_std:.4f}"
        compile_speedup_str = f"{compile_speedup:.2f}x"
        
        print(
            f"{shape_str:>25} | {py_str:>18} | {compiled_str:>18} | "
            f"{compile_speedup_str:>15}"
        )


def main():
    global mxfp8_e4m3_quantize_compiled
    
    parser = argparse.ArgumentParser(
        description="Benchmark MXFP8 quantization implementations"
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--swizzled",
        action="store_true",
        help="Use swizzled scale factor layout",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip correctness verification",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Input data type (default: bfloat16)",
    )
    parser.add_argument(
        "--include-3d",
        action="store_true",
        help="Include 3D (batched) benchmarks",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (default: reduce-overhead)",
    )
    args = parser.parse_args()
    
    # Initialize the compiled function with the selected mode
    print(f"Initializing torch.compile with mode='{args.compile_mode}'...")
    mxfp8_e4m3_quantize_compiled = get_compiled_quantize(args.compile_mode)
    
    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    # Define test shapes (M, N) - common sizes in LLM inference
    shapes_2d = [
        # Small shapes
        (128, 128),
        (256, 256),
        (512, 512),
        # Medium shapes (typical hidden sizes)
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        # Large shapes (larger models)
        (8192, 8192),
        # Rectangular shapes (common in attention/FFN)
        (1024, 4096),
        (4096, 1024),
        (2048, 8192),
        (8192, 2048),
        # Batch-like shapes (seq_len x hidden_dim)
        (1, 4096),
        (32, 4096),
        (128, 4096),
        (512, 4096),
        (1024, 4096),
        (2048, 4096),
    ]
    
    # Run 2D benchmarks
    run_benchmark(
        shapes_2d,
        dtype=dtype,
        is_sf_swizzled_layout=args.swizzled,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
        verify=not args.no_verify,
    )
    
    # Optionally run 3D benchmarks
    if args.include_3d:
        shapes_3d = [
            (2, 512, 4096),
            (4, 512, 4096),
            (8, 512, 4096),
            (16, 256, 4096),
            (32, 128, 4096),
        ]
        run_3d_benchmark(
            shapes_3d,
            dtype=dtype,
            is_sf_swizzled_layout=args.swizzled,
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
        )


if __name__ == "__main__":
    main()

