#!/usr/bin/env python3
"""GEAK Test Harness - Softmax PoC

Loads baseline and optimized CK softmax kernel .so files via ctypes,
verifies the optimized kernel against the baseline (ground truth), and
reports bandwidth and speedup.

Usage:
    # Compare baseline vs optimized
    python test_harness.py libbaseline.so liboptimized.so

    # Custom shapes (semicolon-separated, dims comma-separated)
    python test_harness.py libbaseline.so liboptimized.so --shapes "8,128,2048;16,64,4096"

    # Control timing parameters
    python test_harness.py libbaseline.so liboptimized.so --warmup 10 --nrepeat 100

    # Skip verification (timing only)
    python test_harness.py libbaseline.so liboptimized.so --no-verify
"""

import argparse
import ctypes
import sys
from pathlib import Path

import torch


# -- Kernel loading via ctypes ------------------------------------------------


def load_kernel(path: str):
    """Load a kernel .so and set up the run_kernel function signature."""
    lib = ctypes.CDLL(path)
    lib.run_kernel.restype = ctypes.c_float
    lib.run_kernel.argtypes = [
        ctypes.c_void_p,  # in_dev
        ctypes.c_void_p,  # out_dev
        ctypes.POINTER(ctypes.c_int64),  # lengths
        ctypes.POINTER(ctypes.c_int64),  # strides
        ctypes.c_int,  # ndims
        ctypes.POINTER(ctypes.c_int),  # reduce_dims
        ctypes.c_int,  # n_reduce_dims
        ctypes.c_double,  # alpha
        ctypes.c_double,  # beta
        ctypes.c_bool,  # time_kernel
        ctypes.c_int,  # warmup
        ctypes.c_int,  # nrepeat
    ]
    return lib


def call_kernel(
    lib,
    x: torch.Tensor,
    y: torch.Tensor,
    reduce_dims: list[int],
    alpha: float = 1.0,
    beta: float = 0.0,
    time_kernel: bool = False,
    warmup: int = 5,
    nrepeat: int = 50,
) -> float:
    """Call run_kernel from a loaded .so. Returns time in ms (0 if not timing)."""
    ndims = x.ndim
    lengths = (ctypes.c_int64 * ndims)(*x.shape)
    strides = (ctypes.c_int64 * ndims)(*x.stride())
    rdims = (ctypes.c_int * len(reduce_dims))(*reduce_dims)

    torch.cuda.synchronize()
    ms = lib.run_kernel(
        x.data_ptr(),
        y.data_ptr(),
        lengths,
        strides,
        ndims,
        rdims,
        len(reduce_dims),
        alpha,
        beta,
        time_kernel,
        warmup,
        nrepeat,
    )
    return ms


# -- Test logic ---------------------------------------------------------------


def run_kernel_output(
    lib, x: torch.Tensor, reduce_dim: int
) -> torch.Tensor | None:
    """Run kernel on input x. Returns output tensor, or None if unsupported."""
    y = torch.empty_like(x)
    ms = call_kernel(lib, x, y, [reduce_dim], alpha=1.0, beta=0.0, time_kernel=False)
    if ms < 0:
        return None
    return y


def compare_outputs(
    y_opt: torch.Tensor, y_ref: torch.Tensor, tolerance: float = 1e-3
) -> tuple[bool, float]:
    """Compare optimized output against baseline. Returns (pass, max_err)."""
    max_err = (y_opt.float() - y_ref.float()).abs().max().item()
    passed = max_err < tolerance
    return passed, max_err


def benchmark_kernel(
    lib,
    shape: list[int],
    reduce_dim: int,
    warmup: int = 5,
    nrepeat: int = 50,
) -> float:
    """Time the kernel. Returns average time in ms, or -1 if unsupported."""
    x = torch.randn(shape, dtype=torch.float16, device="cuda")
    y = torch.empty_like(x)

    ms = call_kernel(
        lib,
        x,
        y,
        [reduce_dim],
        alpha=1.0,
        beta=0.0,
        time_kernel=True,
        warmup=warmup,
        nrepeat=nrepeat,
    )
    return ms


def compute_bandwidth(shape: list[int], time_ms: float) -> float:
    """Compute effective bandwidth in GB/s (read input + write output, FP16)."""
    numel = 1
    for s in shape:
        numel *= s
    bytes_moved = numel * 2 * 2  # 2 bytes per FP16, read + write
    return bytes_moved / time_ms / 1e6  # bytes / ms = KB/s, / 1e6 = GB/s


# -- Main ---------------------------------------------------------------------


def parse_shapes(shapes_str: str) -> list[list[int]]:
    """Parse semicolon-separated shape specs: '8,128,2048;16,64,4096'"""
    result = []
    for part in shapes_str.split(";"):
        dims = [int(d.strip()) for d in part.split(",")]
        result.append(dims)
    return result


def main():
    parser = argparse.ArgumentParser(description="GEAK Softmax Test Harness")
    parser.add_argument("baseline", help="Path to baseline kernel .so (ground truth)")
    parser.add_argument("optimized", help="Path to optimized kernel .so")
    parser.add_argument(
        "--shapes",
        default="8,128,2048",
        help="Semicolon-separated shapes (default: 8,128,2048)",
    )
    parser.add_argument(
        "--reduce-dim",
        type=int,
        default=-1,
        help="Reduction dimension (default: last dim)",
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip verification, timing only"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations before timing (default: 5)",
    )
    parser.add_argument(
        "--nrepeat",
        type=int,
        default=50,
        help="Timed iterations to average (default: 50)",
    )
    args = parser.parse_args()

    shapes = parse_shapes(args.shapes)

    # Load kernel .so files
    for path in [args.baseline, args.optimized]:
        if not Path(path).exists():
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)

    base_label = Path(args.baseline).name
    opt_label = Path(args.optimized).name
    base_lib = load_kernel(str(Path(args.baseline).resolve()))
    opt_lib = load_kernel(str(Path(args.optimized).resolve()))

    any_failed = False
    for shape in shapes:
        ndims = len(shape)
        reduce_dim = args.reduce_dim if args.reduce_dim >= 0 else ndims - 1

        print(f"\nShape: {shape}, reduce_dim={reduce_dim}")
        print("-" * 60)

        # -- Verification: run both on the same input, compare outputs --------
        status, max_err = "SKIP", 0.0
        if not args.no_verify:
            x = torch.randn(shape, dtype=torch.float16, device="cuda")

            y_base = run_kernel_output(base_lib, x, reduce_dim)
            if y_base is None:
                print(f"  {base_label}: UNSUPPORTED — skipping this shape")
                continue

            y_opt = run_kernel_output(opt_lib, x, reduce_dim)
            if y_opt is None:
                print(f"  {opt_label}: UNSUPPORTED — skipping this shape")
                continue

            passed, max_err = compare_outputs(y_opt, y_base)
            status = "PASS" if passed else "FAIL"
            if not passed:
                any_failed = True

        # -- Benchmarking -----------------------------------------------------
        ms_base = benchmark_kernel(
            base_lib, shape, reduce_dim, warmup=args.warmup, nrepeat=args.nrepeat
        )
        if ms_base < 0:
            print(f"  {base_label}: UNSUPPORTED")
            continue

        ms_opt = benchmark_kernel(
            opt_lib, shape, reduce_dim, warmup=args.warmup, nrepeat=args.nrepeat
        )
        if ms_opt < 0:
            print(f"  {opt_label}: UNSUPPORTED")
            continue

        bw_base = compute_bandwidth(shape, ms_base)
        bw_opt = compute_bandwidth(shape, ms_opt)
        speedup = ms_base / ms_opt if ms_opt > 0 else float("inf")

        print(f"  {base_label:30s}  {ms_base:8.3f} ms  {bw_base:8.1f} GB/s")
        print(
            f"  {opt_label:30s}  {ms_opt:8.3f} ms  {bw_opt:8.1f} GB/s"
            f"  [{status}, max_err={max_err:.2e}]"
        )
        print()
        print(f"  Speedup: {speedup:.2f}x")

    print()
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
