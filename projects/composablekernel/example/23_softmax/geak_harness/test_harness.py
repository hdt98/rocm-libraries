#!/usr/bin/env python3
"""GEAK Test Harness - Softmax PoC

Loads CK softmax kernel .so files via ctypes and tests them against
PyTorch's reference implementation.

Usage:
    # Verify a single kernel
    ./test_harness.py libbaseline.so

    # Compare baseline vs optimized
    ./test_harness.py libbaseline.so liboptimized.so

    # Custom shapes (semicolon-separated, dims comma-separated)
    ./test_harness.py liboptimized.so --shapes "8,128,2048;16,64,4096"

    # Control timing parameters
    ./test_harness.py liboptimized.so --warmup 10 --nrepeat 100

    # Skip verification (timing only)
    ./test_harness.py liboptimized.so --no-verify
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


def verify_kernel(
    lib, shape: list[int], reduce_dim: int, label: str
) -> tuple[bool, float]:
    """Run kernel and compare against PyTorch softmax. Returns (pass, max_err)."""
    x = torch.randn(shape, dtype=torch.float16, device="cuda")
    y = torch.empty_like(x)

    # PyTorch reference (compute in float32 for accuracy, then cast back)
    y_ref = torch.softmax(x.float(), dim=reduce_dim).half()

    ms = call_kernel(lib, x, y, [reduce_dim], alpha=1.0, beta=0.0, time_kernel=False)
    if ms < 0:
        print(f"  {label}: UNSUPPORTED (IsSupportedArgument returned false)")
        return False, float("inf")

    max_err = (y.float() - y_ref.float()).abs().max().item()
    passed = max_err < 1e-2  # FP16 softmax tolerance
    return passed, max_err


def benchmark_kernel(
    lib,
    shape: list[int],
    reduce_dim: int,
    label: str,
    warmup: int = 5,
    nrepeat: int = 50,
) -> float:
    """Time the kernel. Returns average time in ms."""
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
    parser.add_argument("kernels", nargs="+", help="Path(s) to kernel .so files")
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

    # Load all kernel .so files
    kernels = []
    for path in args.kernels:
        p = Path(path)
        if not p.exists():
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)
        lib = load_kernel(str(p.resolve()))
        kernels.append((p.name, lib))

    # Run tests
    any_failed = False
    for shape in shapes:
        ndims = len(shape)
        reduce_dim = args.reduce_dim if args.reduce_dim >= 0 else ndims - 1

        print(f"\nShape: {shape}, reduce_dim={reduce_dim}")
        print("-" * 60)

        results = []
        for label, lib in kernels:
            # Verification (also serves as support check)
            if not args.no_verify:
                passed, max_err = verify_kernel(lib, shape, reduce_dim, label)
                if max_err == float("inf"):
                    # UNSUPPORTED — skip benchmarking for this kernel+shape
                    continue
                status = "PASS" if passed else "FAIL"
                if not passed:
                    any_failed = True
            else:
                passed, max_err, status = True, 0.0, "SKIP"

            # Timing
            ms = benchmark_kernel(
                lib, shape, reduce_dim, label, warmup=args.warmup, nrepeat=args.nrepeat
            )
            if ms < 0:
                print(f"  {label}: UNSUPPORTED")
                continue

            bw = compute_bandwidth(shape, ms)
            results.append((label, ms, bw, status, max_err))
            print(
                f"  {label:30s}  {ms:8.3f} ms  {bw:8.1f} GB/s"
                f"  [{status}, max_err={max_err:.2e}]"
            )

        # Speedup comparison if multiple kernels
        if len(results) >= 2:
            base_ms = results[0][1]
            print()
            for label, ms, bw, status, _ in results[1:]:
                speedup = base_ms / ms if ms > 0 else float("inf")
                print(f"  {label} vs {results[0][0]}: {speedup:.2f}x")

    print()
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
