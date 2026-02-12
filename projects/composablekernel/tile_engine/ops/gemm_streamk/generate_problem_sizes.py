#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Standalone helper: generate a JSON file with random GEMM problem sizes.

This can be fed into the benchmark script via --problem-sizes-file, or
used as test_params for the CK Tile Engine test config format.

Usage:
  python generate_problem_sizes.py \
      --num-samples 10000 --seed 42 \
      --output benchmark_problem_sizes.json
"""

import argparse
import json
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Generate random GEMM problem sizes (M, N, K) for benchmarking."
    )
    parser.add_argument(
        "--num-samples", type=int, default=10000,
        help="Number of problem sizes to generate (default: 10000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--dim-min", type=int, default=1,
        help="Minimum dimension size before alignment (default: 1).",
    )
    parser.add_argument(
        "--dim-max", type=int, default=16384,
        help="Maximum dimension size (default: 16384).",
    )
    parser.add_argument(
        "--alignment", type=int, default=None,
        help="Align ALL dimensions to this multiple. "
             "Overridden by --alignment-m/n/k if those are also set. "
             "Default: use --alignment-m/n/k values.",
    )
    parser.add_argument(
        "--alignment-m", type=int, default=256,
        help="Align M dimension to this multiple (default: 256, matching "
             "common tile_m size for compiled kernels without padding).",
    )
    parser.add_argument(
        "--alignment-n", type=int, default=256,
        help="Align N dimension to this multiple (default: 256).",
    )
    parser.add_argument(
        "--alignment-k", type=int, default=32,
        help="Align K dimension to this multiple (default: 32).",
    )
    parser.add_argument(
        "--output", "-o", default="benchmark_problem_sizes.json",
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--ck-format", action="store_true",
        help="Output in CK test config format (with test_params wrapper).",
    )

    args = parser.parse_args()

    # Determine per-dimension alignment
    if args.alignment is not None:
        align_m = align_n = align_k = args.alignment
    else:
        align_m = args.alignment_m
        align_n = args.alignment_n
        align_k = args.alignment_k

    rng = np.random.RandomState(args.seed)

    sizes = []
    for _ in range(args.num_samples):
        m = int(rng.randint(args.dim_min, args.dim_max + 1))
        n = int(rng.randint(args.dim_min, args.dim_max + 1))
        k = int(rng.randint(args.dim_min, args.dim_max + 1))

        # Align (round up)
        m = max(align_m, ((m + align_m - 1) // align_m) * align_m)
        n = max(align_n, ((n + align_n - 1) // align_n) * align_n)
        k = max(align_k, ((k + align_k - 1) // align_k) * align_k)

        # Clamp (round down to alignment)
        m = min(m, (args.dim_max // align_m) * align_m)
        n = min(n, (args.dim_max // align_n) * align_n)
        k = min(k, (args.dim_max // align_k) * align_k)

        sizes.append({"m": m, "n": n, "k": k, "split_k": 1})

    if args.ck_format:
        output = {
            "problem": {
                "description": (
                    f"Randomly generated {args.num_samples} problem sizes "
                    f"(seed={args.seed}, range=[{args.dim_min}..{args.dim_max}], "
                    f"alignment M%{align_m} N%{align_n} K%{align_k})"
                )
            },
            "test_params": {
                "problem_sizes": sizes
            },
        }
    else:
        output = sizes

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(sizes)} problem sizes -> {args.output}")
    print(f"  Seed: {args.seed}")
    print(f"  Range: [{args.dim_min}, {args.dim_max}]")
    print(f"  Alignment: M%{align_m}, N%{align_n}, K%{align_k}")

    # Show sample
    print(f"\n  Sample (first 5):")
    for s in sizes[:5]:
        print(f"    M={s['m']:>5d}  N={s['n']:>5d}  K={s['k']:>5d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
