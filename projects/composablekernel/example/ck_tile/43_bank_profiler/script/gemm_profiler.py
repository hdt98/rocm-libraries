#!/usr/bin/env python3
"""
GEMM Bank Conflict Profiler: Profile multiple CK tile GEMM configurations
under rocprofv3 to measure LDS bank conflicts.
"""

import subprocess
import pandas as pd
import os
import argparse
import json

CONFIGS = [
    "128x128x32",
    "256x128x32",
    "128x256x32",
    "256x256x32",
    "64x64x32",
    "128x128x64",
]

PROBLEM_SIZES = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (3840, 4096, 2048),
    (4096, 4096, 4096),
]


def find_binary(build_dir, config):
    """Locate a compiled GEMM binary for a given tile config."""
    name = f"tile_bank_profiler_gemm_{config}"
    candidates = [
        os.path.join(build_dir, name),
        os.path.join(build_dir, "bin", name),
        os.path.join(build_dir, "example", "ck_tile", "43_bank_profiler",
                     name),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    for root, dirs, files in os.walk(build_dir):
        if name in files:
            return os.path.join(root, name)
    return None


def profile_config(binary, config, m, n, k, out_dir="out"):
    """Run a single GEMM config under rocprofv3 and return counter values."""
    os.makedirs(out_dir, exist_ok=True)

    out_name = f"gemm_{config}_{m}x{n}x{k}"

    cmd = [
        "rocprofv3",
        "--pmc", "SQ_INSTS_LDS", "SQ_LDS_BANK_CONFLICT",
        "--output-format", "csv",
        "--output-file", out_name,
        "-d", out_dir,
        "--",
        binary,
        f"-m={m}", f"-n={n}", f"-k={k}",
        "-warmup=1", "-repeat=1",
    ]

    try:
        result = subprocess.run(cmd, timeout=120,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"  FAILED (return code {result.returncode})")
            return None
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return None

    try:
        csv_path = os.path.join(out_dir,
                                f"{out_name}_counter_collection.csv")
        df = pd.read_csv(csv_path)
        df = df[df["Counter_Name"].isin(
            ["SQ_INSTS_LDS", "SQ_LDS_BANK_CONFLICT"])]

        pivot = df.pivot_table(
            index=["Dispatch_Id", "Kernel_Name"],
            columns="Counter_Name",
            values="Counter_Value",
            aggfunc="first").reset_index()

        # Filter for the GEMM kernel (contains "kentry")
        gemm_rows = pivot[pivot["Kernel_Name"].str.contains(
            "kentry", na=False)]
        if len(gemm_rows) == 0:
            # Fallback: take last dispatch
            gemm_rows = pivot

        last = gemm_rows.iloc[-1]

        lds_insts = int(last.get("SQ_INSTS_LDS", 0))
        conflicts = int(last.get("SQ_LDS_BANK_CONFLICT", 0))
        conflict_ratio = conflicts / lds_insts if lds_insts > 0 else 0

        return {
            "lds_insts": lds_insts,
            "conflicts": conflicts,
            "conflict_ratio": conflict_ratio,
        }
    except Exception as e:
        print(f"  Error parsing results: {e}")
        return None


def run_all(build_dir, configs=None, problem_sizes=None):
    """Profile all configurations across all problem sizes."""
    if configs is None:
        configs = CONFIGS
    if problem_sizes is None:
        problem_sizes = PROBLEM_SIZES

    results = []
    skipped = []

    for config in configs:
        binary = find_binary(build_dir, config)
        if binary is None:
            print(f"WARNING: Binary not found for config {config}, skipping")
            skipped.append(config)
            continue

        print(f"\n{'='*60}")
        print(f"Config: {config}")
        print(f"Binary: {binary}")
        print(f"{'='*60}")

        for m, n, k in problem_sizes:
            print(f"  M={m}, N={n}, K={k}... ", end="", flush=True)

            data = profile_config(binary, config, m, n, k)

            if data is not None:
                data.update({"config": config, "M": m, "N": n, "K": k})
                results.append(data)
                print(f"LDS={data['lds_insts']}, "
                      f"Conflicts={data['conflicts']}, "
                      f"Ratio={data['conflict_ratio']:.4f}")
            else:
                print("  FAILED")

    if results:
        df = pd.DataFrame(results)
        out_path = "out/gemm_bank_profile_results.csv"
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")

        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(df.to_string(index=False))

        return df
    else:
        print("\nNo results collected!")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GEMM Bank Conflict Profiler")
    parser.add_argument("--build-dir", default=".",
                        help="Directory containing compiled GEMM binaries")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Specific configs to test (default: all)")
    parser.add_argument("--sizes", nargs="*", default=None,
                        help="Problem sizes as MxNxK (e.g., 1024x1024x1024)")
    args = parser.parse_args()

    configs = args.configs
    problem_sizes = None
    if args.sizes:
        problem_sizes = []
        for s in args.sizes:
            parts = s.split("x")
            problem_sizes.append((int(parts[0]), int(parts[1]), int(parts[2])))

    run_all(args.build_dir, configs, problem_sizes)
