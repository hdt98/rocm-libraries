#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CK Tile Engine GEMM StreamK Performance Sweep & Visualization

Generates random GEMM problem sizes (M, N, K) in [1..16384] with a fixed seed,
benchmarks every compiled kernel instance on each problem size, and produces a
Roofline-style chart: Y-axis = GFLOPS, X-axis = arithmetic intensity (FLOP/byte).

Usage (typical workflow):
  # 1. Build the GEMM StreamK kernels (from the build directory)
  cmake --build . --target benchmark_gemm_streamk_all -j$(nproc)

  # 2. Run the perf sweep (10 000 random MxNxK, fixed seed 42)
  python gemm_streamk_perf_sweep.py <build_dir> \
        --num-samples 10000 --seed 42 \
        --warmup 5 --repeat 20 \
        --json results.json --csv results.csv \
        --chart perf_chart.png

  # For a quick sanity-check with fewer samples:
  python gemm_streamk_perf_sweep.py <build_dir> --num-samples 50 --seed 42
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Tile-alignment helpers
# ---------------------------------------------------------------------------

def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def extract_tile_alignment_from_kernels(
    kernel_paths: List[Path],
) -> Tuple[int, int, int]:
    """
    Parse kernel executable names to discover tile dimensions and return the
    LCM of all tile_m, tile_n, tile_k values found.

    Kernel names follow the pattern:
      benchmark_gemm_streamk_..._<TileM>x<TileN>x<TileK>_<WarpM>x<WarpN>x<WarpK>_...

    Returns (align_m, align_n, align_k).  Falls back to (256, 256, 32) if
    nothing can be parsed.
    """
    tile_ms, tile_ns, tile_ks = set(), set(), set()

    # Match groups of 3 integers joined by 'x' (e.g. 256x256x32)
    dim_pattern = re.compile(r"(\d+)x(\d+)x(\d+)")

    for kp in kernel_paths:
        groups = dim_pattern.findall(kp.name)
        if groups:
            # First NxNxN group in the name is the tile dimensions
            # (the largest-magnitude group)
            parsed = [(int(a), int(b), int(c)) for a, b, c in groups]
            # Sort by product descending — tile dims are always the biggest
            parsed.sort(key=lambda t: t[0] * t[1] * t[2], reverse=True)
            tm, tn, tk = parsed[0]
            tile_ms.add(tm)
            tile_ns.add(tn)
            tile_ks.add(tk)

    if not tile_ms:
        return (256, 256, 32)  # safe default

    align_m = 1
    for v in tile_ms:
        align_m = _lcm(align_m, v)

    align_n = 1
    for v in tile_ns:
        align_n = _lcm(align_n, v)

    align_k = 1
    for v in tile_ks:
        align_k = _lcm(align_k, v)

    return (align_m, align_n, align_k)


# ---------------------------------------------------------------------------
# Problem-size generation
# ---------------------------------------------------------------------------

def generate_problem_sizes(
    num_samples: int,
    seed: int,
    dim_min: int = 1,
    dim_max: int = 16384,
    align_m: int = 256,
    align_n: int = 256,
    align_k: int = 32,
) -> List[Tuple[int, int, int]]:
    """
    Generate *num_samples* random (M, N, K) triples with a fixed *seed*.

    Each dimension is sampled uniformly from [dim_min, dim_max] and then
    rounded up to the nearest multiple of the corresponding tile alignment
    (align_m for M, align_n for N, align_k for K).  This ensures generated
    sizes are compatible with compiled kernels that have padding disabled.

    We additionally filter out any sizes whose total tensor footprint would
    exceed ~32 GiB in fp16 to avoid OOM.
    """
    rng = np.random.RandomState(seed)
    # ~32 GiB in fp16 elements ≈ 16 G elements
    max_elements = 16 * (1024 ** 3)

    sizes: List[Tuple[int, int, int]] = []
    attempts = 0
    max_attempts = num_samples * 5  # safety valve

    while len(sizes) < num_samples and attempts < max_attempts:
        attempts += 1
        m = int(rng.randint(dim_min, dim_max + 1))
        n = int(rng.randint(dim_min, dim_max + 1))
        k = int(rng.randint(dim_min, dim_max + 1))

        # Align to tile dimensions (round up)
        m = max(align_m, ((m + align_m - 1) // align_m) * align_m)
        n = max(align_n, ((n + align_n - 1) // align_n) * align_n)
        k = max(align_k, ((k + align_k - 1) // align_k) * align_k)

        # Clamp to dim_max (round down to alignment)
        m = min(m, (dim_max // align_m) * align_m)
        n = min(n, (dim_max // align_n) * align_n)
        k = min(k, (dim_max // align_k) * align_k)

        # Ensure we didn't clamp below minimum
        if m < align_m or n < align_n or k < align_k:
            continue

        # OOM guard (conservative for fp16 = 2 bytes)
        total_elems = m * k + k * n + m * n
        if total_elems > max_elements:
            continue

        sizes.append((m, n, k))

    if len(sizes) < num_samples:
        print(
            f"[WARN] Only generated {len(sizes)}/{num_samples} valid problem sizes "
            f"after {attempts} attempts."
        )

    return sizes


def align_problem_sizes(
    sizes: List[Tuple[int, int, int]],
    align_m: int,
    align_n: int,
    align_k: int,
) -> List[Tuple[int, int, int]]:
    """
    Re-align externally loaded problem sizes to the required tile alignment.

    Sizes that are already aligned are kept as-is.  Unaligned sizes are
    rounded up.  Sizes that would become 0 are dropped.
    """
    aligned = []
    skipped = 0
    for m, n, k in sizes:
        am = ((m + align_m - 1) // align_m) * align_m
        an = ((n + align_n - 1) // align_n) * align_n
        ak = ((k + align_k - 1) // align_k) * align_k
        if am <= 0 or an <= 0 or ak <= 0:
            skipped += 1
            continue
        aligned.append((am, an, ak))
    if skipped:
        print(f"[WARN] Dropped {skipped} problem sizes that could not be aligned.")
    return aligned


# ---------------------------------------------------------------------------
# Arithmetic-intensity helpers
# ---------------------------------------------------------------------------

def compute_arithmetic_intensity(m: int, n: int, k: int, dtype_bytes: int = 2) -> float:
    """
    Arithmetic intensity = FLOPs / bytes_transferred.

    For GEMM C = A * B:
      FLOPs           = 2 * M * N * K
      bytes_transferred = dtype_bytes * (M*K + K*N + M*N)
    """
    flops = 2.0 * m * n * k
    byte_count = dtype_bytes * (m * k + k * n + m * n)
    if byte_count == 0:
        return 0.0
    return flops / byte_count


def compute_gflops(m: int, n: int, k: int, time_ms: float) -> float:
    """GFLOPS = 2*M*N*K / (time_s * 1e9)."""
    if time_ms <= 0:
        return 0.0
    flops = 2.0 * m * n * k
    return flops / (time_ms * 1e-3) / 1e9


# ---------------------------------------------------------------------------
# Kernel discovery & execution
# ---------------------------------------------------------------------------

def discover_kernels(build_dir: Path, pattern: str = "benchmark_gemm_streamk_*") -> List[Path]:
    """Find all benchmark_gemm_streamk_* executables under <build_dir>/bin."""
    bin_dir = build_dir / "bin"
    if not bin_dir.exists():
        # Some build systems put executables directly in the build dir
        bin_dir = build_dir

    kernels = sorted(bin_dir.glob(pattern))
    # Filter out non-executable files and directories
    kernels = [k for k in kernels if k.is_file() and os.access(k, os.X_OK)]
    return kernels


def run_kernel(
    kernel_path: Path,
    m: int,
    n: int,
    k: int,
    warmup: int = 5,
    repeat: int = 20,
    timeout: int = 120,
    verbose: bool = False,
) -> Optional[Dict]:
    """
    Run a single kernel executable with the given problem size.

    Returns parsed JSON result dict or None on failure.
    """
    cmd = [
        str(kernel_path),
        f"-m={m}",
        f"-n={n}",
        f"-k={k}",
        f"-warmup={warmup}",
        f"-repeat={repeat}",
        "-verify=0",
        "-timer=true",
        "-flush_cache=true",
        "-rotating_count=1000",
        "-json_output=true",
        "-metric=1",  # TFLOPS metric
    ]

    if verbose:
        print(f"  CMD: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )

        if result.returncode != 0:
            if verbose:
                print(f"  [ERR] {kernel_path.name} returned {result.returncode}")
                if result.stderr:
                    print(f"        stderr: {result.stderr.strip()[:300]}")
            return None

        output = result.stdout.strip()
        if not output:
            # Kernel produced no stdout — likely IsSupportedArgument() failed
            # and the exception was caught internally.  Show stderr hint.
            if verbose:
                stderr_msg = result.stderr.strip()[:300] if result.stderr else "(none)"
                print(f"  [NO OUTPUT] {kernel_path.name}  stderr: {stderr_msg}")
            return None

        # The executable with -json_output=true prints a JSON blob
        data = json.loads(output)
        return data

    except subprocess.TimeoutExpired:
        if verbose:
            print(f"  [TIMEOUT] {kernel_path.name}")
        return None
    except json.JSONDecodeError as exc:
        if verbose:
            print(f"  [JSON ERR] {kernel_path.name}: {exc}")
            print(f"        raw stdout: {result.stdout.strip()[:300]}")
        return None
    except Exception as exc:
        if verbose:
            print(f"  [ERR] {kernel_path.name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main benchmarking loop
# ---------------------------------------------------------------------------

def benchmark_sweep(
    build_dir: Path,
    problem_sizes: List[Tuple[int, int, int]],
    warmup: int = 5,
    repeat: int = 20,
    timeout: int = 120,
    verbose: bool = False,
    dtype_bytes: int = 2,
    kernel_pattern: str = "benchmark_gemm_streamk_*",
) -> List[Dict]:
    """
    Run every discovered kernel on every problem size.

    Returns a flat list of result dicts, each enriched with:
      - m, n, k
      - arithmetic_intensity
      - gflops
      - kernel_name
      - latency_ms
      - bandwidth_gb_s
    """
    kernels = discover_kernels(build_dir, pattern=kernel_pattern)
    if not kernels:
        print("[ERROR] No kernel executables found. Did you build with "
              "`cmake --build . --target benchmark_gemm_streamk_all`?")
        return []

    print(f"Discovered {len(kernels)} kernel(s):")
    for k in kernels:
        print(f"  - {k.name}")

    # Show detected tile alignment so the user knows what constraints apply
    align_m, align_n, align_k = extract_tile_alignment_from_kernels(kernels)
    print(f"\nDetected tile alignment from kernel names: "
          f"M%{align_m}==0, N%{align_n}==0, K%{align_k}==0")
    print(f"  (Kernels compiled without padding require problem sizes "
          f"to be multiples of these tile dimensions.)\n")

    total_runs = len(kernels) * len(problem_sizes)
    print(f"Total benchmark runs: {total_runs} "
          f"({len(kernels)} kernels x {len(problem_sizes)} problems)")
    print(f"Warmup={warmup}, Repeat={repeat}\n")

    all_results: List[Dict] = []
    completed = 0
    failed_per_problem = 0
    total_failures = 0
    start_wall = time.time()

    for prob_idx, (m, n, k) in enumerate(problem_sizes):
        ai = compute_arithmetic_intensity(m, n, k, dtype_bytes)

        problem_successes = 0
        for kern_idx, kernel_path in enumerate(kernels):
            completed += 1
            if completed % 100 == 0 or completed <= 5:
                elapsed = time.time() - start_wall
                eta = (elapsed / completed) * (total_runs - completed) if completed else 0
                print(
                    f"[{completed}/{total_runs}] "
                    f"M={m} N={n} K={k} | {kernel_path.name} "
                    f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)"
                )

            raw = run_kernel(
                kernel_path, m, n, k,
                warmup=warmup, repeat=repeat,
                timeout=timeout, verbose=verbose,
            )

            if raw is None:
                total_failures += 1
                continue

            problem_successes += 1

            # Extract perf fields from the JSON blob
            perf = raw.get("perf_result", {})
            latency_ms = perf.get("latency(ms)", 0.0)
            tflops = perf.get("tflops(TFlops)", 0.0)
            bandwidth = perf.get("bandwidth(GB/s)", 0.0)

            gflops = tflops * 1000.0  # TFlops -> GFlops

            # If tflops is zero, compute from latency
            if gflops <= 0 and latency_ms > 0:
                gflops = compute_gflops(m, n, k, latency_ms)

            record = {
                "kernel_name": raw.get("name", kernel_path.stem),
                "m": m,
                "n": n,
                "k": k,
                "arithmetic_intensity": ai,
                "gflops": gflops,
                "tflops": tflops,
                "latency_ms": latency_ms,
                "bandwidth_gb_s": bandwidth,
                "kernel_executable": kernel_path.name,
            }

            all_results.append(record)

        if problem_successes == 0:
            failed_per_problem += 1
            # Show a diagnostic for the first few fully-failed problems
            if failed_per_problem <= 3:
                print(
                    f"  [WARN] ALL {len(kernels)} kernels failed for "
                    f"M={m} N={n} K={k}  "
                    f"(M%{align_m}={m % align_m}, "
                    f"N%{align_n}={n % align_n}, "
                    f"K%{align_k}={k % align_k})"
                )
                if m % align_m != 0 or n % align_n != 0 or k % align_k != 0:
                    print(
                        f"         ^ Dimensions are NOT multiples of tile sizes! "
                        f"Re-run with --auto-align or aligned problem sizes."
                    )
            elif failed_per_problem == 4:
                print(f"  [WARN] (suppressing further per-problem warnings ...)")

    wall_time = time.time() - start_wall
    print(f"\nBenchmark sweep completed: {len(all_results)} successful runs, "
          f"{total_failures} failures in {wall_time:.1f}s")

    if failed_per_problem > 0:
        print(f"[WARN] {failed_per_problem}/{len(problem_sizes)} problem sizes "
              f"had ALL kernels fail.")
        if failed_per_problem == len(problem_sizes):
            print(
                "\n[HINT] Every single problem size failed for every kernel.\n"
                "       This almost certainly means the problem dimensions are not\n"
                f"       aligned to the tile sizes (M%{align_m}, N%{align_n}, K%{align_k}).\n"
                "       The compiled kernels have padding disabled (pad_m/n/k=False)\n"
                "       so they reject misaligned sizes via IsSupportedArgument().\n"
                "\n"
                "       Fix: re-run with --auto-align to automatically align sizes,\n"
                "            or regenerate sizes with the correct alignment:\n"
                f"              python generate_problem_sizes.py "
                f"--alignment-m {align_m} --alignment-n {align_n} "
                f"--alignment-k {align_k} ...\n"
            )

    return all_results


# ---------------------------------------------------------------------------
# Best-kernel selection per problem size
# ---------------------------------------------------------------------------

def select_best_per_problem(results: List[Dict]) -> List[Dict]:
    """
    For each unique (m, n, k) keep only the result with the highest GFLOPS.
    """
    best: Dict[Tuple[int, int, int], Dict] = {}
    for r in results:
        key = (r["m"], r["n"], r["k"])
        if key not in best or r["gflops"] > best[key]["gflops"]:
            best[key] = r
    return sorted(best.values(), key=lambda x: x["arithmetic_intensity"])


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_csv(results: List[Dict], filename: str):
    if not results:
        return
    fieldnames = [
        "kernel_name", "m", "n", "k",
        "arithmetic_intensity", "gflops", "tflops",
        "latency_ms", "bandwidth_gb_s", "kernel_executable",
    ]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV exported to {filename}")


def export_json(results: List[Dict], best_results: List[Dict], filename: str,
                problem_sizes: List[Tuple[int, int, int]], seed: int):
    out = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "num_problem_sizes": len(problem_sizes),
            "total_runs": len(results),
            "successful_runs": len([r for r in results if r["gflops"] > 0]),
        },
        "summary": {},
        "best_per_problem": best_results,
        "all_results": results,
    }

    gflops_vals = [r["gflops"] for r in results if r["gflops"] > 0]
    if gflops_vals:
        out["summary"] = {
            "best_gflops": max(gflops_vals),
            "avg_gflops": sum(gflops_vals) / len(gflops_vals),
            "median_gflops": float(np.median(gflops_vals)),
            "min_gflops": min(gflops_vals),
        }

    with open(filename, "w") as f:
        json.dump(out, f, indent=2)
    print(f"JSON exported to {filename}")


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def generate_chart(
    results: List[Dict],
    best_results: List[Dict],
    filename: str,
    title: str = "CK Tile Engine GEMM StreamK — Roofline Performance",
):
    """
    Produce a scatter plot: X = arithmetic intensity (FLOP/byte),
                            Y = performance (GFLOPS).

    - Light dots = individual kernel results
    - Highlighted dots = best kernel per problem size
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[WARN] matplotlib not available — skipping chart generation. "
              "Install with: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    # --- All results (faded) ---
    ai_all = [r["arithmetic_intensity"] for r in results if r["gflops"] > 0]
    gf_all = [r["gflops"] for r in results if r["gflops"] > 0]

    if not ai_all:
        print("[WARN] No valid data points to plot.")
        return

    ax.scatter(
        ai_all, gf_all,
        s=6, alpha=0.15, color="steelblue", label="All kernels", rasterized=True,
    )

    # --- Best per problem (bold) ---
    ai_best = [r["arithmetic_intensity"] for r in best_results if r["gflops"] > 0]
    gf_best = [r["gflops"] for r in best_results if r["gflops"] > 0]

    ax.scatter(
        ai_best, gf_best,
        s=14, alpha=0.7, color="crimson", label="Best kernel per problem",
        edgecolors="darkred", linewidths=0.3, rasterized=True,
    )

    # --- Formatting ---
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)

    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=13)
    ax.set_ylabel("Performance (GFLOPS)", fontsize=13)
    ax.set_title(title, fontsize=15, pad=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.6)

    # Nicer tick labels
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # Annotate peak
    if gf_best:
        peak_idx = int(np.argmax(gf_best))
        ax.annotate(
            f"Peak: {gf_best[peak_idx]:,.0f} GFLOPS",
            xy=(ai_best[peak_idx], gf_best[peak_idx]),
            xytext=(20, 20), textcoords="offset points",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", lw=0.5),
        )

    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Chart saved to {filename}")
    plt.close(fig)

    # Also produce a by-kernel-name breakdown chart
    _generate_per_kernel_chart(results, filename)


def _generate_per_kernel_chart(results: List[Dict], base_filename: str):
    """Optional second chart: colour-code by kernel executable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        return

    # Group by kernel name
    kernel_names = sorted(set(r["kernel_executable"] for r in results if r["gflops"] > 0))
    if len(kernel_names) <= 1:
        return  # nothing interesting to colour-code

    colours = cm.get_cmap("tab20", len(kernel_names))

    fig, ax = plt.subplots(figsize=(14, 8))

    for idx, kname in enumerate(kernel_names):
        subset = [r for r in results if r["kernel_executable"] == kname and r["gflops"] > 0]
        ai = [r["arithmetic_intensity"] for r in subset]
        gf = [r["gflops"] for r in subset]
        ax.scatter(
            ai, gf, s=8, alpha=0.5,
            color=colours(idx), label=kname[:60], rasterized=True,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=13)
    ax.set_ylabel("Performance (GFLOPS)", fontsize=13)
    ax.set_title("CK Tile Engine GEMM StreamK — Per-Kernel Comparison", fontsize=15, pad=12)
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.6)

    # Only show legend if manageable number of kernels
    if len(kernel_names) <= 20:
        ax.legend(loc="upper left", fontsize=7, ncol=2)

    fig.tight_layout()
    stem = Path(base_filename).stem
    suffix = Path(base_filename).suffix
    per_kernel_file = str(Path(base_filename).parent / f"{stem}_per_kernel{suffix}")
    fig.savefig(per_kernel_file, dpi=200, bbox_inches="tight")
    print(f"Per-kernel chart saved to {per_kernel_file}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CK Tile Engine GEMM StreamK Performance Sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "build_dir",
        help="Path to the CMake build directory containing compiled kernel executables.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10000,
        help="Number of random (M,N,K) problem sizes to generate (default: 10000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible problem-size generation (default: 42).",
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
        "--warmup", type=int, default=5,
        help="Warmup iterations per kernel launch (default: 5).",
    )
    parser.add_argument(
        "--repeat", type=int, default=20,
        help="Benchmark iterations per kernel launch (default: 20).",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Timeout in seconds per kernel execution (default: 120).",
    )
    parser.add_argument(
        "--dtype-bytes", type=int, default=2,
        help="Element size in bytes for arithmetic-intensity calculation "
             "(default: 2 for fp16).",
    )
    parser.add_argument(
        "--csv", default="gemm_streamk_perf_sweep.csv",
        help="CSV output file (default: gemm_streamk_perf_sweep.csv).",
    )
    parser.add_argument(
        "--json", default="",
        help="JSON output file (default: none).",
    )
    parser.add_argument(
        "--chart", default="gemm_streamk_perf_chart.png",
        help="Performance chart output file (default: gemm_streamk_perf_chart.png).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-kernel command lines and errors.",
    )
    parser.add_argument(
        "--kernel-pattern", default="benchmark_gemm_streamk_*",
        help="Glob pattern for kernel executables (default: benchmark_gemm_streamk_*).",
    )
    parser.add_argument(
        "--problem-sizes-file", default="",
        help="Optional JSON file with pre-defined problem sizes "
             "(list of {\"m\": ..., \"n\": ..., \"k\": ...} dicts). "
             "If provided, --num-samples and --seed are ignored.",
    )
    parser.add_argument(
        "--auto-align", action="store_true", default=True,
        help="Automatically align problem sizes to the tile dimensions "
             "detected from compiled kernels (default: enabled). "
             "Use --no-auto-align to disable.",
    )
    parser.add_argument(
        "--no-auto-align", dest="auto_align", action="store_false",
        help="Disable automatic alignment of problem sizes to tile dimensions.",
    )

    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve()
    if not build_dir.exists():
        print(f"[ERROR] Build directory does not exist: {build_dir}")
        return 1

    # ---- Discover kernels early to detect tile alignment ----
    kernels = discover_kernels(build_dir, pattern=args.kernel_pattern)
    if kernels:
        align_m, align_n, align_k = extract_tile_alignment_from_kernels(kernels)
        print(f"Auto-detected tile alignment: M%{align_m}==0, "
              f"N%{align_n}==0, K%{align_k}==0")
    else:
        align_m, align_n, align_k = 256, 256, 32
        print(f"[WARN] No kernels found yet — using default alignment: "
              f"M%{align_m}, N%{align_n}, K%{align_k}")

    # ---- Problem sizes ----
    if args.problem_sizes_file:
        print(f"Loading problem sizes from {args.problem_sizes_file} ...")
        with open(args.problem_sizes_file) as f:
            raw_sizes = json.load(f)
        if isinstance(raw_sizes, dict) and "test_params" in raw_sizes:
            # Handle the CK test config format
            raw_sizes = raw_sizes["test_params"].get("problem_sizes", [])
        problem_sizes = [(s["m"], s["n"], s["k"]) for s in raw_sizes]
        print(f"Loaded {len(problem_sizes)} problem sizes from file.")

        # Check and optionally fix alignment
        if args.auto_align:
            misaligned = sum(
                1 for m, n, k in problem_sizes
                if m % align_m != 0 or n % align_n != 0 or k % align_k != 0
            )
            if misaligned:
                print(f"[INFO] {misaligned}/{len(problem_sizes)} sizes are not "
                      f"aligned to tile dims — re-aligning (round up).")
                problem_sizes = align_problem_sizes(
                    problem_sizes, align_m, align_n, align_k
                )
    else:
        print(f"Generating {args.num_samples} random problem sizes "
              f"(seed={args.seed}, range=[{args.dim_min}..{args.dim_max}], "
              f"aligned to M%{align_m}, N%{align_n}, K%{align_k}) ...")
        problem_sizes = generate_problem_sizes(
            args.num_samples, args.seed,
            dim_min=args.dim_min, dim_max=args.dim_max,
            align_m=align_m, align_n=align_n, align_k=align_k,
        )
        print(f"Generated {len(problem_sizes)} problem sizes.")

    if not problem_sizes:
        print("[ERROR] No problem sizes to benchmark.")
        return 1

    # Show a sample
    print("\nSample problem sizes (first 5):")
    for m, n, k in problem_sizes[:5]:
        ai = compute_arithmetic_intensity(m, n, k, args.dtype_bytes)
        print(f"  M={m:>5d}  N={n:>5d}  K={k:>5d}  AI={ai:.2f}")

    # ---- Run sweep ----
    results = benchmark_sweep(
        build_dir,
        problem_sizes,
        warmup=args.warmup,
        repeat=args.repeat,
        timeout=args.timeout,
        verbose=args.verbose,
        dtype_bytes=args.dtype_bytes,
        kernel_pattern=args.kernel_pattern,
    )

    if not results:
        print("[ERROR] No benchmark results collected.")
        return 1

    # ---- Best per problem ----
    best_results = select_best_per_problem(results)

    # ---- Summary ----
    gflops_vals = [r["gflops"] for r in results if r["gflops"] > 0]
    if gflops_vals:
        print(f"\n{'='*60}")
        print(f"  Performance Summary")
        print(f"{'='*60}")
        print(f"  Total runs:     {len(results)}")
        print(f"  Successful:     {len(gflops_vals)}")
        print(f"  Peak GFLOPS:    {max(gflops_vals):,.1f}")
        print(f"  Avg  GFLOPS:    {sum(gflops_vals)/len(gflops_vals):,.1f}")
        print(f"  Median GFLOPS:  {float(np.median(gflops_vals)):,.1f}")
        print(f"  Min  GFLOPS:    {min(gflops_vals):,.1f}")
        print(f"{'='*60}\n")

    # ---- Export ----
    if args.csv:
        export_csv(results, args.csv)

    if args.json:
        export_json(results, best_results, args.json, problem_sizes, args.seed)

    if args.chart:
        generate_chart(results, best_results, args.chart)

    return 0


if __name__ == "__main__":
    sys.exit(main())
