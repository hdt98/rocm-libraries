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
# GPU architecture detection & binary compatibility
# ---------------------------------------------------------------------------

def detect_gpu_arch() -> str:
    """Detect GPU architecture (e.g. 'gfx942') using rocminfo."""
    try:
        output = subprocess.check_output(
            ["rocminfo"], text=True, stderr=subprocess.PIPE, timeout=10
        )
        for line in output.splitlines():
            if "Name:" in line and "gfx" in line:
                match = re.search(r"(gfx\w+)", line)
                if match:
                    return match.group(1)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def check_binary_gpu_targets(binary_path: Path) -> List[str]:
    """
    Check what GPU architectures a HIP binary was compiled for.

    Uses roc-obj-ls (preferred) or llvm-objdump to list embedded code objects.
    Returns a list of architecture strings like ['gfx90a', 'gfx942'].
    """
    archs = set()

    # Try roc-obj-ls first (ROCm 6.x+)
    for tool in ["roc-obj-ls", "roc-obj-extract"]:
        try:
            out = subprocess.check_output(
                [tool, str(binary_path)],
                text=True, stderr=subprocess.PIPE, timeout=10,
            )
            for line in out.splitlines():
                match = re.search(r"(gfx\w+)", line)
                if match:
                    archs.add(match.group(1))
            if archs:
                return sorted(archs)
        except (subprocess.CalledProcessError, FileNotFoundError,
                subprocess.TimeoutExpired):
            continue

    # Fallback: use strings to find gfx arch markers in the binary
    try:
        out = subprocess.check_output(
            ["strings", str(binary_path)],
            text=True, stderr=subprocess.PIPE, timeout=10,
        )
        for match in re.finditer(r"amdhsa--.+?(gfx\w+)", out):
            archs.add(match.group(1))
        if archs:
            return sorted(archs)
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        pass

    return []


def print_gpu_arch_diagnostic(
    gpu_arch: str, binary_targets: List[str], kernels: List[Path]
) -> bool:
    """
    Print a diagnostic comparing the running GPU architecture with the binary
    targets. Returns True if everything looks OK, False if there's a mismatch.
    """
    print(f"\n--- GPU Architecture Diagnostic ---")
    print(f"  Running GPU:     {gpu_arch or '(unknown - rocminfo failed)'}")
    if binary_targets:
        print(f"  Binary targets:  {', '.join(binary_targets)}")
    else:
        print(f"  Binary targets:  (could not detect)")

    if not gpu_arch:
        print(f"  [WARN] Could not detect GPU architecture. "
              f"Run 'rocminfo' to verify.")
        return True  # can't tell, proceed anyway

    if binary_targets and gpu_arch not in binary_targets:
        print(f"\n  *** ARCHITECTURE MISMATCH ***")
        print(f"  The kernel binaries were compiled for: {', '.join(binary_targets)}")
        print(f"  But the running GPU is: {gpu_arch}")
        print(f"  This causes 'Cannot find Symbol' errors for ALL kernels.")
        print(f"  FIX: Rebuild with the correct GPU target:")
        print(f"    cd <build_dir>")
        print(f"    cmake .. -DGPU_TARGETS={gpu_arch}")
        print(f"    cmake --build . --target benchmark_gemm_streamk_all -j$(nproc)")
        return False
    elif binary_targets:
        print(f"  [OK] Binary targets include {gpu_arch}")
    return True


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
) -> Tuple[Optional[Dict], str]:
    """
    Run a single kernel executable with the given problem size.

    Returns (parsed_result_dict_or_None, failure_reason_string).
    On success the failure_reason is "".
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

        stderr_text = result.stderr.strip() if result.stderr else ""

        if result.returncode != 0:
            reason = f"exit code {result.returncode}"
            if stderr_text:
                reason += f": {stderr_text[:200]}"
            if verbose:
                print(f"  [ERR] {kernel_path.name}: {reason}")
            return None, reason

        output = result.stdout.strip()
        if not output:
            # Kernel produced no stdout — IsSupportedArgument() failed or
            # another internal exception was caught.
            reason = stderr_text[:200] if stderr_text else "no stdout (kernel likely rejected args)"
            if verbose:
                print(f"  [NO OUTPUT] {kernel_path.name}  reason: {reason}")
            return None, reason

        # The executable with -json_output=true prints a JSON blob.
        # Sometimes HIP runtime or the kernel may print extra lines before
        # the JSON.  Try to find the outermost { ... } to handle that.
        json_str = output
        first_brace = output.find("{")
        last_brace = output.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            json_str = output[first_brace : last_brace + 1]

        data = json.loads(json_str)
        return data, ""

    except subprocess.TimeoutExpired:
        reason = f"timeout ({timeout}s)"
        if verbose:
            print(f"  [TIMEOUT] {kernel_path.name}")
        return None, reason
    except json.JSONDecodeError as exc:
        reason = f"JSON parse error: {exc}"
        if verbose:
            print(f"  [JSON ERR] {kernel_path.name}: {exc}")
            print(f"        raw stdout: {output[:300]}")
        return None, reason
    except Exception as exc:
        reason = str(exc)[:200]
        if verbose:
            print(f"  [ERR] {kernel_path.name}: {exc}")
        return None, reason


# ---------------------------------------------------------------------------
# Kernel probe / diagnostic
# ---------------------------------------------------------------------------

def _probe_single_kernel(
    kernel_path: Path,
    m: int, n: int, k: int,
    timeout: int = 30,
) -> Tuple[bool, str]:
    """
    Run a kernel once with minimal iterations to check if it works.

    Returns (success, stderr_output).
    """
    cmd = [
        str(kernel_path),
        f"-m={m}", f"-n={n}", f"-k={k}",
        "-warmup=0", "-repeat=1",
        "-verify=0", "-timer=true",
        "-flush_cache=false",
        "-rotating_count=0",
        "-json_output=true", "-metric=1",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        stderr_text = result.stderr.strip() if result.stderr else ""
        stdout_text = result.stdout.strip() if result.stdout else ""

        if result.returncode != 0:
            return False, (
                f"exit code {result.returncode}\n"
                f"  stderr: {stderr_text[:500]}\n"
                f"  stdout: {stdout_text[:200]}"
            )

        if not stdout_text:
            return False, (
                f"no JSON output (exit 0)\n"
                f"  stderr: {stderr_text[:500]}"
            )

        # Try to parse JSON
        first_brace = stdout_text.find("{")
        last_brace = stdout_text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            json.loads(stdout_text[first_brace : last_brace + 1])
            return True, ""

        return False, f"stdout not valid JSON: {stdout_text[:200]}"

    except subprocess.TimeoutExpired:
        return False, f"timeout ({timeout}s)"
    except json.JSONDecodeError as exc:
        return False, f"JSON parse error: {exc}"
    except Exception as exc:
        return False, str(exc)[:300]


def probe_kernels(
    kernels: List[Path],
    align_m: int = 256,
    align_n: int = 256,
    align_k: int = 32,
    timeout: int = 30,
) -> Tuple[List[Path], Dict[str, str]]:
    """
    Run each kernel once with a small problem to see which ones are functional.

    Uses a problem size of 4x tile dimensions (e.g. 1024x1024x128 for
    256x256x32 tiles) so stream-K has enough tiles to distribute work.

    Returns:
      - working_kernels: list of paths that produced valid output
      - failure_map: {kernel_name: reason_string} for failed kernels
    """
    # Use several tiles per dimension so stream-K has a realistic workload.
    probe_m = align_m * 4
    probe_n = align_n * 4
    probe_k = align_k * 4

    print(f"\n--- Probing {len(kernels)} kernel executables "
          f"with M={probe_m}, N={probe_n}, K={probe_k} ---")
    print(f"    (Each executable is a different tile/warp configuration")
    print(f"     compiled from default_config.json — not related to --num-samples)")

    working: List[Path] = []
    failure_map: Dict[str, str] = {}

    for idx, kp in enumerate(kernels):
        ok, detail = _probe_single_kernel(kp, probe_m, probe_n, probe_k,
                                          timeout=timeout)
        if ok:
            working.append(kp)
        else:
            failure_map[kp.name] = detail

        # Progress for large sets
        if (idx + 1) % 20 == 0:
            print(f"    probed {idx + 1}/{len(kernels)} ...")

    # ---- Summary ----
    print(f"\nProbe results: {len(working)}/{len(kernels)} kernel executables operational")

    if failure_map:
        # Group by the first line of the failure detail
        reason_groups: Dict[str, List[str]] = {}
        for name, detail in failure_map.items():
            first_line = detail.split("\n")[0]
            reason_groups.setdefault(first_line, []).append(name)

        print(f"\nFailed kernel executables ({len(failure_map)}):")
        for reason, names in sorted(reason_groups.items(),
                                     key=lambda x: -len(x[1])):
            print(f"\n  [{len(names)} kernel(s)] {reason}")

            # Classify
            if "exit code -6" in reason or "exit code -11" in reason:
                signal_name = "SIGABRT" if "-6" in reason else "SIGSEGV"
                # Check if it's a "Cannot find Symbol" error (arch mismatch)
                any_detail = next(iter(names))
                detail_text = failure_map.get(any_detail, "")
                if "Cannot find Symbol" in detail_text:
                    print(f"    Cause: HIP runtime cannot find kernel symbol in the binary.")
                    print(f"           This means the binary was NOT compiled for the running GPU.")
                    print(f"           Check GPU arch vs binary targets (see diagnostic above).")
                    print(f"           FIX: rebuild with correct -DGPU_TARGETS=<your_gpu_arch>")
                else:
                    print(f"    Cause: kernel crashed ({signal_name}). "
                          f"This usually means the warp tile config")
                    print(f"           generates MFMA instructions not supported "
                          f"on this GPU architecture.")
            elif "no JSON output" in reason:
                print(f"    Cause: IsSupportedArgument() rejected the "
                      f"configuration for this problem size.")

            for nm in names[:5]:
                print(f"    - {nm}")
            if len(names) > 5:
                print(f"    ... and {len(names) - 5} more")

        # Show full stderr from the first failure for diagnostics
        first_failed = next(iter(failure_map))
        full_detail = failure_map[first_failed]
        print(f"\n  Full diagnostic from first failed kernel ({first_failed}):")
        for line in full_detail.split("\n"):
            print(f"    {line}")

    if working:
        print(f"\nWorking kernel executables ({len(working)}):")
        for kp in working:
            print(f"  + {kp.name}")
    else:
        print(f"\n[WARN] No kernels work. Try running one manually to see the full error:")
        if kernels:
            print(f"  {kernels[0]} -m={probe_m} -n={probe_n} -k={probe_k} "
                  f"-warmup=0 -repeat=1 -verify=0")

    print()
    return working, failure_map


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
    skip_probe: bool = False,
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
    all_kernels = discover_kernels(build_dir, pattern=kernel_pattern)
    if not all_kernels:
        print("[ERROR] No kernel executables found. Did you build with "
              "`cmake --build . --target benchmark_gemm_streamk_all`?")
        return []

    # Count fp16 vs fp8 etc. for a useful summary
    dtype_counts: Dict[str, int] = {}
    for k in all_kernels:
        for dt in ("fp16", "fp8", "bf16", "fp32"):
            if f"_{dt}_" in k.name:
                dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
                break
    dtype_summary = ", ".join(f"{v} {k}" for k, v in sorted(dtype_counts.items()))

    print(f"\nDiscovered {len(all_kernels)} kernel executable(s) ({dtype_summary}):")
    print(f"  (These are compiled binaries from default_config.json — each is a")
    print(f"   different tile/warp configuration. NOT related to --num-samples.)")
    for k in all_kernels:
        print(f"  - {k.name}")

    # Show detected tile alignment so the user knows what constraints apply
    align_m, align_n, align_k = extract_tile_alignment_from_kernels(all_kernels)
    print(f"\nDetected tile alignment from kernel names: "
          f"M%{align_m}==0, N%{align_n}==0, K%{align_k}==0")
    print(f"  (Kernels compiled without padding require problem sizes "
          f"to be multiples of these tile dimensions.)")

    # --- GPU architecture compatibility check ---
    gpu_arch = detect_gpu_arch()
    binary_targets = check_binary_gpu_targets(all_kernels[0])
    arch_ok = print_gpu_arch_diagnostic(gpu_arch, binary_targets, all_kernels)

    if not arch_ok:
        print("\n[ERROR] GPU architecture mismatch detected. Most kernels will fail.")
        print("        Proceeding with probe to show which (if any) work...\n")

    # --- Probe phase: identify working vs broken kernels ---
    if skip_probe:
        kernels = all_kernels
    else:
        kernels, failure_map = probe_kernels(
            all_kernels,
            align_m=align_m,
            align_n=align_n,
            align_k=align_k,
            timeout=min(timeout, 30),
        )

        if not kernels:
            print("[ERROR] No kernels passed the probe. Nothing to benchmark.")
            if not arch_ok:
                print("\n  This is likely caused by the GPU architecture mismatch above.")
                print("  Rebuild with: cmake .. -DGPU_TARGETS=<your_gpu> && "
                      "cmake --build . --target benchmark_gemm_streamk_all -j$(nproc)")
            return []

    total_runs = len(kernels) * len(problem_sizes)
    print(f"Total benchmark runs: {total_runs} "
          f"({len(kernels)} kernels x {len(problem_sizes)} problems)")
    print(f"Warmup={warmup}, Repeat={repeat}\n")

    all_results: List[Dict] = []
    completed = 0
    total_failures = 0
    failed_per_problem = 0
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

            raw, reason = run_kernel(
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
            if failed_per_problem <= 3:
                print(
                    f"  [WARN] ALL {len(kernels)} kernels failed for "
                    f"M={m} N={n} K={k}"
                )

    wall_time = time.time() - start_wall
    print(f"\nBenchmark sweep completed: {len(all_results)} successful runs, "
          f"{total_failures} failures in {wall_time:.1f}s")

    if failed_per_problem > 0:
        print(f"[WARN] {failed_per_problem}/{len(problem_sizes)} problem sizes "
              f"had ALL kernels fail.")

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
    parser.add_argument(
        "--skip-probe", action="store_true",
        help="Skip the initial kernel probe phase that checks which kernels "
             "are functional. Use if you know all kernels work.",
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
        skip_probe=args.skip_probe,
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
