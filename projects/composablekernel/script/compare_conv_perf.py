#!/usr/bin/env python3
"""
Compare convolution performance between two TraceLens CSV files.
Identifies the dominant compute kernel (CK implicit-GEMM or HIPBLAS explicit-GEMM)
and compares mean TFLOPS values per (input_shape, filter_shape) problem.

Usage:
    python compare_conv_perf.py [--new NEW_CSV] [--baseline BASELINE_CSV] [--output OUTPUT_CSV]
"""

import ast
import re
import subprocess
import argparse
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Kernel classification helpers
# ---------------------------------------------------------------------------

CK_PREFIX = "_ZN2ck16tensor_operation6device"
HIPBLAS_PREFIX = "Cijk_Ailk_Bljk"


def is_compute_kernel(name: str) -> bool:
    return name.startswith(CK_PREFIX) or name.startswith(HIPBLAS_PREFIX)


def is_ck_kernel(name: str) -> bool:
    return name.startswith(CK_PREFIX)


def demangle(mangled_name: str) -> str:
    """Demangle a C++ mangled name using llvm-cxxfilt."""
    try:
        result = subprocess.run(
            ["llvm-cxxfilt", mangled_name],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception as e:
        return f"[demangle failed: {e}]"


# Cache demangled names to avoid repeated subprocess calls
_demangle_cache: dict[str, str] = {}


def demangle_cached(name: str) -> str:
    if name not in _demangle_cache:
        _demangle_cache[name] = demangle(name)
    return _demangle_cache[name]


# ---------------------------------------------------------------------------
# Tiling info extraction from demangled kernel names
# ---------------------------------------------------------------------------

def _split_top_level_args(s: str) -> list[str]:
    """Split template args on commas, respecting nested <> and ()."""
    args, depth, current = [], 0, []
    for ch in s:
        if ch in "<(":
            depth += 1
            current.append(ch)
        elif ch in ">)":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args.append("".join(current).strip())
    return args


def _extract_gridwise_args(demangled: str, struct_name: str) -> list[str] | None:
    """Return the top-level template args of the first occurrence of struct_name<...>."""
    marker = struct_name + "<"
    idx = demangled.find(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    depth, i = 1, start
    while i < len(demangled) and depth > 0:
        if demangled[i] == "<":
            depth += 1
        elif demangled[i] == ">":
            depth -= 1
        i += 1
    return _split_top_level_args(demangled[start : i - 1])


def _fmt(args: list[str], indices: dict[str, int]) -> str:
    """Format selected positional args as 'key=value' pairs."""
    parts = []
    for label, idx in indices.items():
        if idx < len(args):
            parts.append(f"{label}={args[idx].strip()}")
    return ", ".join(parts)


# Template parameter positions for the tiling params of interest.
# GridwiseGemmMultipleD_xdl_cshuffle: 10 leading non-integer type params
#   [0-6]  data types, [7-9] element-wise ops, [10] NumKPrefetch (first int)
_V1_INDICES = {
    "BlockSize": 11, "MPerBlock": 12, "NPerBlock": 13, "KPerBlock": 14,
    "MPerXdl": 17, "NPerXdl": 18, "MXdlPerWave": 19, "NXdlPerWave": 20,
}

# GridwiseGemmMultiD_xdl_cshuffle_v3: 14 leading non-integer type params
#   [0-3]  layouts, [4-9] data types, [10-12] element-wise ops, [13] GemmSpec
_V3_INDICES = {
    "BlockSize": 14, "MPerBlock": 15, "NPerBlock": 16, "KPerBlock": 17,
    "MPerXdl": 20, "NPerXdl": 21, "MXdlPerWave": 22, "NXdlPerWave": 23,
}


def extract_double_buffering(demangled: str) -> str:
    """Return 'true'/'false' for GridwiseGemmMultipleD_xdl_cshuffle DoubleBuffer param, else ''."""
    args = _extract_gridwise_args(demangled, "GridwiseGemmMultipleD_xdl_cshuffle")
    if args is None or len(args) <= 45:
        return ""
    return args[45].strip()  # 'true' or 'false'


def extract_tiling(demangled: str) -> str:
    """Return a compact tiling string extracted from a demangled kernel name."""
    if not demangled:
        return ""

    # CK v3
    args = _extract_gridwise_args(demangled, "GridwiseGemmMultiD_xdl_cshuffle_v3")
    if args is not None:
        return _fmt(args, _V3_INDICES)

    # CK v1
    args = _extract_gridwise_args(demangled, "GridwiseGemmMultipleD_xdl_cshuffle")
    if args is not None:
        return _fmt(args, _V1_INDICES)

    # HIPBLAS: MT<MPerBlock>x<NPerBlock>x<KPerBlock>  MI<MPerXdl>x<NPerXdl>
    if demangled.startswith("Cijk_Ailk_Bljk"):
        parts = []
        mt = re.search(r"MT(\d+)x(\d+)x(\d+)", demangled)
        if mt:
            parts += [f"MPerBlock={mt.group(1)}", f"NPerBlock={mt.group(2)}", f"KPerBlock={mt.group(3)}"]
        mi = re.search(r"MI(\d+)x(\d+)", demangled)
        if mi:
            parts += [f"MPerXdl={mi.group(1)}", f"NPerXdl={mi.group(2)}"]
        return ", ".join(parts)

    return ""


# ---------------------------------------------------------------------------
# Kernel extraction from kernel_details__summarize_kernel_stats
# ---------------------------------------------------------------------------

def parse_kernel_details(raw: str) -> list[dict]:
    """
    Parse the kernel_details__summarize_kernel_stats column (a Python-repr list of dicts).
    Returns list of dicts with at least 'name' and 'mean_duration_us'.
    """
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        # The field contains Python repr with np.float64(...) calls — evaluate safely.
        import numpy as np  # noqa: F401 – needed for eval
        return ast.literal_eval(raw)
    except Exception:
        try:
            # Fall back: replace np.float64(...) wrappers before eval
            import re
            cleaned = re.sub(r"np\.float64\(([^)]+)\)", r"\1", raw)
            return ast.literal_eval(cleaned)
        except Exception:
            return []


def extract_compute_kernel(raw: str) -> tuple[str, str]:
    """
    Return (mangled_name, demangled_name) of the dominant compute kernel,
    or ('', '') if none found.
    HIPBLAS names are not mangled, so demangled == mangled for them.
    """
    kernels = parse_kernel_details(raw)
    compute = [k for k in kernels if is_compute_kernel(k.get("name", ""))]
    if not compute:
        return "", ""

    # Pick the kernel with the longest total / mean duration (most dominant)
    dominant = max(compute, key=lambda k: k.get("mean_duration_us", 0))
    mangled = dominant["name"]

    if is_ck_kernel(mangled):
        demangled = demangle_cached(mangled)
    else:
        demangled = mangled  # HIPBLAS names are already human-readable

    return mangled, demangled


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    # Try UTF-8 first, fall back to latin-1 (handles µ / non-ASCII characters in column names)
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    return df


def build_key(row: pd.Series) -> str:
    """Composite key: (input_shape, filter_shape, stride, padding, dilation, groups)."""
    return "|".join([
        str(row.get("param: input_shape", "")),
        str(row.get("param: filter_shape", "")),
        str(row.get("param: stride", "")),
        str(row.get("param: padding", "")),
        str(row.get("param: dilation", "")),
        str(row.get("param: groups", "")),
    ])


def process_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Extract relevant columns and compute kernel info."""
    print(f"[{label}] Processing {len(df)} rows ...")

    rows = []
    for _, row in df.iterrows():
        key = build_key(row)
        tflops_mean = row.get("TFLOPS/s_mean", float("nan"))
        raw_kernels = row.get("kernel_details__summarize_kernel_stats", "")
        mangled, demangled = extract_compute_kernel(raw_kernels)

        rows.append({
            "problem_key": key,
            "input_shape": row.get("param: input_shape", ""),
            "filter_shape": row.get("param: filter_shape", ""),
            "stride": row.get("param: stride", ""),
            "padding": row.get("param: padding", ""),
            "dilation": row.get("param: dilation", ""),
            "groups": row.get("param: groups", ""),
            f"TFLOPS_mean_{label}": tflops_mean,
            f"kernel_demangled_{label}": demangled,
            f"tiling_{label}": extract_tiling(demangled),
            f"double_buffering_{label}": extract_double_buffering(demangled),
        })

    result = pd.DataFrame(rows)
    print(f"[{label}] Done.")
    return result


def compare(new_csv: str, baseline_csv: str, output_csv: str) -> None:
    df_new = load_csv(new_csv)
    df_base = load_csv(baseline_csv)

    proc_new = process_df(df_new, "new")
    proc_base = process_df(df_base, "baseline")

    # Merge on problem key
    merged = pd.merge(
        proc_new,
        proc_base[["problem_key", "TFLOPS_mean_baseline", "kernel_demangled_baseline", "tiling_baseline", "double_buffering_baseline"]],
        on="problem_key",
        how="outer",
    )

    # Compute relative difference: (new - baseline) / baseline * 100
    merged["TFLOPS_delta_pct"] = (
        (merged["TFLOPS_mean_new"] - merged["TFLOPS_mean_baseline"])
        / merged["TFLOPS_mean_baseline"] * 100
    ).round(2)

    # Friendly sort: descending TFLOPS delta (regressions first)
    merged.sort_values("TFLOPS_delta_pct", ascending=True, inplace=True)

    merged["kernel name changed"] = (
        merged["kernel_demangled_new"] != merged["kernel_demangled_baseline"]
    ).map({True: "yes", False: "no"})

    merged.rename(columns={
        "kernel_demangled_baseline": "kernel_baseline",
        "kernel_demangled_new": "kernel_new",
        "double_buffering_baseline": "explicit double buffering_baseline",
        "double_buffering_new": "explicit double buffering_new",
    }, inplace=True)

    # Column order for the output CSV
    cols = [
        "input_shape",
        "filter_shape",
        "stride",
        "padding",
        "dilation",
        "groups",
        "TFLOPS_mean_baseline",
        "TFLOPS_mean_new",
        "TFLOPS_delta_pct",
        "kernel name changed",
        "explicit double buffering_baseline",
        "explicit double buffering_new",
        "tiling_baseline",
        "tiling_new",
        "kernel_baseline",
        "kernel_new",
    ]
    # Keep only columns that exist in merged
    cols = [c for c in cols if c in merged.columns]
    merged = merged[cols]

    merged.to_csv(output_csv, index=False)
    print(f"\nResults written to: {output_csv}")

    # Summary
    n_total = len(merged)
    n_regression = (merged["TFLOPS_delta_pct"] < -1).sum()
    n_improvement = (merged["TFLOPS_delta_pct"] > 1).sum()
    print(f"Total problems: {n_total}")
    print(f"  Regressions  (delta < -1%): {n_regression}")
    print(f"  Improvements (delta >  1%): {n_improvement}")
    print(f"  Neutral      (|delta| ≤ 1%): {n_total - n_regression - n_improvement}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    data_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new",
        default=str(data_dir / "conv_ops_comparison_new_img_I2V.csv"),
        help="Path to the 'new image' CSV (default: conv_ops_comparison_new_img_I2V.csv)",
    )
    parser.add_argument(
        "--baseline",
        default=str(data_dir / "conv_ops_comparison_xdit26.4_I2V.csv"),
        help="Path to the baseline CSV (default: conv_ops_comparison_xdit26.4_I2V.csv)",
    )
    parser.add_argument(
        "--output",
        default=str(data_dir / "conv_perf_comparison.csv"),
        help="Output CSV path (default: conv_perf_comparison.csv)",
    )
    args = parser.parse_args()

    compare(args.new, args.baseline, args.output)


if __name__ == "__main__":
    main()
