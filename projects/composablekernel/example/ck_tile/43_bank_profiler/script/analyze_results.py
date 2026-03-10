#!/usr/bin/env python3
"""
Analyze bank conflict profiling results from GEMM kernels.
Generates a comparative report across tile configurations and problem sizes.
"""

import pandas as pd
import os
import sys


# Tile configuration metadata
CONFIG_INFO = {
    "128x128x32": {"M_Tile": 128, "N_Tile": 128, "K_Tile": 32,
                    "M_Warp": 2, "N_Warp": 2, "Warps": 4, "BlockSize": 256},
    "256x128x32": {"M_Tile": 256, "N_Tile": 128, "K_Tile": 32,
                    "M_Warp": 4, "N_Warp": 2, "Warps": 8, "BlockSize": 512},
    "128x256x32": {"M_Tile": 128, "N_Tile": 256, "K_Tile": 32,
                    "M_Warp": 2, "N_Warp": 4, "Warps": 8, "BlockSize": 512},
    "256x256x32": {"M_Tile": 256, "N_Tile": 256, "K_Tile": 32,
                    "M_Warp": 4, "N_Warp": 4, "Warps": 16, "BlockSize": 1024},
    "64x64x32":   {"M_Tile": 64, "N_Tile": 64, "K_Tile": 32,
                    "M_Warp": 1, "N_Warp": 1, "Warps": 1, "BlockSize": 64},
    "128x128x64": {"M_Tile": 128, "N_Tile": 128, "K_Tile": 64,
                    "M_Warp": 2, "N_Warp": 2, "Warps": 4, "BlockSize": 256},
}


def compute_lds_layout_info(config):
    """Compute theoretical LDS layout properties from the 3D+padding policy."""
    info = CONFIG_INFO.get(config)
    if info is None:
        return None

    m_tile = info["M_Tile"]
    n_tile = info["N_Tile"]
    k_tile = info["K_Tile"]

    # A: shape [K/8, M, 8], stride [(M+1)*8, 8, 1]  (fp16 = 2 bytes)
    a_k0 = k_tile // 8
    a_stride_k0 = (m_tile + 1) * 8  # in elements
    a_space = a_k0 * a_stride_k0
    a_bytes = a_space * 2  # fp16

    # B: shape [K/8, N, 8], stride [(N+1)*8, 8, 1]
    b_k0 = k_tile // 8
    b_stride_k0 = (n_tile + 1) * 8
    b_space = b_k0 * b_stride_k0
    b_bytes = b_space * 2

    # Alignment between A and B
    a_aligned = ((a_bytes + 15) // 16) * 16
    total_lds = a_aligned + b_bytes

    return {
        "config": config,
        "M_Tile": m_tile,
        "N_Tile": n_tile,
        "K_Tile": k_tile,
        "A_bytes": a_bytes,
        "B_bytes": b_bytes,
        "Total_LDS_bytes": total_lds,
        "Total_LDS_KB": total_lds / 1024,
        "A_padding": f"+1 on M={m_tile} dim -> stride={(m_tile+1)*8}",
        "B_padding": f"+1 on N={n_tile} dim -> stride={(n_tile+1)*8}",
        "BlockSize": info["BlockSize"],
        "Warps": info["Warps"],
    }


def analyze(csv_path="out/gemm_bank_profile_results.csv",
            bank_results_path="bank_results.txt"):
    """Generate analysis report from profiling results."""
    if not os.path.isfile(csv_path):
        print(f"Results file not found: {csv_path}")
        print("Run gemm_profiler.py first.")
        return

    df = pd.read_csv(csv_path)

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("CK Tile GEMM Bank Conflict Analysis Report")
    report_lines.append("=" * 70)
    report_lines.append("")

    # 1. Hardware bank count (if available)
    if os.path.isfile(bank_results_path):
        with open(bank_results_path) as f:
            bank_info = f.read()
        report_lines.append("Hardware Bank Information:")
        for line in bank_info.split("\n")[:5]:
            report_lines.append(f"  {line}")
        report_lines.append("")

    # 2. LDS layout info per config
    report_lines.append("-" * 70)
    report_lines.append("Theoretical LDS Layout (3D + Padding Policy)")
    report_lines.append("-" * 70)
    lds_info_rows = []
    for config in sorted(CONFIG_INFO.keys()):
        info = compute_lds_layout_info(config)
        if info:
            lds_info_rows.append(info)
            report_lines.append(
                f"  {config}: A={info['A_bytes']}B, B={info['B_bytes']}B, "
                f"Total={info['Total_LDS_KB']:.1f}KB, "
                f"BlockSize={info['BlockSize']}, Warps={info['Warps']}")
    report_lines.append("")

    # 3. Conflict ratio table (config x problem size)
    report_lines.append("-" * 70)
    report_lines.append("Conflict Ratios (conflicts / LDS instructions)")
    report_lines.append("-" * 70)

    pivot = df.pivot_table(
        index="config",
        columns=df.apply(lambda r: f"{r['M']}x{r['N']}x{r['K']}", axis=1),
        values="conflict_ratio",
        aggfunc="first")
    report_lines.append(pivot.to_string())
    report_lines.append("")

    # 4. Raw conflict counts
    report_lines.append("-" * 70)
    report_lines.append("Raw Conflict Counts")
    report_lines.append("-" * 70)
    pivot_raw = df.pivot_table(
        index="config",
        columns=df.apply(lambda r: f"{r['M']}x{r['N']}x{r['K']}", axis=1),
        values="conflicts",
        aggfunc="first")
    report_lines.append(pivot_raw.to_string())
    report_lines.append("")

    # 5. Average conflict ratio per config
    report_lines.append("-" * 70)
    report_lines.append("Average Conflict Ratio per Configuration")
    report_lines.append("-" * 70)
    avg = df.groupby("config")["conflict_ratio"].mean().sort_values()
    for config, ratio in avg.items():
        report_lines.append(f"  {config}: {ratio:.6f}")
    report_lines.append("")

    # 6. Best/worst configs
    best_config = avg.idxmin()
    worst_config = avg.idxmax()
    report_lines.append(f"Best (lowest conflicts):  {best_config} "
                        f"({avg[best_config]:.6f})")
    report_lines.append(f"Worst (highest conflicts): {worst_config} "
                        f"({avg[worst_config]:.6f})")
    report_lines.append("")

    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)
    print(report_text)

    out_path = "out/analysis_report.txt"
    with open(out_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 \
        else "out/gemm_bank_profile_results.csv"
    analyze(csv_path)
