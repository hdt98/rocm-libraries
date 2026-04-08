#!/usr/bin/env python3
"""
Compare CK baseline vs improved benchmark results.

Usage:
    python3 plot_baseline_vs_improved.py \\
        --baseline data/i2v_baseline.txt \\
        --improved data/i2v_improved.txt \\
        --title "I2V: improved vs baseline" \\
        --output i2v_comparison.png

Or run without arguments to generate the standard i2v and t2v plots using
paths relative to the script location (../build-gfx950/data/).
"""

import argparse
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_file(path):
    """Return dict mapping normalised command string -> (name, tflops, avg_time_ms)."""
    results = {}
    with open(path) as f:
        text = f.read()

    entries = re.split(r'={40,}\s*\n(?=Input command:)', text)
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        cmd_match = re.search(r'Input command:(.*)', entry)
        tflops_match = re.search(r'\ntflops:\s+(\S+)', entry)
        time_match = re.search(r'\navg_time:\s+(\S+)', entry)
        name_match = re.search(r'\nname:\s+(.*)', entry)

        if not (cmd_match and tflops_match and time_match and name_match):
            continue

        cmd = cmd_match.group(1).strip()
        # Normalise: strip the MIOpenDriver binary prefix so paths don't matter
        cmd = re.sub(r'^.*MIOpenDriver\s+', '', cmd)

        results[cmd] = (
            name_match.group(1).strip(),
            float(tflops_match.group(1)),
            float(time_match.group(1)),
        )

    return results


def build_label(cmd):
    """Build a compact human-readable label from a convolution command string."""
    n  = re.search(r'\s-n\s+(\S+)', cmd)
    c  = re.search(r'(?<!\w)-c\s+(\S+)', cmd)
    H  = re.search(r'(?<!\w)-H\s+(\S+)', cmd)
    W  = re.search(r'(?<!\w)-W\s+(\S+)', cmd)
    k  = re.search(r'(?<!\w)-k\s+(\S+)', cmd)
    y  = re.search(r'(?<!\w)-y\s+(\S+)', cmd)
    x  = re.search(r'(?<!\w)-x\s+(\S+)', cmd)
    d  = re.search(r'--in_d\s+(\S+)', cmd)
    fd = re.search(r'--fil_d\s+(\S+)', cmd)

    parts = []
    if n and n.group(1) != '1':
        parts.append(f"n={n.group(1)}")
    if d:
        parts.append(f"D={d.group(1)}")
    parts.append(f"H={H.group(1)}" if H else "")
    parts.append(f"W={W.group(1)}" if W else "")
    parts.append(f"C={c.group(1)}" if c else "")
    parts.append(f"K={k.group(1)}" if k else "")
    fy = y.group(1) if y else '1'
    fx = x.group(1) if x else '1'
    ffd = fd.group(1) if fd else '1'
    if ffd != '1' or fy != '1' or fx != '1':
        parts.append(f"fil={ffd}x{fy}x{fx}")
    return " ".join(p for p in parts if p)


def make_plot(baseline_path, improved_path, title, out_path):
    baseline = parse_file(baseline_path)
    improved = parse_file(improved_path)

    # Only keep commands present in both files
    common = sorted(set(baseline) & set(improved))
    if not common:
        print(f"No matching commands found between the two files for '{title}'")
        return

    labels   = [build_label(cmd) for cmd in common]
    tfl_base = np.array([baseline[cmd][1] for cmd in common])
    tfl_impr = np.array([improved[cmd][1] for cmd in common])
    speedup  = tfl_impr / tfl_base

    n = len(common)
    x = np.arange(n)
    bar_w = 0.38

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, n * 0.9), 10),
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # --- TFLOPS bars ---
    b1 = ax1.bar(x - bar_w/2, tfl_base, bar_w, label='Baseline', color='#e07b54')
    b2 = ax1.bar(x + bar_w/2, tfl_impr, bar_w, label='Improved', color='#4c8cbf')

    ax1.set_ylabel('TFLOPS')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=7.5)
    ax1.legend()
    ax1.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax1.set_axisbelow(True)

    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1, f'{h:.0f}',
                 ha='center', va='bottom', fontsize=6, color='#a05030')
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1, f'{h:.0f}',
                 ha='center', va='bottom', fontsize=6, color='#2060a0')

    # --- Speedup bars ---
    colors = ['#2ca02c' if s >= 1.0 else '#d62728' for s in speedup]
    ax2.bar(x, speedup, 0.6, color=colors)
    ax2.axhline(1.0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_ylabel('Improved / Baseline')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=7.5)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax2.set_axisbelow(True)

    for i, s in enumerate(speedup):
        ax2.text(i, s + 0.005, f'{s:.2f}x', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}  ({n} matched cases, "
          f"avg speedup {speedup.mean():.2f}x, "
          f"max {speedup.max():.2f}x, "
          f"min {speedup.min():.2f}x)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline', help='Baseline benchmark file')
    parser.add_argument('--improved', help='Improved benchmark file')
    parser.add_argument('--title',   help='Plot title', default='Improved vs Baseline')
    parser.add_argument('--output',  help='Output PNG path')
    args = parser.parse_args()

    if args.baseline and args.improved and args.output:
        make_plot(args.baseline, args.improved, args.title, args.output)
    else:
        # Default: generate both i2v and t2v plots
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, '..', 'build-gfx950', 'data')
        data_dir = os.path.normpath(data_dir)

        for prefix, label in [('i2v', 'I2V'), ('t2v', 'T2V')]:
            make_plot(
                os.path.join(data_dir, f'{prefix}_baseline.txt'),
                os.path.join(data_dir, f'{prefix}_improved.txt'),
                f'{label}: improved vs baseline',
                os.path.join(data_dir, f'{prefix}_baseline_vs_improved.png'),
            )


if __name__ == '__main__':
    main()
