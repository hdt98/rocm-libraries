#!/usr/bin/env python3
"""Plot CK benchmark TFLOPS vs TraceLens baseline TFLOPS for matched cases."""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_file(path):
    """Parse benchmark file, return list of (label, tracelens_tflops, ck_tflops) for matched cases."""
    records = []
    with open(path) as f:
        text = f.read()

    # Each entry starts with the separator line followed by "Input command:"
    # Split on separator+Input-command boundary to get self-contained entries
    entries = re.split(r'={40,}\s*\n(?=Input command:)', text)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        if '# TraceLens: no match found' in entry:
            continue
        if '# TraceLens (model runtime)' not in entry:
            continue

        cmd_match = re.search(r'Input command:(.*)', entry)
        if not cmd_match:
            continue
        cmd = cmd_match.group(1)

        # Build short label
        n  = re.search(r'\s-n\s+(\S+)', cmd)
        c  = re.search(r'\s-c\s+(\S+)', cmd)
        H  = re.search(r'\s-H\s+(\S+)', cmd)
        W  = re.search(r'\s-W\s+(\S+)', cmd)
        k  = re.search(r'\s-k\s+(\S+)', cmd)
        y  = re.search(r'\s-y\s+(\S+)', cmd)
        x  = re.search(r'\s-x\s+(\S+)', cmd)
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
        fil = f"{fd.group(1) if fd else '1'}x{y.group(1) if y else '1'}x{x.group(1) if x else '1'}"
        parts.append(f"fil={fil}")
        label = " ".join(p for p in parts if p)

        tl_match = re.search(r'# TraceLens \(model runtime\)\s*\navg_time[^\n]*\ntflops:\s+(\S+)', entry)
        if not tl_match:
            continue
        tl_tflops = float(tl_match.group(1))

        ck_match = re.search(r'# CK benchmark\s*\nname:[^\n]*\navg_time[^\n]*\ntflops:\s+(\S+)', entry)
        if not ck_match:
            continue
        ck_tflops = float(ck_match.group(1))

        records.append((label, tl_tflops, ck_tflops))

    return records


def make_plot(records, title, out_path):
    n = len(records)
    if n == 0:
        print(f"No matched records for {title}")
        return

    labels = [r[0] for r in records]
    tl     = np.array([r[1] for r in records])
    ck     = np.array([r[2] for r in records])
    speedup = ck / tl

    x = np.arange(n)
    bar_w = 0.38

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, n * 0.9), 10),
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # --- TFLOPS bars ---
    b1 = ax1.bar(x - bar_w/2, tl, bar_w, label='TraceLens baseline', color='#e07b54')
    b2 = ax1.bar(x + bar_w/2, ck, bar_w, label='CK benchmark',       color='#4c8cbf')

    ax1.set_ylabel('TFLOPS')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=7.5)
    ax1.legend()
    ax1.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax1.set_axisbelow(True)

    # Annotate bars with values
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.0f}',
                 ha='center', va='bottom', fontsize=6, color='#e07b54')
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.0f}',
                 ha='center', va='bottom', fontsize=6, color='#4c8cbf')

    # --- Speedup bars ---
    colors = ['#2ca02c' if s >= 1.0 else '#d62728' for s in speedup]
    ax2.bar(x, speedup, 0.6, color=colors)
    ax2.axhline(1.0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_ylabel('CK / TraceLens')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=7.5)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax2.set_axisbelow(True)

    for i, s in enumerate(speedup):
        ax2.text(i, s + 0.01, f'{s:.2f}x', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}  ({n} matched cases)")


base = '/home/AMD/vpietila/git/rocm-libraries-ck2/projects/composablekernel/build-gfx950/data'

t2v = parse_file(f'{base}/t2v_improved_vs_trace_lens.txt')
i2v = parse_file(f'{base}/i2v_improved_vs_trace_lens.txt')

make_plot(t2v, 'T2V: CK benchmark vs TraceLens baseline (matched cases)',
          f'{base}/t2v_vs_tracelens.png')
make_plot(i2v, 'I2V: CK benchmark vs TraceLens baseline (matched cases)',
          f'{base}/i2v_vs_tracelens.png')
