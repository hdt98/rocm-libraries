#!/usr/bin/env python3
"""
Extract avg_time and best instance name from CK / CK-Tile profiler output files.
Also finds and prints the maximum error across all [Error] entries.

Usage:
    python script/extract_perf.py <output_file> [<output_file> ...]

Outputs:
    <input>_avg_times.txt   - one avg_time value per line
    <input>_best_names.txt  - one best-instance name per line
    Prints max error to stdout.
"""

import re
import sys
from pathlib import Path


# ── regex patterns ────────────────────────────────────────────────────────────

# CK:       "avg_time: 0.159877"
# CK-Tile:  "\tavg_time: 0.0396603, SplitK 128"
RE_AVG_TIME = re.compile(r"avg_time:\s*([\d.e+\-]+)")

# CK:       "name: DeviceGrouped... (instance N)"
# CK-Tile:  "\tname: GroupedConvolution..."
RE_NAME = re.compile(r"^\s*name:\s*(.+)$")

# CK-Tile error line: "\tNumber of incorrect values: N Is all zero:N max err: 0.012"
# CK-Tile (older):   "\tmax err: 0.012"
RE_MAX_ERR = re.compile(r"max err:\s*([\d.e+\-]+)")


def process_file(path: Path):
    avg_times = []
    best_names = []
    max_errors = []

    in_best_block = False
    pending_name = None

    with open(path, errors="replace") as f:
        for line in f:
            # ── best-config block ────────────────────────────────────────────
            if "Best configuration parameters:" in line:
                in_best_block = True
                pending_name = None
                continue

            if in_best_block:
                m_name = RE_NAME.match(line)
                if m_name:
                    pending_name = m_name.group(1).strip()
                    continue

                m_time = RE_AVG_TIME.search(line)
                if m_time:
                    avg_times.append(float(m_time.group(1)))
                    if pending_name is not None:
                        best_names.append(pending_name)
                    in_best_block = False
                    pending_name = None
                    continue

                # blank line or unrelated line ends the block
                if line.strip() == "" or (
                    not line.startswith("\t") and "name" not in line.lower()
                    and "avg_time" not in line and "tflops" not in line
                    and "GB/s" not in line and "Ignored" not in line
                ):
                    in_best_block = False
                continue

            # ── error lines ──────────────────────────────────────────────────
            m_err = RE_MAX_ERR.search(line)
            if m_err:
                max_errors.append(float(m_err.group(1)))

    return avg_times, best_names, max_errors


def main(files):
    all_errors = []

    for fname in files:
        path = Path(fname)
        if not path.exists():
            print(f"[WARN] File not found: {fname}", file=sys.stderr)
            continue

        avg_times, best_names, errors = process_file(path)
        all_errors.extend(errors)

        # write avg_times
        out_times = path.parent / (path.name + "_avg_times.txt")
        out_times.write_text("\n".join(str(t) for t in avg_times) + "\n")
        print(f"[{path.name}] avg_times ({len(avg_times)} entries) → {out_times}")

        # write best names
        out_names = path.parent / (path.name + "_best_names.txt")
        out_names.write_text("\n".join(best_names) + "\n")
        print(f"[{path.name}] best_names ({len(best_names)} entries) → {out_names}")

    # top 10 largest errors
    if all_errors:
        top10 = sorted(all_errors, reverse=True)[:10]
        print(f"\nTop {len(top10)} largest errors across all files:")
        for i, err in enumerate(top10, 1):
            print(f"  {i:2}. {err}")
    else:
        print("\nNo error entries found.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
