#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Display a per-lane LDS bank access table from gpu_trace.py output.

Reads one or more first_bank arrays from stdin and prints an access table.
Columns are bank indices; cells show which thread accesses that bank.
Row groups reflect the physical thread layout. If multiple threads in the
same group hit the same bank they are serialized into additional sub-rows.

Usage:
  jq 'select(.address == "0x..." and .wave == 0) | .derived.first_bank' \\
      trace.jsonl | python3 lds_bank_table.py
"""

import argparse
import json
import sys

import pandas as pd

ROW_GROUPS = [
    [ 0,  1,  2,  3, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27],
    [32, 33, 34, 35, 44, 45, 46, 47, 52, 53, 54, 55, 56, 57, 58, 59],
    [ 4,  5,  6,  7,  8,  9, 10, 11, 16, 17, 18, 19, 28, 29, 30, 31],
    [36, 37, 38, 39, 40, 41, 42, 43, 48, 49, 50, 51, 60, 61, 62, 63],
]


def render_table(first_bank, n_banks):

    columns = [str(b) for b in range(n_banks)]

    # Build all data rows, tracking where group dividers go.
    data_rows = []
    divider_before = []  # data-row indices before which to insert a divider

    for group in ROW_GROUPS:
        divider_before.append(len(data_rows))

        bank_threads: dict[int, list[int]] = {}
        for t in group:
            bank_threads.setdefault(first_bank[t], []).append(t)

        n_subrows = max(len(v) for v in bank_threads.values())
        for sub in range(n_subrows):
            row = {
                str(b): (
                    f"T{bank_threads[b][sub]}" if sub < len(bank_threads.get(b, [])) else ""
                )
                for b in range(n_banks)
            }
            data_rows.append(row)

    df = pd.DataFrame(data_rows, columns=columns)
    lines = df.to_string(index=False, col_space=3).splitlines()
    header, data_lines = lines[0], lines[1:]

    divider = "-" * len(header)
    divider_set = set(divider_before)

    result = [header]
    for i, line in enumerate(data_lines):
        if i in divider_set:
            result.append(divider)
        result.append(line)

    print("\n".join(result))


def parse_records(text):
    """Extract all top-level JSON values from text (handles pretty-printed jq output)."""
    decoder = json.JSONDecoder()
    records = []
    idx = 0
    text = text.strip()
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        records.append(obj)
        idx = end
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Instruction data width in bits (e.g. 128 for ds_read_b128). Default: 128.",
    )
    args = parser.parse_args()

    # 64 LDS banks, each 4 bytes (1 dword) wide.
    # A b<W> instruction spans W/32 dwords, so the number of bank groups is 64 / (W/32).
    dwords_per_instr = args.width // 32
    n_banks = 64 // dwords_per_instr

    records = parse_records(sys.stdin.read())
    for i, record in enumerate(records):
        if i > 0:
            print()
        first_bank = record if isinstance(record, list) else record["first_bank"]
        render_table(first_bank, n_banks)


if __name__ == "__main__":
    main()
