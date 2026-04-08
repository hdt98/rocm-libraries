#!/usr/bin/env python3
"""
tracelens_to_baseline.py

Map TraceLens CSV rows to CK benchmark (.txt) entries and produce a merged
comparison file in the same order as the .txt file.

Usage:
    python3 tracelens_to_miopendriver.py <tracelens.csv> <baseline.txt> [--output out.txt]

Output format (one block per conv):
    ================================================================================
    Input command: <MIOpenDriver command>
    ================================================================================
    # TraceLens (model runtime)
    avg_time (us): <mean kernel time from CSV>
    TFLOPS: <TFLOPS/s_mean from CSV>
    GB/s: <TB/s_mean * 1000 from CSV>
    # CK benchmark
    name: <best kernel name>
    avg_time: <ms>
    tflops: <TFLOPS>
    GB/s: <GB/s>
"""

import argparse
import ast
import csv
import re
import sys


SEP = "=" * 80


# ---------------------------------------------------------------------------
# Key extraction helpers
# ---------------------------------------------------------------------------

def _parse_tuple(s: str):
    return ast.literal_eval(s.strip())


def miopen_key(cmd: str) -> tuple:
    """Extract a canonical shape key from a MIOpenDriver command string."""
    args: dict = {}
    tokens = cmd.split()
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("-"):
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                args[t] = tokens[i + 1]
                i += 2
            else:
                args[t] = True
                i += 1
        else:
            i += 1

    return (
        args.get("-n", "1"),
        args.get("-c", "?"),
        args.get("-k", "?"),
        args.get("--in_d", "1"),
        args.get("-H", "?"),
        args.get("-W", "?"),
        args.get("--fil_d", "1"),
        args.get("-y", "?"),
        args.get("-x", "?"),
        args.get("-g", "1"),
        args.get("--conv_stride_d", "1"),
        args.get("-u", "1"),
        args.get("-v", "1"),
        args.get("--pad_d", "0"),
        args.get("-p", "0"),
        args.get("-q", "0"),
        args.get("--dilation_d", "1"),
        args.get("-l", "1"),
        args.get("-j", "1"),
        args.get("--spatial_dim", "2"),
    )


def csv_key(row: dict) -> tuple:
    """Extract a canonical shape key from a TraceLens CSV row."""
    input_shape = _parse_tuple(row["param: input_shape (NCDHW)"])
    filter_shape = _parse_tuple(row["param: filter_shape (KCZYX)"])
    stride = _parse_tuple(row["param: stride"])
    padding = _parse_tuple(row["param: padding"])
    dilation = _parse_tuple(row["param: dilation"])
    g = row["param: groups"].strip()
    conv_type = row["param: convNd"].strip()

    N = str(input_shape[0])
    C = str(input_shape[1])
    K = str(filter_shape[0])

    if conv_type == "conv3d":
        D = str(input_shape[2])
        H = str(input_shape[3])
        W = str(input_shape[4])
        Z = str(filter_shape[2])
        Y = str(filter_shape[3])
        X = str(filter_shape[4])
        sd = str(stride[0])
        su = str(stride[1])
        sv = str(stride[2])
        pd = str(padding[0])
        p  = str(padding[1])
        q  = str(padding[2])
        dl  = str(dilation[0])
        dl2 = str(dilation[1])
        dl3 = str(dilation[2])
        spatial = "3"
    else:  # conv2d — no depth dimension
        D = "1"; Z = "1"; sd = "1"; pd = "0"; dl = "1"
        H = str(input_shape[2])
        W = str(input_shape[3])
        Y = str(filter_shape[2])
        X = str(filter_shape[3])
        su = str(stride[0])
        sv = str(stride[1])
        p  = str(padding[0])
        q  = str(padding[1])
        dl2 = str(dilation[0])
        dl3 = str(dilation[1])
        spatial = "2"

    return (N, C, K, D, H, W, Z, Y, X, g, sd, su, sv, pd, p, q, dl, dl2, dl3, spatial)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _time_col(fieldnames):
    """Return the Kernel Time (µs)_mean column name (encoding-robust)."""
    for col in fieldnames:
        if "Kernel Time" in col and "_mean" in col:
            return col
    return None


def parse_baseline(path: str):
    """
    Parse a CK benchmark .txt file.

    Returns a list of dicts, one per conv entry:
        {command, name, avg_time, tflops, gb_s}
    """
    text = open(path).read()
    blocks = re.split(r"={40,}", text)
    blocks = [b.strip() for b in blocks if b.strip()]

    entries = []
    i = 0
    while i < len(blocks):
        cmd_match = re.search(r"Input command:\s*(.*)", blocks[i])
        if cmd_match:
            cmd = cmd_match.group(1).strip()
            # The next block should contain the kernel results
            result = {}
            if i + 1 < len(blocks):
                b = blocks[i + 1]
                nm = re.search(r"name:\s*(.*)", b)
                at = re.search(r"avg_time:\s*([\d.eE+\-]+)", b)
                tf = re.search(r"tflops:\s*([\d.eE+\-]+)", b)
                gb = re.search(r"GB/s:\s*([\d.eE+\-]+)", b)
                result = {
                    "name":     nm.group(1).strip() if nm else "",
                    "avg_time": at.group(1) if at else "",
                    "tflops":   tf.group(1) if tf else "",
                    "gb_s":     gb.group(1) if gb else "",
                }
                i += 2
            else:
                i += 1
            entries.append({"command": cmd, **result})
        else:
            i += 1

    return entries


def parse_tracelens(path: str):
    """
    Parse a TraceLens CSV file.

    Returns:
        (list of dicts with keys: key, avg_time_us, tflops, gb_s)
        where key is the canonical shape tuple.
    """
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    time_col = _time_col(fieldnames)

    result = []
    for row in rows:
        key = csv_key(row)
        avg_us = row.get(time_col, "") if time_col else ""
        tflops = row.get("TFLOPS/s_mean", "")
        tb_s   = row.get("TB/s_mean", "")
        gb_s   = f"{float(tb_s) * 1000:.3f}" if tb_s else ""
        result.append({
            "key": key,
            "avg_time_us": avg_us,
            "tflops": tflops,
            "gb_s": gb_s,
        })

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge TraceLens CSV with CK benchmark .txt in baseline order."
    )
    parser.add_argument("csv", help="TraceLens CSV file")
    parser.add_argument("baseline", help="CK benchmark .txt file")
    parser.add_argument("--output", "-o", default="-", help="Output file (default: stdout)")
    args = parser.parse_args()

    baseline_entries = parse_baseline(args.baseline)
    tracelens_rows   = parse_tracelens(args.csv)

    # Build lookup: shape_key -> tracelens row (last wins for duplicates)
    tl_lookup: dict = {}
    for tl in tracelens_rows:
        tl_lookup[tl["key"]] = tl

    out = sys.stdout if args.output == "-" else open(args.output, "w")

    unmatched = 0
    for entry in baseline_entries:
        key = miopen_key(entry["command"])
        tl  = tl_lookup.get(key)

        print(SEP, file=out)
        print(f"Input command: {entry['command']}", file=out)
        print(SEP, file=out)

        if tl:
            ms_str = f"{float(tl['avg_time_us']) / 1000:.6f}" if tl["avg_time_us"] else "N/A"
            tf_str = f"{float(tl['tflops']):.4f}"             if tl["tflops"]       else "N/A"
            gb_str = tl["gb_s"] or "N/A"
            print(f"# TraceLens (model runtime)", file=out)
            print(f"avg_time (ms): {ms_str}", file=out)
            print(f"tflops:        {tf_str}", file=out)
            print(f"GB/s:          {gb_str}", file=out)
        else:
            print("# TraceLens: no match found", file=out)
            unmatched += 1

        print(f"# CK benchmark", file=out)
        if entry.get("name"):
            print(f"name:          {entry['name']}", file=out)
        if entry.get("avg_time"):
            print(f"avg_time (ms): {entry['avg_time']}", file=out)
        if entry.get("tflops"):
            print(f"tflops:        {entry['tflops']}", file=out)
        if entry.get("gb_s"):
            print(f"GB/s:          {entry['gb_s']}", file=out)
        print(file=out)

    if out is not sys.stdout:
        out.close()

    n = len(baseline_entries)
    matched = n - unmatched
    print(
        f"Done: {matched}/{n} baseline entries matched to TraceLens rows"
        + (f" ({unmatched} unmatched)" if unmatched else ""),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
