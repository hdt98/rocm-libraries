#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Derive macro-tile access patterns empirically from a gpu_trace.py trace.

Given trace.jsonl and matrix metadata, produces tables showing which
wave.lane accesses each (row, K-chunk) of the A and B tiles for:

  GR  buffer_load...lds  global memory -> LDS
  LR  ds_read_b128       LDS -> VGPRs

For LR the mapping is derived by composing GR data (global addr ->
(m_row, k_chunk)) with the LDS address written during GR. This means
the tables reflect the actual data flow regardless of swizzle pattern.

Requires: polars

Usage (256x256x256 mxfp4 GEMM, LDA = K = 32768 elements):
  python3 trace_tile_map.py trace.jsonl --mac-m 256 --mac-k 256 --lda 32768

For B tile (LDB = M = 4096 elements, transposed B):
  python3 trace_tile_map.py trace.jsonl --mac-m 256 --mac-k 256 --lda 32768 \\
      --ldb 4096 --mac-n 256

Filter to a specific workgroup:
  python3 trace_tile_map.py trace.jsonl --lda 32768 --workgroup 15,15,0
"""

import argparse
import json
import sys

import polars as pl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_trace(path: str) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def truncate(df: pl.DataFrame, head: int = 20, tail: int = 3) -> pl.DataFrame:
    if len(df) <= head + tail:
        return df
    ellipsis_row = {col: "..." for col in df.columns}
    return pl.concat([df.head(head), pl.DataFrame([ellipsis_row]), df.tail(tail)])


def decode_global(addr: int, base: int, row_bytes: int, chunk_bytes: int):
    """Return (row, k_chunk) from a byte address in global memory."""
    off = addr - base
    row = off // row_bytes
    chunk = (off % row_bytes) // chunk_bytes
    return int(row), int(chunk)


def build_df(mapping: dict, n_rows: int, n_chunks: int, row_label: str = "row") -> pl.DataFrame:
    """Build a Polars DataFrame: rows = matrix rows, cols = K-chunks."""
    rows = []
    for r in range(n_rows):
        row = {row_label: str(r)}
        for c in range(n_chunks):
            row[f"K{c}"] = mapping.get((r, c), "")
        rows.append(row)
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def analyse(
    records: list,
    mac_rows: int,
    n_k_chunks: int,
    row_bytes: int,
    chunk_bytes: int,
    row_label: str,
) -> tuple[dict, dict, dict]:
    """
    Analyse GR and LR records for one tile (A or B).

    Returns:
      gr_mapping   (row, chunk) -> "w{wave}.L{lane}" from buffer_load...lds
      lr_mapping   (row, chunk) -> "w{wave}.L{lane}" from ds_read via LDS composition
      lds_to_tile  LDS byte address -> (row, chunk) side-product for inspection
    """
    # Separate record types
    gr_recs = [
        r for r in records
        if r.get("family") == "buffer"
        and r.get("derived", {}).get("lds_direct")
    ]
    lr_recs = [
        r for r in records
        if r.get("family") == "lds"
        and r.get("instruction", "").split()[0].startswith("ds_read")
    ]

    if not gr_recs and not lr_recs:
        return {}, {}, {}

    # ---- GR pass: decode (row, chunk) from global address ------------------
    # Find global base as minimum effective address across all GR records.
    all_global = [
        addr
        for rec in gr_recs
        for addr in rec["derived"]["effective_addr"]
    ]
    if not all_global:
        global_base = 0
    else:
        global_base = min(all_global)

    gr_mapping: dict = {}
    lds_to_tile: dict = {}  # LDS byte addr -> (row, chunk)

    for rec in gr_recs:
        wave = rec["wave"]
        global_addrs = rec["derived"]["effective_addr"]
        lds_addrs = rec["derived"].get("lds_effective_addr", [])

        for lane, gaddr in enumerate(global_addrs):
            row, chunk = decode_global(gaddr, global_base, row_bytes, chunk_bytes)
            if 0 <= row < mac_rows and 0 <= chunk < n_k_chunks:
                key = (row, chunk)
                gr_mapping[key] = f"w{wave}.L{lane}"
                if lane < len(lds_addrs):
                    lds_to_tile[lds_addrs[lane]] = key

    # ---- LR pass: look up tile position via LDS address --------------------
    lr_mapping: dict = {}

    for rec in lr_recs:
        wave = rec["wave"]
        lds_addrs = rec["derived"]["effective_addr"]

        for lane, laddr in enumerate(lds_addrs):
            key = lds_to_tile.get(laddr)
            if key is not None:
                row, chunk = key
                if 0 <= row < mac_rows and 0 <= chunk < n_k_chunks:
                    lr_mapping[(row, chunk)] = f"w{wave}.L{lane}"

    return gr_mapping, lr_mapping, lds_to_tile


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_EPILOG = """
examples:
  # 256x256x256 mxfp4 TN GEMM, captured with gpu_trace.py --workgroup 15,15,0
  ./trace_tile_map.py trace.jsonl --lda 32768 --workgroup 15,15,0

  # same, only show A tile
  ./trace_tile_map.py trace.jsonl --lda 32768 --workgroup 15,15,0 --tile a

  # capture the trace (run from the kernel binary directory):
  rocgdb --batch \\
    -ex "source path/to/gpu_trace.py" \\
    -ex 'gpu_trace --workgroup 15,15,0 --output trace.jsonl \\
         --kernel <kernel_label> --families buffer lds' \\
    --args ./rocroller-gemm <gemm args>
"""


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("trace", help="trace.jsonl from gpu_trace.py")
    p.add_argument("--mac-m", type=int, default=256, help="Macro-tile M dimension (default 256)")
    p.add_argument("--mac-n", type=int, default=256, help="Macro-tile N dimension (default 256)")
    p.add_argument("--mac-k", type=int, default=256, help="Macro-tile K dimension in elements (default 256)")
    p.add_argument(
        "--lda",
        type=int,
        required=True,
        help="Leading dimension of A in global memory (elements per row, e.g. full K for TN layout)",
    )
    p.add_argument(
        "--ldb",
        type=int,
        default=None,
        help="Leading dimension of B in global memory (elements per row). "
             "Defaults to --lda if not given.",
    )
    p.add_argument("--dt-bits", type=int, default=4, help="Bits per element (default 4 for fp4)")
    p.add_argument(
        "--chunk-bytes",
        type=int,
        default=16,
        help="Bytes per ds_read_b128 / buffer_load_dwordx4 (default 16)",
    )
    p.add_argument(
        "--workgroup",
        type=str,
        default=None,
        help="Workgroup to analyse, e.g. '15,15,0'. Default: first workgroup in trace.",
    )
    p.add_argument(
        "--tile",
        choices=["a", "b", "both"],
        default="both",
        help="Which matrix tile to show (default: both). "
             "A and B are separated by instruction program-order address.",
    )
    return p.parse_args()


def select_workgroup(records: list, target: list | None) -> list:
    """Filter records to a single workgroup."""
    if target is not None:
        return [r for r in records if r.get("workgroup") == target]
    # Use the first workgroup that appears
    for r in records:
        wg = r.get("workgroup")
        if wg is not None:
            first = wg
            break
    else:
        return records
    filtered = [r for r in records if r.get("workgroup") == first]
    print(f"Using workgroup {first}  ({len(filtered)} records)")
    return filtered


def _most_common(counter: dict) -> list:
    """Return keys sorted by descending count."""
    return sorted(counter, key=lambda k: -counter[k])


def split_ab(records: list) -> tuple[list, list]:
    """
    Split buffer_load...lds (GR) and ds_read (LR) records into A and B tiles.

    GR split: cluster by Shader Resource Descriptor (SRD) SGPR range in the
    instruction text (e.g. s[8:11] vs s[12:15]).  The two most common SRDs
    among dwordx4 loads correspond to A and B.

    LR split: cluster ds_read_b128 records by their address VGPR.  The two
    most common address registers correspond to A and B.

    Both approaches are robust to A/B instruction interleaving in the kernel.
    """
    import re

    gr_recs = [
        r for r in records
        if r.get("family") == "buffer"
        and r.get("derived", {}).get("lds_direct")
        and "dwordx4" in r.get("instruction", "")
    ]
    lr_recs = [
        r for r in records
        if r.get("family") == "lds"
        and r.get("instruction", "").split()[0] == "ds_read_b128"
    ]

    # --- GR split by SRD ---
    srd_re = re.compile(r"s\[(\d+):\d+\]")
    srd_count: dict[str, int] = {}
    addr_to_srd: dict[str, str] = {}
    for r in gr_recs:
        m = srd_re.search(r["instruction"])
        srd = m.group(0) if m else "?"
        srd_count[srd] = srd_count.get(srd, 0) + 1
        addr_to_srd[r["address"]] = srd

    top_srds = _most_common(srd_count)
    srd_a = top_srds[0] if len(top_srds) > 0 else None
    srd_b = top_srds[1] if len(top_srds) > 1 else None
    print(f"GR split: SRD A={srd_a} ({srd_count.get(srd_a,0)} instrs), "
          f"B={srd_b} ({srd_count.get(srd_b,0)} instrs)")

    a_gr_addrs = {r["address"] for r in gr_recs if addr_to_srd.get(r["address"]) == srd_a}
    b_gr_addrs = {r["address"] for r in gr_recs if addr_to_srd.get(r["address"]) == srd_b}

    # --- LR split by address VGPR ---
    areg_re = re.compile(r"ds_read_b128\s+\S+,\s*(v\d+)")
    areg_count: dict[str, int] = {}
    addr_to_areg: dict[str, str] = {}
    for r in lr_recs:
        m = areg_re.search(r["instruction"])
        areg = m.group(1) if m else "?"
        areg_count[areg] = areg_count.get(areg, 0) + 1
        addr_to_areg[r["address"]] = areg

    top_aregs = _most_common(areg_count)
    areg_a = top_aregs[0] if len(top_aregs) > 0 else None
    areg_b = top_aregs[1] if len(top_aregs) > 1 else None
    print(f"LR split: addr-reg A={areg_a} ({areg_count.get(areg_a,0)} instrs), "
          f"B={areg_b} ({areg_count.get(areg_b,0)} instrs)")

    a_lr_addrs = {r["address"] for r in lr_recs if addr_to_areg.get(r["address"]) == areg_a}
    b_lr_addrs = {r["address"] for r in lr_recs if addr_to_areg.get(r["address"]) == areg_b}

    a_recs = [r for r in records if r.get("address") in a_gr_addrs or r.get("address") in a_lr_addrs]
    b_recs = [r for r in records if r.get("address") in b_gr_addrs or r.get("address") in b_lr_addrs]
    return a_recs, b_recs


def main():
    args = parse_args()

    records = load_trace(args.trace)
    print(f"Loaded {len(records)} records from {args.trace}")

    # Filter to target workgroup
    target_wg = list(map(int, args.workgroup.split(","))) if args.workgroup else None
    records = select_workgroup(records, target_wg)

    dt_bytes = args.dt_bits / 8.0                          # e.g. 0.5 for fp4
    chunk_bytes = args.chunk_bytes                          # 16 for dwordx4
    chunk_elems = int(chunk_bytes / dt_bytes)               # fp4 elements per chunk

    n_k_chunks_a = args.mac_k // chunk_elems               # K-chunks in A
    n_k_chunks_b = args.mac_k // chunk_elems               # K-chunks in B (same K)

    row_bytes_a = int(args.lda * dt_bytes)                  # global A row stride in bytes
    ldb = args.ldb if args.ldb is not None else args.lda
    row_bytes_b = int(ldb * dt_bytes)                       # global B row stride in bytes

    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_width_chars(300)

    # Split records into A and B tile instructions
    a_recs, b_recs = split_ab(records)
    print(f"A records: {len(a_recs)}, B records: {len(b_recs)}")

    def show_tile(recs, mac_rows, n_chunks, row_bytes, name, row_label):
        gr_map, lr_map, _ = analyse(recs, mac_rows, n_chunks, row_bytes, chunk_bytes, row_label)
        covered_gr = sum(1 for v in gr_map.values() if v)
        covered_lr = sum(1 for v in lr_map.values() if v)
        total = mac_rows * n_chunks

        print()
        print("=" * 80)
        print(f"GR -- {name} global tile -> LDS  (buffer_load_dwordx4 ... lds)")
        print(f"  {covered_gr}/{total} cells covered")
        print(f"  Cell = wave.lane that loaded (row, K-chunk) from global memory")
        if gr_map:
            print(truncate(build_df(gr_map, mac_rows, n_chunks, row_label)))
        else:
            print("  [no data]")

        print()
        print("=" * 80)
        print(f"LR -- {name} LDS tile -> VGPRs  (ds_read_b128)")
        print(f"  {covered_lr}/{total} cells covered  [derived by composing GR LDS addr -> tile]")
        print(f"  Cell = wave.lane that reads (row, K-chunk) from LDS")
        if lr_map:
            print(truncate(build_df(lr_map, mac_rows, n_chunks, row_label)))
        else:
            print("  [no data -- need GR records with lds_effective_addr]")

    if args.tile in ("a", "both"):
        show_tile(a_recs, args.mac_m, n_k_chunks_a, row_bytes_a, "A", "M-row")

    if args.tile in ("b", "both"):
        show_tile(b_recs, args.mac_n, n_k_chunks_b, row_bytes_b, "B", "N-row")


if __name__ == "__main__":
    main()
