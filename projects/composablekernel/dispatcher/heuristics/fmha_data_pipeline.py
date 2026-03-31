#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
FMHA benchmark CSV to training parquet converter.

Reads fmha_bench_all.csv (produced by tile_engine/ops/fmha/fmha_full_benchmark.py)
and outputs a canonical parquet file ready for LightGBM training.

Usage:
    python fmha_data_pipeline.py --input fmha_bench_all.csv --output fmha_fwd_gfx950.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BOOL_COLS = [
    "lse",
    "dropout",
    "logits",
    "sink",
    "skip",
    "paged_kv",
    "deterministic",
    "dbias",
]


def load_fmha_csv(path: str | Path) -> pd.DataFrame:
    """Load FMHA benchmark CSV and normalize types."""
    df = pd.read_csv(path)

    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: 1 if str(v).lower() in ("true", "1", "yes") else 0
            )

    int_cols = [
        "batch",
        "seqlen_q",
        "seqlen_k",
        "nhead_q",
        "nhead_k",
        "hdim_q",
        "hdim_v",
        "tile_m0",
        "tile_n0",
        "tile_k0",
        "tile_n1",
        "tile_k1",
        "tile_k0max",
        "pad_s",
        "pad_sk",
        "pad_d",
        "pad_dv",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    float_cols = ["latency_ms", "tflops", "bandwidth_gb_s"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df[df["tflops"] > 0].reset_index(drop=True)

    return df


def compute_group_keys(df: pd.DataFrame) -> np.ndarray:
    """Compute group keys for GroupKFold CV (one group per problem shape)."""
    keys = (
        df["batch"].astype(str)
        + "_"
        + df["seqlen_q"].astype(str)
        + "_"
        + df["seqlen_k"].astype(str)
        + "_"
        + df["nhead_q"].astype(str)
        + "_"
        + df["nhead_k"].astype(str)
        + "_"
        + df["hdim_q"].astype(str)
        + "_"
        + df["hdim_v"].astype(str)
    )
    _, groups = np.unique(keys.values, return_inverse=True)
    return groups


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns useful for analysis."""
    df = df.copy()
    df["gqa_ratio"] = df["nhead_q"] / df["nhead_k"].clip(lower=1)
    df["num_ops"] = (
        2.0
        * df["batch"]
        * df["nhead_q"]
        * df["seqlen_q"]
        * df["seqlen_k"]
        * (df["hdim_q"] + df["hdim_v"])
    )
    df["group_key"] = compute_group_keys(df)
    return df


def save_parquet(df: pd.DataFrame, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} rows to {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="FMHA CSV to parquet converter")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument(
        "--family", default="fwd", help="Filter by family (default: fwd)"
    )
    args = parser.parse_args()

    df = load_fmha_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    if args.family:
        df = df[df["family"] == args.family].reset_index(drop=True)
        print(f"Filtered to family={args.family}: {len(df)} rows")

    df = add_derived_columns(df)

    print(f"Unique shapes: {df['group_key'].nunique()}")
    print(f"Unique kernels: {df['kernel'].nunique()}")
    print(f"Dtypes: {df['dtype'].value_counts().to_dict()}")
    print(f"Pipelines: {df['pipeline'].value_counts().to_dict()}")
    print(f"TFLOPS range: {df['tflops'].min():.2f} - {df['tflops'].max():.2f}")

    save_parquet(df, args.output)


if __name__ == "__main__":
    main()
