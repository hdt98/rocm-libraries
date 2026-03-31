#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 23: ML-based FMHA kernel selection.

Loads a trained LightGBM model and ranks all candidate kernels for a
given problem shape, comparing the ML top-1 pick against the oracle-best
from benchmark data.

Usage:
    python 23_ml_heuristic.py [--model-dir <path>] [--data <parquet>]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fmha_feature_engine import FmhaFwdFeatureEngine

_THIS_DIR = Path(__file__).resolve().parent
_DISPATCHER_ROOT = _THIS_DIR.parents[2]
sys.path.insert(0, str(_DISPATCHER_ROOT / "heuristics"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=str(_DISPATCHER_ROOT / "heuristics" / "models" / "fmha_fwd_gfx950"),
    )
    parser.add_argument(
        "--data",
        default=str(
            _DISPATCHER_ROOT / "heuristics" / "data" / "fmha_fwd_gfx950.parquet"
        ),
    )
    args = parser.parse_args()

    import json
    import lightgbm as lgb

    model_path = Path(args.model_dir) / "model_tflops.lgbm"
    spec_path = Path(args.model_dir) / "feature_spec.json"

    print(f"Loading model from {model_path}")
    booster = lgb.Booster(model_file=str(model_path))
    with open(spec_path) as f:
        spec = json.load(f)
    log_transform = "tflops" in spec.get("log_targets", [])
    print(f"Model loaded: {booster.num_trees()} trees, log_transform={log_transform}")

    engine = FmhaFwdFeatureEngine()

    df = pd.read_parquet(args.data)
    print(f"Loaded {len(df)} benchmark rows")

    test_shapes = [
        {
            "batch": 1,
            "seqlen_q": 2048,
            "seqlen_k": 2048,
            "nhead_q": 32,
            "nhead_k": 8,
            "hdim_q": 128,
            "hdim_v": 128,
        },
        {
            "batch": 4,
            "seqlen_q": 2048,
            "seqlen_k": 2048,
            "nhead_q": 64,
            "nhead_k": 4,
            "hdim_q": 128,
            "hdim_v": 128,
        },
        {
            "batch": 2,
            "seqlen_q": 1024,
            "seqlen_k": 1024,
            "nhead_q": 8,
            "nhead_k": 4,
            "hdim_q": 64,
            "hdim_v": 64,
        },
        {
            "batch": 2,
            "seqlen_q": 1024,
            "seqlen_k": 1024,
            "nhead_q": 8,
            "nhead_k": 4,
            "hdim_q": 256,
            "hdim_v": 256,
        },
    ]

    print(f"\n{'Shape':<55} {'ML Top-1':>10} {'Oracle':>10} {'Eff%':>8}")
    print("-" * 88)

    for shape in test_shapes:
        mask = True
        for k, v in shape.items():
            mask &= df[k] == v
        shape_df = df[mask & (df["dtype"] == "fp16")]

        if len(shape_df) == 0:
            label = f"B={shape['batch']} Hq={shape['nhead_q']} Sq={shape['seqlen_q']} D={shape['hdim_q']}"
            print(f"{label:<55} {'N/A':>10} {'N/A':>10}")
            continue

        X = engine.extract_batch(shape_df)
        preds = booster.predict(X)
        if log_transform:
            preds = np.expm1(preds)

        best_pred_idx = preds.argmax()
        ml_tflops = shape_df.iloc[best_pred_idx]["tflops"]
        oracle_tflops = shape_df["tflops"].max()
        efficiency = ml_tflops / oracle_tflops * 100 if oracle_tflops > 0 else 0

        ml_kernel = shape_df.iloc[best_pred_idx]["kernel"]
        print(f"ML kernel: {ml_kernel}")
        label = f"B={shape['batch']} Hq={shape['nhead_q']} Hk={shape['nhead_k']} Sq={shape['seqlen_q']} D={shape['hdim_q']}"
        print(
            f"{label:<55} {ml_tflops:>10.2f} {oracle_tflops:>10.2f} {efficiency:>7.1f}%"
        )

    # Summary statistics across all shapes
    print("\n--- Per-shape efficiency summary ---")
    groups = df.groupby(
        [
            "batch",
            "seqlen_q",
            "seqlen_k",
            "nhead_q",
            "nhead_k",
            "hdim_q",
            "hdim_v",
            "dtype",
        ]
    )
    efficiencies = []
    for name, group in groups:
        if len(group) < 2:
            continue
        X = engine.extract_batch(group)
        preds = booster.predict(X)
        if log_transform:
            preds = np.expm1(preds)
        best_pred_idx = preds.argmax()
        ml_tflops = group.iloc[best_pred_idx]["tflops"]
        oracle = group["tflops"].max()
        if oracle > 0:
            efficiencies.append(ml_tflops / oracle)

    effs = np.array(efficiencies)
    print(f"Shapes evaluated: {len(effs)}")
    print(f"Mean efficiency:  {effs.mean() * 100:.2f}%")
    print(f"P10 efficiency:   {np.percentile(effs, 10) * 100:.2f}%")
    print(f"Min efficiency:   {effs.min() * 100:.2f}%")
    print(f"Max efficiency:   {effs.max() * 100:.2f}%")


if __name__ == "__main__":
    main()
