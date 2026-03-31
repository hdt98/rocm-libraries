#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Training script for FMHA kernel performance prediction.

Trains LightGBM models on FMHA benchmark data to predict TFLOPS.
Uses log1p transform for scale-invariant accuracy across the wide
TFLOPS range (0.01 for decode to 480+ for large prefill).

Usage:
    # Convert CSV to parquet first
    python fmha_data_pipeline.py --input fmha_bench_all.csv --output data/fmha_fwd.parquet

    # Train
    python fmha_train.py --data data/fmha_fwd.parquet --output models/fmha_fwd_gfx950
"""

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from fmha_feature_engine import FmhaFwdFeatureEngine
from fmha_data_pipeline import compute_group_keys


TARGET_COL = "tflops"
LOG_TRANSFORM = True

DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": ["rmse", "mae"],
    "num_leaves": 255,
    "max_depth": 15,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "min_child_samples": 10,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.05,
    "reg_lambda": 0.5,
    "verbose": -1,
    "n_jobs": 16,
    "seed": 42,
}


def train_model(df: pd.DataFrame, output_dir: Path, n_folds: int = 5):
    engine = FmhaFwdFeatureEngine()
    feature_names = engine.get_feature_names()
    cat_features = engine.get_categorical_features()

    print(
        f"Extracting {len(feature_names)} features from {len(df)} rows...", flush=True
    )
    t0 = time.time()
    X = engine.extract_batch(df)
    print(f"Feature extraction: {time.time() - t0:.1f}s", flush=True)

    y = df[TARGET_COL].values.copy()
    if LOG_TRANSFORM:
        y = np.log1p(y)

    groups = compute_group_keys(df)
    n_groups = len(np.unique(groups))
    print(
        f"Groups: {n_groups}, Features: {X.shape[1]}, Target: {TARGET_COL} (log={LOG_TRANSFORM})"
    )

    # Cross-validation
    gkf = GroupKFold(n_splits=min(n_folds, n_groups))
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        cat_indices = [
            feature_names.index(c) for c in cat_features if c in feature_names
        ]

        model = lgb.LGBMRegressor(**DEFAULT_PARAMS)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(500)],
            categorical_feature=cat_indices,
        )

        preds = model.predict(X_val)
        if LOG_TRANSFORM:
            y_real = np.expm1(y_val)
            p_real = np.expm1(preds)
        else:
            y_real = y_val
            p_real = preds

        # Per-group (shape) efficiency: best_predicted / best_actual
        val_groups = groups[val_idx]
        efficiencies = []
        for g in np.unique(val_groups):
            mask = val_groups == g
            best_actual = y_real[mask].max()
            best_pred_idx = p_real[mask].argmax()
            selected_actual = y_real[mask][best_pred_idx]
            if best_actual > 0:
                efficiencies.append(selected_actual / best_actual)

        mean_eff = np.mean(efficiencies) * 100
        p10_eff = np.percentile(efficiencies, 10) * 100
        min_eff = np.min(efficiencies) * 100
        cv_scores.append(mean_eff)
        print(
            f"  Fold {fold}: mean_eff={mean_eff:.2f}%, P10={p10_eff:.2f}%, min={min_eff:.2f}%"
        )

    print(
        f"\nCV mean efficiency: {np.mean(cv_scores):.2f}% (+/- {np.std(cv_scores):.2f}%)"
    )

    # Train final model on all data
    print("\nTraining final model on all data...")
    cat_indices = [feature_names.index(c) for c in cat_features if c in feature_names]
    final_model = lgb.LGBMRegressor(**DEFAULT_PARAMS)
    final_model.fit(
        X,
        y,
        categorical_feature=cat_indices,
        callbacks=[lgb.log_evaluation(1000)],
    )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model_tflops.lgbm"
    final_model.booster_.save_model(str(model_path))
    print(f"Saved model to {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")

    spec = {
        "op": "fmha_fwd",
        "feature_names": feature_names,
        "categorical_features": cat_features,
        "num_features": len(feature_names),
        "log_targets": ["tflops"] if LOG_TRANSFORM else [],
        "hw_profile": engine._hw,
    }
    spec_path = output_dir / "feature_spec.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    manifest = {
        "op": "fmha_fwd",
        "arch": "gfx950",
        "n_samples": len(df),
        "n_groups": int(n_groups),
        "n_features": len(feature_names),
        "cv_mean_efficiency": float(np.mean(cv_scores)),
        "cv_std_efficiency": float(np.std(cv_scores)),
        "n_estimators": final_model.n_estimators_,
        "log_transform": LOG_TRANSFORM,
    }
    manifest_path = output_dir / "train_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved: {model_path}, {spec_path}, {manifest_path}")
    return final_model


def main():
    parser = argparse.ArgumentParser(description="Train FMHA performance model")
    parser.add_argument("--data", required=True, help="Training parquet file")
    parser.add_argument("--output", required=True, help="Output model directory")
    parser.add_argument("--folds", type=int, default=5, help="CV folds")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")

    train_model(df, Path(args.output), n_folds=args.folds)


if __name__ == "__main__":
    main()
