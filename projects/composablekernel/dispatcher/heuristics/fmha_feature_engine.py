#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
FMHA-specific feature engineering for kernel performance prediction.

Extracts ~68 features from (problem_shape, kernel_config) pairs for
LightGBM-based TFLOPS prediction and kernel ranking.
"""

import math

import numpy as np
import pandas as pd

from feature_engine import FeatureEngine, DTYPE_BYTES


FMHA_PIPELINE_MAP = {
    "qr": 0,
    "qr_async": 1,
    "qr_async_trload": 2,
    "qr_async_trload_v3": 3,
    "qr_pagedkv": 4,
    "qr_nwarp_sshuffle": 5,
    "appendkv": 6,
}

FMHA_MASK_MAP = {"no": 0, "top_left": 1, "bottom_right": 2, "generic": 3}
FMHA_BIAS_MAP = {"no": 0, "elementwise": 1, "alibi": 2}
FMHA_DTYPE_MAP = {"fp16": 0, "bf16": 1, "fp8bf16": 2, "fp8fp32": 3}


class FmhaFwdFeatureEngine(FeatureEngine):
    """Feature engine for FMHA forward kernels."""

    def __init__(
        self,
        num_cus: int = 304,
        lds_capacity: int = 65536,
        max_clock_mhz: int = 2400,
        simds_per_cu: int = 4,
        shader_engines: int = 32,
        wavefront_size: int = 64,
        num_xcd: int = 8,
    ):
        self._hw = {
            "num_cus": num_cus,
            "lds_capacity": lds_capacity,
            "max_clock_mhz": max_clock_mhz,
            "simds_per_cu": simds_per_cu,
            "shader_engines": shader_engines,
            "wavefront_size": wavefront_size,
            "num_xcd": num_xcd,
            "total_simds": num_cus * simds_per_cu,
        }

    def get_feature_names(self) -> list[str]:
        return [
            # Problem features (20)
            "batch",
            "seqlen_q",
            "seqlen_k",
            "nhead_q",
            "nhead_k",
            "hdim_q",
            "hdim_v",
            "dtype",
            "log2_batch",
            "log2_sq",
            "log2_sk",
            "log2_hq",
            "log2_hk",
            "log2_dq",
            "log2_dv",
            "gqa_ratio",
            "aspect_ratio_sq_sk",
            "num_ops_log2",
            "arithmetic_intensity",
            "is_decode",
            # Kernel features (20)
            "pipeline",
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
            "mask",
            "bias",
            "lse",
            "dropout",
            "logits",
            "sink",
            "skip",
            "qscale",
            "paged_kv",
            # Interaction features (20)
            "num_tiles_sq",
            "num_tiles_sk",
            "total_output_tiles",
            "tile_eff_sq",
            "tile_eff_sk",
            "overall_tile_efficiency",
            "cu_utilization",
            "tile_volume",
            "tile_area_m0_n0",
            "lds_estimate",
            "lds_ratio",
            "ratio_dq_to_k0",
            "ratio_dv_to_n1",
            "sq_fits_single_tile",
            "sk_fits_single_tile",
            "hdim_q_eq_hdim_v",
            "is_gqa",
            "total_q_elements",
            "total_kv_elements",
            "feature_count",
            # Hardware features (8)
            "hw_num_cus",
            "hw_simds_per_cu",
            "hw_total_simds",
            "hw_shader_engines",
            "hw_max_clock_mhz",
            "hw_wavefront_size",
            "hw_lds_capacity",
            "hw_num_xcd",
        ]

    def get_categorical_features(self) -> list[str]:
        return ["dtype", "pipeline", "mask", "bias", "qscale"]

    def extract(self, problem: dict, kernel: dict) -> np.ndarray:
        def dict_float(d: dict, *keys: str) -> float:
            return next((float(d[k]) for k in keys if k in d), 0.0)

        def dict_int(d: dict, *keys: str) -> int:
            return next((int(d[k]) for k in keys if k in d), 0)

        def log2_sc(x: float) -> float:
            return math.log2(max(x, 1))

        batch = dict_int(problem, "batch")
        sq = dict_int(problem, "seqlen_q")
        sk = dict_int(problem, "seqlen_k")
        hq = dict_int(problem, "nhead_q")
        hk = dict_int(problem, "nhead_k")
        dq = dict_int(problem, "hdim_q")
        dv = dict_int(problem, "hdim_v")
        dtype_str = str(problem.get("dtype", "fp16"))
        dtype_enc = FMHA_DTYPE_MAP.get(dtype_str, 0)
        bpe = DTYPE_BYTES.get(dtype_str, 2.0)

        gqa = hq / max(hk, 1)
        aspect_sq_sk = sq / max(sk, 1)
        num_ops = 2.0 * batch * hq * sq * sk * (dq + dv)
        mem_bytes = (
            batch * hq * sq * dq
            + batch * hk * sk * dq
            + batch * hk * sk * dv
            + batch * hq * sq * dv
        ) * bpe
        ai = num_ops / max(mem_bytes, 1.0)
        is_decode = 1.0 if sq <= 1 else 0.0

        pip_str = str(kernel.get("pipeline", "qr_async"))
        pip = FMHA_PIPELINE_MAP.get(pip_str, 1)
        tm0 = dict_float(kernel, "tile_m0")
        tn0 = dict_float(kernel, "tile_n0")
        tk0 = dict_float(kernel, "tile_k0")
        tn1 = dict_float(kernel, "tile_n1")
        tk1 = dict_float(kernel, "tile_k1")
        tk0max = dict_float(kernel, "tile_k0max")
        ps = dict_float(kernel, "pad_s")
        psk = dict_float(kernel, "pad_sk")
        pd_ = dict_float(kernel, "pad_d")
        pdv = dict_float(kernel, "pad_dv")

        mask_str = str(kernel.get("mask", "no"))
        mask_enc = FMHA_MASK_MAP.get(mask_str, 0)
        bias_str = str(kernel.get("bias", "no"))
        bias_enc = FMHA_BIAS_MAP.get(bias_str, 0)

        def bval(d, k):
            v = d.get(k, False)
            return (
                1.0 if v and str(v).lower() not in ("false", "0", "no", "none") else 0.0
            )

        lse = bval(kernel, "lse")
        dropout = bval(kernel, "dropout")
        logits = bval(kernel, "logits")
        sink = bval(kernel, "sink")
        skip = bval(kernel, "skip")
        qscale_str = str(kernel.get("qscale", "no"))
        qscale_enc = 1.0 if qscale_str not in ("no", "False", "0") else 0.0
        paged_kv = bval(kernel, "paged_kv")

        # Interaction features
        ntm = math.ceil(sq / max(tm0, 1))
        ntk = math.ceil(sk / max(tn0, 1))
        total_tiles = batch * hq * ntm * ntk

        def tile_eff(dim, tile):
            if tile <= 0:
                return 1.0
            r = dim % tile
            return r / tile if r > 0 else 1.0

        eff_sq = tile_eff(sq, tm0)
        eff_sk = tile_eff(sk, tn0)
        overall_eff = eff_sq * eff_sk
        cu_util = total_tiles / max(self._hw["num_cus"], 1)
        tile_vol = tm0 * tn0 * tk0
        tile_area = tm0 * tn0
        lds_est = (tm0 * tk0 + tn0 * tk0) * bpe
        lds_ratio = lds_est / max(self._hw["lds_capacity"], 1)
        ratio_dq_k0 = dq / max(tk0, 1)
        ratio_dv_n1 = dv / max(tn1, 1) if tn1 > 0 else 0.0
        sq_single = 1.0 if sq <= tm0 else 0.0
        sk_single = 1.0 if sk <= tn0 else 0.0
        dq_eq_dv = 1.0 if dq == dv else 0.0
        is_gqa = 1.0 if hq != hk else 0.0
        total_q = batch * hq * sq * dq
        total_kv = batch * hk * sk * (dq + dv)
        feat_count = float(
            lse
            + dropout
            + logits
            + sink
            + skip
            + paged_kv
            + (1.0 if mask_enc > 0 else 0.0)
            + (1.0 if bias_enc > 0 else 0.0)
        )

        hw = self._hw
        return np.array(
            [
                batch,
                sq,
                sk,
                hq,
                hk,
                dq,
                dv,
                dtype_enc,
                log2_sc(batch),
                log2_sc(sq),
                log2_sc(sk),
                log2_sc(hq),
                log2_sc(hk),
                log2_sc(dq),
                log2_sc(dv),
                gqa,
                aspect_sq_sk,
                log2_sc(num_ops),
                ai,
                is_decode,
                pip,
                tm0,
                tn0,
                tk0,
                tn1,
                tk1,
                tk0max,
                ps,
                psk,
                pd_,
                pdv,
                mask_enc,
                bias_enc,
                lse,
                dropout,
                logits,
                sink,
                skip,
                qscale_enc,
                paged_kv,
                ntm,
                ntk,
                total_tiles,
                eff_sq,
                eff_sk,
                overall_eff,
                cu_util,
                tile_vol,
                tile_area,
                lds_est,
                lds_ratio,
                ratio_dq_k0,
                ratio_dv_n1,
                sq_single,
                sk_single,
                dq_eq_dv,
                is_gqa,
                total_q,
                total_kv,
                feat_count,
                hw["num_cus"],
                hw["simds_per_cu"],
                hw["total_simds"],
                hw["shader_engines"],
                hw["max_clock_mhz"],
                hw["wavefront_size"],
                hw["lds_capacity"],
                hw["num_xcd"],
            ],
            dtype=np.float64,
        )

    def extract_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Fully vectorized batch extraction -- handles 190K rows in seconds."""
        n = len(df)
        hw = self._hw

        batch = df["batch"].values.astype(np.float64)
        sq = df["seqlen_q"].values.astype(np.float64)
        sk = df["seqlen_k"].values.astype(np.float64)
        hq = df["nhead_q"].values.astype(np.float64)
        hk = np.maximum(df["nhead_k"].values.astype(np.float64), 1.0)
        dq = df["hdim_q"].values.astype(np.float64)
        dv = df["hdim_v"].values.astype(np.float64)

        dtype_enc = df["dtype"].map(FMHA_DTYPE_MAP).fillna(0).values.astype(np.float64)
        bpe = df["dtype"].map(DTYPE_BYTES).fillna(2.0).values.astype(np.float64)

        def log2_arr(x: np.ndarray) -> np.ndarray:
            return np.log2(np.maximum(x, 1.0))

        gqa = hq / hk
        aspect = sq / np.maximum(sk, 1.0)
        num_ops = 2.0 * batch * hq * sq * sk * (dq + dv)
        mem = (
            batch * hq * sq * dq
            + batch * hk * sk * dq
            + batch * hk * sk * dv
            + batch * hq * sq * dv
        ) * bpe
        ai = num_ops / np.maximum(mem, 1.0)
        is_decode = (sq <= 1).astype(np.float64)

        pip = df["pipeline"].map(FMHA_PIPELINE_MAP).fillna(1).values.astype(np.float64)
        tm0 = df["tile_m0"].values.astype(np.float64)
        tn0 = df["tile_n0"].values.astype(np.float64)
        tk0 = df["tile_k0"].values.astype(np.float64)
        tn1 = df["tile_n1"].values.astype(np.float64)
        tk1 = df["tile_k1"].values.astype(np.float64)
        tk0max = df["tile_k0max"].values.astype(np.float64)
        ps = df["pad_s"].values.astype(np.float64)
        psk = df["pad_sk"].values.astype(np.float64)
        pd_ = df["pad_d"].values.astype(np.float64)
        pdv = df["pad_dv"].values.astype(np.float64)

        mask_enc = df["mask"].map(FMHA_MASK_MAP).fillna(0).values.astype(np.float64)
        bias_enc = df["bias"].map(FMHA_BIAS_MAP).fillna(0).values.astype(np.float64)

        def boolcol(name):
            if name not in df.columns:
                return np.zeros(n, dtype=np.float64)
            return (
                df[name]
                .apply(lambda v: 1.0 if str(v).lower() in ("true", "1", "yes") else 0.0)
                .values
            )

        lse = boolcol("lse")
        dropout = boolcol("dropout")
        logits = boolcol("logits")
        sink = boolcol("sink")
        skip = boolcol("skip")
        qscale_enc = (
            df["qscale"]
            .apply(lambda v: 0.0 if str(v) in ("no", "False", "0") else 1.0)
            .values
            if "qscale" in df.columns
            else np.zeros(n)
        )
        paged_kv = boolcol("paged_kv")

        ntm = np.ceil(sq / np.maximum(tm0, 1.0))
        ntk = np.ceil(sk / np.maximum(tn0, 1.0))
        total_tiles = batch * hq * ntm * ntk

        def tile_eff(dim, tile):
            t = np.maximum(tile, 1.0)
            r = np.fmod(dim, t)
            return np.where(r > 0, r / t, 1.0)

        eff_sq = tile_eff(sq, tm0)
        eff_sk = tile_eff(sk, tn0)
        overall_eff = eff_sq * eff_sk
        cu_util = total_tiles / hw["num_cus"]
        tile_vol = tm0 * tn0 * tk0
        tile_area = tm0 * tn0
        lds_est = (tm0 * tk0 + tn0 * tk0) * bpe
        lds_ratio = lds_est / hw["lds_capacity"]
        ratio_dq_k0 = dq / np.maximum(tk0, 1.0)
        ratio_dv_n1 = np.where(tn1 > 0, dv / tn1, 0.0)
        sq_single = (sq <= tm0).astype(np.float64)
        sk_single = (sk <= tn0).astype(np.float64)
        dq_eq_dv = (dq == dv).astype(np.float64)
        is_gqa = (hq != hk).astype(np.float64)
        total_q = batch * hq * sq * dq
        total_kv = batch * hk * sk * (dq + dv)
        feat_count = (
            lse
            + dropout
            + logits
            + sink
            + skip
            + paged_kv
            + (mask_enc > 0).astype(np.float64)
            + (bias_enc > 0).astype(np.float64)
        )

        result = np.column_stack(
            [
                batch,
                sq,
                sk,
                hq,
                hk,
                dq,
                dv,
                dtype_enc,
                log2_arr(batch),
                log2_arr(sq),
                log2_arr(sk),
                log2_arr(hq),
                log2_arr(hk),
                log2_arr(dq),
                log2_arr(dv),
                gqa,
                aspect,
                log2_arr(num_ops),
                ai,
                is_decode,
                pip,
                tm0,
                tn0,
                tk0,
                tn1,
                tk1,
                tk0max,
                ps,
                psk,
                pd_,
                pdv,
                mask_enc,
                bias_enc,
                lse,
                dropout,
                logits,
                sink,
                skip,
                qscale_enc,
                paged_kv,
                ntm,
                ntk,
                total_tiles,
                eff_sq,
                eff_sk,
                overall_eff,
                cu_util,
                tile_vol,
                tile_area,
                lds_est,
                lds_ratio,
                ratio_dq_k0,
                ratio_dv_n1,
                sq_single,
                sk_single,
                dq_eq_dv,
                is_gqa,
                total_q,
                total_kv,
                feat_count,
                np.full(n, hw["num_cus"]),
                np.full(n, hw["simds_per_cu"]),
                np.full(n, hw["total_simds"]),
                np.full(n, hw["shader_engines"]),
                np.full(n, hw["max_clock_mhz"]),
                np.full(n, hw["wavefront_size"]),
                np.full(n, hw["lds_capacity"]),
                np.full(n, hw["num_xcd"]),
            ]
        )
        return result

    def get_parameter_space(self) -> dict[str, list]:
        return {
            "pipeline": list(FMHA_PIPELINE_MAP.values()),
            "tile_m0": [32, 64, 128, 256],
            "tile_n0": [64, 128, 256],
            "tile_k0": [16, 32, 64, 128],
            "mask": list(FMHA_MASK_MAP.values()),
            "bias": list(FMHA_BIAS_MAP.values()),
        }
