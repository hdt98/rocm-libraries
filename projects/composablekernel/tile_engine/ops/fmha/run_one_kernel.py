#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run a single FMHA kernel on GPU and report timing.

Reads JSON from stdin: {"so_path": "...", "shape": {...}, "cfg": {...}}
Prints JSON to stdout: {"ok": true, "ms": 0.123, "tflops": 456.7}
                    or: {"ok": false}

Designed to be called by fmha_full_benchmark.py as an isolated subprocess.
GPU faults in this process do NOT propagate to the parent.
"""

import json
import os
import sys

import numpy as np

for p in [os.environ.get("FMHA_PYPATH_1", ""), os.environ.get("FMHA_PYPATH_2", "")]:
    if p and p not in sys.path:
        sys.path.insert(0, p)

from fmha_utils import FmhaProblem, FmhaRunner  # noqa: E402

DTYPE_NP = {
    "fp16": np.float16,
    "bf16": np.float16,
    "fp32": np.float32,
    "fp8bf16": np.float16,
    "fp8fp32": np.float16,
}


def main():
    d = json.loads(sys.stdin.buffer.read())
    s = d["shape"]
    cfg = d["cfg"]

    prob = FmhaProblem(
        batch=s["batch"],
        nhead_q=s["nhead_q"],
        nhead_k=s["nhead_k"],
        seqlen_q=s["seqlen_q"],
        seqlen_k=s["seqlen_k"],
        hdim_q=s["hdim_q"],
        hdim_v=s["hdim_v"],
    )
    dt = DTYPE_NP.get(s.get("dtype", "fp16"), np.float16)
    np.random.seed(42)
    Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(dt)
    K = (np.random.randn(*prob.k_shape()) * 0.1).astype(dt)
    V = (np.random.randn(*prob.v_shape()) * 0.1).astype(dt)

    runner = FmhaRunner.from_library(cfg["so_path"])
    api = cfg.get("api_family", "fwd")

    if api == "bwd":
        out_buf = (
            np.random.randn(s["batch"], s["nhead_q"], s["seqlen_q"], s["hdim_v"]) * 0.1
        ).astype(dt)
        LSE = np.random.randn(s["batch"], s["nhead_q"], s["seqlen_q"]).astype(
            np.float32
        )
        dO = (np.random.randn(*out_buf.shape) * 0.1).astype(dt)
        result = runner.run_bwd(
            Q,
            K,
            V,
            out_buf,
            LSE,
            dO,
            prob,
            data_type=cfg.get("data_type", "fp16"),
            mask_type=cfg.get("mask_int", 0),
            bias_type=cfg.get("bias_int", 0),
            has_dropout=cfg.get("has_dropout", 0),
            has_dbias=cfg.get("has_dbias", 0),
            is_deterministic=cfg.get("deterministic", 0),
            is_group_mode=cfg.get("mode", "batch") == "group",
            is_store_randval=cfg.get("is_store_randval", 0),
            tile_n0=cfg.get("tile_n0", 128),
        )
    else:
        result = runner.run(
            Q,
            K,
            V,
            prob,
            mask_type=cfg.get("mask_int", 0),
            bias_type=cfg.get("bias_int", 0),
            has_lse=cfg.get("has_lse", 0),
            has_dropout=cfg.get("has_dropout", 0),
            has_logits=cfg.get("has_logits", 0),
            has_sink=cfg.get("has_sink", 0),
            has_skip=cfg.get("has_skip", 0),
            api_family=api,
            data_type=cfg.get("data_type", "fp16"),
            page_size=cfg.get("page_size", 16),
            kv_layout=cfg.get("kv_layout", 0),
            kv_lookup=cfg.get("kv_lookup", 1),
        )

    if result.success:
        print(json.dumps({"ok": True, "ms": result.time_ms, "tflops": result.tflops}))
    else:
        print(json.dumps({"ok": False}))


if __name__ == "__main__":
    main()
