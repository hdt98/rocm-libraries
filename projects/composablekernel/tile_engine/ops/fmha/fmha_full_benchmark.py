#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Full FMHA benchmark sweep.

JIT-compiles FMHA kernels, then for EACH test shape finds all matching
kernels and benchmarks them, streaming results incrementally to CSV/JSON.

Results are printed live per-shape with the best kernel highlighted.
TFLOPS and latency come directly from CK's HIP event timing.

Usage:
    # Full sweep
    python fmha_full_benchmark.py --workers 256

    # Quick end-to-end test
    python fmha_full_benchmark.py --category smoke --variant fwd --max-kernels 10 --workers 4

    # Filter to h128 fp16
    python fmha_full_benchmark.py --filter "c.hdim_q == 128 and c.data_type == 'fp16'"
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_DISPATCHER_ROOT = _THIS_DIR.parents[2] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))
sys.path.insert(0, str(_DISPATCHER_ROOT / "codegen"))
sys.path.insert(0, str(_THIS_DIR))

from fmha_utils import (  # noqa: E402
    detect_gpu_arch,
    setup_multiple_fmha_dispatchers,
)
from fmha_instance_builder import expand_sweep, apply_filter  # noqa: E402

YAML_PATH = _THIS_DIR / "ck_fmha_testing_matrix.yaml"

VARIANT_CONFIGS = {
    "fwd": "configs/receipt0_fwd.json",
    "splitkv": "configs/splitkv.json",
    "pagedkv": "configs/pagedkv.json",
    "appendkv": "configs/appendkv.json",
    "batch_prefill": "configs/batch_prefill.json",
    "bwd": "configs/bwd.json",
}

# Variant -> YAML section mapping. KV-cache variants use forward_tests shapes.
VARIANT_YAML_SECTIONS = {
    "fwd": ["forward_tests"],
    "splitkv": ["forward_tests"],
    "pagedkv": ["forward_tests"],
    "appendkv": ["forward_tests"],
    "batch_prefill": ["forward_tests"],
    "bwd": ["backward_tests"],
}

DTYPE_CK = {"fp16": "fp16", "bf16": "bf16", "fp8bf16": "fp8bf16", "fp8fp32": "fp8fp32"}
DTYPE_NP = {
    "fp16": np.float16,
    "bf16": np.float16,
    "fp32": np.float32,
    "fp8bf16": np.float16,
    "fp8fp32": np.float16,
}
ELEM_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "fp8bf16": 1, "fp8fp32": 1}

MASK_INT = {"no": 0, "top_left": 1, "generic": 3}
BIAS_INT = {"no": 0, "bias": 1, "alibi": 2}
KV_LAYOUT_INT = {"vectorized": 0, "linear": 1}
KV_LOOKUP_INT = {"vllm": 0, "sglang": 1}


@dataclass
class TestShape:
    name: str
    category: str
    variant: str
    batch: int
    seqlen_q: int
    seqlen_k: int
    nhead_q: int
    nhead_k: int
    hdim_q: int
    hdim_v: int
    dtype: str
    mask: str = "no_mask"
    bias: str = "none"
    dropout: float = 0.0
    lse: bool = False


def parse_yaml(
    yaml_path: Path, category: str = "smoke", sections: Optional[List[str]] = None
) -> List[TestShape]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    shapes = []
    cats = ["smoke"]
    if category in ("full", "nightly"):
        cats.append("full")
    if category == "nightly":
        cats.append("nightly")

    section_variant_map = [("forward_tests", "fwd"), ("backward_tests", "bwd")]
    if sections:
        section_variant_map = [(s, v) for s, v in section_variant_map if s in sections]

    for section, variant in section_variant_map:
        if section not in data:
            continue
        for cat in cats:
            for test in data[section].get(cat, []):
                for combo in itertools.product(
                    test.get("batch", [1]),
                    test.get("seqlen_q", [1024]),
                    test.get("seqlen_k", [1024]),
                    test.get("nhead_q", [16]),
                    test.get("nhead_k", [16]),
                    test.get("hdim_q", [128]),
                    test.get("hdim_v", [128]),
                    test.get("dtype", ["fp16"]),
                    test.get("mask", ["no_mask"]),
                    test.get("bias", ["none"]),
                    test.get("dropout", [0.0]),
                    test.get("lse", [False]),
                ):
                    b, sq, sk, hq, hk, dq, dv, dt, m, bi, dr, ls = combo
                    shapes.append(
                        TestShape(
                            test["name"],
                            cat,
                            variant,
                            b,
                            sq,
                            sk,
                            hq,
                            hk,
                            dq,
                            dv,
                            dt,
                            mask=m,
                            bias=bi,
                            dropout=dr,
                            lse=ls,
                        )
                    )
    return shapes


def bandwidth_gb_s(shape: TestShape, latency_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    eb = ELEM_BYTES.get(shape.dtype, 2)
    total = (
        shape.batch
        * (
            shape.nhead_q * shape.seqlen_q * shape.hdim_q
            + shape.nhead_k * shape.seqlen_k * shape.hdim_q
            + shape.nhead_k * shape.seqlen_k * shape.hdim_v
            + shape.nhead_q * shape.seqlen_q * shape.hdim_v
        )
        * eb
    )
    return total / (latency_ms * 1e6)


# ---------------------------------------------------------------------------
# Subprocess worker code: runs all kernels for ONE shape in a separate process.
# Reads JSON from stdin, writes JSON result rows to stdout.
# If a GPU fault kills this process, the parent survives and moves on.
# ---------------------------------------------------------------------------

_WORKER_CODE = r"""
import json, sys, os, numpy as np
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent if "__file__" in dir() else Path(".")
_DISPATCHER_ROOT = Path(os.environ.get("FMHA_DISPATCHER_ROOT",
    str(Path(__file__).resolve().parents[2] / "dispatcher") if "__file__" in dir() else ""))

# Paths are passed via env or inferred
for p in [os.environ.get("FMHA_PYPATH_1", ""), os.environ.get("FMHA_PYPATH_2", "")]:
    if p and p not in sys.path:
        sys.path.insert(0, p)

from fmha_utils import FmhaRunner, FmhaProblem

DTYPE_NP = {"fp16": np.float16, "bf16": np.float16, "fp32": np.float32,
            "fp8bf16": np.float16, "fp8fp32": np.float16}
ELEM_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "fp8bf16": 1, "fp8fp32": 1}

def bandwidth_gb_s(s, lat):
    if lat <= 0: return 0.0
    eb = ELEM_BYTES.get(s["dtype"], 2)
    total = s["batch"] * (
        s["nhead_q"]*s["seqlen_q"]*s["hdim_q"] + s["nhead_k"]*s["seqlen_k"]*s["hdim_q"] +
        s["nhead_k"]*s["seqlen_k"]*s["hdim_v"] + s["nhead_q"]*s["seqlen_q"]*s["hdim_v"]
    ) * eb
    return total / (lat * 1e6)

data = json.loads(sys.stdin.read())
s = data["shape"]
kernels = data["kernels"]

prob = FmhaProblem(batch=s["batch"], nhead_q=s["nhead_q"], nhead_k=s["nhead_k"],
                   seqlen_q=s["seqlen_q"], seqlen_k=s["seqlen_k"],
                   hdim_q=s["hdim_q"], hdim_v=s["hdim_v"])
np_dt = DTYPE_NP.get(s["dtype"], np.float16)
np.random.seed(42)
Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np_dt)
K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np_dt)
V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np_dt)

out_dt = np_dt
O = (np.random.randn(*prob.q_shape()[:3] + (s["hdim_v"],)) * 0.1).astype(out_dt)
LSE = np.random.randn(s["batch"], s["nhead_q"], s["seqlen_q"]).astype(np.float32)
dO = (np.random.randn(*O.shape) * 0.1).astype(out_dt)

rows = []
for so_path, cfg in kernels:
    try:
        runner = FmhaRunner.from_library(so_path)
        api = cfg.get("api_family", "fwd")
        if api == "bwd":
            is_grp = cfg.get("mode", "batch") == "group"
            result = runner.run_bwd(Q, K, V, O, LSE, dO, prob,
                data_type=cfg.get("data_type", "fp16"),
                mask_type=cfg["mask_int"], bias_type=cfg["bias_int"],
                has_dropout=cfg["has_dropout"],
                has_dbias=cfg.get("has_dbias", 0),
                is_deterministic=cfg.get("deterministic", 0),
                is_group_mode=is_grp,
                is_store_randval=cfg.get("is_store_randval", 0),
                tile_n0=cfg.get("tile_n0", 128))
        else:
            result = runner.run(Q, K, V, prob,
                mask_type=cfg["mask_int"], bias_type=cfg["bias_int"],
                has_lse=cfg["has_lse"], has_dropout=cfg["has_dropout"],
                has_logits=cfg["has_logits"], has_sink=cfg["has_sink"],
                has_skip=cfg["has_skip"],
                api_family=api,
                data_type=cfg.get("data_type", "fp16"),
                page_size=cfg.get("page_size", 16),
                kv_layout=cfg.get("kv_layout", 0),
                kv_lookup=cfg.get("kv_lookup", 1))
    except Exception as exc:
        print(f"  WARN: kernel {cfg.get('name','?')} exception: {exc}", file=sys.stderr)
        continue
    if not result.success:
        continue
    bw = bandwidth_gb_s(s, result.time_ms)
    row = {
        "problem_name": s["name"], "batch": s["batch"],
        "seqlen_q": s["seqlen_q"], "seqlen_k": s["seqlen_k"],
        "nhead_q": s["nhead_q"], "nhead_k": s["nhead_k"],
        "hdim_q": s["hdim_q"], "hdim_v": s["hdim_v"], "dtype": s["dtype"],
    }
    for k in ["kernel","family","mode","pipeline",
              "tile_m0","tile_n0","tile_k0","tile_n1","tile_k1","tile_k0max",
              "pad_s","pad_sk","pad_d","pad_dv",
              "mask","bias","lse","dropout","logits","sink","skip",
              "qscale","paged_kv","rope","deterministic","dbias"]:
        row[k] = cfg[k]
    row["latency_ms"] = round(result.time_ms, 4)
    row["tflops"] = round(result.tflops, 2)
    row["bandwidth_gb_s"] = round(bw, 2)
    rows.append(row)

print(json.dumps(rows))
"""


FAMILY_TO_API = {
    "fwd": "fwd",
    "fwd_splitkv": "splitkv",
    "fwd_splitkv_combine": "splitkv",
    "fwd_pagedkv": "pagedkv",
    "fwd_appendkv": "appendkv",
    "batch_prefill": "batch_prefill",
    "bwd_dot_do_o": "bwd",
    "bwd_dq_dk_dv": "bwd",
    "bwd_convert_dq": "bwd",
}


def _config_to_serializable(config, so_path: str) -> dict:
    """Convert FmhaKernelConfig + so_path to a picklable dict for subprocess."""
    return {
        "so_path": so_path,
        "api_family": FAMILY_TO_API.get(config.family, "fwd"),
        "data_type": config.data_type,
        "kernel": config.name,
        "family": config.family,
        "mode": config.mode,
        "pipeline": config.pipeline,
        "tile_m0": config.tile_m0,
        "tile_n0": config.tile_n0,
        "tile_k0": config.tile_k0,
        "tile_n1": config.tile_n1,
        "tile_k1": config.tile_k1,
        "tile_k0max": config.tile_k0max,
        "pad_s": config.pad_s,
        "pad_sk": config.pad_sk,
        "pad_d": config.pad_d,
        "pad_dv": config.pad_dv,
        "mask": config.mask,
        "bias": config.bias,
        "lse": config.lse,
        "dropout": config.dropout,
        "logits": config.logits,
        "sink": config.sink,
        "skip": config.skip_min_seqlen_q,
        "qscale": config.qscale,
        "paged_kv": config.paged_kv,
        "rope": config.rope,
        "deterministic": config.deterministic,
        "dbias": config.dbias,
        "mask_int": MASK_INT.get(config.mask, 0),
        "bias_int": BIAS_INT.get(config.bias, 0),
        "has_lse": int(config.lse),
        "has_dropout": int(config.dropout not in (False, 0, "no", "False")),
        "has_logits": int(config.logits),
        "has_sink": int(config.sink),
        "has_skip": int(config.skip_min_seqlen_q),
        "has_dbias": int(getattr(config, "dbias", False)),
        "is_store_randval": int(getattr(config, "store_randval", False)),
        "page_size": getattr(config, "page_size", 16),
        "kv_layout": KV_LAYOUT_INT.get(
            getattr(config, "kv_memory_layout", "vectorized"), 0
        ),
        "kv_lookup": KV_LOOKUP_INT.get(getattr(config, "kv_lookup_table", "sglang"), 1),
    }


def _shape_to_dict(shape: TestShape) -> dict:
    return {
        "name": shape.name,
        "category": shape.category,
        "variant": shape.variant,
        "batch": shape.batch,
        "seqlen_q": shape.seqlen_q,
        "seqlen_k": shape.seqlen_k,
        "nhead_q": shape.nhead_q,
        "nhead_k": shape.nhead_k,
        "hdim_q": shape.hdim_q,
        "hdim_v": shape.hdim_v,
        "dtype": shape.dtype,
        "mask": shape.mask,
        "bias": shape.bias,
        "dropout": shape.dropout,
        "lse": shape.lse,
    }


def main():
    p = argparse.ArgumentParser(description="Full FMHA Benchmark Sweep")
    p.add_argument("--arch", default=detect_gpu_arch())
    p.add_argument("--category", default="smoke", choices=["smoke", "full", "nightly"])
    p.add_argument("--variant", default="all")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--build-dir", default="/tmp/fmha_full_bench")
    p.add_argument("--filter", dest="filter_expr", default="")
    p.add_argument("--filter-file", default="")
    p.add_argument("--csv", default="fmha_sweep_results.csv")
    p.add_argument("--json", default="fmha_sweep_results.json")
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--max-kernels", type=int, default=0)
    p.add_argument(
        "--shape-timeout",
        type=int,
        default=600,
        help="Per-shape timeout in seconds (0=none)",
    )
    args = p.parse_args()

    build_dir = Path(args.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    variants = list(VARIANT_CONFIGS.keys()) if args.variant == "all" else [args.variant]

    # ---- Phase 1: Parse shapes ----
    print(f"\n{'=' * 80}")
    print("Phase 1: Parse test shapes")
    print(f"{'=' * 80}")

    all_shapes: List[TestShape] = []
    for variant in variants:
        sections = VARIANT_YAML_SECTIONS.get(variant, ["forward_tests"])
        vshapes = parse_yaml(YAML_PATH, args.category, sections=sections)
        for s in vshapes:
            s.variant = variant
        all_shapes.extend(vshapes)

    print(f"  Category: {args.category}")
    print(f"  Variants: {variants}")
    print(f"  Total shapes: {len(all_shapes)}")

    # ---- Phase 2: Compile ----
    print(f"\n{'=' * 80}")
    print("Phase 2: Compile kernels")
    print(f"{'=' * 80}")

    # kernel_index: (hdim_q, hdim_v, dtype, variant) -> list of (so_path, cfg_dict)
    kernel_index: Dict[tuple, List[tuple]] = {}

    for variant in variants:
        cfg_path = str(_THIS_DIR / VARIANT_CONFIGS[variant])
        if not Path(cfg_path).exists():
            continue
        configs = expand_sweep(cfg_path, args.arch)
        if args.filter_expr or args.filter_file:
            configs = apply_filter(configs, args.filter_expr, args.filter_file)
        if args.max_kernels > 0:
            configs = configs[: args.max_kernels]
        if not configs:
            continue

        print(f"\n  {variant}: {len(configs)} configs, {args.workers} workers...")
        t0 = time.perf_counter()
        setups = setup_multiple_fmha_dispatchers(
            configs, output_dir=build_dir, max_workers=args.workers
        )
        ok = sum(1 for s in setups if s.success)
        print(f"    Built {ok}/{len(configs)} in {time.perf_counter() - t0:.0f}s")

        for config, setup in zip(configs, setups):
            if not setup.success or setup.runner is None:
                continue
            so_path = getattr(setup, "library_path", "") or ""
            if not so_path:
                candidate = build_dir / f"libdispatcher_fmha_{config.name}.so"
                if candidate.exists():
                    so_path = str(candidate)
            if not so_path:
                continue
            key = (config.hdim_q, config.hdim_v, config.data_type, variant)
            cfg_dict = _config_to_serializable(config, so_path)
            kernel_index.setdefault(key, []).append((so_path, cfg_dict))

    total_built = sum(len(v) for v in kernel_index.values())
    print(f"\n  Total compiled: {total_built}")
    print(f"  Unique (hdim,dtype,variant) groups: {len(kernel_index)}")

    if args.compile_only:
        print(f"\n  Compile-only. {total_built} kernels ready.")
        return

    # ---- Phase 3: Shape-first benchmark sweep (subprocess-isolated) ----
    print(f"\n{'=' * 80}")
    print("Phase 3: Benchmark sweep (subprocess-isolated, shape-first)")
    print(f"{'=' * 80}")

    csv_path = Path(args.csv) if os.path.isabs(args.csv) else _THIS_DIR / args.csv
    csv_file = open(csv_path, "w", newline="")
    csv_fields = [
        "problem_name",
        "batch",
        "seqlen_q",
        "seqlen_k",
        "nhead_q",
        "nhead_k",
        "hdim_q",
        "hdim_v",
        "dtype",
        "kernel",
        "family",
        "mode",
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
        "rope",
        "deterministic",
        "dbias",
        "latency_ms",
        "tflops",
        "bandwidth_gb_s",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    json_results = []
    total_measurements = 0
    total_shapes_run = 0
    total_gpu_faults = 0
    bench_t0 = time.perf_counter()

    print(f"  Shapes to run: {len(all_shapes)}")
    print(f"  Shape timeout: {args.shape_timeout}s")
    print()

    for si, shape in enumerate(all_shapes):
        ck_dtype = DTYPE_CK.get(shape.dtype, shape.dtype)
        key = (shape.hdim_q, shape.hdim_v, ck_dtype, shape.variant)
        kernel_entries = kernel_index.get(key, [])
        if not kernel_entries:
            continue

        shape_dict = _shape_to_dict(shape)

        # Run in isolated subprocess via subprocess.run + json IPC.
        # This gives full process isolation: GPU faults kill the child, not us.
        worker_input = json.dumps(
            {
                "shape": shape_dict,
                "kernels": kernel_entries,
                "timeout": args.shape_timeout,
            }
        )
        worker_env = os.environ.copy()
        worker_env["FMHA_PYPATH_1"] = str(_DISPATCHER_ROOT / "python")
        worker_env["FMHA_PYPATH_2"] = str(_DISPATCHER_ROOT / "codegen")
        try:
            proc_result = subprocess.run(
                [sys.executable, "-c", _WORKER_CODE],
                input=worker_input,
                capture_output=True,
                text=True,
                env=worker_env,
                timeout=args.shape_timeout + 30 if args.shape_timeout > 0 else None,
            )
        except subprocess.TimeoutExpired:
            total_gpu_faults += 1
            print(
                f"  [{si + 1}/{len(all_shapes)}] {shape.name} B={shape.batch} S={shape.seqlen_q} "
                f"H={shape.hdim_q} {shape.dtype} {shape.variant} -> TIMEOUT",
                flush=True,
            )
            continue

        if proc_result.returncode != 0:
            total_gpu_faults += 1
            print(
                f"  [{si + 1}/{len(all_shapes)}] {shape.name} B={shape.batch} S={shape.seqlen_q} "
                f"H={shape.hdim_q} {shape.dtype} {shape.variant} -> GPU FAULT (exit={proc_result.returncode})",
                flush=True,
            )
            continue

        try:
            rows = json.loads(proc_result.stdout)
        except (json.JSONDecodeError, ValueError):
            rows = []

        if rows:
            total_shapes_run += 1
            for row in rows:
                writer.writerow(row)
                json_results.append(row)
                total_measurements += 1
            csv_file.flush()
            best = max(rows, key=lambda r: r["tflops"])
            print(
                f"  [{si + 1}/{len(all_shapes)}] {shape.name} "
                f"B={shape.batch} S={shape.seqlen_q} H={shape.hdim_q} {shape.dtype} "
                f"{shape.variant} -> {len(rows)} kernels, best={best['tflops']:.3g} TFLOPS "
                f"({best['latency_ms']:.4f} ms) ({best['kernel'][:40]})",
                flush=True,
            )

    csv_file.close()
    bench_time = time.perf_counter() - bench_t0

    # ---- Phase 4: Summary ----
    print(f"\n{'=' * 80}")
    print("Results")
    print(f"{'=' * 80}")
    print(f"  Shapes benchmarked: {total_shapes_run}")
    print(f"  Total measurements: {total_measurements}")
    print(f"  GPU faults survived: {total_gpu_faults}")
    print(f"  Benchmark time: {bench_time:.1f}s")
    print(f"  CSV: {csv_path}")

    if json_results:
        json_path = (
            Path(args.json) if os.path.isabs(args.json) else _THIS_DIR / args.json
        )
        report = {
            "metadata": {
                "arch": args.arch,
                "category": args.category,
                "variants": variants,
                "total_kernels": total_built,
                "total_shapes": len(all_shapes),
                "shapes_benchmarked": total_shapes_run,
                "total_measurements": total_measurements,
                "gpu_faults": total_gpu_faults,
                "bench_time_s": round(bench_time, 1),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "results": json_results,
        }
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  JSON: {json_path}")

        from collections import defaultdict

        by_shape = defaultdict(lambda: {"best": 0, "n": 0})
        for r in json_results:
            k = f"{r['problem_name']} ({r['dtype']})"
            by_shape[k]["n"] += 1
            by_shape[k]["best"] = max(by_shape[k]["best"], r["tflops"])
        print("\n  Top shapes by best TFLOPS:")
        for name, info in sorted(by_shape.items(), key=lambda x: -x[1]["best"])[:15]:
            print(f"    {name:50s} {info['best']:>10.3g} TFLOPS ({info['n']} kernels)")

    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
