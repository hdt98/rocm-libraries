#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Full FMHA Parity Test -- parallel JIT build, sequential GPU test.

Phase 1: JIT-compile every unique kernel config in parallel (hipcc only, no GPU).
Phase 2: Run each test case sequentially through CK Tile and the dispatcher
          (each dispatcher invocation in its own subprocess for HIP isolation).

Usage:
    python3 full_parity_test.py --max-cases 100
    python3 full_parity_test.py --max-cases 0       # all ~3500 cases
    python3 full_parity_test.py --workers 8          # parallel JIT build
    python3 full_parity_test.py --skip-jit           # reuse previous build
"""

import sys
import os
import time
import argparse
import subprocess
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Tuple
from fmha_smoke_matrix import (
    generate_fwd_fp16_bf16_matrix,
    to_ck_cli_args,
    TestCase,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DISPATCHER_DIR = SCRIPT_DIR.parent
PYTHON_DIR = DISPATCHER_DIR / "python"

sys.path.insert(0, str(SCRIPT_DIR))


# =========================================================================
# Config dedup + tile lookup
# =========================================================================

HDIM_TILE_TABLE = {
    (32, 32): (128, 64, 16, 32, 32, 32),
    (64, 64): (128, 64, 32, 64, 32, 64),
    (128, 128): (128, 128, 32, 128, 32, 128),
    (192, 128): (128, 128, 32, 128, 32, 192),
    (192, 192): (128, 128, 32, 192, 32, 192),
    (256, 256): (128, 128, 32, 256, 32, 256),
    (80, 96): (128, 128, 16, 96, 32, 96),
    (96, 128): (128, 128, 32, 128, 32, 96),
}


def _round_hdim(d: int) -> int:
    for t in [32, 64, 96, 128, 192, 256]:
        if d <= t:
            return t
    return 256


def _lookup_tile(dq: int, dv: int):
    key = (dq, dv)
    if key in HDIM_TILE_TABLE:
        return HDIM_TILE_TABLE[key]
    sq = max(dq, dv)
    key2 = (sq, sq)
    if key2 in HDIM_TILE_TABLE:
        t = list(HDIM_TILE_TABLE[key2])
        t[3] = dv
        t[5] = sq
        return tuple(t)
    return (128, 64, 16, dv, 32, sq)


def _mask_str(m: str) -> str:
    return "no" if m == "0" else "top_left"


def _bias_str(b: str) -> str:
    return {"n": "no", "e": "bias", "a": "alibi"}.get(b, "no")


def config_key(c: TestCase) -> tuple:
    tdq = _round_hdim(c.hdim_q)
    tdv = _round_hdim(c.effective_hdim_v())
    # GQA (nhead_q != nhead_k) is a runtime property handled via strides,
    # NOT a compile-time kernel variant.  is_group_mode refers to
    # variable-length batching (mode=1), not GQA.
    is_varlen = c.mode == 1
    return (
        c.prec,
        tdq,
        tdv,
        _mask_str(c.mask),
        _bias_str(c.bias),
        bool(c.lse),
        c.p_drop > 0,
        is_varlen,
    )


def config_name(key: tuple) -> str:
    prec, dq, dv, mask, bias, lse, drop, varlen = key
    n = f"{prec}_h{dq}x{dv}_{'grp' if varlen else 'bat'}_{mask}_{bias}"
    if lse:
        n += "_lse"
    if drop:
        n += "_drop"
    return n


def config_to_codegen_json(key: tuple, arch: str) -> str:
    """Produce the JSON string that generate_fmha_fallback.py expects."""
    prec, dq, dv, mask, bias, lse, drop, is_varlen = key
    tile = _lookup_tile(dq, dv)
    return json.dumps(
        {
            "arch": arch,
            "signature": {
                "family": "fwd",
                "data_type": prec,
                "mode": "group" if is_varlen else "batch",
                "vlayout": "r",
                "hdim_q": dq,
                "hdim_v": dv,
                "mask": mask,
                "bias": bias,
                "lse": lse,
                "dropout": drop,
                "qscale": "no",
                "rope": "none",
                "logits": False,
                "paged_kv": False,
                "fp8_static_quant": False,
                "skip_min_seqlen_q": False,
                "sink": False,
                "dbias": False,
                "store_randval": False,
                "deterministic": False,
                "kv_memory_layout": "vectorized",
                "kv_lookup_table": "sglang",
                "page_size": 1,
            },
            "algorithm": {
                "pipeline": "qr_async" if dq >= 64 else "qr",
                "tile": list(tile),
                "wave": [4, 1, 1, 4, 1, 1, 1, 1, 1],
                "warp": [32, 32, 16, 32, 32, 16, 16, 16, 16],
                "padding": [True, True, True, True],
                "block_per_cu": 1,
                "num_wave_groups": 1,
                "max_splits_log2": 0,
                "max_seq_len_q": 0,
            },
        }
    )


# =========================================================================
# Phase 1 -- JIT build (no GPU, pure hipcc subprocesses)
# =========================================================================


def _jit_one(key: tuple, out_dir: Path, arch: str) -> Tuple[bool, str, float]:
    """JIT-compile a single kernel config. Runs hipcc only, never touches GPU."""
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    codegen_dir = DISPATCHER_DIR / "codegen"
    ctypes_src = DISPATCHER_DIR / "bindings" / "ctypes" / "fmha_ctypes_lib.cpp"
    static_lib = DISPATCHER_DIR / "build" / "libck_tile_dispatcher.a"
    if not static_lib.exists():
        return (False, "libck_tile_dispatcher.a not found", time.perf_counter() - t0)

    hipcc = "hipcc"
    cfg_json = config_to_codegen_json(key, arch)

    # 1. codegen
    r = subprocess.run(
        [
            sys.executable,
            str(codegen_dir / "generate_fmha_fallback.py"),
            "--output-dir",
            str(out_dir),
            "--gpu-target",
            arch,
            "--config-json",
            cfg_json,
        ],
        capture_output=True,
        text=True,
        cwd=str(codegen_dir),
    )
    if r.returncode != 0:
        return (False, f"codegen: {r.stderr[:200]}", time.perf_counter() - t0)

    dispatch_hdr = out_dir / "fmha_python_dispatch.hpp"
    if not dispatch_hdr.exists():
        return (False, "no dispatch header", time.perf_counter() - t0)

    inc = [
        f"-I{DISPATCHER_DIR.parent / 'include'}",
        f"-I{DISPATCHER_DIR / 'include'}",
        f"-I{DISPATCHER_DIR.parent}",
        f"-I{out_dir}",
        f"-I{out_dir / 'dispatcher_wrappers'}",
    ]
    base_flags = [
        "-fPIC",
        "-O3",
        f"--offload-arch={arch}",
        "-std=c++17",
        "-mllvm",
        "-enable-noalias-to-md-conversion=0",
        "-Wno-undefined-func-template",
        "-Wno-float-equal",
        "--offload-compress",
    ]

    # 2. compile kernel .cpp files
    kernel_objs = []
    for cpp in sorted(out_dir.glob("fmha_*.cpp")):
        obj = cpp.with_suffix(".o")
        r = subprocess.run(
            [hipcc, "-c", *base_flags, *inc, str(cpp), "-o", str(obj)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return (False, f"kernel: {r.stderr[:200]}", time.perf_counter() - t0)
        kernel_objs.append(str(obj))

    # 3. compile ctypes lib
    ctypes_obj = out_dir / "fmha_ctypes_lib.o"
    r = subprocess.run(
        [
            hipcc,
            "-c",
            *base_flags,
            *inc,
            f"-include{dispatch_hdr}",
            f'-DGFX_ARCH="{arch}"',
            str(ctypes_src),
            "-o",
            str(ctypes_obj),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return (False, f"ctypes: {r.stderr[:200]}", time.perf_counter() - t0)

    # 4. link .so
    name = config_name(key)
    so_path = out_dir / f"libdispatcher_fmha_{name}.so"
    r = subprocess.run(
        [
            hipcc,
            "-shared",
            "-fPIC",
            str(ctypes_obj),
            *kernel_objs,
            str(static_lib),
            "-lamdhip64",
            "-o",
            str(so_path),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return (False, f"link: {r.stderr[:200]}", time.perf_counter() - t0)

    return (True, str(so_path), time.perf_counter() - t0)


# =========================================================================
# Phase 2 -- GPU tests (sequential, each in its own subprocess)
# =========================================================================


def find_ck_exe() -> Optional[str]:
    for p in [
        "/tmp/ck_fmha_full/bin/tile_example_fmha_fwd",
        "/tmp/ck_fmha_build/bin/tile_example_fmha_fwd",
    ]:
        if os.path.exists(p):
            return p
    return None


def run_ck_test(exe: str, case: TestCase) -> Tuple[bool, str]:
    cmd = [exe] + to_ck_cli_args(case)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (r.returncode == 0, "")
    except subprocess.TimeoutExpired:
        return (False, "timeout")
    except Exception as e:
        return (False, str(e)[:60])


MASK_INT = {"0": 0, "1": 1, "2": 2}
BIAS_INT = {"n": 0, "e": 1, "a": 2}


def run_dispatcher_test(so_path: str, case: TestCase, arch: str) -> Tuple[bool, str]:
    """Run one test in an isolated subprocess -- never touches our process's HIP."""
    dq = case.hdim_q
    dv = case.effective_hdim_v()
    nk = case.effective_nhead_k()

    if case.seqlen_k <= 0 or case.seqlen_q <= 0:
        return (True, "edge-case-ok")

    mi = MASK_INT.get(case.mask, 1 if case.mask.startswith(("t:", "b:")) else 0)
    bi = BIAS_INT.get(case.bias, 0)
    scale = 1.0 / (dq**0.5)

    # Build a tiny runner script executed in a fresh process
    runner = f"""\
import ctypes, numpy as np, sys
lib = ctypes.CDLL("{so_path}")
lib.fmha_dispatcher_initialize.argtypes = [ctypes.c_char_p]
lib.fmha_dispatcher_initialize.restype = ctypes.c_int
lib.fmha_dispatcher_run_fwd.argtypes = [
    ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
    ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,
    ctypes.c_int,ctypes.c_int,ctypes.c_float,
    ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,
    ctypes.POINTER(ctypes.c_float)]
lib.fmha_dispatcher_run_fwd.restype = ctypes.c_int
lib.fmha_dispatcher_cleanup.argtypes = []
lib.fmha_dispatcher_cleanup.restype = None
rc = lib.fmha_dispatcher_initialize(b"{arch}")
if rc != 0: print("INIT_FAIL"); sys.exit(1)
np.random.seed(42)
Q=np.ascontiguousarray((np.random.randn({case.batch},{case.nhead_q},{case.seqlen_q},{dq})*0.3).astype(np.float16))
K=np.ascontiguousarray((np.random.randn({case.batch},{nk},{case.seqlen_k},{dq})*0.3).astype(np.float16))
V=np.ascontiguousarray((np.random.randn({case.batch},{nk},{case.seqlen_k},{dv})*0.3).astype(np.float16))
O=np.ascontiguousarray(np.zeros(({case.batch},{case.nhead_q},{case.seqlen_q},{dv}),dtype=np.float16))
t=ctypes.c_float(0.0)
rc=lib.fmha_dispatcher_run_fwd(Q.ctypes.data,K.ctypes.data,V.ctypes.data,O.ctypes.data,\
{case.batch},{case.nhead_q},{nk},{case.seqlen_q},{case.seqlen_k},{dq},{dv},\
{scale},{mi},{bi},{case.lse},{int(case.p_drop > 0)},ctypes.byref(t))
lib.fmha_dispatcher_cleanup()
if rc!=0: print(f"RC{{rc}}"); sys.exit(1)
nz=int(np.count_nonzero(O))
if nz==0: print("ZEROS"); sys.exit(1)
print(f"OK {{t.value:.3f}}ms nz={{nz}}")
"""
    try:
        r = subprocess.run(
            [sys.executable, "-c", runner],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "HIP_VISIBLE_DEVICES": "0"},
        )
        out = r.stdout.strip()
        err = r.stderr.strip()
        if r.returncode == 0 and out.startswith("OK"):
            return (True, out)
        msg = out or err[:120]
        return (False, msg[:120])
    except subprocess.TimeoutExpired:
        return (False, "timeout")


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=0, help="0 = all ~3500")
    parser.add_argument("--max-configs", type=int, default=0, help="0 = all needed")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--skip-jit", action="store_true")
    parser.add_argument("--skip-ck", action="store_true")
    parser.add_argument("--report", default="parity_report.json")
    args = parser.parse_args()

    ck_exe = find_ck_exe() if not args.skip_ck else None

    print("=" * 80)
    print("FMHA Full Parity Test")
    print("=" * 80)
    print(f"  CK Tile exe:  {ck_exe or 'NOT FOUND / SKIPPED'}")
    print(f"  GPU arch:     {args.arch}")
    print(f"  JIT workers:  {args.workers}")

    # ---- generate test matrix ----
    all_fwd = generate_fwd_fp16_bf16_matrix()
    # Filter to batch-mode (mode=0) only; group-mode (mode=1) requires
    # seqstart arrays which the ctypes lib doesn't yet support.
    fwd_cases = [c for c in all_fwd if c.mode == 0]
    print(f"  Total matrix: {len(all_fwd)} (batch-mode: {len(fwd_cases)})")
    if args.max_cases > 0:
        fwd_cases = fwd_cases[: args.max_cases]

    configs: Dict[tuple, dict] = {}
    case_key: Dict[int, tuple] = {}
    for i, c in enumerate(fwd_cases):
        k = config_key(c)
        configs[k] = configs.get(k, {})
        case_key[i] = k

    if args.max_configs > 0:
        configs = dict(list(configs.items())[: args.max_configs])

    print(f"  Test cases:   {len(fwd_cases)}")
    print(f"  Unique cfgs:  {len(configs)}")

    # ---- Phase 1: parallel JIT ----
    jit_root = Path("/tmp/fmha_parity_jit")
    jit_root.mkdir(parents=True, exist_ok=True)

    lib_for: Dict[tuple, Optional[str]] = {}
    jit_stats = Counter()
    jit_t0 = time.perf_counter()

    if not args.skip_jit:
        print(
            f"\n--- Phase 1: JIT compile ({len(configs)} configs, {args.workers} workers) ---"
        )
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for key in configs:
                name = config_name(key)
                out = jit_root / name
                futures[pool.submit(_jit_one, key, out, args.arch)] = (key, name, out)

            done = 0
            for f in as_completed(futures):
                key, name, out = futures[f]
                ok, msg, elapsed = f.result()
                done += 1
                if ok:
                    lib_for[key] = msg  # msg = so_path on success
                    jit_stats["ok"] += 1
                else:
                    lib_for[key] = None
                    jit_stats["fail"] += 1
                if done % max(1, len(configs) // 20) == 0 or done <= 3 or not ok:
                    tag = "OK" if ok else f"FAIL({msg[:50]})"
                    print(f"  [{done}/{len(configs)}] {name}  {elapsed:.1f}s  {tag}")

    else:
        print("\n--- Phase 1: reusing existing JIT artifacts ---")
        for key in configs:
            name = config_name(key)
            out = jit_root / name
            sos = sorted(out.glob("libdispatcher_fmha_*.so")) if out.exists() else []
            if sos:
                lib_for[key] = str(sos[0])
                jit_stats["ok"] += 1
            else:
                lib_for[key] = None
                jit_stats["missing"] += 1

    jit_elapsed = time.perf_counter() - jit_t0
    print(f"  JIT done: {dict(jit_stats)}  ({jit_elapsed:.0f}s)")

    # ---- Phase 2: sequential GPU tests ----
    print(f"\n--- Phase 2: running {len(fwd_cases)} tests (sequential) ---")
    ck_cnt = Counter()
    disp_cnt = Counter()
    par_cnt = Counter()
    failures = []
    test_t0 = time.perf_counter()

    for i, case in enumerate(fwd_cases):
        if (i + 1) % 50 == 0 or i == 0:
            el = time.perf_counter() - test_t0
            rate = (i + 1) / max(el, 0.01)
            print(f"  [{i + 1}/{len(fwd_cases)}] {el:.0f}s  ({rate:.1f} cases/s)")

        # CK Tile
        if ck_exe:
            ck_ok, _ = run_ck_test(ck_exe, case)
        else:
            ck_ok = None

        # Dispatcher
        key = case_key.get(i)
        so = lib_for.get(key) if key else None
        if so:
            d_ok, d_msg = run_dispatcher_test(so, case, args.arch)
        else:
            d_ok, d_msg = None, "no-lib"

        # tally
        ck_cnt["pass" if ck_ok else ("fail" if ck_ok is False else "skip")] += 1
        disp_cnt["pass" if d_ok else ("fail" if d_ok is False else "skip")] += 1

        if ck_ok is not None and d_ok is not None:
            if ck_ok == d_ok:
                par_cnt["match"] += 1
            else:
                par_cnt["mismatch"] += 1
                failures.append(
                    dict(
                        idx=i,
                        ck=ck_ok,
                        disp=d_ok,
                        msg=d_msg,
                        hq=case.hdim_q,
                        hv=case.effective_hdim_v(),
                        mask=case.mask,
                        bias=case.bias,
                        nq=case.nhead_q,
                        nk=case.effective_nhead_k(),
                        sq=case.seqlen_q,
                        sk=case.seqlen_k,
                    )
                )
        else:
            par_cnt["n/a"] += 1

        if d_ok is False:
            dv = case.effective_hdim_v()
            nk = case.effective_nhead_k()
            print(
                f"    FAIL[{i}] h={case.hdim_q}x{dv} m={case.mask} b={case.bias}"
                f" nq={case.nhead_q} nk={nk}"
                f" sq={case.seqlen_q} sk={case.seqlen_k} -> {d_msg[:80]}"
            )

    test_elapsed = time.perf_counter() - test_t0

    # ---- report ----
    print(f"\n{'=' * 80}")
    print("FMHA Parity Report")
    print(f"{'=' * 80}")
    print(
        f"  JIT build:   {jit_elapsed:.0f}s  ({jit_stats.get('ok', 0)} ok,"
        f" {jit_stats.get('fail', 0)} fail)"
    )
    print(f"  GPU tests:   {test_elapsed:.0f}s  ({len(fwd_cases)} cases)")
    print(f"  Total:       {jit_elapsed + test_elapsed:.0f}s")
    print()
    print(
        f"  CK Tile:     {ck_cnt.get('pass', 0)} pass,"
        f" {ck_cnt.get('fail', 0)} fail, {ck_cnt.get('skip', 0)} skip"
    )
    print(
        f"  Dispatcher:  {disp_cnt.get('pass', 0)} pass,"
        f" {disp_cnt.get('fail', 0)} fail, {disp_cnt.get('skip', 0)} skip"
    )
    print(
        f"  Parity:      {par_cnt.get('match', 0)} match,"
        f" {par_cnt.get('mismatch', 0)} mismatch, {par_cnt.get('n/a', 0)} n/a"
    )
    print(f"{'=' * 80}")

    if failures:
        print("\nFirst 10 mismatches:")
        for f in failures[:10]:
            print(
                f"  [{f['idx']}] ck={f['ck']} disp={f['disp']}"
                f" h={f['hq']}x{f['hv']} m={f['mask']} b={f['bias']}"
                f" nq={f['nq']} nk={f['nk']} -> {f['msg'][:60]}"
            )

    with open(args.report, "w") as fp:
        json.dump(
            dict(
                jit_time_s=jit_elapsed,
                test_time_s=test_elapsed,
                cases=len(fwd_cases),
                configs=len(configs),
                jit=dict(jit_stats),
                ck=dict(ck_cnt),
                dispatcher=dict(disp_cnt),
                parity=dict(par_cnt),
                failures=failures[:100],
            ),
            fp,
            indent=2,
        )
    print(f"\nSaved {args.report}")

    skip_or_mismatch = par_cnt.get("mismatch", 0) + disp_cnt.get("skip", 0)
    return 1 if skip_or_mismatch > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
