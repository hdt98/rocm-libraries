#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
FMHA tile engine benchmark runner.

JIT-compiles kernel configs from sweep JSONs using the dispatcher's Python
interface, runs GPU benchmarks, and reports results.

Build pipeline is 3-stage for maximum parallelism:
  Stage 1: Codegen (fast, parallel) - generate .cpp/.hpp per kernel
  Stage 2: hipcc compile (slow, fully parallel) - all .cpp -> .o at once
  Stage 3: Link (fast, parallel) - .o files -> .so per kernel

Usage:
    python fmha_benchmark.py configs/fwd.json
    python fmha_benchmark.py configs/receipt0_fwd.json --workers 256 --build-dir /tmp/fmha_build
    python fmha_benchmark.py configs/fwd.json --problems "2,8,1024,128" --verify
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_DISPATCHER_ROOT = Path(__file__).resolve().parents[3] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))

from fmha_utils import (  # noqa: E402
    FmhaKernelConfig,
    FmhaRunner,
    FmhaProblem,
    FmhaSetupResult,
    cpu_attention_fwd,
    detect_gpu_arch,
    get_dispatcher_root,
    _find_static_lib,
    _find_hipcc,
)

from fmha_instance_builder import expand_sweep  # noqa: E402


def parse_problems(spec: str) -> List[FmhaProblem]:
    """Parse problem specs: 'batch,nhead,seqlen,hdim;...'"""
    problems = []
    for part in spec.split(";"):
        vals = [int(x) for x in part.split(",")]
        if len(vals) == 4:
            b, h, s, d = vals
            problems.append(
                FmhaProblem(
                    batch=b,
                    nhead_q=h,
                    nhead_k=h,
                    seqlen_q=s,
                    seqlen_k=s,
                    hdim_q=d,
                    hdim_v=d,
                )
            )
        elif len(vals) == 6:
            b, hq, hk, sq, sk, d = vals
            problems.append(
                FmhaProblem(
                    batch=b,
                    nhead_q=hq,
                    nhead_k=hk,
                    seqlen_q=sq,
                    seqlen_k=sk,
                    hdim_q=d,
                    hdim_v=d,
                )
            )
    return problems


class PipelinedJIT:
    """3-stage pipelined JIT: codegen -> compile -> link, each fully parallel."""

    def __init__(self, configs: List[FmhaKernelConfig], build_dir: Path, workers: int):
        self.configs = configs
        self.build_dir = build_dir
        self.workers = workers
        self.root = get_dispatcher_root()
        self.hipcc = _find_hipcc()
        self.static_lib = _find_static_lib()
        self.ctypes_src = self.root / "bindings" / "ctypes" / "fmha_ctypes_lib.cpp"
        self.codegen_dir = self.root / "codegen"
        self.inc_flags = [
            f"-I{self.root.parent / 'include'}",
            f"-I{self.root / 'include'}",
            f"-I{self.root.parent}",
        ]
        self._lock = threading.Lock()
        self._done = 0
        self._phase = ""
        self._t0 = 0.0

    def _tick(self, ok: bool = True):
        with self._lock:
            self._done += 1
            if self._done % 500 == 0 or self._done == len(self.configs):
                elapsed = time.perf_counter() - self._t0
                rate = self._done / elapsed if elapsed > 0 else 0
                print(
                    f"    [{self._done}/{len(self.configs)}]"
                    f" {elapsed:.0f}s ({rate:.1f}/s)",
                    flush=True,
                )

    def _codegen_one(self, config: FmhaKernelConfig) -> Optional[Path]:
        out = self.build_dir / config.name
        out.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [
                sys.executable,
                str(self.codegen_dir / "generate_fmha_fallback.py"),
                "--output-dir",
                str(out),
                "--gpu-target",
                config.gfx_arch,
                "--config-json",
                config.to_codegen_json(),
            ],
            capture_output=True,
            text=True,
            cwd=str(self.codegen_dir),
        )
        self._tick()
        if r.returncode != 0:
            return None
        if not (out / "fmha_python_dispatch.hpp").exists():
            return None
        return out

    def _compile_one(self, cpp: Path, arch: str) -> Optional[Path]:
        obj = cpp.with_suffix(".o")
        if obj.exists():
            self._tick()
            return obj
        cmd = [
            self.hipcc,
            "-c",
            "-fPIC",
            "-O3",
            f"--offload-arch={arch}",
            "-std=c++17",
            *self.inc_flags,
            "-mllvm",
            "-enable-noalias-to-md-conversion=0",
            "-Wno-undefined-func-template",
            "-Wno-float-equal",
            "--offload-compress",
        ]
        if arch.startswith("gfx9"):
            cmd.append("-DCK_TILE_FMHA_FWD_FAST_EXP2=1")
        cmd += [str(cpp), "-o", str(obj)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        self._tick()
        return obj if r.returncode == 0 else None

    def _compile_ctypes(self, out_dir: Path, arch: str) -> Optional[Path]:
        obj = out_dir / "fmha_ctypes_lib.o"
        if obj.exists():
            self._tick()
            return obj
        dispatch = out_dir / "fmha_python_dispatch.hpp"
        cmd = [
            self.hipcc,
            "-c",
            "-fPIC",
            "-O3",
            f"--offload-arch={arch}",
            "-std=c++17",
            *self.inc_flags,
            f"-I{out_dir}",
            f"-I{out_dir / 'dispatcher_wrappers'}",
            f"-include{dispatch}",
            f'-DGFX_ARCH="{arch}"',
            "-mllvm",
            "-enable-noalias-to-md-conversion=0",
            "-Wno-undefined-func-template",
            "-Wno-float-equal",
            "--offload-compress",
        ]
        if arch.startswith("gfx9"):
            cmd.append("-DCK_TILE_FMHA_FWD_FAST_EXP2=1")
        cmd += [str(self.ctypes_src), "-o", str(obj)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        self._tick()
        return obj if r.returncode == 0 else None

    def _link_one(self, out_dir: Path, config: FmhaKernelConfig) -> Optional[Path]:
        objs = list(out_dir.glob("*.o"))
        if not objs:
            self._tick()
            return None
        lib = out_dir / f"lib_{config.name}.so"
        if lib.exists():
            self._tick()
            return lib
        r = subprocess.run(
            [
                self.hipcc,
                "-shared",
                "-fPIC",
                *[str(o) for o in objs],
                str(self.static_lib),
                "-o",
                str(lib),
            ],
            capture_output=True,
            text=True,
        )
        self._tick()
        return lib if r.returncode == 0 else None

    def run(self) -> Dict[str, FmhaSetupResult]:
        results: Dict[str, FmhaSetupResult] = {}
        arch = self.configs[0].gfx_arch if self.configs else "gfx950"
        n = len(self.configs)

        # Stage 1: Parallel codegen
        print(f"  Stage 1: Codegen ({n} kernels, {self.workers} workers)")
        self._done = 0
        self._t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            codegen_dirs = list(pool.map(self._codegen_one, self.configs))
        t1 = time.perf_counter() - self._t0
        codegen_ok = sum(1 for d in codegen_dirs if d is not None)
        print(f"    Done: {codegen_ok}/{n} in {t1:.0f}s")

        # Collect all .cpp files and ctypes compile jobs
        kernel_cpps: List[Tuple[Path, str]] = []  # (cpp, arch)
        ctypes_jobs: List[Tuple[Path, str]] = []  # (out_dir, arch)
        config_map: Dict[str, Tuple[FmhaKernelConfig, Path]] = {}

        for config, out_dir in zip(self.configs, codegen_dirs):
            if out_dir is None:
                results[config.name] = FmhaSetupResult(
                    success=False, config=config, error="codegen failed"
                )
                continue
            config_map[config.name] = (config, out_dir)
            for cpp in out_dir.glob("fmha_*.cpp"):
                kernel_cpps.append((cpp, arch))
            ctypes_jobs.append((out_dir, arch))

        # Stage 2: Parallel compile ALL .cpp + ctypes at once
        total_compile = len(kernel_cpps) + len(ctypes_jobs)
        print(
            f"  Stage 2: Compile ({len(kernel_cpps)} kernels"
            f" + {len(ctypes_jobs)} ctypes = {total_compile} files,"
            f" {self.workers} workers)"
        )
        self._done = 0
        self._t0 = time.perf_counter()

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            kernel_futs = {
                pool.submit(self._compile_one, cpp, a): cpp for cpp, a in kernel_cpps
            }
            ctypes_futs = {
                pool.submit(self._compile_ctypes, d, a): d for d, a in ctypes_jobs
            }

            kernel_results = {}
            for fut in as_completed(kernel_futs):
                cpp = kernel_futs[fut]
                kernel_results[cpp] = fut.result()

            ctypes_results = {}
            for fut in as_completed(ctypes_futs):
                d = ctypes_futs[fut]
                ctypes_results[d] = fut.result()

        t2 = time.perf_counter() - self._t0
        kernel_ok = sum(1 for v in kernel_results.values() if v is not None)
        ctypes_ok = sum(1 for v in ctypes_results.values() if v is not None)
        print(
            f"    Done: kernels={kernel_ok}/{len(kernel_cpps)}"
            f" ctypes={ctypes_ok}/{len(ctypes_jobs)} in {t2:.0f}s"
        )

        # Mark failed compiles
        for name, (config, out_dir) in config_map.items():
            if ctypes_results.get(out_dir) is None:
                results[name] = FmhaSetupResult(
                    success=False, config=config, error="compile failed"
                )

        # Stage 3: Parallel link
        link_jobs = [
            (name, config, out_dir)
            for name, (config, out_dir) in config_map.items()
            if name not in results
        ]
        print(f"  Stage 3: Link ({len(link_jobs)} libraries, {self.workers} workers)")
        self._done = 0
        self._t0 = time.perf_counter()

        def _do_link(item):
            name, config, out_dir = item
            lib = self._link_one(out_dir, config)
            return name, config, lib

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for name, config, lib in pool.map(_do_link, link_jobs):
                if lib is None:
                    results[name] = FmhaSetupResult(
                        success=False, config=config, error="link failed"
                    )
                else:
                    try:
                        runner = FmhaRunner.from_library(str(lib), arch)
                        results[name] = FmhaSetupResult(
                            success=True,
                            config=config,
                            runner=runner,
                            library_path=str(lib),
                        )
                    except Exception as e:
                        results[name] = FmhaSetupResult(
                            success=False, config=config, error=f"load failed: {e}"
                        )

        t3 = time.perf_counter() - self._t0
        loaded = sum(1 for r in results.values() if r.success)
        print(f"    Done: {loaded} loaded in {t3:.0f}s")

        return results


def main():
    parser = argparse.ArgumentParser(description="FMHA Tile Engine Benchmark")
    parser.add_argument("configs", nargs="+", help="Sweep config JSON(s)")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--workers", type=int, default=8, help="Parallel JIT workers")
    parser.add_argument(
        "--problems",
        default="2,8,1024,128",
        help="Problem sizes: batch,nhead,seqlen,hdim",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--verify", action="store_true", help="Verify against CPU reference"
    )
    parser.add_argument(
        "--best", action="store_true", help="Show best kernel per problem"
    )
    parser.add_argument("--csv", type=str, default=None, help="Write CSV to file")
    parser.add_argument("--json", type=str, default=None, help="Write JSON to file")
    parser.add_argument(
        "--build-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "build"),
        help="JIT build output directory",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Remove build dir before starting"
    )
    parser.add_argument(
        "--compile-only", action="store_true", help="Only compile, skip benchmark"
    )
    args = parser.parse_args()

    problems = parse_problems(args.problems)
    build_dir = Path(args.build_dir).resolve()

    if args.clean and build_dir.exists():
        print(f"  Cleaning {build_dir} ...")
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    # Phase 0: Expand all configs
    all_configs = []
    for cfg_path in args.configs:
        configs = expand_sweep(cfg_path, args.arch)
        all_configs.extend(configs)
        print(f"  {cfg_path}: {len(configs)} kernel configs")

    print(f"\n{'=' * 70}")
    print("FMHA Tile Engine Benchmark")
    print(f"{'=' * 70}")
    print(f"  Arch:     {args.arch}")
    print(f"  Kernels:  {len(all_configs)}")
    print(f"  Problems: {len(problems)}")
    print(f"  Workers:  {args.workers}")
    print(f"  Build:    {build_dir}")

    # Phase 1: Pipelined JIT
    print("\n--- Phase 1: Pipelined JIT compile ---")
    jit_t0 = time.perf_counter()

    pipeline = PipelinedJIT(all_configs, build_dir, args.workers)
    setup_map = pipeline.run()

    jit_time = time.perf_counter() - jit_t0
    built = sum(1 for r in setup_map.values() if r.success)
    failed = len(all_configs) - built
    print(f"\n  Total: {built}/{len(all_configs)} in {jit_time:.0f}s ({failed} failed)")

    if args.compile_only:
        print(f"\n{'=' * 70}")
        print(f"  Compile-only mode. {built} kernels ready.")
        print(f"{'=' * 70}")
        return

    # Phase 2: Sequential GPU benchmark
    print(f"\n--- Phase 2: Benchmark ({built} kernels x {len(problems)} problems) ---")

    np.random.seed(42)
    all_results = []
    bench_t0 = time.perf_counter()

    for prob_idx, prob in enumerate(problems):
        Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float16)
        K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float16)
        V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float16)

        ref = None
        if args.verify:
            ref = cpu_attention_fwd(
                Q.astype(np.float32),
                K.astype(np.float32),
                V.astype(np.float32),
                prob.scale,
            )

        prob_str = f"B={prob.batch} H={prob.nhead_q} S={prob.seqlen_q} D={prob.hdim_q}"
        print(f"\n  Problem [{prob_idx}]: {prob_str}")
        print(
            f"  {'Kernel':<50} {'Time(ms)':>10} {'TFLOPS':>10}"
            f" {'MaxErr':>10} {'Status':>6}"
        )
        print(f"  {'-' * 90}")

        for config in all_configs:
            setup = setup_map.get(config.name)
            if setup is None or not setup.success or setup.runner is None:
                continue

            result = setup.runner.run(Q, K, V, prob)
            if not result.success:
                continue

            max_err = 0.0
            status = "OK"
            if ref is not None and result.output is not None:
                max_err = float(np.abs(result.output.astype(np.float32) - ref).max())
                status = "PASS" if max_err < 0.05 else "FAIL"

            print(
                f"  {config.name:<50} {result.time_ms:>10.3f}"
                f" {result.tflops:>10.2f} {max_err:>10.2e} {status:>6}"
            )

            all_results.append(
                {
                    "kernel": config.name,
                    "dtype": config.data_type,
                    "hdim": config.hdim_q,
                    "pipeline": config.pipeline,
                    "problem": {
                        "batch": prob.batch,
                        "nhead_q": prob.nhead_q,
                        "seqlen_q": prob.seqlen_q,
                        "hdim_q": prob.hdim_q,
                    },
                    "latency_ms": result.time_ms,
                    "tflops": result.tflops,
                    "max_err": max_err,
                }
            )

    bench_time = time.perf_counter() - bench_t0

    # Cleanup
    for setup in setup_map.values():
        if setup.success and setup.runner:
            try:
                setup.runner.cleanup()
            except Exception:
                pass

    # Report
    print(f"\n{'=' * 70}")
    print(f"  JIT:       {jit_time:.0f}s ({built} kernels)")
    print(f"  Benchmark: {bench_time:.1f}s")
    print(f"  Results:   {len(all_results)} measurements")

    if args.best and all_results:
        from collections import defaultdict

        by_problem = defaultdict(list)
        for r in all_results:
            key = json.dumps(r["problem"], sort_keys=True)
            by_problem[key].append(r)

        print("\n  Best kernel per problem:")
        for key, results in by_problem.items():
            best = max(results, key=lambda x: x["tflops"])
            prob = json.loads(key)
            print(
                f"    B={prob['batch']} H={prob['nhead_q']}"
                f" S={prob['seqlen_q']} D={prob['hdim_q']}"
                f" -> {best['kernel']} ({best['tflops']:.2f} TFLOPS)"
            )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "kernel",
                    "dtype",
                    "hdim",
                    "pipeline",
                    "batch",
                    "nhead_q",
                    "seqlen_q",
                    "hdim_q",
                    "latency_ms",
                    "tflops",
                    "max_err",
                ],
            )
            writer.writeheader()
            for r in all_results:
                row = {**r, **r["problem"]}
                del row["problem"]
                writer.writerow(row)
        print(f"\n  CSV: {args.csv}")

    if args.json:
        report = {
            "metadata": {
                "arch": args.arch,
                "jit_time_s": jit_time,
                "bench_time_s": bench_time,
                "num_kernels": len(all_configs),
                "num_built": built,
                "num_problems": len(problems),
            },
            "results": all_results,
        }
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  JSON: {args.json}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
