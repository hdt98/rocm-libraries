#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
tune_grouped_conv_fwd_im2win.py
===============================
Benchmark and validate all im2win forward-conv kernel configs for a given
convolution problem.

For each config the script:
  1. Runs the binary with the user-supplied conv args (v=0) for timing.
  2. Runs the binary again with v=2 (GPU reference) to validate correctness.
  3. Reports timing, throughput, bandwidth, and pass/fail status.
  4. Prints a ranked summary at the end.

Usage
-----
  python3 tune_grouped_conv_fwd_im2win.py [options] [-- conv_args...]

Options
-------
  --bin-dir DIR   Directory containing the im2win binaries
                  (default: ./bin relative to CWD, i.e. the build dir)
  --warmup N      Warm-up iterations  (default: 10)
  --repeat  N     Timing iterations   (default: 50)
  --no-validate   Skip GPU-reference validation (saves time)
  --configs A,B   Comma-separated subset of config names to run
                  (default: all available configs)

Conv args
---------
  Everything after '--' is forwarded verbatim to every binary, e.g.:
    -prec=fp16 -in_layout=GNCHW -wei_layout=GKCYX -out_layout=NHWGK
    -g=32 -n=32 -k=4 -c=4 -y=3 -x=3 -h=200 -w=200
    -lpad_h=1 -lpad_w=1 -rpad_h=1 -rpad_w=1

Example
-------
  cd <build-dir>
  python3 .../script/tune_grouped_conv_fwd_im2win.py \\
      --warmup 10 --repeat 50 \\
      -- \\
      -prec=fp16 -in_layout=GNCHW -wei_layout=GKCYX -out_layout=NHWGK \\
      -g=32 -n=32 -k=4 -c=4 -y=3 -x=3 -h=200 -w=200 \\
      -lpad_h=1 -lpad_w=1 -rpad_h=1 -rpad_w=1
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Config registry ──────────────────────────────────────────────────────────
# Must stay in sync with im2win_conv_configs.hpp and CMakeLists.txt.
ALL_CONFIGS: list[str] = [
    # True im2win (M=K, N=N×Ho×Wo) — correct algorithm
    "Ti_M4N256K64",
    "Ti_M4N128K64",
    "Ti_M8N128K64",
    "Ti_M16N64K64",
    # Old-shape configs (M=N×Ho×Wo, N=K) — kept for comparison
    "CV3_M16N64K64",
    "CV3_M64N64K64",
    "CV3_M64N32K64",
    "Mem_M128N32K64",
    "Mem_M128N32K64_IW",
    "CV3_M128N128K64_Occ2",
    "CV3_M64N16K64",
    "CV3_M128N16K64",
    "Mem_M128N16K64",
    "Mem_M128N16K64_IW",
    "CV3_M64N16K64_Occ2",
    "Mem_M128N4K64",
    "Mem_M256N4K64",
    # Old sequential group-merging (GNCHW, loop per group)
    "CV3_M128N16K64_Gm2",
    "CV3_M128N16K64_Gm4",
    "CV3_M128N16K64_Gm8",
    "CV3_M128N16K64_Gm32",
    "CV3_M64N16K64_Gm32",
    # True group merging (NHWGC/GKYXC + XOR diagonal C descriptor)
    "Merge_Gm2_M8N128K64",
    "Merge_Gm4_M16N64K64",
    "Merge_Gm8_M32N128K64",
    "Merge_Gm32_M128N32K64",
    "Merge_Gm32_M128N64K64",
    "Merge_Gm8_M128N32K32",
]

BINARY_PREFIX = "tile_example_grouped_conv_fwd_im2win_"

# Patterns for parsing binary stdout
_TIMING_RE  = re.compile(
    r"^(?P<ms>[0-9]+\.[0-9]+)\s+ms,\s+"
    r"(?P<tflops>[0-9]+\.[0-9]+)\s+TFlops,\s+"
    r"(?P<gbs>[0-9]+\.[0-9]+)\s+GB/s",
    re.MULTILINE,
)
_PASS_RE    = re.compile(r"verification:\s+PASSED", re.IGNORECASE)
_FAIL_RE    = re.compile(r"verification:\s+FAILED", re.IGNORECASE)
_SKIP_RE    = re.compile(r"not supported|unsupported|Wrong!", re.IGNORECASE)


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class ConfigResult:
    name:    str
    ms:      Optional[float] = None
    tflops:  Optional[float] = None
    gbs:     Optional[float] = None
    valid:   Optional[bool]  = None   # None = not validated
    skip_reason: str = ""


# ── Runner ────────────────────────────────────────────────────────────────────
def run_binary(bin_path: Path, extra_args: list[str], timeout: int = 120) -> tuple[int, str]:
    """Run a binary and return (returncode, stdout+stderr)."""
    cmd = [str(bin_path)] + extra_args
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
        )
        return result.returncode, result.stdout
    except subprocess.TimeoutExpired:
        return -1, f"[TIMEOUT after {timeout}s]"
    except Exception as exc:
        return -1, f"[EXCEPTION: {exc}]"


def benchmark_config(
    bin_path:     Path,
    conv_args:    list[str],
    warmup:       int,
    repeat:       int,
    validate:     bool,
) -> ConfigResult:
    name = bin_path.name.removeprefix(BINARY_PREFIX)
    result = ConfigResult(name=name)

    # ── Step 1: timing run (v=0) ──────────────────────────────────────────
    timing_args = conv_args + [f"-warmup={warmup}", f"-repeat={repeat}", "-v=0"]
    rc, output = run_binary(bin_path, timing_args)

    if _SKIP_RE.search(output):
        result.skip_reason = "kernel reported unsupported arguments"
        return result

    m = _TIMING_RE.search(output)
    if not m:
        result.skip_reason = "no timing line in output"
        return result

    result.ms     = float(m.group("ms"))
    result.tflops = float(m.group("tflops"))
    result.gbs    = float(m.group("gbs"))

    # ── Step 2: validation run (v=2, warmup=1, repeat=1) ─────────────────
    if validate:
        val_args = conv_args + ["-warmup=1", "-repeat=1", "-v=2"]
        _, val_output = run_binary(bin_path, val_args)

        if _PASS_RE.search(val_output):
            result.valid = True
        elif _FAIL_RE.search(val_output):
            result.valid = False
        else:
            # validation was requested but no verdict found — treat as error
            result.valid = False

    return result


# ── Formatting helpers ────────────────────────────────────────────────────────
_COL_NAME   = 26
_COL_MS     = 10
_COL_TFLOPS = 10
_COL_GBS    = 10
_COL_VALID  = 8
_SEP        = "═" * 76
_DASH       = "─" * 76


def _validation_str(valid: Optional[bool], requested: bool) -> str:
    if not requested:
        return "  n/a"
    if valid is True:
        return " PASS"
    if valid is False:
        return " FAIL"
    return "  ???"


def print_header(validate: bool) -> None:
    print(_SEP)
    print(" im2win config sweep — tile_example_grouped_conv_fwd_im2win_<config>")
    print(_SEP)
    hdr = (
        f"{'Config':<{_COL_NAME}}"
        f"{'ms':>{_COL_MS}}"
        f"{'TFlops':>{_COL_TFLOPS}}"
        f"{'GB/s':>{_COL_GBS}}"
    )
    if validate:
        hdr += f"{'Valid':>{_COL_VALID}}"
    print(hdr)
    print(_DASH)


def print_result(r: ConfigResult, validate: bool) -> None:
    if r.ms is None:
        print(f"  {r.name:<{_COL_NAME-2}}  [SKIP: {r.skip_reason}]")
        return

    line = (
        f"  {r.name:<{_COL_NAME-2}}"
        f"  {r.ms:{_COL_MS-2}.5f}"
        f"  {r.tflops:{_COL_TFLOPS-2}.5f}"
        f"  {r.gbs:{_COL_GBS-2}.3f}"
    )
    if validate:
        line += f"  {_validation_str(r.valid, validate)}"
    print(line)


def print_summary(results: list[ConfigResult], validate: bool) -> None:
    timed   = [r for r in results if r.ms is not None]
    failed  = [r for r in timed  if validate and r.valid is False]
    valid   = [r for r in timed  if not validate or r.valid is True]
    skipped = [r for r in results if r.ms is None]

    print(_SEP)

    if not timed:
        print(" No configs produced timing output.")
        print(_SEP)
        return

    if valid:
        best = min(valid, key=lambda r: r.ms)
        print(f" Best (correct):  {best.name}  →  {best.ms:.5f} ms"
              f"  |  {best.tflops:.3f} TFlops  |  {best.gbs:.1f} GB/s")
    else:
        print(" No correctly-validated configs found.")

    if failed:
        names = ", ".join(r.name for r in failed)
        print(f" FAILED validation: {names}")

    if skipped:
        names = ", ".join(r.name for r in skipped)
        print(f" Skipped ({len(skipped)}): {names}")

    print(_SEP)

    # Exit code: non-zero if any validated config failed
    if failed:
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--bin-dir",
        default=None,
        help="Directory containing the im2win binaries "
             "(default: $IM2WIN_BIN_DIR or ./bin)",
    )
    p.add_argument(
        "--warmup", type=int, default=10,
        help="Warm-up iterations before timing (default: 10)",
    )
    p.add_argument(
        "--repeat", type=int, default=50,
        help="Timing iterations (default: 50)",
    )
    p.add_argument(
        "--no-validate", action="store_true",
        help="Skip GPU-reference validation (faster, but no correctness check)",
    )
    p.add_argument(
        "--configs",
        default=None,
        help="Comma-separated subset of config names to run "
             "(default: all configs)",
    )
    # Everything after '--' is forwarded to the binaries as conv args.
    return p.parse_known_args()


def main() -> None:
    ns, extra = parse_args()

    # ── Binary directory ──────────────────────────────────────────────────
    if ns.bin_dir:
        bin_dir = Path(ns.bin_dir)
    elif "IM2WIN_BIN_DIR" in os.environ:
        bin_dir = Path(os.environ["IM2WIN_BIN_DIR"])
    else:
        bin_dir = Path.cwd() / "bin"

    if not bin_dir.is_dir():
        sys.exit(f"Error: binary directory not found: {bin_dir}\n"
                 "Run this script from the build directory or pass --bin-dir.")

    # ── Config selection ──────────────────────────────────────────────────
    if ns.configs:
        configs = [c.strip() for c in ns.configs.split(",") if c.strip()]
        unknown = set(configs) - set(ALL_CONFIGS)
        if unknown:
            sys.exit(f"Error: unknown configs: {', '.join(sorted(unknown))}\n"
                     f"Available: {', '.join(ALL_CONFIGS)}")
    else:
        configs = ALL_CONFIGS

    validate = not ns.no_validate

    # Extra conv args: everything after '--' on the command line.
    # argparse puts them in `extra` (unknown args); strip the '--' sentinel.
    conv_args = [a for a in extra if a != "--"]

    # ── Print header ──────────────────────────────────────────────────────
    print(_SEP)
    print(" im2win config sweep")
    print(f" Binary dir: {bin_dir}")
    print(f" Conv args:  {' '.join(conv_args)}")
    print(f" Validation: {'GPU reference (v=2)' if validate else 'disabled (--no-validate)'}")
    print(_SEP)
    print_header(validate)

    # ── Run configs ───────────────────────────────────────────────────────
    results: list[ConfigResult] = []
    for cfg_name in configs:
        bin_path = bin_dir / f"{BINARY_PREFIX}{cfg_name}"
        if not bin_path.exists():
            r = ConfigResult(name=cfg_name, skip_reason="binary not found")
            results.append(r)
            print_result(r, validate)
            continue

        r = benchmark_config(
            bin_path=bin_path,
            conv_args=conv_args,
            warmup=ns.warmup,
            repeat=ns.repeat,
            validate=validate,
        )
        results.append(r)
        print_result(r, validate)
        # Flush immediately so progress is visible for long sweeps.
        sys.stdout.flush()

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(results, validate)


if __name__ == "__main__":
    main()
