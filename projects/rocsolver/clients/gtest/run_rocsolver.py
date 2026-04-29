#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import platform
import shlex
import subprocess
from pathlib import Path


logging.basicConfig(level=logging.INFO)


TEST_EXE = "rocsolver-test"

QUICK_TESTS = [
    "checkin*BDSQR*",
    "checkin*STEBZ*",
    "checkin*STEIN*",
    "checkin*STERF*",
    "checkin*STEQR*",
    "checkin*SYEVJ*",
    "checkin*HEEVJ*",
    "checkin*LARFG*",
    "checkin*LARF*",
    "checkin*LARFT*",
    "checkin*GETF2*",
    "checkin*POTF2*",
    "checkin*GEQR2*",
    "checkin*GELQ2*",
    "checkin*SPLITLU*",
    "checkin*REFACTLU*",
    "checkin*REFACTCHOL*",
]


def exe_name(name: str) -> str:
    if platform.system() == "Windows":
        return f"{name}.exe"
    return name


def derive_rocm_path(script_dir: Path) -> Path:
    test_exe = exe_name(TEST_EXE)
    for candidate in (script_dir, *script_dir.parents):
        if (candidate / "bin" / test_exe).is_file():
            return candidate
        if candidate.name == "bin" and (candidate / test_exe).is_file():
            return candidate.parent
    if script_dir.name == "bin":
        return script_dir.parent
    return script_dir.parent


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    rocm_path = Path(
        os.environ.get("ROCM_PATH", derive_rocm_path(script_dir))
    ).resolve()
    rocm_bin_dir = Path(os.environ.get("ROCM_BIN_DIR", rocm_path / "bin")).resolve()

    env = os.environ.copy()
    # GitHub Actions shard arrays are 1-indexed; GTest shard indexes are 0-indexed.
    env["GTEST_SHARD_INDEX"] = str(int(os.getenv("SHARD_INDEX", "1")) - 1)
    env["GTEST_TOTAL_SHARDS"] = os.getenv("TOTAL_SHARDS", "1")

    cmd = [str(rocm_bin_dir / exe_name(TEST_EXE))]
    if os.getenv("TEST_TYPE", "full") == "quick":
        cmd.append(f"--gtest_filter={':'.join(QUICK_TESTS)}-*LARFB*:*known_bug*")
    else:
        cmd.append(
            "--gtest_filter=checkin*-*known_bug*:checkin_lapack/SYGVDX_INPLACE.__float/41"
        )

    logging.info(f"++ Exec [{rocm_path}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=rocm_path, check=True, env=env)


if __name__ == "__main__":
    main()
