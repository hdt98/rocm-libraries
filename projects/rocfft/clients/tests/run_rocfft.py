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


TEST_EXE = "rocfft-test"


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
    raise RuntimeError(
        "ROCM_PATH is required when run_rocfft.py is not executed from an "
        "installed ROCm tree containing bin/rocfft-test."
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    rocm_path_env = os.getenv("ROCM_PATH")
    rocm_path = (
        Path(rocm_path_env).resolve()
        if rocm_path_env
        else derive_rocm_path(script_dir)
    )
    rocm_bin_dir = Path(os.environ.get("ROCM_BIN_DIR", rocm_path / "bin")).resolve()
    test_exe = rocm_bin_dir / exe_name(TEST_EXE)
    if not test_exe.is_file():
        raise FileNotFoundError(f"rocFFT test executable not found at {test_exe}")

    env = os.environ.copy()
    # GitHub Actions shard arrays are 1-indexed; GTest shard indexes are 0-indexed.
    env["GTEST_SHARD_INDEX"] = str(int(os.getenv("SHARD_INDEX", "1")) - 1)
    env["GTEST_TOTAL_SHARDS"] = os.getenv("TOTAL_SHARDS", "1")

    if os.getenv("TEST_TYPE", "full") == "quick":
        test_filter = ["--smoketest"]
    else:
        # Due to the large number of tests for rocFFT, run a subset.
        test_filter = [
            "--gtest_filter=-*multi_gpu*",
            "--test_prob",
            "0.02",
        ]

    cmd = [str(test_exe), *test_filter]
    logging.info(f"++ Exec [{rocm_path}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=rocm_path, check=True, env=env)


if __name__ == "__main__":
    main()
