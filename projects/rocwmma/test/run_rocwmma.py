#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
from pathlib import Path


logging.basicConfig(level=logging.INFO)


TEST_DIR_NAME = "rocwmma"

# TODO(#2823): Re-enable test once flaky issue is resolved.
TESTS_TO_IGNORE = ["unpack_util_test"]


def derive_rocm_path(script_dir: Path) -> Path:
    for candidate in (script_dir, *script_dir.parents):
        if (candidate / "bin" / TEST_DIR_NAME / "CTestTestfile.cmake").is_file():
            return candidate
    if script_dir.name == TEST_DIR_NAME and script_dir.parent.name == "bin":
        return script_dir.parent.parent
    return script_dir.parent.parent


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
    env["GTEST_BRIEF"] = "1"
    env["ROCM_PATH"] = str(rocm_path)

    test_subdir = ""
    timeout = "3600"
    test_type = os.getenv("TEST_TYPE", "full")
    if test_type in ("quick", "regression"):
        test_subdir = "regression"
        timeout = "720"

    ctest_parallelism = "2"
    if os.getenv("AMDGPU_FAMILIES") == "gfx1153":
        ctest_parallelism = "1"

    test_dir = rocm_bin_dir / TEST_DIR_NAME
    if test_subdir:
        test_dir /= test_subdir

    cmd = [
        "ctest",
        "--test-dir",
        str(test_dir),
        "--output-on-failure",
        "--parallel",
        ctest_parallelism,
        "--timeout",
        timeout,
        "--exclude-regex",
        "|".join(TESTS_TO_IGNORE),
    ]
    logging.info(f"++ Exec [{rocm_path}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=rocm_path, check=True, env=env)


if __name__ == "__main__":
    main()
