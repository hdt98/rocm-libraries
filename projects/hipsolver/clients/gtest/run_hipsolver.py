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


TEST_EXE = "hipsolver-test"

TESTS_TO_EXCLUDE = [
    "*known_bug*",
    "*HEEVD*float_complex*",
    "*HEEVJ*float_complex*",
    "*HEGVD*float_complex*",
    "*HEGVJ*float_complex*",
    "*HEEVDX*float_complex*",
    "*SYTRF*float_complex*",
    "*HEEVD*double_complex*",
    "*HEEVJ*double_complex*",
    "*HEGVD*double_complex*",
    "*HEGVJ*double_complex*",
    "*HEEVDX*double_complex*",
    "*SYTRF*double_complex*",
    # TODO(#2824): Re-enable test once flaky issue is resolved.
    "checkin_lapack/POTRF_FORTRAN.batched__float_complex/9",
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

    exclusion_list = ":".join(TESTS_TO_EXCLUDE)
    cmd = [
        str(rocm_bin_dir / exe_name(TEST_EXE)),
        f"--gtest_filter=-{exclusion_list}",
    ]

    logging.info(f"++ Exec [{rocm_path}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=rocm_path, check=True, env=env)


if __name__ == "__main__":
    main()
