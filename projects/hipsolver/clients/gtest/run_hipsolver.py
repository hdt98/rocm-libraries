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


def format_command(cmd) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in cmd)


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
    raise RuntimeError(
        "Could not derive ROCM_PATH from an installed hipsolver-test layout. "
        "Set ROCM_PATH explicitly."
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    rocm_path_env = os.getenv("ROCM_PATH")
    rocm_path = Path(rocm_path_env).resolve() if rocm_path_env else derive_rocm_path(script_dir)
    rocm_bin_dir = Path(os.getenv("ROCM_BIN_DIR") or rocm_path / "bin").resolve()
    test_exe_path = rocm_bin_dir / exe_name(TEST_EXE)
    if not test_exe_path.is_file():
        raise FileNotFoundError(f"Could not find test executable: {test_exe_path}")

    env = os.environ.copy()
    # GitHub Actions shard arrays are 1-indexed; GTest shard indexes are 0-indexed.
    env["GTEST_SHARD_INDEX"] = str(int(os.getenv("SHARD_INDEX", "1")) - 1)
    env["GTEST_TOTAL_SHARDS"] = os.getenv("TOTAL_SHARDS", "1")

    exclusion_list = ":".join(TESTS_TO_EXCLUDE)
    cmd = [
        str(test_exe_path),
        f"--gtest_filter=-{exclusion_list}",
    ]

    logging.info(f"++ Exec [{rocm_path}]$ {format_command(cmd)}")
    subprocess.run(cmd, cwd=rocm_path, check=True, env=env)


if __name__ == "__main__":
    main()
