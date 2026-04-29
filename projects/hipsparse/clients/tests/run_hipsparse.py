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


TEST_EXE = "hipsparse-test"

TEST_TO_IGNORE = {
    "gfx1151": {
        # TODO(#3621): Include test once out of resource errors are resolved.
        "windows": ["*spmm*"]
    },
}


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

    output_artifacts_dir = os.getenv("OUTPUT_ARTIFACTS_DIR")
    if output_artifacts_dir:
        env["HIPSPARSE_CLIENTS_MATRICES_DIR"] = str(
            Path(output_artifacts_dir) / "clients" / "matrices"
        )

    gtest_filter = "--gtest_filter="
    if os.getenv("TEST_TYPE", "full") == "quick":
        gtest_filter += "*spmv*:*spsv*:*spsm*:*spmm*:*csric0*:*csrilu0*:-known_bug*"
    else:
        gtest_filter += "*quick*:-known_bug*"

    amdgpu_families = os.getenv("AMDGPU_FAMILIES")
    os_type = platform.system().lower()
    if amdgpu_families in TEST_TO_IGNORE and os_type in TEST_TO_IGNORE[amdgpu_families]:
        ignored_tests = TEST_TO_IGNORE[amdgpu_families][os_type]
        gtest_filter += ":" + ":".join(ignored_tests)

    cmd = [str(rocm_bin_dir / exe_name(TEST_EXE)), gtest_filter]
    logging.info(f"++ Exec [{rocm_path}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=rocm_path, check=True, env=env)


if __name__ == "__main__":
    main()
