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
    raise RuntimeError(
        "Could not derive ROCM_PATH from an installed hipsparse-test layout. "
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

    output_artifacts_dir = os.getenv("OUTPUT_ARTIFACTS_DIR")
    if output_artifacts_dir:
        env["HIPSPARSE_CLIENTS_MATRICES_DIR"] = str(
            Path(output_artifacts_dir) / "clients" / "matrices"
        )

    if os.getenv("TEST_TYPE", "full") == "quick":
        tests_to_run = ["*spmv*", "*spsv*", "*spsm*", "*spmm*", "*csric0*", "*csrilu0*"]
    else:
        tests_to_run = ["*quick*"]

    tests_to_skip = ["*known_bug*"]
    amdgpu_families = os.getenv("AMDGPU_FAMILIES")
    os_type = platform.system().lower()
    if amdgpu_families in TEST_TO_IGNORE and os_type in TEST_TO_IGNORE[amdgpu_families]:
        tests_to_skip.extend(TEST_TO_IGNORE[amdgpu_families][os_type])

    gtest_filter = f"--gtest_filter={':'.join(tests_to_run)}-{':'.join(tests_to_skip)}"
    cmd = [str(test_exe_path), gtest_filter]
    logging.info(f"++ Exec [{rocm_path}]$ {format_command(cmd)}")
    subprocess.run(cmd, cwd=rocm_path, check=True, env=env)


if __name__ == "__main__":
    main()
