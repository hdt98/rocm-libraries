#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
import logging
import os
import platform
import shlex
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def format_command(cmd) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in cmd)


def load_shared_test_utils():
    tools_dir = os.getenv("THEROCK_TEST_TOOLS_DIR")
    if not tools_dir:
        return None

    utils_path = Path(tools_dir) / "test_utils.py"
    if not utils_path.is_file():
        return None

    spec = importlib.util.spec_from_file_location("therock_test_utils", utils_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load shared test utilities from {utils_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logging.info(f"++ Loaded shared test utilities from {utils_path}")
    return module


def gtest_shard_env():
    shard_index = os.getenv("SHARD_INDEX", "1")
    total_shards = os.getenv("TOTAL_SHARDS", "1")
    test_utils = load_shared_test_utils()
    if test_utils is not None:
        return test_utils.gtest_shard_env(shard_index, total_shards)

    # GitHub Actions shard arrays are 1-indexed; GTest shard indexes are 0-indexed.
    return {
        "GTEST_SHARD_INDEX": str(int(shard_index) - 1),
        "GTEST_TOTAL_SHARDS": total_shards,
    }


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
        Path(rocm_path_env).resolve() if rocm_path_env else derive_rocm_path(script_dir)
    )
    rocm_bin_dir = Path(os.environ.get("ROCM_BIN_DIR", rocm_path / "bin")).resolve()
    test_exe = rocm_bin_dir / exe_name(TEST_EXE)
    if not test_exe.is_file():
        raise FileNotFoundError(f"rocFFT test executable not found at {test_exe}")

    env = os.environ.copy()
    env.update(gtest_shard_env())

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
    logging.info(f"++ Exec [{rocm_path}]$ {format_command(cmd)}")
    subprocess.run(cmd, cwd=rocm_path, check=True, env=env)


if __name__ == "__main__":
    main()
