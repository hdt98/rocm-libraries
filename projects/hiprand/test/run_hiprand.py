#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
from pathlib import Path


logging.basicConfig(level=logging.INFO)


def format_command(cmd) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in cmd)


TEST_DIR_NAME = "hipRAND"


def derive_rocm_path(script_dir: Path) -> Path:
    for candidate in (script_dir, *script_dir.parents):
        if (candidate / "bin" / TEST_DIR_NAME / "CTestTestfile.cmake").is_file():
            return candidate
    raise RuntimeError(
        "ROCM_PATH is required when run_hiprand.py is not executed from an "
        "installed ROCm tree containing bin/hipRAND/CTestTestfile.cmake."
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    rocm_path_env = os.getenv("ROCM_PATH")
    rocm_path = (
        Path(rocm_path_env).resolve()
        if rocm_path_env
        else derive_rocm_path(script_dir)
    )
    test_dir = rocm_path / "bin" / TEST_DIR_NAME
    if not (test_dir / "CTestTestfile.cmake").is_file():
        raise FileNotFoundError(
            f"hipRAND CTest file not found at {test_dir / 'CTestTestfile.cmake'}"
        )

    cmd = [
        "ctest",
        "--test-dir",
        str(test_dir),
        "--output-on-failure",
        "--parallel",
        "8",
        "--timeout",
        "60",
    ]
    logging.info(f"++ Exec [{rocm_path}]$ {format_command(cmd)}")
    subprocess.run(cmd, cwd=rocm_path, check=True)


if __name__ == "__main__":
    main()
