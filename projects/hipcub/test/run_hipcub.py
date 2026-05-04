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


TEST_DIR_NAME = "hipcub"

QUICK_TESTS = [
    "*ShuffleTests/*.*",
    "*WarpStoreTest/*.*",
    "AdjacentDifference/*.*",
    "AdjacentDifferenceSubtract/*.*",
    "BatchCopyTests/*.*",
    "BatchMemcpyTests/*.*",
    "BlockScan*",
    "DeviceScanTests/*.*",
    "Discontinuity/*.*",
    "DivisionOperatorTests/*.*",
    "ExchangeTests",
    "GridTests/*.*",
    "HistogramEven/*.*",
    "HistogramInputArrayTests/*.*",
    "HistogramRange/*.*",
    "IteratorTests/*.*",
    "LoadStoreTestsDirect/*.*",
    "LoadStoreTestsStriped/*.*",
    "LoadStoreTestsTranspose/*.*",
    "LoadStoreTestsVectorize/*.*",
    "MergeSort/*.*",
    "NCThreadOperatorsTests/*",
    "RadixRank/*.*",
    "RadixSort/*.*",
    "ReduceArgMinMaxSpecialTests/*.*",
    "ReduceInputArrayTests/*.*",
    "ReduceLargeIndicesTests/*.*",
    "ReduceSingleValueTests/*.*",
    "ReduceTests/*.*",
    "RunLengthDecodeTest/*.*",
    "RunLengthEncode/*.*",
    "SegmentedReduce/*.*",
    "SegmentedReduceArgMinMaxSpecialTests/*.*",
    "SegmentedReduceOp/*.*",
    "SelectTests/*.*",
    "ThreadOperationTests/*.*",
    "ThreadOperatorsTests/*.*",
    "UtilPtxTests/*.*",
    "WarpExchangeTest/*.*",
    "WarpLoadTest/*.*",
    "WarpMergeSort/*.*",
    "WarpReduceTests/*.*",
    "WarpScanTests*",
]


def exe_name(name: str) -> str:
    if platform.system() == "Windows":
        return f"{name}.exe"
    return name


def derive_rocm_path(script_dir: Path) -> Path:
    for candidate in (script_dir, *script_dir.parents):
        if (candidate / "bin" / TEST_DIR_NAME / "CTestTestfile.cmake").is_file():
            return candidate
    raise RuntimeError(
        "ROCM_PATH is required when run_hipcub.py is not executed from an "
        "installed ROCm tree containing bin/hipcub/CTestTestfile.cmake."
    )


def get_rocm_lib_dir(rocm_path: Path) -> Path:
    for name in ("lib", "lib64"):
        lib_dir = rocm_path / name
        if lib_dir.is_dir():
            return lib_dir
    return rocm_path / "lib"


def build_env(rocm_path: Path, rocm_bin_dir: Path):
    env = os.environ.copy()
    env["PATH"] = (
        f"{rocm_bin_dir}{os.pathsep}{env['PATH']}"
        if env.get("PATH")
        else str(rocm_bin_dir)
    )
    env["ROCM_PATH"] = str(rocm_path)

    lib_path = get_rocm_lib_dir(rocm_path)
    existing_ld_path = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = (
        f"{lib_path}{os.pathsep}{existing_ld_path}"
        if existing_ld_path
        else str(lib_path)
    )
    return env


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    rocm_path_env = os.getenv("ROCM_PATH")
    rocm_path = (
        Path(rocm_path_env).resolve()
        if rocm_path_env
        else derive_rocm_path(script_dir)
    )
    rocm_bin_dir = Path(os.environ.get("ROCM_BIN_DIR", rocm_path / "bin")).resolve()
    test_dir = rocm_bin_dir / TEST_DIR_NAME
    resource_spec_file = test_dir / "resources.json"
    ctest_file = test_dir / "CTestTestfile.cmake"
    resource_spec_generator = test_dir / exe_name("generate_resource_spec")
    if not ctest_file.is_file():
        raise FileNotFoundError(f"hipCUB CTest file not found at {ctest_file}")
    if not resource_spec_generator.is_file():
        raise FileNotFoundError(
            f"hipCUB resource spec generator not found at {resource_spec_generator}"
        )

    env = build_env(rocm_path, rocm_bin_dir)

    res_gen_cmd = [
        str(resource_spec_generator),
        str(resource_spec_file),
    ]
    logging.info(f"++ Exec [{rocm_path}]$ {format_command(res_gen_cmd)}")
    subprocess.run(res_gen_cmd, cwd=rocm_path, check=True, env=env)

    cmd = [
        "ctest",
        "--test-dir",
        str(test_dir),
        "--output-on-failure",
        "--parallel",
        "8",
        "--resource-spec-file",
        str(resource_spec_file),
        "--timeout",
        "300",
    ]

    if os.getenv("TEST_TYPE", "full") == "quick":
        env["GTEST_FILTER"] = ":".join(QUICK_TESTS)

    logging.info(f"++ Exec [{rocm_path}]$ {format_command(cmd)}")
    subprocess.run(cmd, cwd=rocm_path, check=True, env=env)


if __name__ == "__main__":
    main()
