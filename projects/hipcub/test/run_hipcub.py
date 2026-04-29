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
    if script_dir.name == TEST_DIR_NAME and script_dir.parent.name == "bin":
        return script_dir.parent.parent
    return script_dir.parent.parent


def build_env(rocm_path: Path, rocm_bin_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = (
        f"{rocm_bin_dir}{os.pathsep}{env['PATH']}"
        if env.get("PATH")
        else str(rocm_bin_dir)
    )
    env["ROCM_PATH"] = str(rocm_path)

    lib_path = rocm_path / "lib"
    existing_ld_path = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = (
        f"{lib_path}{os.pathsep}{existing_ld_path}"
        if existing_ld_path
        else str(lib_path)
    )
    return env


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    rocm_path = Path(
        os.environ.get("ROCM_PATH", derive_rocm_path(script_dir))
    ).resolve()
    rocm_bin_dir = Path(os.environ.get("ROCM_BIN_DIR", rocm_path / "bin")).resolve()
    test_dir = rocm_bin_dir / TEST_DIR_NAME
    resource_spec_file = test_dir / "resources.json"

    env = build_env(rocm_path, rocm_bin_dir)

    res_gen_cmd = [
        str(test_dir / exe_name("generate_resource_spec")),
        str(resource_spec_file),
    ]
    logging.info(f"++ Exec [{rocm_path}]$ {shlex.join(res_gen_cmd)}")
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

    logging.info(f"++ Exec [{rocm_path}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=rocm_path, check=True, env=env)


if __name__ == "__main__":
    main()
