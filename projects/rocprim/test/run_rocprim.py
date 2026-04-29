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


TEST_DIR_NAME = "rocprim"

TEST_TO_IGNORE = {
    # TODO(#2836): Re-enable gfx110X tests once issues are resolved.
    "gfx110X-all": {
        "windows": [
            "rocprim.block_discontinuity",
            "rocprim.device_merge_sort",
            "rocprim.device_reduce",
        ]
    },
    "gfx1151": {
        "windows": [
            # TODO(#2836): Re-enable test once issues are resolved.
            "rocprim.device_merge_sort",
            # TODO(#2836): Re-enable test once issues are resolved.
            "rocprim.device_radix_sort",
        ]
    },
}

QUICK_TESTS = [
    "*ArgIndexIterator",
    "*BasicTests.GetVersion",
    "*BatchMemcpyTests/*",
    "*BlockScan",
    "*ConfigDispatchTests.*",
    "*ConstantIteratorTests/*",
    "*CountingIteratorTests/*",
    "*DeviceScanTests/*",
    "*DiscardIteratorTests.Less",
    "*ExchangeTests*",
    "*FirstPart",
    "*HipcubBlockRunLengthDecodeTest/*",
    "*Histogram*",
    "*HistogramAtomic*",
    "*HistogramSortInput*",
    "*IntrinsicsTests*",
    "*InvokeResultBinOpTests/*",
    "*InvokeResultUnOpTests/*",
    "*MergeTests/*",
    "*PartitionLargeInputTest/*",
    "*PartitionTests/*",
    "*PredicateIteratorTests.*",
    "*RadixKeyCodecTest.*",
    "*RadixMergeCompareTest/*",
    "*RadixSort/*",
    "*RadixSortIntegral/*",
    "*ReduceByKey*",
    "*ReduceInputArrayTestsFloating",
    "*ReduceInputArrayTestsIntegral/*",
    "*ReducePrecisionTests/*",
    "*ReduceSingleValueTestsFloating",
    "*ReduceSingleValueTestsIntegral",
    "*ReduceTests/*",
    "*ReverseIteratorTests.*",
    "*RunLengthEncode/*",
    "*SecondPart/*",
    "*SegmentedReduce/*",
    "*SelectLargeInputFlaggedTest/*",
    "*SelectTests/*",
    "*ShuffleTestsFloating/*",
    "*ShuffleTestsIntegral*",
    "*SortBitonicTestsIntegral/*",
    "*ThirdPart/*",
    "*ThreadOperationTests/*",
    "*ThreadTests/*",
    "*TransformIteratorTests/*",
    "*TransformTests/*",
    "*VectorizationTests*",
    "*WarpExchangeScatterTest/*",
    "*WarpExchangeTest/*",
    "*WarpLoadTest/*",
    "*WarpReduceTestsFloating/*",
    "*WarpReduceTestsIntegral/*",
    "*WarpScanTests*",
    "*WarpSortShuffleBasedTestsIntegral/*",
    "*ceIntegral/*",
    "*tyIntegral/*",
    "TestHipGraphBasic",
]


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
    test_dir = rocm_bin_dir / TEST_DIR_NAME

    shard_index = int(os.getenv("SHARD_INDEX", "1")) - 1
    total_shards = int(os.getenv("TOTAL_SHARDS", "1"))

    cmd = [
        "ctest",
        "--test-dir",
        str(test_dir),
        "--output-on-failure",
        "--parallel",
        "8",
        "--timeout",
        "900",
        "--repeat",
        "until-pass:6",
        # Start test and stride used for sharding.
        "--tests-information",
        f"{shard_index},,{total_shards}",
    ]

    amdgpu_families = os.getenv("AMDGPU_FAMILIES")
    os_type = platform.system().lower()
    if amdgpu_families in TEST_TO_IGNORE and os_type in TEST_TO_IGNORE[amdgpu_families]:
        ignored_tests = TEST_TO_IGNORE[amdgpu_families][os_type]
        cmd.extend(["--exclude-regex", "|".join(ignored_tests)])

    env = os.environ.copy()
    if os.getenv("TEST_TYPE", "full") == "quick":
        env["GTEST_FILTER"] = ":".join(QUICK_TESTS)

    logging.info(f"++ Exec [{rocm_path}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=rocm_path, check=True, env=env)


if __name__ == "__main__":
    main()
