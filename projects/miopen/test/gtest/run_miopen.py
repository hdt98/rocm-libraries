#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess

logging.basicConfig(level=logging.INFO)

VALID_TEST_CATEGORIES = {"quick", "standard", "comprehensive", "full"}
TEST_DIR_NAME = "MIOpen"
CTEST_TIMEOUT_SECONDS = 7200


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
    logging.info("++ Loaded shared test utilities from %s", utils_path)
    return module


def derive_rocm_path(script_dir: Path) -> Path:
    for candidate in (script_dir, *script_dir.parents):
        if (candidate / "bin" / TEST_DIR_NAME / "CTestTestfile.cmake").is_file():
            return candidate
        if (
            candidate.name == "bin"
            and (candidate / TEST_DIR_NAME / "CTestTestfile.cmake").is_file()
        ):
            return candidate.parent
    raise RuntimeError(
        "ROCM_PATH is required when run_miopen.py is not executed from an "
        "installed ROCm tree containing bin/MIOpen/CTestTestfile.cmake."
    )


def find_matching_gpu_arch(gpu_arch: str, available_gpu_archs: set[str]) -> str | None:
    if gpu_arch in available_gpu_archs:
        return gpu_arch

    for i in range(len(gpu_arch) - 1, 4, -1):
        pattern = gpu_arch[:i] + "X"
        if pattern in available_gpu_archs:
            return pattern

    return None


def normalize_test_category(test_type: str | None) -> str:
    if not test_type:
        return "quick"
    category = test_type.strip().lower()
    return category if category in VALID_TEST_CATEGORIES else "quick"


def extract_gpu_arch(amdgpu_families: str | None) -> str:
    if not amdgpu_families:
        return ""
    match = re.search(r"gfx[0-9a-zA-Z]+", amdgpu_families)
    return match.group(0).lower() if match else ""


def positive_int(name: str, value: str | int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be an integer, got {value!r}") from e
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1, got {parsed}")
    return parsed


def ctest_shard_args(shard_index: str | int, total_shards: str | int) -> list[str]:
    parsed_shard_index = positive_int("shard_index", shard_index)
    parsed_total_shards = positive_int("total_shards", total_shards)
    if parsed_shard_index > parsed_total_shards:
        raise ValueError(
            "shard_index must be less than or equal to total_shards, "
            f"got {parsed_shard_index} > {parsed_total_shards}"
        )
    return ["--tests-information", f"{parsed_shard_index},,{parsed_total_shards}"]


def gtest_shard_env(shard_index: str | int, total_shards: str | int) -> dict[str, str]:
    parsed_shard_index = positive_int("shard_index", shard_index)
    parsed_total_shards = positive_int("total_shards", total_shards)
    if parsed_shard_index > parsed_total_shards:
        raise ValueError(
            "shard_index must be less than or equal to total_shards, "
            f"got {parsed_shard_index} > {parsed_total_shards}"
        )
    return {
        "GTEST_SHARD_INDEX": str(parsed_shard_index - 1),
        "GTEST_TOTAL_SHARDS": str(parsed_total_shards),
    }


def count_ctest_tests(test_dir: Path) -> int:
    result = subprocess.run(
        ["ctest", "-N", "--test-dir", str(test_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    return sum(
        1 for line in result.stdout.splitlines() if re.search(r"Test\s+#\d+:", line)
    )


def read_ctest_labels(test_dir: Path) -> set[str]:
    result = subprocess.run(
        ["ctest", "--print-labels", "--test-dir", str(test_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def discover_ctest_labels(test_dir: Path) -> tuple[set[str], set[str]]:
    if count_ctest_tests(test_dir) == 0:
        raise RuntimeError(f"No CTest tests found in {test_dir}")

    gpu_archs = set()
    exclude_labels = set()
    for label in read_ctest_labels(test_dir):
        if label.startswith("ex_gpu_"):
            gpu_arch = label.removeprefix("ex_gpu_")
            if gpu_arch.startswith("gfx"):
                gpu_archs.add(gpu_arch)
        elif label.endswith("_exclude"):
            exclude_labels.add(label)
    return gpu_archs, exclude_labels


def build_ctest_label_args(
    category: str,
    gpu_arch: str,
    available_gpu_archs: set[str],
    exclude_labels: set[str],
) -> list[str]:
    args = ["-L", category]
    exclude_patterns = []

    category_exclude_label = f"{category}_exclude"
    if category_exclude_label in exclude_labels:
        exclude_patterns.append(category_exclude_label)

    if gpu_arch in ("", "generic", "none"):
        exclude_patterns.append("ex_gpu")
    else:
        matching_arch = find_matching_gpu_arch(gpu_arch, available_gpu_archs)
        if matching_arch:
            args.extend(["-L", f"ex_gpu_{matching_arch}"])
        else:
            exclude_patterns.append("ex_gpu")

    if exclude_patterns:
        args.extend(["-LE", "|".join(exclude_patterns)])
    return args


def ctest_parallel_count() -> int:
    amdgpu_families = os.getenv("AMDGPU_FAMILIES", "")
    if "gfx1152" in amdgpu_families or "gfx1153" in amdgpu_families:
        return 4
    return 8


def build_fallback_ctest_command(test_dir: Path) -> list[str]:
    category = normalize_test_category(os.getenv("TEST_TYPE", "quick"))
    gpu_arch = extract_gpu_arch(os.getenv("AMDGPU_FAMILIES"))
    available_gpu_archs, exclude_labels = discover_ctest_labels(test_dir)

    cmd = ["ctest"]
    cmd.extend(
        build_ctest_label_args(
            category,
            gpu_arch,
            available_gpu_archs,
            exclude_labels,
        )
    )
    cmd.extend(
        [
            "--output-on-failure",
            "--parallel",
            str(ctest_parallel_count()),
            "--timeout",
            str(CTEST_TIMEOUT_SECONDS),
            "--test-dir",
            str(test_dir),
            "-V",
        ]
    )
    cmd.extend(
        ctest_shard_args(os.getenv("SHARD_INDEX", 1), os.getenv("TOTAL_SHARDS", 1))
    )
    return cmd


def build_env(rocm_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ROCM_PATH"] = str(rocm_path)
    env.update(
        gtest_shard_env(os.getenv("SHARD_INDEX", 1), os.getenv("TOTAL_SHARDS", 1))
    )
    return env


def rocm_bin_dir(rocm_path: Path) -> Path:
    bin_dir_env = os.getenv("ROCM_BIN_DIR") or os.getenv("THEROCK_BIN_DIR")
    return Path(bin_dir_env).resolve() if bin_dir_env else rocm_path / "bin"


def run_with_shared_utils(test_utils, rocm_path: Path, test_dir: Path) -> int:
    settings = test_utils.TestRunSettings.from_env(
        test_dir=test_dir,
        rocm_path=rocm_path,
    ).with_ctest(
        parallel=ctest_parallel_count(),
        timeout_seconds=CTEST_TIMEOUT_SECONDS,
    )
    env = test_utils.build_test_env(settings, base_env=os.environ)
    result = test_utils.run_ctest(
        settings,
        cwd=rocm_path,
        env=env,
        check=False,
        discover_labels=True,
    )
    return result.returncode


def run_with_fallback(rocm_path: Path, test_dir: Path) -> int:
    cmd = build_fallback_ctest_command(test_dir)
    logging.info("++ Exec [%s]$ %s", rocm_path, shlex.join(cmd))
    result = subprocess.run(cmd, cwd=rocm_path, env=build_env(rocm_path), check=False)
    return result.returncode


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    rocm_path_env = os.getenv("ROCM_PATH")
    rocm_path = (
        Path(rocm_path_env).resolve() if rocm_path_env else derive_rocm_path(script_dir)
    )
    test_dir = rocm_bin_dir(rocm_path) / TEST_DIR_NAME
    if not (test_dir / "CTestTestfile.cmake").is_file():
        raise FileNotFoundError(f"MIOpen CTest metadata not found in {test_dir}")

    test_utils = load_shared_test_utils()
    if test_utils is not None:
        return run_with_shared_utils(test_utils, rocm_path, test_dir)
    return run_with_fallback(rocm_path, test_dir)


if __name__ == "__main__":
    raise SystemExit(main())
