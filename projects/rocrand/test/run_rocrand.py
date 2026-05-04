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


TEST_DIR_NAME = "rocRAND"

# If quick tests are enabled, run quick tests only. Otherwise, run the full suite.
QUICK_TESTS = [
    "*basic_tests*",
    "*config_dispatch_tests.*",
    "*cpp_utils_tests.*",
    "*cpp_wrapper*",
    "*distributions/*",
    "*generate_host_test/*",
    "*generate_long_long_tests/*",
    "*generate_normal_tests/*",
    "*generate_uniform_tests/*",
    "*generator_type_tests.*",
    "*kernel_lfsr113*",
    "*kernel_lfsr113_poisson/*",
    "*kernel_mrg/*",
    "*kernel_mtgp32*",
    "*kernel_mtgp32_poisson/*",
    "*kernel_philox4x32_10*",
    "*kernel_philox4x32_10_poisson/*",
    "*kernel_scrambled_sobol32*",
    "*kernel_scrambled_sobol32_poisson/*",
    "*kernel_scrambled_sobol64*",
    "*kernel_scrambled_sobol64_poisson/*",
    "*kernel_sobol32*",
    "*kernel_sobol32_poisson/*",
    "*kernel_sobol64*",
    "*kernel_sobol64_poisson/*",
    "*kernel_threefry2x32_20*",
    "*kernel_threefry2x32_20_poisson/*",
    "*kernel_threefry2x64_20*",
    "*kernel_threefry2x64_20_poisson/*",
    "*kernel_threefry4x32_20*",
    "*kernel_threefry4x32_20_poisson/*",
    "*kernel_threefry4x64_20*",
    "*kernel_threefry4x64_20_poisson/*",
    "*kernel_xorwow*",
    "*kernel_xorwow_poisson/*",
    "*lfsr113_engine_api_tests.*",
    "*lfsr113_generator/*",
    "*lfsr113_generator_prng_tests/*",
    "*linkage_tests.*",
    "*log_normal_distribution_tests.*",
    "*log_normal_tests.*",
    "*mrg/*",
    "*mrg_generator_prng_tests.*",
    "*mrg_log_normal_distribution_tests/*",
    "*mrg_normal_distribution_tests/*",
    "*mrg_prng_engine_tests/*",
    "*mrg_uniform_distribution_tests/*",
    "*mtgp32_generator/*",
    "*normal_distribution_tests.*",
    "*philox4x32_10_generator/*",
    "*philox_prng_state_tests.*",
    "*poisson_distribution_tests/*",
    "*poisson_tests.*",
    "*rocrand_generate_tests.*",
    "*rocrand_hipgraph_generate_tests.*",
    "*sobol_log_normal_distribution_tests/*",
    "*sobol_normal_distribution_tests.*",
    "*sobol_qrng_tests/*",
    "*threefry2x32_20_generator/*",
    "*threefry2x64_20_generator/*",
    "*threefry4x32_20_generator/*",
    "*threefry4x64_20_generator/*",
    "*threefry_prng_state_tests.*",
    "*xorwow_engine_type_test.*",
    "*xorwow_generator/*",
    "-*basic_tests/rocrand_basic_tests.rocrand_create_destroy_generator_test/10*",
]


def derive_rocm_path(script_dir: Path) -> Path:
    for candidate in (script_dir, *script_dir.parents):
        if (candidate / "bin" / TEST_DIR_NAME / "CTestTestfile.cmake").is_file():
            return candidate
    raise RuntimeError(
        "ROCM_PATH is required when run_rocrand.py is not executed from an "
        "installed ROCm tree containing bin/rocRAND/CTestTestfile.cmake."
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
            f"rocRAND CTest file not found at {test_dir / 'CTestTestfile.cmake'}"
        )

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
        "until-pass:3",
    ]

    env = os.environ.copy()
    if os.getenv("TEST_TYPE", "full") == "quick":
        env["GTEST_FILTER"] = ":".join(QUICK_TESTS)

    logging.info(f"++ Exec [{rocm_path}]$ {format_command(cmd)}")
    subprocess.run(cmd, cwd=rocm_path, check=True, env=env)


if __name__ == "__main__":
    main()
