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


TEST_EXE = "rocsparse-test"


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
        "Could not derive ROCM_PATH from an installed rocsparse-test layout. "
        "Set ROCM_PATH explicitly."
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    rocm_path_env = os.getenv("ROCM_PATH")
    rocm_path = Path(rocm_path_env).resolve() if rocm_path_env else derive_rocm_path(script_dir)
    rocm_bin_dir = Path(os.getenv("ROCM_BIN_DIR") or rocm_path / "bin").resolve()
    test_exe_path = rocm_bin_dir / exe_name(TEST_EXE)
    smoke_yaml = rocm_path / "share" / "rocsparse" / "test" / "rocsparse_smoke.yaml"
    if not test_exe_path.is_file():
        raise FileNotFoundError(f"Could not find test executable: {test_exe_path}")
    if not smoke_yaml.is_file():
        raise FileNotFoundError(f"Could not find smoke test data: {smoke_yaml}")

    env = os.environ.copy()
    # GitHub Actions shard arrays are 1-indexed; GTest shard indexes are 0-indexed.
    env["GTEST_SHARD_INDEX"] = str(int(os.getenv("SHARD_INDEX", "1")) - 1)
    env["GTEST_TOTAL_SHARDS"] = os.getenv("TOTAL_SHARDS", "1")

    cmd = [str(test_exe_path)]
    output_artifacts_dir = os.getenv("OUTPUT_ARTIFACTS_DIR")
    if output_artifacts_dir:
        cmd.extend(
            [
                "--matrices-dir",
                str(Path(output_artifacts_dir) / "clients" / "matrices"),
            ]
        )

    # The current quick and full paths both use the smoke suite.
    cmd.extend(["--yaml", str(smoke_yaml)])

    logging.info(f"++ Exec [{rocm_path}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=rocm_path, check=True, env=env)


if __name__ == "__main__":
    main()
