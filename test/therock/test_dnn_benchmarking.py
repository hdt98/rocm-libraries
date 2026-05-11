#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TheRock test runner for dnn-benchmarking.

Installs the dnn-benchmarking wheel into the active venv, configures the
ROCm environment from the unpacked artifact tree, and runs a curated
pytest target.

Environment variables used:
    THEROCK_BIN_DIR: Root of the unpacked artifact tree (contains lib/, bin/, etc.)
    THEROCK_DIR:     Repository root (fallback: three directories above this script)
"""

import logging
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = Path(
    os.environ.get("THEROCK_DIR") or SCRIPT_DIR.parent.parent.parent
).resolve()

logging.basicConfig(level=logging.INFO)

if not THEROCK_BIN_DIR:
    logging.error("THEROCK_BIN_DIR is not set")
    sys.exit(1)

artifact_root = Path(THEROCK_BIN_DIR).resolve().parent
bundle = artifact_root / "share" / "hipdnn" / "dnn-benchmarking"
wheelhouse = bundle / "wheels"
tests_dir = bundle / "tests"

if not bundle.exists():
    logging.error(f"dnn-benchmarking test bundle not found: {bundle}")
    sys.exit(1)
if not tests_dir.exists():
    logging.error(f"dnn-benchmarking tests not found: {tests_dir}")
    sys.exit(1)

env = os.environ.copy()
env["ROCM_PATH"] = str(artifact_root)
env["HIP_PLATFORM"] = "amd"

workspace = Path(env.get("RUNNER_TEMP", str(artifact_root / "tmp"))) / "dnn-benchmarking"
workspace.mkdir(parents=True, exist_ok=True)
env["DNN_BENCH_WORKSPACE"] = str(workspace)
env["PYTHONPYCACHEPREFIX"] = str(workspace / "pycache")
env["XDG_CACHE_HOME"] = str(workspace / "cache")
env["MIOPEN_USER_DB_PATH"] = str(workspace / "miopen_cache")
env["MIOPEN_CUSTOM_CACHE_DIR"] = str(workspace / "miopen_cache")
env["AMD_COMGR_CACHE_DIR"] = str(workspace / "comgr_cache")

plugin_dir = artifact_root / "lib" / "hipdnn_plugins" / "engines"
if plugin_dir.exists():
    env["DNN_BENCHMARKING_HIPDNN_PLUGIN_PATH"] = str(plugin_dir)

if platform.system() == "Windows":
    env["PATH"] = os.pathsep.join(
        [str(artifact_root / "bin"), str(artifact_root / "lib"), env.get("PATH", "")]
    )
else:
    env["PATH"] = os.pathsep.join(
        [str(artifact_root / "bin"), env.get("PATH", "")]
    )
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [
            str(artifact_root / "lib"),
            str(artifact_root / "lib" / "llvm" / "lib"),
            env.get("LD_LIBRARY_PATH", ""),
        ]
    )

# Install dnn-benchmarking (and hipdnn-frontend if bundled) from the wheelhouse.
install_cmd = [
    sys.executable, "-m", "pip", "install",
    "--no-index", "--find-links", str(wheelhouse),
    "dnn-benchmarking[test]",
]
if any(wheelhouse.glob("hipdnn_frontend-*.whl")):
    install_cmd.append("hipdnn-frontend")

logging.info(f"++ Install: {shlex.join(install_cmd)}")
subprocess.run(install_cmd, check=True, env=env)

# Run the no-torch smoke suite: CLI parsing, graph loading, config, reporting,
# shape conversion, and non-GPU tests. These don't require torch or a GPU.
pytest_cmd = [
    sys.executable, "-m", "pytest", "-q",
    str(tests_dir / "unit" / "cli"),
    str(tests_dir / "unit" / "config"),
    str(tests_dir / "unit" / "graph"),
    str(tests_dir / "unit" / "reporting"),
    str(tests_dir / "unit" / "tools"),
    str(tests_dir / "unit" / "validation" / "test_comparison.py"),
    str(tests_dir / "integration" / "test_graph_loading.py"),
]

logging.info(f"++ Test: {shlex.join(pytest_cmd)}")
result = subprocess.run(pytest_cmd, env=env, cwd=str(bundle))
sys.exit(result.returncode)
