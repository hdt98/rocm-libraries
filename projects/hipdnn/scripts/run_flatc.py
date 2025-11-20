# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import sys, subprocess, shutil, os

REQUIRED_VER = "25.9.23"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMAS_DIR = os.path.join(SCRIPT_DIR, "..", "sdk", "schemas")
OUTPUT_DIR = os.path.join(
    SCRIPT_DIR, "..", "sdk", "include", "hipdnn_sdk", "data_objects"
)

# Find flatc in PATH
flatc_path = shutil.which("flatc")
current_ver = ""

if flatc_path:
    try:
        current_ver = subprocess.check_output(
            [flatc_path, "--version"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        pass

if REQUIRED_VER not in current_ver:
    print(
        f'ERROR: flatc version {REQUIRED_VER} required. Found: {current_ver or "None"}',
        file=sys.stderr,
    )
    print("Download the following and include the executable in PATH:", file=sys.stderr)
    print(
        f"  Windows: Download https://github.com/google/flatbuffers/releases/download/v{REQUIRED_VER}/Windows.flatc.binary.zip",
        file=sys.stderr,
    )
    print(
        f"  Linux:   wget https://github.com/google/flatbuffers/releases/download/v{REQUIRED_VER}/Linux.flatc.binary.g++-13.zip",
        file=sys.stderr,
    )
    sys.exit(1)

for f in sys.argv[1:]:
    result = subprocess.run(
        [
            flatc_path,
            "-I",
            SCHEMAS_DIR,
            "--cpp",
            "--gen-object-api",
            "--gen-mutable",
            "--gen-compare",
            "--defaults-json",
            "--scoped-enums",
            "-o",
            OUTPUT_DIR,
            f,
        ]
    )
    if result.returncode != 0:
        sys.exit(1)
