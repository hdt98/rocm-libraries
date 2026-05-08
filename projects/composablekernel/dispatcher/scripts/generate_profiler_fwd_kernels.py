#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Generates dispatcher-based forward kernels for the CK Profiler.
#
# This script:
# 1. Reads JSON config files (from convert_builder_configs.py)
# 2. Calls unified_grouped_conv_codegen.py --config-file for each JSON
# 3. Generates include_all_grouped_conv_fwd_kernels.hpp
# 4. Generates chunked register_*_chunk_N.cpp files + register_all_grouped_conv_kernels.cpp
#
# Usage:
#   python3 generate_profiler_fwd_kernels.py \
#     --config-dir <path-to-json-configs> \
#     --codegen <path-to-unified_grouped_conv_codegen.py> \
#     --output-dir <generated-kernel-output-dir> \
#     --arch gfx950 \
#     [--config-set tests|profiler]

import argparse
import subprocess
import sys
from pathlib import Path

from registration_codegen import generate_chunked_registration


def generate_kernels_from_config(codegen_script, config_file, output_dir, arch):
    """Run unified_grouped_conv_codegen.py --config-file for a single JSON config."""
    cmd = [
        sys.executable,
        str(codegen_script),
        "--config-file", str(config_file),
        "--arch", arch,
        "--output", str(output_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR generating from {config_file}:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return False
    if result.stdout.strip():
        print(result.stdout.strip())
    return True


def collect_kernel_headers(output_dir):
    """Collect all generated .hpp forward kernel headers."""
    headers = sorted(Path(output_dir).glob("grouped_conv_fwd_*.hpp"))
    return headers


def generate_include_all_header(headers, output_dir):
    """Generate include_all_grouped_conv_fwd_kernels.hpp."""
    lines = [
        "// Auto-generated — do not edit",
        "// Includes all generated forward kernel headers.",
        "#pragma once",
        "",
    ]
    for h in headers:
        lines.append(f'#include "{h.name}"')
    lines.append("")

    path = Path(output_dir) / "include_all_grouped_conv_fwd_kernels.hpp"
    path.write_text("\n".join(lines))
    print(f"Generated {path} ({len(headers)} includes)")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Generate dispatcher-based forward kernels for CK Profiler."
    )
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--codegen", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--config-set", default="tests", choices=["tests", "profiler"])

    args = parser.parse_args()

    config_dir = Path(args.config_dir) / args.config_set
    codegen = Path(args.codegen)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config_dir.exists():
        print(f"ERROR: Config directory not found: {config_dir}", file=sys.stderr)
        sys.exit(1)
    if not codegen.exists():
        print(f"ERROR: Codegen script not found: {codegen}", file=sys.stderr)
        sys.exit(1)

    json_configs = sorted(config_dir.glob("*.json"))
    if not json_configs:
        print(f"ERROR: No JSON config files in {config_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(json_configs)} config files in {config_dir}")

    success = True
    for config_file in json_configs:
        print(f"Generating from {config_file.name}...")
        if not generate_kernels_from_config(codegen, config_file, output_dir, args.arch):
            success = False

    if not success:
        print("ERROR: Some kernel generations failed", file=sys.stderr)
        sys.exit(1)

    headers = collect_kernel_headers(output_dir)
    print(f"Found {len(headers)} generated kernel headers")

    if not headers:
        print("ERROR: No kernel headers generated", file=sys.stderr)
        sys.exit(1)

    generate_include_all_header(headers, output_dir)
    generate_chunked_registration(
        headers, output_dir,
        variant="fwd",
        op_enum="GroupedConvOp::Forward",
        run_fn_maker="backends::make_conv_fwd_run_fn",
        is_supported_fn_maker="backends::make_conv_fwd_is_supported_fn",
        register_fn_name="register_all_grouped_conv_fwd_kernels",
    )

    print(f"\nDone. {len(headers)} kernels ready in {output_dir}")


if __name__ == "__main__":
    main()
