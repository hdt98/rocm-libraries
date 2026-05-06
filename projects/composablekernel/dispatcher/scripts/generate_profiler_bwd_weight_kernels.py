#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Generates dispatcher-based backward weight kernels for the CK Profiler.
#
# This script:
# 1. Reads JSON config files (from convert_builder_configs.py)
# 2. Calls unified_grouped_conv_codegen.py --config-file for each JSON
# 3. Generates include_all_grouped_conv_bwd_weight_kernels.hpp
# 4. Generates register_all_grouped_conv_kernels.hpp
#
# Usage:
#   python3 generate_profiler_bwd_weight_kernels.py \
#     --config-dir <path-to-json-configs> \
#     --codegen <path-to-unified_grouped_conv_codegen.py> \
#     --output-dir <generated-kernel-output-dir> \
#     --arch gfx950 \
#     [--config-set tests|profiler]

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


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
    """Collect all generated .hpp kernel headers."""
    headers = sorted(Path(output_dir).glob("grouped_conv_bwd_weight_*.hpp"))
    return headers


def generate_include_all_header(headers, output_dir):
    """Generate include_all_grouped_conv_bwd_weight_kernels.hpp."""
    lines = [
        "// Auto-generated — do not edit",
        "// Includes all generated backward weight kernel headers.",
        "#pragma once",
        "",
    ]
    for h in headers:
        lines.append(f'#include "{h.name}"')
    lines.append("")

    path = Path(output_dir) / "include_all_grouped_conv_bwd_weight_kernels.hpp"
    path.write_text("\n".join(lines))
    print(f"Generated {path} ({len(headers)} includes)")
    return path


def generate_registration_header(headers, output_dir):
    """Generate register_all_grouped_conv_kernels.hpp with registration function."""
    lines = [
        "// Auto-generated — do not edit",
        "// Registration function for all generated backward weight kernels.",
        "#pragma once",
        "",
        '#include "ck_tile/dispatcher/grouped_conv_registry.hpp"',
        '#include "ck_tile/dispatcher/backends/generated_conv_backend.hpp"',
        "",
        "namespace ck_tile {",
        "namespace dispatcher {",
        "",
    ]

    # Build registration function body
    reg_lines = []
    for i, h in enumerate(headers):
        kname = h.stem
        ns = f"ns_{kname}"
        launcher = f"{ns}::{kname}_Launcher"

        ndim = 3 if "_3d_" in kname else 2

        # Parse dtype
        dtype = "fp16"
        for dt in ["fp16", "bf16", "fp32"]:
            if f"_{dt}_" in kname:
                dtype = dt
                break

        # Parse tile, wave, warp from name triplets
        triplets = re.findall(r"_(\d+)x(\d+)x(\d+)", kname)
        tile_m, tile_n, tile_k = 128, 128, 32
        wave_m, wave_n, wave_k = 2, 2, 1
        warp_m, warp_n, warp_k = 32, 32, 16
        if len(triplets) >= 1:
            tile_m, tile_n, tile_k = int(triplets[0][0]), int(triplets[0][1]), int(triplets[0][2])
        if len(triplets) >= 2:
            wave_m, wave_n, wave_k = int(triplets[1][0]), int(triplets[1][1]), int(triplets[1][2])
        if len(triplets) >= 3:
            warp_m, warp_n, warp_k = int(triplets[2][0]), int(triplets[2][1]), int(triplets[2][2])

        # Parse pipeline, scheduler, epilogue from name
        pipeline = "compv3"
        for p in ["compv1", "compv2", "compv3", "compv4", "compv5", "mem"]:
            if f"_{p}_" in kname:
                pipeline = p
                break
        scheduler = "interwave" if "interwave" in kname else "intrawave"
        epilogue = "cshuffle"

        # Parse vector sizes from name: vec{a}_{b}_{c} format
        vec_a, vec_b, vec_c = 4, 8, 8
        vec_match = re.search(r"_vec(\d+)_(\d+)_(\d+)", kname)
        if vec_match:
            vec_a = int(vec_match.group(1))
            vec_b = int(vec_match.group(2))
            vec_c = int(vec_match.group(3))

        block_per_cu = 1
        num_wave_groups = 1
        # Parse num_groups_to_merge from name: _gm{N}_ format
        num_groups_to_merge = 1
        gm_match = re.search(r"_gm(\d+)", kname)
        if gm_match:
            num_groups_to_merge = int(gm_match.group(1))

        reg_lines.append(f"    // Kernel {i}: {kname}")
        reg_lines.append("    {")
        reg_lines.append(f"        GroupedConvKernelKey key;")
        reg_lines.append(f'        key.dtype_in     = "{dtype}";')
        reg_lines.append(f'        key.dtype_wei    = "{dtype}";')
        reg_lines.append(f'        key.dtype_out    = "{dtype}";')
        reg_lines.append(f'        key.layout       = "nhwgc";')
        reg_lines.append(f"        key.ndim_spatial = {ndim};")
        reg_lines.append(f"        key.op           = GroupedConvOp::BackwardWeight;")
        reg_lines.append(f"        key.tile_m       = {tile_m};")
        reg_lines.append(f"        key.tile_n       = {tile_n};")
        reg_lines.append(f"        key.tile_k       = {tile_k};")
        reg_lines.append(f"        key.wave_m       = {wave_m};")
        reg_lines.append(f"        key.wave_n       = {wave_n};")
        reg_lines.append(f"        key.wave_k       = {wave_k};")
        reg_lines.append(f"        key.warp_m       = {warp_m};")
        reg_lines.append(f"        key.warp_n       = {warp_n};")
        reg_lines.append(f"        key.warp_k       = {warp_k};")
        reg_lines.append(f'        key.pipeline     = "{pipeline}";')
        reg_lines.append(f'        key.scheduler    = "{scheduler}";')
        reg_lines.append(f'        key.epilogue     = "{epilogue}";')
        reg_lines.append(f"        key.vector_size_a      = {vec_a};")
        reg_lines.append(f"        key.vector_size_b      = {vec_b};")
        reg_lines.append(f"        key.vector_size_c      = {vec_c};")
        reg_lines.append(f"        key.block_per_cu       = {block_per_cu};")
        reg_lines.append(f"        key.num_wave_groups    = {num_wave_groups};")
        reg_lines.append(f"        key.num_groups_to_merge = {num_groups_to_merge};")
        reg_lines.append(f"        key.arch         = arch;")
        reg_lines.append(
            f"        auto run_fn = backends::make_conv_bwd_weight_run_fn<{launcher}, {ndim}>();"
        )
        reg_lines.append(
            f"        auto is_supported_fn = backends::make_conv_bwd_weight_is_supported_fn<{launcher}, {ndim}>();"
        )
        reg_lines.append(
            f"#ifdef CK_EXPERIMENTAL_BUILDER"
        )
        reg_lines.append(
            f"        auto instance_str = backends::get_instance_string<{launcher}>();"
        )
        reg_lines.append(
            f'        auto inst = std::make_shared<GroupedConvKernelInstance>(key, "{kname}", std::move(run_fn), std::move(is_supported_fn), instance_str);'
        )
        reg_lines.append(
            f"#else"
        )
        reg_lines.append(
            f'        auto inst = std::make_shared<GroupedConvKernelInstance>(key, "{kname}", std::move(run_fn), std::move(is_supported_fn));'
        )
        reg_lines.append(
            f"#endif"
        )
        reg_lines.append(f"        registry.register_kernel(key, inst);")
        reg_lines.append("    }")

    lines.append("inline void register_all_grouped_conv_bwd_weight_kernels(")
    lines.append("    GroupedConvRegistry& registry, const std::string& arch)")
    lines.append("{")
    lines.extend(reg_lines)
    lines.append("}")
    lines.append("")

    # Convenience overload that uses singleton registry
    lines.append("inline void register_all_grouped_conv_bwd_weight_kernels(const std::string& arch)")
    lines.append("{")
    lines.append("    auto& registry = GroupedConvRegistry::instance();")
    lines.append("    register_all_grouped_conv_bwd_weight_kernels(registry, arch);")
    lines.append("}")
    lines.append("")
    lines.append("} // namespace dispatcher")
    lines.append("} // namespace ck_tile")
    lines.append("")

    path = Path(output_dir) / "register_all_grouped_conv_kernels.hpp"
    path.write_text("\n".join(lines))
    print(f"Generated {path} ({len(headers)} registrations)")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Generate dispatcher-based bwd_weight kernels for CK Profiler."
    )
    parser.add_argument(
        "--config-dir", required=True,
        help="Directory containing JSON config files (e.g. configs/grouped_conv/backward_weight/tests/)"
    )
    parser.add_argument(
        "--codegen", required=True,
        help="Path to unified_grouped_conv_codegen.py"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for generated kernel .hpp files"
    )
    parser.add_argument(
        "--arch", default="gfx950",
        help="Target GPU architecture"
    )
    parser.add_argument(
        "--config-set", default="tests", choices=["tests", "profiler"],
        help="Config set to use (tests or profiler)"
    )

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

    # Find all JSON configs
    json_configs = sorted(config_dir.glob("*.json"))
    if not json_configs:
        print(f"ERROR: No JSON config files in {config_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(json_configs)} config files in {config_dir}")

    # Generate kernels from each config
    success = True
    for config_file in json_configs:
        print(f"Generating from {config_file.name}...")
        if not generate_kernels_from_config(codegen, config_file, output_dir, args.arch):
            success = False

    if not success:
        print("ERROR: Some kernel generations failed", file=sys.stderr)
        sys.exit(1)

    # Collect generated headers
    headers = collect_kernel_headers(output_dir)
    print(f"Found {len(headers)} generated kernel headers")

    if not headers:
        print("ERROR: No kernel headers generated", file=sys.stderr)
        sys.exit(1)

    # Generate the two profiler headers
    generate_include_all_header(headers, output_dir)
    generate_registration_header(headers, output_dir)

    print(f"\nDone. {len(headers)} kernels ready in {output_dir}")


if __name__ == "__main__":
    main()
