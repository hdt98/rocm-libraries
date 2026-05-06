#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Converts CK Builder .conf config files to JSON format for the CK Dispatcher codegen.
#
# Usage:
#   python3 convert_builder_configs.py \
#     --input ../experimental/.../configs/backward_weight/profiler/nhwgc_bf16.conf \
#     --output configs/grouped_conv/backward_weight/profiler/nhwgc_bf16.json \
#     --variant bwd_weight --layout nhwgc --datatype bf16 --ndim 2
#
# Or convert all configs at once:
#   python3 convert_builder_configs.py --convert-all

import argparse
import json
import sys
from pathlib import Path


def get_dtype_str(problem_name):
    """Map problem name to dtype string used in get_k_mfma."""
    if "fp32" in problem_name:
        return "float"
    elif "fp16" in problem_name:
        return "ck_tile::half_t"
    elif "bf16" in problem_name:
        return "ck_tile::bf16_t"
    else:
        raise RuntimeError(f"Cannot parse data type from problem name: {problem_name}")


def get_k_mfma(dtype, m_per_xdl, n_per_xdl):
    """Get k dimension for MFMA instruction based on dtype and xdl tile."""
    if m_per_xdl != n_per_xdl:
        raise RuntimeError("m_per_xdl != n_per_xdl not supported")
    if dtype == "float":
        return 2 if m_per_xdl == 32 else 4
    else:
        return 8 if m_per_xdl == 32 else 16


def check_vectors(a, b, c):
    """Validate vector sizes (must be 1 or even)."""
    for v in [a, b, c]:
        if v != 1 and v % 2 != 0:
            return False
    return True


def parse_instance_string(instance_string):
    """Parse instance string, treating Seq(...) as a single parameter."""
    params = []
    current_param = ""
    paren_depth = 0
    for char in instance_string:
        if char == '(':
            paren_depth += 1
            current_param += char
        elif char == ')':
            paren_depth -= 1
            current_param += char
        elif char == ',' and paren_depth == 0:
            params.append(current_param.strip())
            current_param = ""
        else:
            current_param += char
    if current_param.strip():
        params.append(current_param.strip())
    return params


def map_pipeline_version(version_str):
    """Map CK Builder pipeline version to dispatcher pipeline string."""
    mapping = {
        "V1": "compv1",
        "V2": "compv2",
        "V3": "compv3",
        "V4": "compv4",
        "V5": "compv5",
        "V6": "compv6",
        "ASYNC_V1": "mem",
        "ASYNC_V4": "mem",
    }
    return mapping.get(version_str, version_str.lower())


def map_scheduler(scheduler_str):
    """Map CK Builder scheduler to dispatcher scheduler string."""
    if "Intrawave" in scheduler_str:
        return "intrawave"
    elif "Interwave" in scheduler_str:
        return "interwave"
    return scheduler_str.lower()


def map_specialization(spec_str):
    """Map CK Builder specialization to dispatcher specialization string."""
    mapping = {
        "Default": "default",
        "OddC": "default",
        "Filter1x1Pad0": "filter1x1_pad0",
        "Filter1x1Stride1Pad0": "filter1x1_stride1_pad0",
        "Filter3x3": "filter3x3",
    }
    return mapping.get(spec_str, spec_str.lower())


def parse_bwd_weight_instance(instance_id, instance, problem_name):
    """Parse a single backward weight instance string. Returns dict or None if skipped."""
    if "#" in instance or ";" in instance:
        return None

    instance = instance.strip()
    if not instance:
        return None

    device_op_name = instance.split("<")[0]
    start = instance.index('<') + 1
    end = instance.rindex('>')
    params_str = instance[start:end]
    args = parse_instance_string(params_str)

    direct_load = False
    is_v3_instance = "Xdl_CShuffleV3" in instance
    is_two_stage_instance = "TwoStage" in instance
    is_explicit_gemm = "Explicit" in device_op_name

    if is_explicit_gemm:
        gemm_params = instance.split("<")[2].split(">")[1].split(",")
        args = [param.split(":")[1].strip() for param in gemm_params]

        spec = "Filter1x1Stride1Pad0"
        block_size = int(args[0])

        mnk = args[1].split("x")
        m_per_block, n_per_block, k_per_block = int(mnk[0]), int(mnk[1]), int(mnk[2])

        wave_tile = args[2].split("x")
        m_per_xdl, n_per_xdl = int(wave_tile[0]), int(wave_tile[1])

        k1_values = args[3].split("x")
        k1 = min(int(k1_values[0]), int(k1_values[1]))

        wave_map = args[4].split("x")
        m_xdl_per_wave, n_xdl_per_wave = int(wave_map[0]), int(wave_map[1])

        vector_read = args[5].split("x")
        a_scalar_per_vector = int(vector_read[0])
        b_scalar_per_vector = int(vector_read[1])
        c_seq = [int(x) for x in vector_read[2].strip("Seq").strip("(").strip(")").split(",")]
        if len(set(c_seq)) != 1:
            raise RuntimeError(f"c_scalar_per_vector not uniform for instance {instance_id}")
        c_scalar_per_vector = c_seq[0]

        num_groups_to_merge = 1
        block_gemm_pipeline_scheduler = args[6]
        blk_gemm_pipeline_version = args[7]
    else:
        spec = args[11]
        block_size = int(args[12])
        m_per_block = int(args[13])
        n_per_block = int(args[14])
        k1 = int(args[16])
        m_per_xdl = int(args[17])
        n_per_xdl = int(args[18])
        m_xdl_per_wave = int(args[19])
        n_xdl_per_wave = int(args[20])
        a_scalar_per_vector = int(args[25])
        b_scalar_per_vector = int(args[32])
        c_scalar_per_vector = int(args[38])

        if is_v3_instance or is_two_stage_instance:
            k_per_block = int(args[15])
        else:
            k0_per_block = int(args[15])
            k_per_block = k0_per_block * k1

        if is_v3_instance:
            if len(args) != 45:
                raise RuntimeError(f"V3 instance expected 45 params, got {len(args)}")
            direct_load = int(args[43]) == 1
            num_groups_to_merge = int(args[44])
            block_gemm_pipeline_scheduler = args[39]
            blk_gemm_pipeline_version = args[40]
        elif is_two_stage_instance:
            if len(args) != 46:
                raise RuntimeError(f"TwoStage instance expected 46 params, got {len(args)}")
            num_groups_to_merge = int(args[41]) if isinstance(args[41], int) or args[41].isdigit() else 1
            block_gemm_pipeline_scheduler = args[39]
            blk_gemm_pipeline_version = args[40]
        else:
            if len(args) != 43:
                raise RuntimeError(f"V1 instance expected 43 params, got {len(args)}")
            num_groups_to_merge = 1
            block_gemm_pipeline_scheduler = "Intrawave"
            blk_gemm_pipeline_version = "v1"

    # Common processing
    if block_gemm_pipeline_scheduler not in ["Intrawave", "Interwave"]:
        raise RuntimeError(f"Invalid scheduler: {block_gemm_pipeline_scheduler}")
    if blk_gemm_pipeline_version not in ["v1", "v2", "v3", "v4", "v5"]:
        raise RuntimeError(f"Invalid pipeline version: {blk_gemm_pipeline_version}")

    split_image = "Large" in instance
    double_smem_buffer = blk_gemm_pipeline_version == "v4"
    scheduler = block_gemm_pipeline_scheduler
    pipeline_version = blk_gemm_pipeline_version.upper()

    # Old CK V5 maps to V6 for CK Tile
    if pipeline_version == "V5":
        pipeline_version = "V6"

    if direct_load:
        if pipeline_version == "V1":
            pipeline_version = "ASYNC_V1"
        elif pipeline_version == "V4":
            pipeline_version = "ASYNC_V4"
        else:
            raise RuntimeError(f"Unsupported pipeline for direct load: {pipeline_version}")

    m_warp = int(m_per_block / (m_per_xdl * m_xdl_per_wave))
    n_warp = int(n_per_block / (n_per_xdl * n_xdl_per_wave))
    warp_size = 64
    k_warp = int(block_size / (warp_size * m_warp * n_warp))
    dtype = get_dtype_str(problem_name)
    k_per_xdl = max(k1, get_k_mfma(dtype, m_per_xdl, n_per_xdl))

    # Apply same skip rules as generate_instances.py
    if not check_vectors(a_scalar_per_vector, b_scalar_per_vector, c_scalar_per_vector):
        print(f"Skipping instance {instance_id}: irregular vector sizes")
        return None
    if pipeline_version == "V6":
        print(f"Skipping instance {instance_id}: V6 not supported")
        return None
    if m_per_block > (warp_size * a_scalar_per_vector) or n_per_block > (warp_size * b_scalar_per_vector):
        print(f"Skipping instance {instance_id}: multiple warps per continuous tile dim")
        return None

    if is_explicit_gemm:
        if dtype != "float" and c_scalar_per_vector % 2 != 0:
            is_two_stage_instance = True

    return {
        "id": instance_id,
        "tile_m": m_per_block,
        "tile_n": n_per_block,
        "tile_k": k_per_block,
        "warp_m": m_warp,
        "warp_n": n_warp,
        "warp_k": k_warp,
        "warp_tile_m": m_per_xdl,
        "warp_tile_n": n_per_xdl,
        "warp_tile_k": k_per_xdl,
        "vector_size_a": a_scalar_per_vector,
        "vector_size_b": b_scalar_per_vector,
        "vector_size_c": c_scalar_per_vector,
        "pipeline": map_pipeline_version(pipeline_version),
        "scheduler": map_scheduler(scheduler),
        "epilogue": "cshuffle",
        "double_smem_buffer": double_smem_buffer,
        "num_groups_to_merge": num_groups_to_merge,
        "block_per_cu": 1,
        "num_wave_groups": 1,
        "specialization": map_specialization(spec),
        "two_stage": is_two_stage_instance,
        "explicit_gemm": is_explicit_gemm,
        "split_image": split_image,
    }


def convert_config_file(input_path, variant, layout, datatype, ndim):
    """Convert a single CK Builder .conf file to JSON format."""
    with open(input_path, "r") as f:
        lines = f.readlines()

    # Determine problem_name for dtype detection
    problem_name = f"grouped_convolution_{variant}_tile_{layout}_{datatype}"

    instances = []
    for instance_id, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        if variant == "bwd_weight":
            result = parse_bwd_weight_instance(instance_id, line, problem_name)
        else:
            raise RuntimeError(f"Variant '{variant}' conversion not yet implemented. "
                             f"Only 'bwd_weight' is supported in this version.")

        if result is not None:
            instances.append(result)

    output = {
        "variant": variant,
        "ndim_spatial": ndim,
        "layout": layout,
        "datatype": datatype,
        "instances": instances,
    }

    print(f"Converted {len(instances)} instances from {input_path}")
    return output


def convert_all(builder_configs_dir, output_dir):
    """Convert all backward_weight config files."""
    builder_dir = Path(builder_configs_dir)
    out_dir = Path(output_dir)

    configs = [
        ("nhwgc_fp32", "nhwgc", "fp32", 2),
        ("nhwgc_fp16", "nhwgc", "fp16", 2),
        ("nhwgc_bf16", "nhwgc", "bf16", 2),
        ("ndhwgc_fp32", "ndhwgc", "fp32", 3),
        ("ndhwgc_fp16", "ndhwgc", "fp16", 3),
        ("ndhwgc_bf16", "ndhwgc", "bf16", 3),
    ]

    for prefix in ["tests", "profiler"]:
        for config_name, layout, datatype, ndim in configs:
            input_path = builder_dir / "backward_weight" / prefix / f"{config_name}.conf"
            if not input_path.exists():
                print(f"Skipping {input_path} (not found)")
                continue

            output_path = out_dir / "backward_weight" / prefix / f"{config_name}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            result = convert_config_file(input_path, "bwd_weight", layout, datatype, ndim)

            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CK Builder .conf config files to JSON for CK Dispatcher codegen."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Single file conversion
    single = subparsers.add_parser("convert", help="Convert a single config file")
    single.add_argument("--input", required=True, help="Input .conf file path")
    single.add_argument("--output", required=True, help="Output .json file path")
    single.add_argument("--variant", required=True, choices=["bwd_weight", "forward", "bwd_data"])
    single.add_argument("--layout", required=True, choices=["nhwgc", "ndhwgc"])
    single.add_argument("--datatype", required=True, choices=["fp32", "fp16", "bf16"])
    single.add_argument("--ndim", required=True, type=int, choices=[2, 3])

    # Batch conversion
    batch = subparsers.add_parser("convert-all", help="Convert all backward_weight configs")
    batch.add_argument(
        "--builder-configs-dir",
        default=str(Path(__file__).resolve().parent.parent.parent /
                    "experimental/grouped_convolution_tile_instances/configs"),
        help="Path to CK Builder configs directory",
    )
    batch.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "configs/grouped_conv"),
        help="Output directory for JSON configs",
    )

    args = parser.parse_args()

    if args.command == "convert":
        result = convert_config_file(args.input, args.variant, args.layout, args.datatype, args.ndim)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  -> {output_path}")
    elif args.command == "convert-all":
        convert_all(args.builder_configs_dir, args.output_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
