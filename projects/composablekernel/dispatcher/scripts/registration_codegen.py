#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Shared registration code generation utilities for dispatcher kernel scripts.
# Generates chunked registration .cpp files for parallel compilation.

import re
from pathlib import Path


# Number of kernels per registration chunk.
# Each chunk is one .cpp file / compilation unit.
# TODO: What is the optimal value?
CHUNK_SIZE = 50


def parse_kernel_metadata(kname):
    """Extract metadata from a kernel header filename stem."""
    ndim = 3 if "_3d_" in kname else 2

    dtype = "fp16"
    for dt in ["fp16", "bf16", "fp32"]:
        if f"_{dt}_" in kname:
            dtype = dt
            break

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

    pipeline = "compv3"
    for p in ["compv1", "compv2", "compv3", "compv4", "compv5", "compv6", "mem"]:
        if f"_{p}_" in kname:
            pipeline = p
            break
    scheduler = "interwave" if "interwave" in kname else "intrawave"
    epilogue = "cshuffle"

    vec_a, vec_b, vec_c = 4, 8, 8
    vec_match = re.search(r"_vec(\d+)_(\d+)_(\d+)", kname)
    if vec_match:
        vec_a = int(vec_match.group(1))
        vec_b = int(vec_match.group(2))
        vec_c = int(vec_match.group(3))

    block_per_cu = 1
    num_wave_groups = 1
    num_groups_to_merge = 1
    gm_match = re.search(r"_gm(\d+)", kname)
    if gm_match:
        num_groups_to_merge = int(gm_match.group(1))

    specialization = "default"
    if "filter1x1_stride1_pad0" in kname:
        specialization = "filter1x1_stride1_pad0"
    elif "filter1x1_pad0" in kname:
        specialization = "filter1x1_pad0"
    elif "filter3x3" in kname:
        specialization = "filter3x3"

    return {
        "ndim": ndim,
        "dtype": dtype,
        "tile_m": tile_m, "tile_n": tile_n, "tile_k": tile_k,
        "wave_m": wave_m, "wave_n": wave_n, "wave_k": wave_k,
        "warp_m": warp_m, "warp_n": warp_n, "warp_k": warp_k,
        "pipeline": pipeline, "scheduler": scheduler, "epilogue": epilogue,
        "vec_a": vec_a, "vec_b": vec_b, "vec_c": vec_c,
        "block_per_cu": block_per_cu, "num_wave_groups": num_wave_groups,
        "num_groups_to_merge": num_groups_to_merge,
        "specialization": specialization,
    }


def make_registration_block(kname, global_idx, op_enum, run_fn_maker, is_supported_fn_maker):
    """Generate C++ registration code for a single kernel."""
    meta = parse_kernel_metadata(kname)
    ns = f"ns_{kname}"
    launcher = f"{ns}::{kname}_Launcher"
    ndim = meta["ndim"]

    lines = []
    lines.append(f"    // Kernel {global_idx}: {kname}")
    lines.append("    {")
    lines.append(f"        GroupedConvKernelKey key;")
    lines.append(f'        key.dtype_in     = "{meta["dtype"]}";')
    lines.append(f'        key.dtype_wei    = "{meta["dtype"]}";')
    lines.append(f'        key.dtype_out    = "{meta["dtype"]}";')
    lines.append(f'        key.layout       = "nhwgc";')
    lines.append(f"        key.ndim_spatial = {ndim};")
    lines.append(f"        key.op           = {op_enum};")
    lines.append(f"        key.tile_m       = {meta['tile_m']};")
    lines.append(f"        key.tile_n       = {meta['tile_n']};")
    lines.append(f"        key.tile_k       = {meta['tile_k']};")
    lines.append(f"        key.wave_m       = {meta['wave_m']};")
    lines.append(f"        key.wave_n       = {meta['wave_n']};")
    lines.append(f"        key.wave_k       = {meta['wave_k']};")
    lines.append(f"        key.warp_m       = {meta['warp_m']};")
    lines.append(f"        key.warp_n       = {meta['warp_n']};")
    lines.append(f"        key.warp_k       = {meta['warp_k']};")
    lines.append(f'        key.pipeline     = "{meta["pipeline"]}";')
    lines.append(f'        key.scheduler    = "{meta["scheduler"]}";')
    lines.append(f'        key.epilogue     = "{meta["epilogue"]}";')
    lines.append(f"        key.vector_size_a      = {meta['vec_a']};")
    lines.append(f"        key.vector_size_b      = {meta['vec_b']};")
    lines.append(f"        key.vector_size_c      = {meta['vec_c']};")
    lines.append(f"        key.block_per_cu       = {meta['block_per_cu']};")
    lines.append(f"        key.num_wave_groups    = {meta['num_wave_groups']};")
    lines.append(f"        key.num_groups_to_merge = {meta['num_groups_to_merge']};")
    lines.append(f'        key.specialization = "{meta["specialization"]}";')
    lines.append(f"        key.arch         = arch;")
    lines.append(f"        auto run_fn = {run_fn_maker}<{launcher}, {ndim}>();")
    lines.append(f"        auto is_supported_fn = {is_supported_fn_maker}<{launcher}, {ndim}>();")
    lines.append(f"#ifdef CK_EXPERIMENTAL_BUILDER")
    lines.append(f"        auto instance_str = backends::get_instance_string<{launcher}>();")
    lines.append(
        f'        auto inst = std::make_shared<GroupedConvKernelInstance>(key, "{kname}", std::move(run_fn), std::move(is_supported_fn), instance_str);'
    )
    lines.append(f"#else")
    lines.append(
        f'        auto inst = std::make_shared<GroupedConvKernelInstance>(key, "{kname}", std::move(run_fn), std::move(is_supported_fn));'
    )
    lines.append(f"#endif")
    lines.append(f"        registry.register_kernel(key, inst);")
    lines.append("    }")
    return lines


def generate_chunked_registration(headers, output_dir, variant, op_enum,
                                   run_fn_maker, is_supported_fn_maker,
                                   register_fn_name, chunk_size=CHUNK_SIZE):
    """Generate chunked registration .cpp files for parallel compilation.

    Args:
        headers: list of Path objects for kernel .hpp files
        output_dir: directory to write generated files
        variant: short name like "bwd_weight", "fwd", "bwd_data"
        op_enum: C++ enum value like "GroupedConvOp::BackwardWeight"
        run_fn_maker: C++ template function like "backends::make_conv_bwd_weight_run_fn"
        is_supported_fn_maker: C++ template function like "backends::make_conv_bwd_weight_is_supported_fn"
        register_fn_name: C++ function name like "register_all_grouped_conv_bwd_weight_kernels"
        chunk_size: number of kernels per chunk file

    Returns:
        list of generated .cpp file paths
    """
    output_dir = Path(output_dir)
    generated_files = []

    # Split headers into chunks
    chunks = [headers[i:i + chunk_size] for i in range(0, len(headers), chunk_size)]

    for chunk_idx, chunk_headers in enumerate(chunks):
        chunk_fn = f"register_{variant}_chunk_{chunk_idx}"
        chunk_cpp = output_dir / f"register_{variant}_chunk_{chunk_idx}.cpp"

        lines = [
            "// Auto-generated — do not edit",
            f"// Registration chunk {chunk_idx} for {variant} kernels ({len(chunk_headers)} kernels).",
            "",
            "#pragma clang diagnostic push",
            '#pragma clang diagnostic ignored "-Wheader-hygiene"',
            '#pragma clang diagnostic ignored "-Wunused-parameter"',
        ]
        # Include only headers for this chunk
        for h in chunk_headers:
            lines.append(f'#include "{h.name}"')
        lines.append("#pragma clang diagnostic pop")
        lines.append("")
        lines.append('#include "ck_tile/dispatcher/grouped_conv_registry.hpp"')
        lines.append('#include "ck_tile/dispatcher/backends/generated_conv_backend.hpp"')
        lines.append("")
        lines.append("namespace ck_tile {")
        lines.append("namespace dispatcher {")
        lines.append("")
        lines.append(f"void {chunk_fn}(GroupedConvRegistry& registry, const std::string& arch)")
        lines.append("{")

        # Global index offset for this chunk
        global_offset = chunk_idx * chunk_size
        for i, h in enumerate(chunk_headers):
            reg_block = make_registration_block(
                h.stem, global_offset + i, op_enum, run_fn_maker, is_supported_fn_maker
            )
            lines.extend(reg_block)

        lines.append("}")
        lines.append("")
        lines.append("} // namespace dispatcher")
        lines.append("} // namespace ck_tile")
        lines.append("")

        chunk_cpp.write_text("\n".join(lines))
        generated_files.append(chunk_cpp)

    # Generate the thin register_all .cpp that calls all chunks
    all_cpp = output_dir / "register_all_grouped_conv_kernels.cpp"
    lines = [
        "// Auto-generated — do not edit",
        f"// Top-level registration that calls {len(chunks)} chunk functions.",
        "",
        '#include "ck_tile/dispatcher/grouped_conv_registry.hpp"',
        "",
        "namespace ck_tile {",
        "namespace dispatcher {",
        "",
    ]
    # Forward-declare chunk functions
    for chunk_idx in range(len(chunks)):
        chunk_fn = f"register_{variant}_chunk_{chunk_idx}"
        lines.append(f"void {chunk_fn}(GroupedConvRegistry& registry, const std::string& arch);")
    lines.append("")

    # Main registration function (with registry parameter)
    lines.append(f"void {register_fn_name}(")
    lines.append(f"    GroupedConvRegistry& registry, const std::string& arch)")
    lines.append("{")
    for chunk_idx in range(len(chunks)):
        chunk_fn = f"register_{variant}_chunk_{chunk_idx}"
        lines.append(f"    {chunk_fn}(registry, arch);")
    lines.append("}")
    lines.append("")

    # Convenience overload
    lines.append(f"void {register_fn_name}(const std::string& arch)")
    lines.append("{")
    lines.append("    auto& registry = GroupedConvRegistry::instance();")
    lines.append(f"    {register_fn_name}(registry, arch);")
    lines.append("}")
    lines.append("")
    lines.append("} // namespace dispatcher")
    lines.append("} // namespace ck_tile")
    lines.append("")

    all_cpp.write_text("\n".join(lines))
    generated_files.append(all_cpp)

    print(f"Generated {len(chunks)} chunk files + register_all ({len(headers)} total registrations)")
    return generated_files
