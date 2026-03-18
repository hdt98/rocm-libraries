// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Variant registry for programmatic kernel selection.
// Host-only header — no CK Tile dependency.

#pragma once

#include "rocm_vector_add_api.hpp"

namespace rocm_ck {

/// Descriptor for a compiled kernel variant in the kpack archive.
struct variant_descriptor
{
    const char* name;
    vector_add_struct kernel;
};

/// Complete table of all compiled variants, matching the kpack archive contents.
/// Each entry corresponds to a .hip file and its make_kernel configuration.
static constexpr variant_descriptor ALL_VARIANTS[] = {
    {"vector_add_fp32_b256", make_kernel({.block_size = 256, .compute_type = DataType::FP32})},
    {"vector_add_fp32_b512", make_kernel({.block_size = 512, .compute_type = DataType::FP32})},
    {"vector_add_fp32_b1024", make_kernel({.block_size = 1024, .compute_type = DataType::FP32})},
    {"vector_add_fp16_b512", make_kernel({.block_size = 512, .compute_type = DataType::FP16})},
    {"vector_add_fp16_b1024", make_kernel({.block_size = 1024, .compute_type = DataType::FP16})},
    {"vector_add_bf16_b512", make_kernel({.block_size = 512, .compute_type = DataType::BF16})},
    {"vector_add_fp32_b256_sa",
     make_kernel(elementwise_signature{DataType::FP32}, elementwise_algorithm{256, 1, 256, true})},
    {"vector_add_fp32_b2048_w8",
     make_kernel(elementwise_signature{DataType::FP32}, elementwise_algorithm{2048, 8, 64, true})},
    {"vector_add_fp16_b1024_w2",
     make_kernel(elementwise_signature{DataType::FP16}, elementwise_algorithm{1024, 2, 512, true})},
};

static constexpr int ALL_VARIANTS_COUNT = sizeof(ALL_VARIANTS) / sizeof(ALL_VARIANTS[0]);

/// Find the best variant for the given data type and problem size.
/// Prefers the largest block_tile that divides problem_size cleanly (aligned).
/// Falls back to the largest padded variant if none aligns.
/// Returns nullptr if no variant matches the data type.
constexpr const variant_descriptor* find_variant(DataType dt, int problem_size)
{
    const variant_descriptor* best_aligned = nullptr;
    const variant_descriptor* best_padded  = nullptr;

    for(int i = 0; i < ALL_VARIANTS_COUNT; ++i)
    {
        const auto& v = ALL_VARIANTS[i];
        if(v.kernel.compute_type != dt)
            continue;

        if(is_aligned(v.kernel, problem_size))
        {
            if(!best_aligned || v.kernel.block_tile > best_aligned->kernel.block_tile)
                best_aligned = &ALL_VARIANTS[i];
        }

        if(v.kernel.pad)
        {
            if(!best_padded || v.kernel.block_tile > best_padded->kernel.block_tile)
                best_padded = &ALL_VARIANTS[i];
        }
    }

    return best_aligned ? best_aligned : best_padded;
}

} // namespace rocm_ck
