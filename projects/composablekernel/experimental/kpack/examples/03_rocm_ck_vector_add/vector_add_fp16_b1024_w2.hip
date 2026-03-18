// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_vector_add_dev.hpp"

// Multi-warp FP16: 2 warps x 64 = 128 threads, same block_tile as fp16_b1024.
// Larger warp_tile (512) preserves kVectorM=8 (max for 16-bit). Compare with
// single-warp fp16_b1024: same work, but kRepeatM drops from 2 to 1.
static constexpr auto kernel = rocm_ck::make_kernel(
    rocm_ck::ElementwiseConfig{
        .signature = {.dtype = rocm_ck::DataType::FP16},
        .algorithm = {.block_tile = 1024, .block_warps = 2, .warp_tile = 512, .pad = true}});

extern "C" __global__ void vector_add_fp16_b1024_w2(rocm_ck::VectorAddArgs args)
{
    rocm_ck::runVectorAdd<kernel>(args);
}
