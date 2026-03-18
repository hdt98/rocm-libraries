// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_vector_add_dev.hpp"

// Baseline single-warp FP32 variant. Smallest tile in the set.
// Geometry: kVectorM=4 (128-bit / 32-bit), kRepeatM=1, 64 threads.
static constexpr auto kernel = rocm_ck::make_kernel(
    rocm_ck::ElementwiseConfig{
        .signature = {.dtype = rocm_ck::DataType::FP32},
        .algorithm = {.block_tile = 256, .block_warps = 1, .warp_tile = 256, .pad = true}});

extern "C" __global__ void vector_add_fp32_b256(rocm_ck::VectorAddArgs args)
{
    rocm_ck::runVectorAdd<kernel>(args);
}
