// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_vector_add_dev.hpp"

// Largest single-warp FP32 tile. kRepeatM=4 — each thread processes 4 chunks.
static constexpr auto kernel = rocm_ck::make_kernel(
    rocm_ck::ElementwiseConfig{
        .signature = {.dtype = rocm_ck::DataType::FP32},
        .algorithm = {.block_tile = 1024, .block_warps = 1, .warp_tile = 1024, .pad = true}});

extern "C" __global__ void vector_add_fp32_b1024(rocm_ck::VectorAddArgs args)
{
    rocm_ck::runVectorAdd<kernel>(args);
}
