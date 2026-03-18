// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_vector_add_dev.hpp"

// Narrower type doubles vectorization: FP16 gets kVectorM=8 (vs FP32's 4)
// because 128-bit registers hold 8 half-precision values instead of 4 floats.
static constexpr auto kernel = rocm_ck::make_kernel(
    rocm_ck::ElementwiseConfig{
        .signature = {.dtype = rocm_ck::DataType::FP16},
        .algorithm = {.block_tile = 512, .block_warps = 1, .warp_tile = 512, .pad = true}});

extern "C" __global__ void vector_add_fp16_b512(rocm_ck::VectorAddArgs args)
{
    rocm_ck::runVectorAdd<kernel>(args);
}
