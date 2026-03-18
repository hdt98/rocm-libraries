// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_vector_add_dev.hpp"

// Narrowing: FP32 inputs -> FP16 output. Same kVectorM=4 constraint as widening
// (wider type always governs). Output stored in FP16 for reduced memory footprint.
static constexpr auto kernel = rocm_ck::make_kernel(
    rocm_ck::ElementwiseConfig{
        .signature = {.in_dtype = rocm_ck::DataType::FP32, .out_dtype = rocm_ck::DataType::FP16},
        .algorithm = {.block_tile = 1024, .block_warps = 1, .warp_tile = 1024, .pad = true}});

extern "C" __global__ void vector_add_fp32_fp16_b1024(rocm_ck::VectorAddArgs args)
{
    rocm_ck::runVectorAdd<kernel>(args);
}
