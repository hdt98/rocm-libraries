// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_vector_add_dev.hpp"

// BF16 widening: BF16 inputs -> FP32 output. The most common mixed-type pattern
// in ML training (BF16 storage, FP32 accumulation). Same geometry as fp16_fp32_b1024.
static constexpr auto kernel = rocm_ck::make_kernel(
    rocm_ck::ElementwiseConfig{
        .signature = {.in_dtype = rocm_ck::DataType::BF16, .out_dtype = rocm_ck::DataType::FP32},
        .algorithm = {.block_tile = 1024, .block_warps = 1, .warp_tile = 1024, .pad = true}});

extern "C" __global__ void vector_add_bf16_fp32_b1024(rocm_ck::VectorAddArgs args)
{
    rocm_ck::runVectorAdd<kernel>(args);
}
