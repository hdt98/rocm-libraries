// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_vector_add_dev.hpp"

// Multi-warp variant matching CK Tile example 21 ElementWise config:
// 8 warps × 64 threads = 512 threads, each block processes 2048 elements.
static constexpr auto kernel = rocm_ck::make_kernel(
    rocm_ck::ElementwiseSignature{.in_type = rocm_ck::DataType::FP32, .out_type = rocm_ck::DataType::FP32},
    rocm_ck::ElementwiseAlgorithm{
        .block_tile = 2048, .block_warps = 8, .warp_tile = 64, .pad = true});

extern "C" __global__ void vector_add_fp32_b2048_w8(rocm_ck::VectorAddArgs args)
{
    rocm_ck::runVectorAdd<kernel>(args);
}
