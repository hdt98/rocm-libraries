// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_vector_add_dev.hpp"

static constexpr auto kernel = rocm_ck::make_kernel(
    rocm_ck::ElementwiseConfig{
        .signature = {rocm_ck::DataType::FP32},
        .algorithm = {.block_tile = 512, .block_warps = 1, .warp_tile = 512, .pad = true}});

extern "C" __global__ void vector_add_fp32_b512(rocm_ck::VectorAddArgs args)
{
    rocm_ck::runVectorAdd<kernel>(args);
}
