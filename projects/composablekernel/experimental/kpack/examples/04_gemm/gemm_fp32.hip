// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// fp32 GEMM kernel variant with 16×16×16 warp tile. All types are fp32
// (input, output, and accumulator).

#include "gemm_dev.hpp"

static constexpr rocm_ck::GemmKernel kernel =
    rocm_ck::make_kernel(rocm_ck::GemmConfig{
        .signature = {.dtype = rocm_ck::DataType::FP32},
        .algorithm = {.block_tile  = {128, 128, 32},
                      .block_warps = {2, 2, 1},
                      .warp_tile   = {16, 16, 16}}});

extern "C" __global__ void gemm_fp32(rocm_ck::GemmArgs args)
{
    rocm_ck::runGemm<kernel>(args);
}
