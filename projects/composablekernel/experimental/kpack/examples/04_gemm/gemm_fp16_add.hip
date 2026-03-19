// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// fp16 GEMM + D0 addition: E = A*B + D0. Demonstrates fused epilogue with
// one D tensor. Accumulates in fp32, stores fp16 output via CShuffleEpilogue.

#include "gemm_dev.hpp"

static constexpr rocm_ck::GemmKernel kernel =
    rocm_ck::make_kernel(rocm_ck::GemmConfig{
        .signature = {.dtype       = rocm_ck::DataType::FP16,
                      .epilogue_op = rocm_ck::EpilogueOp::Add,
                      .d0_dtype    = rocm_ck::DataType::FP16},
        .algorithm = {.block_tile  = {128, 128, 32},
                      .block_warps = {2, 2, 1},
                      .warp_tile   = {16, 16, 16}}});

extern "C" __global__ void gemm_fp16_add(rocm_ck::GemmArgs1D args)
{
    rocm_ck::runGemm<kernel>(args);
}
