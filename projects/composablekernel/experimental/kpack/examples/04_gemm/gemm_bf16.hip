// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// bf16 GEMM kernel variant. Same tile config as fp32 (128×128×32, 16×16×16
// warp tile). Accumulates in fp32, stores bf16 output via CShuffleEpilogue.

#include "gemm_dev.hpp"

static constexpr rocm_ck::GemmKernel kernel =
    rocm_ck::make_kernel(rocm_ck::GemmSignature{.dtype = rocm_ck::DataType::BF16});

extern "C" __global__ void gemm_bf16(rocm_ck::GemmArgs args)
{
    rocm_ck::runGemm<kernel>(args);
}
