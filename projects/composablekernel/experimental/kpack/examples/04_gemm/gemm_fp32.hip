// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// fp32 GEMM kernel variant. Uses GemmSignature schema to configure types
// and layouts, then delegates to runGemm<K> for CK Tile type wiring.

#include "gemm_dev.hpp"

static constexpr rocm_ck::GemmKernel kernel =
    rocm_ck::make_kernel(rocm_ck::GemmSignature{.dtype = rocm_ck::DataType::FP32});

extern "C" __global__ void gemm_fp32(rocm_ck::GemmArgs args)
{
    rocm_ck::runGemm<kernel>(args);
}
