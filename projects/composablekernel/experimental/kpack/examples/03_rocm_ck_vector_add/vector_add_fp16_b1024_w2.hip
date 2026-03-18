// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_vector_add_dev.hpp"

// Multi-warp fp16 variant: 2 warps × 64 threads = 128 threads,
// each block processes 1024 elements with large vector width (kVectorM=8).
static constexpr auto kernel = rocm_ck::make_kernel(
    rocm_ck::ElementwiseSignature{rocm_ck::DataType::FP16},
    rocm_ck::ElementwiseAlgorithm{
        .block_tile = 1024, .block_warps = 2, .warp_tile = 512, .pad = true});

extern "C" __global__ void vector_add_fp16_b1024_w2(rocm_ck::VectorAddArgs args)
{
    rocm_ck::runVectorAdd<kernel>(args);
}
