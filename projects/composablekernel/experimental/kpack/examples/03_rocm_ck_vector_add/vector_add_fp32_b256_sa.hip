// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_vector_add_dev.hpp"

// Explicit sig+algo form: same config as fp32_b256, demonstrates the
// Signature/Algorithm separation API.
static constexpr auto kernel = rocm_ck::make_kernel(
    rocm_ck::elementwise_signature{rocm_ck::DataType::FP32},
    rocm_ck::elementwise_algorithm{
        .block_tile = 256, .block_warps = 1, .warp_tile = 256, .pad = true});

extern "C" __global__ void vector_add_fp32_b256_sa(rocm_ck::VectorAddArgs args)
{
    rocm_ck::run_vector_add<kernel>(args);
}
