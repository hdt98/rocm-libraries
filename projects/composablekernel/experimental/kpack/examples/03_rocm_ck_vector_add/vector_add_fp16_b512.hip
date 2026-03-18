// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_vector_add_dev.hpp"

static constexpr auto kernel =
    rocm_ck::make_kernel({.block_size = 512, .compute_type = rocm_ck::DataType::FP16});

extern "C" __global__ void vector_add_fp16_b512(rocm_ck::VectorAddArgs args)
{
    rocm_ck::run_vector_add<kernel>(args);
}
