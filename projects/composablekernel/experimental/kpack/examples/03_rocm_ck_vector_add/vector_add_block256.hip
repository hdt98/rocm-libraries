// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_vector_add_dev.hpp"

static constexpr auto kernel =
    rocm_ck::make_kernel({.block_size = 256, .compute_type = rocm_ck::DataType::FP32});

extern "C" __global__ void vector_add_block256(rocm_ck::VectorAddArgs args)
{
    rocm_ck::run_vector_add<kernel>(args);
}
