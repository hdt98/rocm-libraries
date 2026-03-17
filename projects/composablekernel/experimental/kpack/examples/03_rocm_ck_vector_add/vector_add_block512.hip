// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocm_vector_add_dev.hpp"

static constexpr rocm_ck::vector_add_config config{.block_size = 512};

extern "C" __global__ void vector_add_block512(rocm_ck::VectorAddArgs args)
{
    rocm_ck::run_vector_add<config>(args);
}
