// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocm_ck/gemm_spec.hpp>

static constexpr const char* kName = "gemm_i8";

static constexpr auto kTargets = rocm_ck::TargetSet::family_gfx94();

static constexpr auto kSpec = rocm_ck::makeSpec(
    rocm_ck::Signature{.dtype   = rocm_ck::DataType::I8,
                       .tensors = {rocm_ck::Tensor{.name = "C", .dtype = rocm_ck::DataType::I32}},
                       .ops     = {rocm_ck::GemmOp{.lhs = "A", .rhs = "B", .out = "C", .acc_dtype = rocm_ck::DataType::I32}}},
    rocm_ck::GemmAlgorithm{.block_tile  = {128, 128, 64},
                           .block_waves = {2, 2, 1},
                           .wave_tile   = {32, 32, 16},
                           .pipeline    = rocm_ck::Pipeline::V3},
    kTargets);

#ifdef __HIP_DEVICE_COMPILE__
#include <rocm_ck/gemm_dev.hpp>

extern "C" __global__ void gemm_i8(rocm_ck::Args args)
{
    rocm_ck::run<kSpec>(args);
}
#endif
