// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocm_ck/gemm_spec.hpp>

static constexpr const char* kName = "gemm_fp8_fnuz";

static constexpr auto kTargets = rocm_ck::TargetSet::family_gfx94();

static constexpr auto kSpec = rocm_ck::makeSpec(
    rocm_ck::Signature{.dtype   = rocm_ck::DataType::FP8_FNUZ,
                       .tensors = {rocm_ck::Tensor{.name = "C", .dtype = rocm_ck::DataType::FP16}},
                       .ops     = {rocm_ck::GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    rocm_ck::GemmAlgorithm{.block_tile  = {128, 128, 32},
                           .block_waves = {2, 2, 1},
                           .wave_tile   = {32, 32, 16}},
    kTargets);

#ifdef __HIP_DEVICE_COMPILE__
#include <rocm_ck/gemm_dev.hpp>

extern "C" __global__ void gemm_fp8_fnuz(rocm_ck::Args args)
{
    rocm_ck::run<kSpec>(args);
}
#endif
