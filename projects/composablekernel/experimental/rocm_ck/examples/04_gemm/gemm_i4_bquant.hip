// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocm_ck/gemm_spec.hpp>

static constexpr const char* kName = "gemm_i4_bquant";

static constexpr auto kTargets = rocm_ck::TargetSet::family_gfx94();

static constexpr auto kSpec = rocm_ck::makeSpec(
    rocm_ck::Signature{.dtype   = rocm_ck::DataType::FP8_FNUZ,
                       .tensors = {rocm_ck::Tensor{.name     = "B",
                                                    .dtype    = rocm_ck::DataType::I4,
                                                    .layout   = rocm_ck::Layout::Col,
                                                    .quantize = rocm_ck::Quantization{.scale_name  = "scale",
                                                                                      .scale_dtype = rocm_ck::DataType::FP8_FNUZ,
                                                                                      .group_size  = 128}},
                                   rocm_ck::Tensor{.name = "C", .dtype = rocm_ck::DataType::FP32}},
                       .ops     = {rocm_ck::GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    rocm_ck::GemmAlgorithm{.block_tile  = {128, 128, 128},
                           .block_waves = {2, 2, 1},
                           .wave_tile   = {32, 32, 16},
                           .pipeline    = rocm_ck::Pipeline::V3},
    kTargets);

#ifdef __HIP_DEVICE_COMPILE__
#include <rocm_ck/gemm_dev.hpp>

extern "C" __global__ void gemm_i4_bquant(rocm_ck::Args args)
{
    rocm_ck::run<kSpec>(args);
}
#endif
