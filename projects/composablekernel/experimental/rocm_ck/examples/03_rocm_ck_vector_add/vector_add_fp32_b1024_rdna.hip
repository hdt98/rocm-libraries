// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocm_ck/elementwise_spec.hpp>

static constexpr const char* kName = "vector_add_fp32_b1024_rdna";

static constexpr auto kTargets = rocm_ck::TargetSet::rdna();

static constexpr auto kSpec = rocm_ck::makeSpec(
    rocm_ck::Signature{.dtype = rocm_ck::DataType::FP32,
                       .ops   = {rocm_ck::AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    rocm_ck::ElementwiseAlgorithm{1024, 1, 1024, true},
    kTargets);

#ifdef __HIP_DEVICE_COMPILE__
#include <rocm_ck/elementwise_dev.hpp>

extern "C" __global__ void vector_add_fp32_b1024_rdna(rocm_ck::Args args)
{
    rocm_ck::run<kSpec>(args);
}
#endif
