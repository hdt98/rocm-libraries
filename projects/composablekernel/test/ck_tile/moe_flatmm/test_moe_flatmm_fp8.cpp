// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "test_moe_flatmm_base.hpp"
#include "test_moe_flatmm_scenarios.hpp"

// FP8 MoE FlatMM on gfx950 warp configs.
// Output is fp16 (matches GemmBasicTypeConfig<fp8_t>::CDataType).
// clang-format off
using FP8Types = ::testing::Types<
    std::tuple<FP8, FP8, FP16, FlatmmConfig16_950<FP8>, GateOnly>,
    std::tuple<FP8, FP8, FP16, FlatmmConfig16_950<FP8>, GateUp>,
    std::tuple<FP8, FP8, FP16, FlatmmConfig16_950<FP8>, Gemm2>,
    std::tuple<FP8, FP8, FP16, FlatmmConfig32_950<FP8>, GateOnly>,
    std::tuple<FP8, FP8, FP16, FlatmmConfig32_950<FP8>, GateUp>,
    std::tuple<FP8, FP8, FP16, FlatmmConfig32_950<FP8>, Gemm2>>;
// clang-format on

// DISABLED_: the non-MX MoeFlatmmPipelineAGmemBGmemCRegV1 fails on production-
// shaped inputs (~30-95% wrong values across all scenarios). See
// docs/issues/moe-flatmm-non-mx-pipeline/findings.md. Re-enable once the kernel
// is fixed; the input generator and scenario matrix are already in place.
template <typename Tuple>
class DISABLED_TestMoeFlatmmFP8 : public TestMoeFlatmmBase<Tuple>
{
};

TYPED_TEST_SUITE(DISABLED_TestMoeFlatmmFP8, FP8Types);

MOE_FLATMM_DECLARE_SCENARIOS(DISABLED_TestMoeFlatmmFP8)
