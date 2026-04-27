// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "test_moe_flatmm_base.hpp"
#include "test_moe_flatmm_scenarios.hpp"

// FP16 MoE FlatMM on gfx950 warp configs (16x16x32 and 32x32x16).
// Covers all three MoE kinds per config.
// clang-format off
using FP16Types = ::testing::Types<
    std::tuple<FP16, FP16, FP16, FlatmmConfig16_950<FP16>, GateOnly>,
    std::tuple<FP16, FP16, FP16, FlatmmConfig16_950<FP16>, GateUp>,
    std::tuple<FP16, FP16, FP16, FlatmmConfig16_950<FP16>, Gemm2>,
    std::tuple<FP16, FP16, FP16, FlatmmConfig32_950<FP16>, GateOnly>,
    std::tuple<FP16, FP16, FP16, FlatmmConfig32_950<FP16>, GateUp>,
    std::tuple<FP16, FP16, FP16, FlatmmConfig32_950<FP16>, Gemm2>>;
// clang-format on

template <typename Tuple>
class TestMoeFlatmmFP16 : public TestMoeFlatmmBase<Tuple>
{
};

TYPED_TEST_SUITE(TestMoeFlatmmFP16, FP16Types);

MOE_FLATMM_DECLARE_SCENARIOS(TestMoeFlatmmFP16)
