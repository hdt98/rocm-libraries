// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "test_moe_flatmm_base.hpp"
#include "test_moe_flatmm_scenarios.hpp"

// BF8 MoE FlatMM on CDNA gfx942/gfx950 warp configs.
// Output is fp16 (matches GemmBasicTypeConfig<bf8_t>::CDataType).
// clang-format off
using BF8Types = ::testing::Types<
    std::tuple<BF8, BF8, FP16, FlatmmConfig16_950<BF8>, GateOnly>,
    std::tuple<BF8, BF8, FP16, FlatmmConfig16_950<BF8>, GateUp>,
    std::tuple<BF8, BF8, FP16, FlatmmConfig16_950<BF8>, Gemm2>,
    std::tuple<BF8, BF8, FP16, FlatmmConfig32_950<BF8>, GateOnly>,
    std::tuple<BF8, BF8, FP16, FlatmmConfig32_950<BF8>, GateUp>,
    std::tuple<BF8, BF8, FP16, FlatmmConfig32_950<BF8>, Gemm2>>;
// clang-format on

template <typename Tuple>
class TestMoeFlatmmBF8 : public TestMoeFlatmmBase<Tuple>
{
};

TYPED_TEST_SUITE(TestMoeFlatmmBF8, BF8Types);

MOE_FLATMM_DECLARE_SCENARIOS(TestMoeFlatmmBF8)
