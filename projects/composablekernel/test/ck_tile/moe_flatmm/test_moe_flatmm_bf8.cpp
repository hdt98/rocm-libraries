// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "test_moe_flatmm_base.hpp"
#include "test_moe_flatmm_scenarios.hpp"

// BF8 MoE FlatMM on CDNA. gfx950 uses larger architecture-specific configs;
// gfx942 falls back to compatible base configs.
// Output is fp16 (matches GemmBasicTypeConfig<bf8_t>::CDataType).
// clang-format off
#if defined(CK_TILE_MOE_FLATMM_USE_GFX942_F8_CONFIGS)
template <typename DataType>
using BF8FlatmmConfig16 = FlatmmConfig16<DataType>;
template <typename DataType>
using BF8FlatmmConfig32 = FlatmmConfig32<DataType>;
#else
template <typename DataType>
using BF8FlatmmConfig16 = FlatmmConfig16_950<DataType>;
template <typename DataType>
using BF8FlatmmConfig32 = FlatmmConfig32_950<DataType>;
#endif

using BF8Types = ::testing::Types<
    std::tuple<BF8, BF8, FP16, BF8FlatmmConfig16<BF8>, GateOnly>,
    std::tuple<BF8, BF8, FP16, BF8FlatmmConfig16<BF8>, GateUp>,
    std::tuple<BF8, BF8, FP16, BF8FlatmmConfig16<BF8>, Gemm2>,
    std::tuple<BF8, BF8, FP16, BF8FlatmmConfig32<BF8>, GateOnly>,
    std::tuple<BF8, BF8, FP16, BF8FlatmmConfig32<BF8>, GateUp>,
    std::tuple<BF8, BF8, FP16, BF8FlatmmConfig32<BF8>, Gemm2>>;
// clang-format on

template <typename Tuple>
class TestMoeFlatmmBF8 : public TestMoeFlatmmBase<Tuple>
{
};

TYPED_TEST_SUITE(TestMoeFlatmmBF8, BF8Types);

MOE_FLATMM_DECLARE_SCENARIOS(TestMoeFlatmmBF8)
