// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "test_moe_flatmm_base.hpp"
#include "test_moe_flatmm_scenarios.hpp"

// FP8 MoE FlatMM on CDNA. gfx950 uses larger architecture-specific configs;
// gfx942 falls back to compatible base configs.
// Output is fp16 (matches GemmBasicTypeConfig<fp8_t>::CDataType).
// clang-format off
#if defined(CK_TILE_MOE_FLATMM_USE_GFX942_F8_CONFIGS)
template <typename DataType>
using FP8FlatmmConfig16 = FlatmmConfig16<DataType>;
template <typename DataType>
using FP8FlatmmConfig32 = FlatmmConfig32<DataType>;
#else
template <typename DataType>
using FP8FlatmmConfig16 = FlatmmConfig16_950<DataType>;
template <typename DataType>
using FP8FlatmmConfig32 = FlatmmConfig32_950<DataType>;
#endif

using FP8Types = ::testing::Types<
    std::tuple<FP8, FP8, FP16, FP8FlatmmConfig16<FP8>, GateOnly>,
    std::tuple<FP8, FP8, FP16, FP8FlatmmConfig16<FP8>, GateUp>,
    std::tuple<FP8, FP8, FP16, FP8FlatmmConfig16<FP8>, Gemm2>,
    std::tuple<FP8, FP8, FP16, FP8FlatmmConfig32<FP8>, GateOnly>,
    std::tuple<FP8, FP8, FP16, FP8FlatmmConfig32<FP8>, GateUp>,
    std::tuple<FP8, FP8, FP16, FP8FlatmmConfig32<FP8>, Gemm2>>;
// clang-format on

template <typename Tuple>
class TestMoeFlatmmFP8 : public TestMoeFlatmmBase<Tuple>
{
};

TYPED_TEST_SUITE(TestMoeFlatmmFP8, FP8Types);

MOE_FLATMM_DECLARE_SCENARIOS(TestMoeFlatmmFP8)
