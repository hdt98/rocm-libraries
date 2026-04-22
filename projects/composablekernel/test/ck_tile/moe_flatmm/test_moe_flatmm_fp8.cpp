// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "test_moe_flatmm_base.hpp"

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

template <typename Tuple>
class TestMoeFlatmmFP8 : public TestMoeFlatmmBase<Tuple>
{
};

TYPED_TEST_SUITE(TestMoeFlatmmFP8, FP8Types);

TYPED_TEST(TestMoeFlatmmFP8, SmallMNK)
{
    this->run_test(/*num_tokens=*/16, /*topk=*/2, /*experts=*/2, /*N=*/512, /*K=*/256);
}

TYPED_TEST(TestMoeFlatmmFP8, MediumMNK)
{
    this->run_test(/*num_tokens=*/32, /*topk=*/2, /*experts=*/4, /*N=*/512, /*K=*/512);
}

TYPED_TEST(TestMoeFlatmmFP8, LargeK)
{
    this->run_test(/*num_tokens=*/32, /*topk=*/2, /*experts=*/4, /*N=*/1024, /*K=*/768);
}
