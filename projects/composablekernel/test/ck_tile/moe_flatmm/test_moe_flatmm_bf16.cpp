// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "test_moe_flatmm_base.hpp"

// BF16 MoE FlatMM on gfx950 warp configs.
// clang-format off
using BF16Types = ::testing::Types<
    std::tuple<BF16, BF16, BF16, FlatmmConfig16_950<BF16>, GateOnly>,
    std::tuple<BF16, BF16, BF16, FlatmmConfig16_950<BF16>, GateUp>,
    std::tuple<BF16, BF16, BF16, FlatmmConfig16_950<BF16>, Gemm2>,
    std::tuple<BF16, BF16, BF16, FlatmmConfig32_950<BF16>, GateOnly>,
    std::tuple<BF16, BF16, BF16, FlatmmConfig32_950<BF16>, GateUp>,
    std::tuple<BF16, BF16, BF16, FlatmmConfig32_950<BF16>, Gemm2>>;
// clang-format on

template <typename Tuple>
class TestMoeFlatmmBF16 : public TestMoeFlatmmBase<Tuple>
{
};

TYPED_TEST_SUITE(TestMoeFlatmmBF16, BF16Types);

TYPED_TEST(TestMoeFlatmmBF16, SmallMNK)
{
    this->run_test(/*num_tokens=*/16, /*topk=*/2, /*experts=*/2, /*N=*/512, /*K=*/256);
}

TYPED_TEST(TestMoeFlatmmBF16, MediumMNK)
{
    this->run_test(/*num_tokens=*/32, /*topk=*/2, /*experts=*/4, /*N=*/512, /*K=*/512);
}

TYPED_TEST(TestMoeFlatmmBF16, LargeK)
{
    this->run_test(/*num_tokens=*/32, /*topk=*/2, /*experts=*/4, /*N=*/1024, /*K=*/768);
}
