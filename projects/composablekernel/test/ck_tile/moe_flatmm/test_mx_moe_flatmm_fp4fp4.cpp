// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "test_mx_moe_flatmm_base.hpp"

// MX MoE FlatMM fp4xfp4 on gfx950. MXMoeFlatmmConfig16 has N_Tile=256, K_Tile=256
// (matches AITER's a8w4 base tile). All three MoE kinds shipped by AITER for
// the dual-MX pipeline are exercised: gate_only (research case), gate_up
// (mlp1 in production), and gemm2 (mlp2). gate_up requires N >= 2*N_Tile = 512
// so that outputN = N/2 stays a multiple of N_Tile.
// clang-format off
using FP4FP4Types = ::testing::Types<
    std::tuple<MXGemmTypeConfig_fp4xfp4, GateOnly>,
    std::tuple<MXGemmTypeConfig_fp4xfp4, GateUp>,
    std::tuple<MXGemmTypeConfig_fp4xfp4, Gemm2>>;
// clang-format on

template <typename Tuple>
class TestMXMoeFlatmmFP4FP4 : public TestMXMoeFlatmmBase<Tuple>
{
};

TYPED_TEST_SUITE(TestMXMoeFlatmmFP4FP4, FP4FP4Types);

TYPED_TEST(TestMXMoeFlatmmFP4FP4, SmallMNK)
{
    this->run_test(/*num_tokens=*/16, /*topk=*/2, /*experts=*/2, /*N=*/512, /*K=*/256);
}

TYPED_TEST(TestMXMoeFlatmmFP4FP4, MediumMNK)
{
    this->run_test(/*num_tokens=*/32, /*topk=*/2, /*experts=*/4, /*N=*/512, /*K=*/512);
}

TYPED_TEST(TestMXMoeFlatmmFP4FP4, LargeMNK)
{
    this->run_test(/*num_tokens=*/32, /*topk=*/2, /*experts=*/4, /*N=*/1024, /*K=*/768);
}
