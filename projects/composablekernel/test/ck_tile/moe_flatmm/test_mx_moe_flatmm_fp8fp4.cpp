// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "test_mx_moe_flatmm_base.hpp"

// MX MoE FlatMM fp8xfp4 on gfx950. A-side scale is unity (0x7F) -- mirrors the
// CLI example. Matches AITER's a8w4_gfx950 kernel set (which ships gate_up and
// gemm2). All three MoE kinds are exercised here; gate_up requires
// N >= 2*N_Tile = 512 so outputN = N/2 stays a multiple of N_Tile.
// clang-format off
using FP8FP4Types = ::testing::Types<
    std::tuple<MXGemmTypeConfig_fp8xfp4, GateOnly>,
    std::tuple<MXGemmTypeConfig_fp8xfp4, GateUp>,
    std::tuple<MXGemmTypeConfig_fp8xfp4, Gemm2>>;
// clang-format on

template <typename Tuple>
class TestMXMoeFlatmmFP8FP4 : public TestMXMoeFlatmmBase<Tuple>
{
};

TYPED_TEST_SUITE(TestMXMoeFlatmmFP8FP4, FP8FP4Types);

TYPED_TEST(TestMXMoeFlatmmFP8FP4, SmallMNK)
{
    this->run_test(/*num_tokens=*/16, /*topk=*/2, /*experts=*/2, /*N=*/512, /*K=*/256);
}

TYPED_TEST(TestMXMoeFlatmmFP8FP4, MediumMNK)
{
    this->run_test(/*num_tokens=*/32, /*topk=*/2, /*experts=*/4, /*N=*/512, /*K=*/512);
}

TYPED_TEST(TestMXMoeFlatmmFP8FP4, LargeMNK)
{
    this->run_test(/*num_tokens=*/32, /*topk=*/2, /*experts=*/4, /*N=*/1024, /*K=*/768);
}
