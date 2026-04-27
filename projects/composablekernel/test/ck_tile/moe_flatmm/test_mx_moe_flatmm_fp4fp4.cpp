// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "test_mx_moe_flatmm_base.hpp"
#include "test_moe_flatmm_scenarios.hpp"

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

MOE_FLATMM_DECLARE_SCENARIOS(TestMXMoeFlatmmFP4FP4)
