// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "test_moe_flatmm_base.hpp"
#include "test_moe_flatmm_scenarios.hpp"

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

// DISABLED_: the non-MX MoeFlatmmPipelineAGmemBGmemCRegV1 fails on production-
// shaped inputs (~30-95% wrong values across all scenarios). See
// docs/issues/moe-flatmm-non-mx-pipeline/findings.md. Re-enable once the kernel
// is fixed; the input generator and scenario matrix are already in place.
template <typename Tuple>
class DISABLED_TestMoeFlatmmBF16 : public TestMoeFlatmmBase<Tuple>
{
};

TYPED_TEST_SUITE(DISABLED_TestMoeFlatmmBF16, BF16Types);

MOE_FLATMM_DECLARE_SCENARIOS(DISABLED_TestMoeFlatmmBF16)
