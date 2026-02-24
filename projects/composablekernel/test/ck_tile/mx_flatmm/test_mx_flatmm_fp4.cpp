// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "test_mx_flatmm_base.hpp"

// clang-format off
using MXFp4Types = ::testing::Types<
    std::tuple<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t, ck_tile::fp16_t, MXfp4_FlatmmConfig16>
>;
// clang-format on

TYPED_TEST_SUITE(TestCkTileMXFlatmm, MXFp4Types);

// FP4 tile config: M_Tile=128, N_Tile=512, K_Tile=256
TYPED_TEST(TestCkTileMXFlatmm, SmallSize) { this->run_test(128, 512, 256); }

TYPED_TEST(TestCkTileMXFlatmm, MediumSize) { this->run_test(256, 512, 512); }

TYPED_TEST(TestCkTileMXFlatmm, LargeSize) { this->run_test(512, 1024, 1024); }

TYPED_TEST(TestCkTileMXFlatmm, SplitK2) { this->run_test(128, 512, 512, 2); }
