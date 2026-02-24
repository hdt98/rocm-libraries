// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "test_mx_flatmm_base.hpp"

// clang-format off
using MXFp8Types = ::testing::Types<
    std::tuple<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::fp16_t, MXfp8_FlatmmConfig16>
>;
// clang-format on

TYPED_TEST_SUITE(TestCkTileMXFlatmm, MXFp8Types);

// FP8 tile config: M_Tile=128, N_Tile=256, K_Tile=256
TYPED_TEST(TestCkTileMXFlatmm, SmallSize) { this->run_test(128, 256, 256); }

TYPED_TEST(TestCkTileMXFlatmm, MediumSize) { this->run_test(256, 256, 512); }

TYPED_TEST(TestCkTileMXFlatmm, LargeSize) { this->run_test(512, 512, 1024); }

TYPED_TEST(TestCkTileMXFlatmm, SplitK2) { this->run_test(128, 256, 512, 2); }
