// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"

#include <gtest/gtest.h>
#include <memory>

#include "test_gemm_quant_fixtures.hpp"

// Type aliases for readability
using RowMajor    = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor = ck_tile::tensor_layout::gemm::ColumnMajor;
using FP8         = ck_tile::fp8_t;
using BF8         = ck_tile::bf8_t;
using Half        = ck_tile::half_t;
using ABQuantGrouped =
    std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::ABQuantGrouped>;

// Group sizes
using GroupSize1D     = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
using GroupSize2D128N = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;

// Type combinations for ABQuant split-K tests - Prefill shape
// Only use 2D BQ groups (1×128×128) to avoid AICK-644 bug
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, AQuantGroupSize, BQuantGroupSize, BQLayout>
// clang-format off
using ABQuantSplitKPrefillTypes = ::testing::Types<
    // FP8 with 2D BQ groups
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigPrefillIntrawave, GroupSize1D, GroupSize2D128N, ColumnMajor>,
    // BF8 with 2D BQ groups
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, ABQuantGrouped, GemmConfigPrefillIntrawave, GroupSize1D, GroupSize2D128N, ColumnMajor>
>;
// clang-format on

// Test suite for ABQuant split-K Prefill
TYPED_TEST_SUITE(TestCkTileGemmABQuant, ABQuantSplitKPrefillTypes);

// ABQuant split-K tests
TYPED_TEST(TestCkTileGemmABQuant, ABQuantGroupedSplitK2Test)
{
    // K=1024 for split_k=2: 1024/2=512=4x128
    this->run_test_with_validation(128, 128, 1024, 2);
}

TYPED_TEST(TestCkTileGemmABQuant, ABQuantGroupedSplitK3Test)
{
    // K=3072 for split_k=3: 3072/3=1024=8x128
    this->run_test_with_validation(128, 128, 3072, 3);
}

TYPED_TEST(TestCkTileGemmABQuant, ABQuantGroupedSplitK4Test)
{
    // K=2048 for split_k=4: 2048/4=512=4x128
    this->run_test_with_validation(128, 128, 2048, 4);
}

TYPED_TEST(TestCkTileGemmABQuant, ABQuantGroupedSplitK5Test)
{
    // K=2560 for split_k=5: 2560/5=512=4x128
    this->run_test_with_validation(128, 128, 2560, 5);
}
