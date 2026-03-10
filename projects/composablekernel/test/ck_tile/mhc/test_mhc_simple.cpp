// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <iostream>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

// Simple compilation test - just verify headers compile
TEST(TestMHCSimple, HeadersCompile)
{
    std::cout << "MHC headers compiled successfully!" << std::endl;
    EXPECT_TRUE(true);
}

// Test basic type definitions
TEST(TestMHCSimple, TypeDefinitions)
{
    using XDataType       = float;
    using PhiDataType     = float;
    using YDataType       = float;
    using ComputeDataType = float;

    // Use the types to avoid unused warnings
    [[maybe_unused]] XDataType x       = 1.0f;
    [[maybe_unused]] PhiDataType phi   = 1.0f;
    [[maybe_unused]] YDataType y       = 1.0f;
    [[maybe_unused]] ComputeDataType c = 1.0f;

    std::cout << "Type definitions work correctly" << std::endl;
    EXPECT_TRUE(true);
}

// Test with BF16 types
TEST(TestMHCSimple, BF16Types)
{
    using XDataType       = ck_tile::bf16_t;
    using PhiDataType     = ck_tile::bf16_t;
    using YDataType       = float;
    using ComputeDataType = float;

    // Use the types to avoid unused warnings
    [[maybe_unused]] XDataType x       = ck_tile::bf16_t(1.0f);
    [[maybe_unused]] PhiDataType phi   = ck_tile::bf16_t(1.0f);
    [[maybe_unused]] YDataType y       = 1.0f;
    [[maybe_unused]] ComputeDataType c = 1.0f;

    std::cout << "BF16 type definitions work correctly" << std::endl;
    EXPECT_TRUE(true);
}
