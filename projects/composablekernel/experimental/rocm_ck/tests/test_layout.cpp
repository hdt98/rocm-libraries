// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/layout.hpp>

#include <gtest/gtest.h>

using namespace rocm_ck;

// ============================================================================
// layoutName
// ============================================================================

TEST(Layout, MapsEnumValuesToExpectedStrings)
{
    EXPECT_STREQ(layoutName(Layout::Contiguous), "Contiguous");
    EXPECT_STREQ(layoutName(Layout::Row), "Row");
    EXPECT_STREQ(layoutName(Layout::Col), "Col");
    EXPECT_STREQ(layoutName(Layout::Auto), "Auto");
}

// ============================================================================
// isValidLayoutForRank
// ============================================================================

TEST(Layout, AllowsContiguousOnlyForRank1)
{
    constexpr bool r1 = isValidLayoutForRank(Layout::Contiguous, 1);
    constexpr bool r2 = isValidLayoutForRank(Layout::Contiguous, 2);
    constexpr bool r0 = isValidLayoutForRank(Layout::Contiguous, 0);
    EXPECT_TRUE(r1);
    EXPECT_FALSE(r2);
    EXPECT_FALSE(r0);
}

TEST(Layout, AllowsRowAndColOnlyForRank2)
{
    constexpr bool row_r1 = isValidLayoutForRank(Layout::Row, 1);
    constexpr bool row_r2 = isValidLayoutForRank(Layout::Row, 2);
    constexpr bool col_r1 = isValidLayoutForRank(Layout::Col, 1);
    constexpr bool col_r2 = isValidLayoutForRank(Layout::Col, 2);

    EXPECT_FALSE(row_r1);
    EXPECT_TRUE(row_r2);
    EXPECT_FALSE(col_r1);
    EXPECT_TRUE(col_r2);
}

TEST(Layout, RejectsAutoForAllRanks)
{
    constexpr bool auto_r0 = isValidLayoutForRank(Layout::Auto, 0);
    constexpr bool auto_r1 = isValidLayoutForRank(Layout::Auto, 1);
    constexpr bool auto_r2 = isValidLayoutForRank(Layout::Auto, 2);
    EXPECT_FALSE(auto_r0);
    EXPECT_FALSE(auto_r1);
    EXPECT_FALSE(auto_r2);
}
