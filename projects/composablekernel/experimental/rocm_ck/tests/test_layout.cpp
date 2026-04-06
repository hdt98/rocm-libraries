// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/layout.hpp>

#include <gtest/gtest.h>

using ::rocm_ck::isValidLayoutForRank;
using ::rocm_ck::Layout;
using ::rocm_ck::layoutName;

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

// ============================================================================
// Edge cases: high ranks
// ============================================================================

TEST(Layout, RejectsRowAndColForRankGreaterThan2)
{
    constexpr bool row_r3 = isValidLayoutForRank(Layout::Row, 3);
    constexpr bool row_r4 = isValidLayoutForRank(Layout::Row, 4);
    constexpr bool row_r6 = isValidLayoutForRank(Layout::Row, 6);
    constexpr bool col_r3 = isValidLayoutForRank(Layout::Col, 3);
    constexpr bool col_r4 = isValidLayoutForRank(Layout::Col, 4);
    constexpr bool col_r6 = isValidLayoutForRank(Layout::Col, 6);

    EXPECT_FALSE(row_r3);
    EXPECT_FALSE(row_r4);
    EXPECT_FALSE(row_r6);
    EXPECT_FALSE(col_r3);
    EXPECT_FALSE(col_r4);
    EXPECT_FALSE(col_r6);
}

TEST(Layout, RejectsContiguousForRankGreaterThan1)
{
    constexpr bool contig_r3 = isValidLayoutForRank(Layout::Contiguous, 3);
    constexpr bool contig_r4 = isValidLayoutForRank(Layout::Contiguous, 4);
    constexpr bool contig_r6 = isValidLayoutForRank(Layout::Contiguous, 6);

    EXPECT_FALSE(contig_r3);
    EXPECT_FALSE(contig_r4);
    EXPECT_FALSE(contig_r6);
}
