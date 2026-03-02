// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>
#include <tuple>
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/utility/functional.hpp"

using namespace ck_tile;

// ============================================================================
// static_ford Tests
// ============================================================================

TEST(CkTileStaticFord, Basic2D)
{
    std::vector<std::pair<index_t, index_t>> visited;

    static_ford<sequence<2, 3>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        constexpr index_t j = multi_id[number<1>{}];
        visited.emplace_back(i, j);
    });

    ASSERT_EQ(visited.size(), 6u);
    EXPECT_EQ(visited[0], std::make_pair(0, 0));
    EXPECT_EQ(visited[1], std::make_pair(0, 1));
    EXPECT_EQ(visited[2], std::make_pair(0, 2));
    EXPECT_EQ(visited[3], std::make_pair(1, 0));
    EXPECT_EQ(visited[4], std::make_pair(1, 1));
    EXPECT_EQ(visited[5], std::make_pair(1, 2));
}

TEST(CkTileStaticFord, ReversedOrder2D)
{
    std::vector<std::pair<index_t, index_t>> visited;

    // Order (1, 0): dim 1 is outer, dim 0 is inner (column-major)
    static_ford<sequence<2, 3>, sequence<1, 0>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        constexpr index_t j = multi_id[number<1>{}];
        visited.emplace_back(i, j);
    });

    ASSERT_EQ(visited.size(), 6u);
    EXPECT_EQ(visited[0], std::make_pair(0, 0));
    EXPECT_EQ(visited[1], std::make_pair(1, 0));
    EXPECT_EQ(visited[2], std::make_pair(0, 1));
    EXPECT_EQ(visited[3], std::make_pair(1, 1));
    EXPECT_EQ(visited[4], std::make_pair(0, 2));
    EXPECT_EQ(visited[5], std::make_pair(1, 2));
}

TEST(CkTileStaticFord, Basic3D)
{
    std::vector<std::tuple<index_t, index_t, index_t>> visited;

    static_ford<sequence<2, 3, 2>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        constexpr index_t j = multi_id[number<1>{}];
        constexpr index_t k = multi_id[number<2>{}];
        visited.emplace_back(i, j, k);
    });

    ASSERT_EQ(visited.size(), 12u);
    EXPECT_EQ(visited[0], std::make_tuple(0, 0, 0));
    EXPECT_EQ(visited[1], std::make_tuple(0, 0, 1));
    EXPECT_EQ(visited[2], std::make_tuple(0, 1, 0));
    EXPECT_EQ(visited[3], std::make_tuple(0, 1, 1));
    EXPECT_EQ(visited[4], std::make_tuple(0, 2, 0));
    EXPECT_EQ(visited[5], std::make_tuple(0, 2, 1));
    EXPECT_EQ(visited[6], std::make_tuple(1, 0, 0));
    EXPECT_EQ(visited[7], std::make_tuple(1, 0, 1));
    EXPECT_EQ(visited[8], std::make_tuple(1, 1, 0));
    EXPECT_EQ(visited[9], std::make_tuple(1, 1, 1));
    EXPECT_EQ(visited[10], std::make_tuple(1, 2, 0));
    EXPECT_EQ(visited[11], std::make_tuple(1, 2, 1));
}

TEST(CkTileStaticFord, NonTrivialOrder3D_201)
{
    std::vector<std::tuple<index_t, index_t, index_t>> visited;

    // Orders<2,0,1>: dim 2 outermost, dim 0 middle, dim 1 innermost
    static_ford<sequence<2, 3, 4>, sequence<2, 0, 1>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        constexpr index_t j = multi_id[number<1>{}];
        constexpr index_t k = multi_id[number<2>{}];
        visited.emplace_back(i, j, k);
    });

    ASSERT_EQ(visited.size(), 24u);
    // With orders (2,0,1): k varies slowest, then i, then j fastest
    EXPECT_EQ(visited[0], std::make_tuple(0, 0, 0));
    EXPECT_EQ(visited[1], std::make_tuple(0, 1, 0));
    EXPECT_EQ(visited[2], std::make_tuple(0, 2, 0));
    EXPECT_EQ(visited[3], std::make_tuple(1, 0, 0));
    EXPECT_EQ(visited[4], std::make_tuple(1, 1, 0));
    EXPECT_EQ(visited[5], std::make_tuple(1, 2, 0));
    EXPECT_EQ(visited[6], std::make_tuple(0, 0, 1));
}

TEST(CkTileStaticFord, DimensionWithSizeOne)
{
    std::vector<std::tuple<index_t, index_t, index_t>> visited;

    static_ford<sequence<2, 1, 3>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        constexpr index_t j = multi_id[number<1>{}];
        constexpr index_t k = multi_id[number<2>{}];
        visited.emplace_back(i, j, k);
    });

    ASSERT_EQ(visited.size(), 6u);
    EXPECT_EQ(visited[0], std::make_tuple(0, 0, 0));
    EXPECT_EQ(visited[1], std::make_tuple(0, 0, 1));
    EXPECT_EQ(visited[2], std::make_tuple(0, 0, 2));
    EXPECT_EQ(visited[3], std::make_tuple(1, 0, 0));
    EXPECT_EQ(visited[4], std::make_tuple(1, 0, 1));
    EXPECT_EQ(visited[5], std::make_tuple(1, 0, 2));
}

TEST(CkTileStaticFord, SingleDimension)
{
    std::vector<index_t> visited;

    static_ford<sequence<5>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        visited.push_back(i);
    });

    ASSERT_EQ(visited.size(), 5u);
    for(index_t i = 0; i < 5; ++i)
    {
        EXPECT_EQ(visited[i], i);
    }
}

TEST(CkTileStaticFord, HigherDimensions4D)
{
    std::vector<std::tuple<index_t, index_t, index_t, index_t>> visited;

    static_ford<sequence<2, 2, 2, 2>>{}([&](auto multi_id) {
        constexpr index_t a = multi_id[number<0>{}];
        constexpr index_t b = multi_id[number<1>{}];
        constexpr index_t c = multi_id[number<2>{}];
        constexpr index_t d = multi_id[number<3>{}];
        visited.emplace_back(a, b, c, d);
    });

    ASSERT_EQ(visited.size(), 16u);
    // Verify lexicographic order: (0,0,0,0), (0,0,0,1), (0,0,1,0), ...
    EXPECT_EQ(visited[0], std::make_tuple(0, 0, 0, 0));
    EXPECT_EQ(visited[1], std::make_tuple(0, 0, 0, 1));
    EXPECT_EQ(visited[2], std::make_tuple(0, 0, 1, 0));
    EXPECT_EQ(visited[3], std::make_tuple(0, 0, 1, 1));
    EXPECT_EQ(visited[4], std::make_tuple(0, 1, 0, 0));
    EXPECT_EQ(visited[15], std::make_tuple(1, 1, 1, 1));
}

TEST(CkTileStaticFord, NonTrivialOrder4D)
{
    std::vector<std::tuple<index_t, index_t, index_t, index_t>> visited;

    // Orders<3,1,0,2>: dim3 outermost, dim1, dim0, dim2 innermost
    static_ford<sequence<2, 3, 2, 4>, sequence<3, 1, 0, 2>>{}([&](auto multi_id) {
        constexpr index_t a = multi_id[number<0>{}];
        constexpr index_t b = multi_id[number<1>{}];
        constexpr index_t c = multi_id[number<2>{}];
        constexpr index_t d = multi_id[number<3>{}];
        visited.emplace_back(a, b, c, d);
    });

    ASSERT_EQ(visited.size(), 48u);
    // Verify against nested for-loop reference with same ordering
    index_t idx = 0;
    for(index_t d = 0; d < 4; ++d)             // dim 3 outermost
        for(index_t b = 0; b < 3; ++b)         // dim 1
            for(index_t a = 0; a < 2; ++a)     // dim 0
                for(index_t c = 0; c < 2; ++c) // dim 2 innermost
                {
                    EXPECT_EQ(visited[idx], std::make_tuple(a, b, c, d))
                        << "Mismatch at index " << idx;
                    ++idx;
                }
}

TEST(CkTileStaticFord, TotalIterationCount)
{
    index_t count = 0;
    static_ford<sequence<3, 5, 2>>{}([&](auto) { ++count; });
    EXPECT_EQ(count, 30);
}

// ============================================================================
// ford Tests (runtime)
// ============================================================================

TEST(CkTileFord, Basic2D)
{
    std::vector<std::pair<index_t, index_t>> visited;

    ford<sequence<2, 3>>{}([&](auto multi_id) { visited.emplace_back(multi_id[0], multi_id[1]); });

    ASSERT_EQ(visited.size(), 6u);
    EXPECT_EQ(visited[0], std::make_pair(0, 0));
    EXPECT_EQ(visited[1], std::make_pair(0, 1));
    EXPECT_EQ(visited[2], std::make_pair(0, 2));
    EXPECT_EQ(visited[3], std::make_pair(1, 0));
    EXPECT_EQ(visited[4], std::make_pair(1, 1));
    EXPECT_EQ(visited[5], std::make_pair(1, 2));
}

TEST(CkTileFord, ReversedOrder2D)
{
    std::vector<std::pair<index_t, index_t>> visited;

    ford<sequence<2, 3>, sequence<1, 0>>{}(
        [&](auto multi_id) { visited.emplace_back(multi_id[0], multi_id[1]); });

    ASSERT_EQ(visited.size(), 6u);
    EXPECT_EQ(visited[0], std::make_pair(0, 0));
    EXPECT_EQ(visited[1], std::make_pair(1, 0));
    EXPECT_EQ(visited[2], std::make_pair(0, 1));
    EXPECT_EQ(visited[3], std::make_pair(1, 1));
    EXPECT_EQ(visited[4], std::make_pair(0, 2));
    EXPECT_EQ(visited[5], std::make_pair(1, 2));
}

TEST(CkTileFord, SingleDimension)
{
    std::vector<index_t> visited;

    ford<sequence<5>>{}([&](auto multi_id) { visited.push_back(multi_id[0]); });

    ASSERT_EQ(visited.size(), 5u);
    for(index_t i = 0; i < 5; ++i)
    {
        EXPECT_EQ(visited[i], i);
    }
}

TEST(CkTileFord, NonTrivialOrder3D)
{
    std::vector<std::tuple<index_t, index_t, index_t>> ford_visited;
    std::vector<std::tuple<index_t, index_t, index_t>> reference;

    // Orders<2,0,1>: dim 2 outermost, dim 0 middle, dim 1 innermost
    ford<sequence<2, 3, 4>, sequence<2, 0, 1>>{}(
        [&](auto multi_id) { ford_visited.emplace_back(multi_id[0], multi_id[1], multi_id[2]); });

    // Build reference with nested loops in the same order
    for(index_t k = 0; k < 4; ++k)         // dim 2 outermost
        for(index_t i = 0; i < 2; ++i)     // dim 0
            for(index_t j = 0; j < 3; ++j) // dim 1 innermost
                reference.emplace_back(i, j, k);

    ASSERT_EQ(ford_visited.size(), reference.size());
    for(size_t idx = 0; idx < ford_visited.size(); ++idx)
    {
        EXPECT_EQ(ford_visited[idx], reference[idx]) << "Mismatch at index " << idx;
    }
}

TEST(CkTileFord, NonTrivialOrder4D)
{
    std::vector<std::tuple<index_t, index_t, index_t, index_t>> ford_visited;
    std::vector<std::tuple<index_t, index_t, index_t, index_t>> reference;

    // Orders<3,1,0,2>: dim3 outermost, dim1, dim0, dim2 innermost
    ford<sequence<2, 3, 2, 4>, sequence<3, 1, 0, 2>>{}([&](auto multi_id) {
        ford_visited.emplace_back(multi_id[0], multi_id[1], multi_id[2], multi_id[3]);
    });

    for(index_t d = 0; d < 4; ++d)
        for(index_t b = 0; b < 3; ++b)
            for(index_t a = 0; a < 2; ++a)
                for(index_t c = 0; c < 2; ++c)
                    reference.emplace_back(a, b, c, d);

    ASSERT_EQ(ford_visited.size(), reference.size());
    for(size_t idx = 0; idx < ford_visited.size(); ++idx)
    {
        EXPECT_EQ(ford_visited[idx], reference[idx]) << "Mismatch at index " << idx;
    }
}

TEST(CkTileFord, ConsistencyWithStaticFord)
{
    // Verify ford and static_ford produce the same iteration order
    std::vector<std::pair<index_t, index_t>> ford_visited;
    std::vector<std::pair<index_t, index_t>> static_ford_visited;

    ford<sequence<3, 4>, sequence<1, 0>>{}(
        [&](auto multi_id) { ford_visited.emplace_back(multi_id[0], multi_id[1]); });

    static_ford<sequence<3, 4>, sequence<1, 0>>{}([&](auto multi_id) {
        constexpr index_t i = multi_id[number<0>{}];
        constexpr index_t j = multi_id[number<1>{}];
        static_ford_visited.emplace_back(i, j);
    });

    ASSERT_EQ(ford_visited.size(), static_ford_visited.size());
    for(size_t idx = 0; idx < ford_visited.size(); ++idx)
    {
        EXPECT_EQ(ford_visited[idx], static_ford_visited[idx]) << "Mismatch at index " << idx;
    }
}

// ============================================================================
// index_decomposer Tests
// ============================================================================

TEST(CkTileIndexDecomposer, Strides2D)
{
    using Decomposer = detail::index_decomposer<sequence<2, 3>, make_index_sequence<2>>;
    static_assert(Decomposer::strides[0] == 3);
    static_assert(Decomposer::strides[1] == 1);
}

TEST(CkTileIndexDecomposer, Strides3D)
{
    using Decomposer = detail::index_decomposer<sequence<2, 3, 4>, make_index_sequence<3>>;
    static_assert(Decomposer::strides[0] == 12);
    static_assert(Decomposer::strides[1] == 4);
    static_assert(Decomposer::strides[2] == 1);
}

TEST(CkTileIndexDecomposer, Decompose2D)
{
    using Decomposer = detail::index_decomposer<sequence<2, 3>, make_index_sequence<2>>;

    using R0 = typename Decomposer::template decompose<0>;
    static_assert(std::is_same_v<R0, sequence<0, 0>>);

    using R5 = typename Decomposer::template decompose<5>;
    static_assert(std::is_same_v<R5, sequence<1, 2>>);
}

TEST(CkTileIndexDecomposer, Decompose3D)
{
    using Decomposer = detail::index_decomposer<sequence<2, 3, 4>, make_index_sequence<3>>;

    using R0 = typename Decomposer::template decompose<0>;
    static_assert(std::is_same_v<R0, sequence<0, 0, 0>>);

    using R23 = typename Decomposer::template decompose<23>;
    static_assert(std::is_same_v<R23, sequence<1, 2, 3>>);
}

TEST(CkTileIndexDecomposer, DecomposeRuntimeWithReorder)
{
    // Ordered lengths for Orders<1,0> applied to Lengths<2,3>: reorders to {3,2}
    using Decomposer = detail::index_decomposer<sequence<3, 2>, make_index_sequence<2>>;
    using Orders     = sequence<1, 0>;

    // linear_idx=3 with ordered lengths {3,2}: (3/2)%3=1, (3/1)%2=1 -> ordered=(1,1)
    // After reorder with Orders<1,0>: result[1]=1, result[0]=1 -> (1,1)
    detail::index_array<2> result{};
    Decomposer::template decompose_runtime<Orders>(3, result);
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[1], 1);

    // linear_idx=0: ordered=(0,0) -> (0,0)
    detail::index_array<2> result0{};
    Decomposer::template decompose_runtime<Orders>(0, result0);
    EXPECT_EQ(result0[0], 0);
    EXPECT_EQ(result0[1], 0);

    // linear_idx=5 (last): ordered=(2,1) -> reordered=(1,2)
    detail::index_array<2> result5{};
    Decomposer::template decompose_runtime<Orders>(5, result5);
    EXPECT_EQ(result5[0], 1);
    EXPECT_EQ(result5[1], 2);
}

TEST(CkTileIndexDecomposer, DimensionWithOne)
{
    using Decomposer = detail::index_decomposer<sequence<2, 1, 3>, make_index_sequence<3>>;
    static_assert(Decomposer::strides[0] == 3);
    static_assert(Decomposer::strides[1] == 3);
    static_assert(Decomposer::strides[2] == 1);
}
