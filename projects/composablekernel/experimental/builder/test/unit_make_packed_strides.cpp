// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ck_tile/builder/testing/conv/args.hpp"

namespace {

using ck_tile::builder::test::Extent;
using ck_tile::builder::test::detail::make_packed_strides_for_order;

TEST(MakePackedStridesForOrder, RowMajor2D)
{
    const Extent<2> lengths{3, 4};
    const std::array<size_t, 2> order{0, 1};

    const auto strides = make_packed_strides_for_order<2>(lengths, order);

    EXPECT_EQ(strides[0], 4u);
    EXPECT_EQ(strides[1], 1u);
}

TEST(MakePackedStridesForOrder, ColMajor2D)
{
    const Extent<2> lengths{3, 4};
    const std::array<size_t, 2> order{1, 0};

    const auto strides = make_packed_strides_for_order<2>(lengths, order);

    EXPECT_EQ(strides[0], 1u);
    EXPECT_EQ(strides[1], 3u);
}

TEST(MakePackedStridesForOrder, Trivial1D)
{
    const Extent<1> lengths{7};
    const std::array<size_t, 1> order{0};

    const auto strides = make_packed_strides_for_order<1>(lengths, order);

    EXPECT_EQ(strides[0], 1u);
}

TEST(MakePackedStridesForOrder, RowMajor3D)
{
    const Extent<3> lengths{2, 3, 4};
    const std::array<size_t, 3> order{0, 1, 2};

    const auto strides = make_packed_strides_for_order<3>(lengths, order);

    EXPECT_EQ(strides[0], 12u); // 3 * 4
    EXPECT_EQ(strides[1], 4u);  // 4
    EXPECT_EQ(strides[2], 1u);
}

TEST(MakePackedStridesForOrder, Permuted3D)
{
    // order {1, 2, 0}: dim 1 is outermost, dim 2 is middle, dim 0 is innermost
    const Extent<3> lengths{2, 3, 4};
    const std::array<size_t, 3> order{1, 2, 0};

    const auto strides = make_packed_strides_for_order<3>(lengths, order);

    EXPECT_EQ(strides[0], 1u); // innermost
    EXPECT_EQ(strides[1], 8u); // 4 * 2
    EXPECT_EQ(strides[2], 2u); // 2
}

// Convolution-like 5D tensor: G=2, N=3, C=4, H=5, W=6
// with GNCHW layout (order {0,1,2,3,4} = row-major)
TEST(MakePackedStridesForOrder, ConvLike5D_GNCHW)
{
    const Extent<5> lengths{2, 3, 4, 5, 6};
    const std::array<size_t, 5> order{0, 1, 2, 3, 4};

    const auto strides = make_packed_strides_for_order<5>(lengths, order);

    EXPECT_EQ(strides[4], 1u);                // W
    EXPECT_EQ(strides[3], 6u);                // W
    EXPECT_EQ(strides[2], 6u * 5u);           // H*W
    EXPECT_EQ(strides[1], 6u * 5u * 4u);      // C*H*W
    EXPECT_EQ(strides[0], 6u * 5u * 4u * 3u); // N*C*H*W
}

// NHWC-like: order {0, 2, 3, 1} means N outermost, then H, W, C innermost
TEST(MakePackedStridesForOrder, ConvLike4D_NHWC)
{
    const Extent<4> lengths{2, 3, 4, 5};           // N=2, C=3, H=4, W=5
    const std::array<size_t, 4> order{0, 2, 3, 1}; // N, H, W, C

    const auto strides = make_packed_strides_for_order<4>(lengths, order);

    EXPECT_EQ(strides[1], 1u);           // C innermost
    EXPECT_EQ(strides[3], 3u);           // W: stride = C
    EXPECT_EQ(strides[2], 3u * 5u);      // H: stride = W*C
    EXPECT_EQ(strides[0], 3u * 5u * 4u); // N: stride = H*W*C
}

} // namespace
