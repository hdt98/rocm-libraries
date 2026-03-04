// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <stdexcept>

#include "ck_tile/builder/testing/conv/args.hpp"

namespace {

using ck_tile::long_index_t;
using ck_tile::builder::test::detail::compute_output_spatial;

// output = floor((input + pad_l + pad_r - dilation*(filter-1) - 1) / stride) + 1

TEST(ComputeOutputSpatial, Basic1D)
{
    // input=8, filter=3, stride=1, dilation=1, pad=0 → (8-3)/1+1 = 6
    const auto result = compute_output_spatial<1>({8}, {3}, {1}, {1}, {0}, {0});
    EXPECT_EQ(result[0], 6);
}

TEST(ComputeOutputSpatial, WithStride)
{
    // input=8, filter=3, stride=2, dilation=1, pad=0 → (8-3)/2+1 = 3
    const auto result = compute_output_spatial<1>({8}, {3}, {2}, {1}, {0}, {0});
    EXPECT_EQ(result[0], 3);
}

TEST(ComputeOutputSpatial, WithDilation)
{
    // effective_filter = 2*(3-1)+1 = 5
    // input=8, pad=0 → (8-5)/1+1 = 4
    const auto result = compute_output_spatial<1>({8}, {3}, {1}, {2}, {0}, {0});
    EXPECT_EQ(result[0], 4);
}

TEST(ComputeOutputSpatial, WithPadding)
{
    // input=8, filter=3, pad_l=1, pad_r=1 → (8+1+1-3)/1+1 = 8
    const auto result = compute_output_spatial<1>({8}, {3}, {1}, {1}, {1}, {1});
    EXPECT_EQ(result[0], 8);
}

TEST(ComputeOutputSpatial, StrideDilationPadding)
{
    // effective_filter = 2*(5-1)+1 = 9
    // (16+1+1-9)/2+1 = 9/2+1 = 4+1 = 5
    const auto result = compute_output_spatial<1>({16}, {5}, {2}, {2}, {1}, {1});
    EXPECT_EQ(result[0], 5);
}

TEST(ComputeOutputSpatial, Case2D)
{
    // height: (64-5)/1+1 = 60
    // width:  (56-3)/1+1 = 54
    const auto result = compute_output_spatial<2>({64, 56}, {5, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0});
    EXPECT_EQ(result[0], 60);
    EXPECT_EQ(result[1], 54);
}

TEST(ComputeOutputSpatial, Case2D_Stride2)
{
    // height: (64-5)/2+1 = 59/2+1 = 29+1 = 30
    // width:  (56-3)/2+1 = 53/2+1 = 26+1 = 27
    const auto result = compute_output_spatial<2>({64, 56}, {5, 3}, {2, 2}, {1, 1}, {0, 0}, {0, 0});
    EXPECT_EQ(result[0], 30);
    EXPECT_EQ(result[1], 27);
}

TEST(ComputeOutputSpatial, InvalidStrideThrows)
{
    EXPECT_THROW(compute_output_spatial<1>({8}, {3}, {0}, {1}, {0}, {0}), std::runtime_error);
}

TEST(ComputeOutputSpatial, NegativeOutputThrows)
{
    // input=4, filter=7 → 4-7 = -3 < 0
    EXPECT_THROW(compute_output_spatial<1>({4}, {7}, {1}, {1}, {0}, {0}), std::runtime_error);
}

} // namespace
