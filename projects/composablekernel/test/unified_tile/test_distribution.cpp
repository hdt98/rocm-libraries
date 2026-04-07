// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_tile/distribution/distribution.hpp"

#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

static constexpr int kNumResults = 12;

__global__ void distribution_test_kernel(int* results)
{
    using namespace unified_tile::distribution;

    // Test 0-1: Named A distribution (BS=256, M=128, K=64, Vec=8)
    {
        constexpr auto dstr =
            make_block_copy_a_distribution<256, 128, 64, 8>();
        results[0] = (get_elements_per_thread(dstr) == 32) ? 1 : 0;
        results[1] = (get_num_tile_dims(dstr) == 2) ? 1 : 0;
    }

    // Test 2-3: Named B distribution (BS=256, K=64, N=128, Vec=8)
    {
        constexpr auto dstr =
            make_block_copy_b_distribution<256, 64, 128, 8>();
        results[2] = (get_elements_per_thread(dstr) == 32) ? 1 : 0;
        results[3] = (get_num_tile_dims(dstr) == 2) ? 1 : 0;
    }

    // Test 4-5: Generic 2D (BS=128, 64x32, Vec=4)
    {
        constexpr auto dstr =
            make_block_copy_2d_distribution<128, 64, 32, 4>();
        results[4] = (get_elements_per_thread(dstr) == 16) ? 1 : 0;
        results[5] = (get_num_tile_dims(dstr) == 2) ? 1 : 0;
    }

    // Test 6-7: Generic 2D larger (BS=256, 256x32, Vec=4)
    {
        constexpr auto dstr =
            make_block_copy_2d_distribution<256, 256, 32, 4>();
        results[6] = (get_elements_per_thread(dstr) == 32) ? 1 : 0;
        results[7] = (get_num_tile_dims(dstr) == 2) ? 1 : 0;
    }

    // Test 8-11: Config struct validation (no device distribution needed)
    {
        using C1 = block_copy_2d_config<256, 128, 64, 8>;
        results[8] = (C1::kElemsPerThread == 32) ? 1 : 0;
        results[9] = (C1::kThreadsInner == 8) ? 1 : 0;

        using C2 = block_copy_2d_config<128, 64, 32, 4>;
        results[10] = (C2::kElemsPerThread == 16) ? 1 : 0;
        results[11] = (C2::kThreadsOuter == 16) ? 1 : 0;
    }
}

class TestUnifiedTileDistribution : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        ck_tile::DeviceMem result_buf(kNumResults * sizeof(int));
        result_buf.SetZero();
        distribution_test_kernel<<<1, 256>>>(
            reinterpret_cast<int*>(result_buf.GetDeviceBuffer()));
        HIP_CHECK_ERROR(hipDeviceSynchronize());
        result_buf.FromDevice(h_results_);
    }

    int h_results_[kNumResults] = {};
};

// Named A distribution
TEST_F(TestUnifiedTileDistribution, BlockCopyAElementsPerThread)
{
    EXPECT_EQ(h_results_[0], 1) << "A(256,128,64,8): expected 32 elems/thread";
}

TEST_F(TestUnifiedTileDistribution, BlockCopyANumTileDims)
{
    EXPECT_EQ(h_results_[1], 1) << "A distribution should have 2 tile dims";
}

// Named B distribution
TEST_F(TestUnifiedTileDistribution, BlockCopyBElementsPerThread)
{
    EXPECT_EQ(h_results_[2], 1) << "B(256,64,128,8): expected 32 elems/thread";
}

TEST_F(TestUnifiedTileDistribution, BlockCopyBNumTileDims)
{
    EXPECT_EQ(h_results_[3], 1) << "B distribution should have 2 tile dims";
}

// Generic 2D distribution
TEST_F(TestUnifiedTileDistribution, Generic2DSmallElements)
{
    EXPECT_EQ(h_results_[4], 1) << "2D(128,64,32,4): expected 16 elems/thread";
}

TEST_F(TestUnifiedTileDistribution, Generic2DSmallDims)
{
    EXPECT_EQ(h_results_[5], 1) << "Generic 2D should have 2 tile dims";
}

TEST_F(TestUnifiedTileDistribution, Generic2DLargeElements)
{
    EXPECT_EQ(h_results_[6], 1) << "2D(256,256,32,4): expected 32 elems/thread";
}

TEST_F(TestUnifiedTileDistribution, Generic2DLargeDims)
{
    EXPECT_EQ(h_results_[7], 1) << "Generic 2D should have 2 tile dims";
}

// Config struct (shared math)
TEST_F(TestUnifiedTileDistribution, ConfigElemsPerThread)
{
    EXPECT_EQ(h_results_[8], 1) << "Config(256,128,64,8): expected 32";
}

TEST_F(TestUnifiedTileDistribution, ConfigThreadsInner)
{
    EXPECT_EQ(h_results_[9], 1) << "Config(256,128,64,8): threads_inner=8";
}

TEST_F(TestUnifiedTileDistribution, ConfigSmallElemsPerThread)
{
    EXPECT_EQ(h_results_[10], 1) << "Config(128,64,32,4): expected 16";
}

TEST_F(TestUnifiedTileDistribution, ConfigSmallThreadsOuter)
{
    EXPECT_EQ(h_results_[11], 1) << "Config(128,64,32,4): threads_outer=16";
}
