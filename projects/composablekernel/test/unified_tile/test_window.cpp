// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_tile/tensor/descriptor.hpp"
#include "unified_tile/tensor/view.hpp"
#include "unified_tile/distribution/distribution.hpp"
#include "unified_tile/tensor/window.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

using DataType = _Float16;

static constexpr int kNumResults = 6;

__global__ void window_test_kernel(const DataType* p_a,
                                    int m_size,
                                    int k_size,
                                    int* results)
{
    using namespace unified_tile;

    // Create descriptor, view, distribution
    auto desc = descriptor::make_descriptor(m_size, k_size);
    auto a_view =
        view::make_tensor_view<address_space::global>(
            const_cast<DataType*>(p_a), desc);

    // --- Test 0-1: A window (M=128, K=64) ---
    {
        constexpr int BS = 256, M = 128, K = 64, V = 8;
        constexpr auto dstr =
            distribution::make_block_copy_a_distribution<BS, M, K, V>();

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
        auto win = window::make_tile_window(
            a_view,
            ck_tile::make_tuple(ck_tile::number<M>{}, ck_tile::number<K>{}),
            ck_tile::multi_index<2>{0, 0},
            dstr);
#else
        auto win = window::make_tile_window(
            a_view,
            mint::nd_index<2>{M, K},
            mint::nd_index<2>{0, 0},
            dstr);
#endif
        results[0] = 1; // creation succeeded

        // Move along K
#ifdef UNIFIED_TILE_BACKEND_CK_TILE
        window::move_window(win, ck_tile::multi_index<2>{0, K});
#else
        window::move_window(win, mint::nd_index<2>{0, K});
#endif
        results[1] = 1; // move succeeded
    }

    // --- Test 2-3: B window (K=64, N=128) ---
    {
        constexpr int BS = 256, K = 64, N = 128, V = 8;
        constexpr auto dstr =
            distribution::make_block_copy_b_distribution<BS, K, N, V>();

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
        auto win = window::make_tile_window(
            a_view,
            ck_tile::make_tuple(ck_tile::number<K>{}, ck_tile::number<N>{}),
            ck_tile::multi_index<2>{0, 0},
            dstr);
#else
        auto win = window::make_tile_window(
            a_view,
            mint::nd_index<2>{K, N},
            mint::nd_index<2>{0, 0},
            dstr);
#endif
        results[2] = 1; // creation succeeded

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
        window::move_window(win, ck_tile::multi_index<2>{K, 0});
#else
        window::move_window(win, mint::nd_index<2>{K, 0});
#endif
        results[3] = 1; // move succeeded
    }

    // --- Test 4-5: Generic 2D window ---
    {
        constexpr int BS = 128, D0 = 64, D1 = 32, V = 4;
        constexpr auto dstr =
            distribution::make_block_copy_2d_distribution<BS, D0, D1, V>();

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
        auto win = window::make_tile_window(
            a_view,
            ck_tile::make_tuple(ck_tile::number<D0>{}, ck_tile::number<D1>{}),
            ck_tile::multi_index<2>{0, 0},
            dstr);
#else
        auto win = window::make_tile_window(
            a_view,
            mint::nd_index<2>{D0, D1},
            mint::nd_index<2>{0, 0},
            dstr);
#endif
        results[4] = 1;

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
        window::move_window(win, ck_tile::multi_index<2>{0, D1});
#else
        window::move_window(win, mint::nd_index<2>{0, D1});
#endif
        results[5] = 1;
    }
}

class TestUnifiedTileWindow : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        hipMalloc(&d_a_, 256 * 256 * sizeof(DataType));
        hipMemset(d_a_, 0, 256 * 256 * sizeof(DataType));
        hipMalloc(&d_results_, kNumResults * sizeof(int));
        hipMemset(d_results_, 0, kNumResults * sizeof(int));
        window_test_kernel<<<1, 256>>>(d_a_, 256, 256, d_results_);
        hipDeviceSynchronize();
        hipMemcpy(h_results_,
                  d_results_,
                  kNumResults * sizeof(int),
                  hipMemcpyDeviceToHost);
    }

    void TearDown() override
    {
        hipFree(d_a_);
        hipFree(d_results_);
    }

    DataType* d_a_ = nullptr;
    int* d_results_ = nullptr;
    int h_results_[kNumResults] = {};
};

TEST_F(TestUnifiedTileWindow, AWindowCreation)
{
    EXPECT_EQ(h_results_[0], 1) << "A window creation failed";
}

TEST_F(TestUnifiedTileWindow, AWindowMove)
{
    EXPECT_EQ(h_results_[1], 1) << "A window move failed";
}

TEST_F(TestUnifiedTileWindow, BWindowCreation)
{
    EXPECT_EQ(h_results_[2], 1) << "B window creation failed";
}

TEST_F(TestUnifiedTileWindow, BWindowMove)
{
    EXPECT_EQ(h_results_[3], 1) << "B window move failed";
}

TEST_F(TestUnifiedTileWindow, Generic2DWindowCreation)
{
    EXPECT_EQ(h_results_[4], 1) << "Generic 2D window creation failed";
}

TEST_F(TestUnifiedTileWindow, Generic2DWindowMove)
{
    EXPECT_EQ(h_results_[5], 1) << "Generic 2D window move failed";
}
