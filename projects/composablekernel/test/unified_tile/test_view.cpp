// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_tile/tensor/descriptor.hpp"
#include "unified_tile/tensor/view.hpp"

#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

using data_t = _Float16;

// Device kernel: create views and verify their sizes
__global__ void view_test_kernel(const data_t* p_global,
                                 int M,
                                 int K,
                                 int* results)
{
    using namespace unified_tile::descriptor;
    using namespace unified_tile::view;

    // Test 0: Global view from packed descriptor [128, 64] -> size == 8192
    {
        auto desc  = make_descriptor(M, K);
        auto gview = make_tensor_view<unified_tile::address_space::global>(
            const_cast<data_t*>(p_global), desc);
        auto size  = get_view_size(gview);
        results[0] = (size == M * K) ? 1 : 0;
    }

    // Test 1: Global view with strides -> size == M * K
    {
#ifdef UNIFIED_TILE_BACKEND_CK_TILE
        auto view = make_tensor_view<unified_tile::address_space::global>(
            const_cast<data_t*>(p_global),
            ck_tile::make_tuple(M, K),
            ck_tile::make_tuple(K, 1));
#else
        auto desc = make_descriptor_with_strides(
            mint::nd_index<2>{M, K}, mint::nd_index<2>{K, 1});
        auto view = make_tensor_view<unified_tile::address_space::global>(
            const_cast<data_t*>(p_global), desc);
#endif
        auto size  = get_view_size(view);
        results[1] = (size == M * K) ? 1 : 0;
    }

    // Test 2: Padding (CK_TILE pads, MINT returns unchanged)
    //   For a [100, 50] tensor padded to [128, 64] tile,
    //   CK_TILE: padded size >= 100*50 (due to right-pad transforms)
    //   MINT: size == 100*50 (padding is no-op)
    {
        int M2    = 100;
        int K2    = 50;
        auto desc = make_descriptor(M2, K2);
        auto view = make_tensor_view<unified_tile::address_space::global>(
            const_cast<data_t*>(p_global), desc);
#ifdef UNIFIED_TILE_BACKEND_CK_TILE
        auto padded = pad_view(
            view,
            ck_tile::make_tuple(ck_tile::number<128>{}, ck_tile::number<64>{}),
            ck_tile::sequence<true, true>{});
        auto padded_size = get_view_size(padded);
        // Padded view keeps original buffer size (padding adds transforms,
        // not memory). Element space size stays M2 * K2 = 5000.
        results[2] = (padded_size == M2 * K2) ? 1 : 0;
#else
        auto padded      = pad_view(view, 0, 0);
        auto padded_size = get_view_size(padded);
        // MINT: no-op, size stays M2 * K2 = 5000
        results[2] = (padded_size == M2 * K2) ? 1 : 0;
#endif
    }
}

class TestUnifiedTileView : public ::testing::Test
{
    protected:
    static constexpr int M         = 128;
    static constexpr int K         = 64;
    static constexpr int kNumTests = 3;
    int host_results[kNumTests]    = {0};

    void SetUp() override
    {
        ck_tile::DeviceMem tensor_buf(M * K * sizeof(data_t));
        tensor_buf.SetZero();

        ck_tile::DeviceMem result_buf(kNumTests * sizeof(int));
        result_buf.SetZero();

        view_test_kernel<<<1, 1>>>(
            reinterpret_cast<const data_t*>(tensor_buf.GetDeviceBuffer()),
            M,
            K,
            reinterpret_cast<int*>(result_buf.GetDeviceBuffer()));
        HIP_CHECK_ERROR(hipDeviceSynchronize());

        result_buf.FromDevice(host_results);
    }
};

TEST_F(TestUnifiedTileView, GlobalViewSize)
{
    EXPECT_EQ(host_results[0], 1);
}

TEST_F(TestUnifiedTileView, StridedViewSize)
{
    EXPECT_EQ(host_results[1], 1);
}

TEST_F(TestUnifiedTileView, PaddedViewSize)
{
    EXPECT_EQ(host_results[2], 1);
}
