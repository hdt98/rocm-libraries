// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_tile/tensor/descriptor.hpp"
#include "unified_tile/tensor/view.hpp"
#include "unified_tile/distribution/distribution.hpp"
#include "unified_tile/tensor/window.hpp"
#include "unified_tile/ops/load.hpp"
#include "unified_tile/ops/store.hpp"

#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

using DataType = _Float16;

static constexpr int kBlockSize = 256;
static constexpr int kM = 128;
static constexpr int kK = 64;
static constexpr int kVecSize = 8;
static constexpr int kNumResults = 4;

__global__ void store_test_kernel(const DataType* p_src,
                                   DataType* p_dst,
                                   int m_size,
                                   int k_size,
                                   int* results)
{
    using namespace unified_tile;

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    auto src_desc = descriptor::make_descriptor(m_size, k_size);
    auto dst_desc = descriptor::make_descriptor(m_size, k_size);
#else
    auto src_desc =
        descriptor::make_aliased_descriptor<"M", "K">(m_size, k_size);
    auto dst_desc =
        descriptor::make_aliased_descriptor<"M", "K">(m_size, k_size);
#endif

    auto src_view = view::make_tensor_view<address_space::global>(
        const_cast<DataType*>(p_src), src_desc);
    auto dst_view = view::make_tensor_view<address_space::global>(
        p_dst, dst_desc);

    constexpr auto dstr =
        distribution::make_block_copy_a_distribution<
            kBlockSize, kM, kK, kVecSize>();

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    auto src_win = window::make_tile_window(
        src_view,
        ck_tile::make_tuple(ck_tile::number<kM>{}, ck_tile::number<kK>{}),
        ck_tile::multi_index<2>{0, 0},
        dstr);
    auto dst_win = window::make_tile_window(
        dst_view,
        ck_tile::make_tuple(ck_tile::number<kM>{}, ck_tile::number<kK>{}),
        ck_tile::multi_index<2>{0, 0},
        dstr);
#else
    auto src_win = window::make_tile_window(
        src_view,
        mint::nd_index<2>{kM, kK},
        mint::nd_index<2>{0, 0},
        dstr);
    auto dst_win = window::make_tile_window(
        dst_view,
        mint::nd_index<2>{kM, kK},
        mint::nd_index<2>{0, 0},
        dstr);
#endif

    // Test 0: Load succeeds
    auto tile = ops::load_tile(src_win);
    results[0] = 1;

    // Test 1: Store succeeds
    ops::store_tile(dst_win, tile);
    results[1] = 1;

    __syncthreads();

    // Test 2-3: Data integrity checks (thread 0 only)
    if(threadIdx.x == 0)
    {
        // Test 2: Check first element
        results[2] = (static_cast<float>(p_dst[0]) ==
                      static_cast<float>(p_src[0]))
                         ? 1
                         : 0;

        // Test 3: Check multiple positions across the tile
        bool all_match = true;
        const int positions[] = {0, 1, kK - 1, kK, kK * 2,
                                 (kM / 2) * k_size + kK / 2,
                                 (kM - 1) * k_size + kK - 1};
        for(int i = 0; i < 7; ++i)
        {
            if(static_cast<float>(p_dst[positions[i]]) !=
               static_cast<float>(p_src[positions[i]]))
            {
                all_match = false;
                break;
            }
        }
        results[3] = all_match ? 1 : 0;
    }
}

class TestUnifiedTileStore : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        constexpr int kTotalElems = kM * kK;

        ck_tile::DeviceMem src_buf(kTotalElems * sizeof(DataType));
        ck_tile::DeviceMem dst_buf(kTotalElems * sizeof(DataType));
        ck_tile::DeviceMem result_buf(kNumResults * sizeof(int));

        // Fill source with known pattern
        std::vector<DataType> h_src(kTotalElems);
        for(int i = 0; i < kTotalElems; ++i)
        {
            h_src[i] = static_cast<DataType>(static_cast<float>(i % 64));
        }
        src_buf.ToDevice(h_src.data());
        dst_buf.SetZero();
        result_buf.SetZero();

        store_test_kernel<<<1, kBlockSize>>>(
            reinterpret_cast<const DataType*>(src_buf.GetDeviceBuffer()),
            reinterpret_cast<DataType*>(dst_buf.GetDeviceBuffer()),
            kM, kK,
            reinterpret_cast<int*>(result_buf.GetDeviceBuffer()));
        HIP_CHECK_ERROR(hipDeviceSynchronize());

        result_buf.FromDevice(h_results_);
    }

    int h_results_[kNumResults] = {};
};

TEST_F(TestUnifiedTileStore, LoadSucceeds)
{
    EXPECT_EQ(h_results_[0], 1) << "Load from global memory failed";
}

TEST_F(TestUnifiedTileStore, StoreSucceeds)
{
    EXPECT_EQ(h_results_[1], 1) << "Store to global memory failed";
}

TEST_F(TestUnifiedTileStore, FirstElementMatch)
{
    EXPECT_EQ(h_results_[2], 1)
        << "dst[0] != src[0] after load+store roundtrip";
}

TEST_F(TestUnifiedTileStore, MultiPositionMatch)
{
    EXPECT_EQ(h_results_[3], 1)
        << "Data mismatch at one or more positions after roundtrip";
}
