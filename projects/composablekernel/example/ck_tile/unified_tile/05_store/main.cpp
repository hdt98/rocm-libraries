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
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

using DataType = _Float16;

static constexpr int kBlockSize = 256;
static constexpr int kMPerBlock = 128;
static constexpr int kKPerBlock = 64;
static constexpr int kVecSize = 8;

// Load from src (global) -> VGPR -> store to dst (global)
// Then verify data matches on device.
__global__ void store_test_kernel(const DataType* p_src,
                                   DataType* p_dst,
                                   int m_size,
                                   int k_size,
                                   int* results)
{
    using namespace unified_tile;

    // Create descriptors and views
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

    // Create distribution
    constexpr auto dstr =
        distribution::make_block_copy_a_distribution<
            kBlockSize, kMPerBlock, kKPerBlock, kVecSize>();

    // Create src window (read) and dst window (write)
#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    auto src_window = window::make_tile_window(
        src_view,
        ck_tile::make_tuple(
            ck_tile::number<kMPerBlock>{},
            ck_tile::number<kKPerBlock>{}),
        ck_tile::multi_index<2>{0, 0},
        dstr);

    auto dst_window = window::make_tile_window(
        dst_view,
        ck_tile::make_tuple(
            ck_tile::number<kMPerBlock>{},
            ck_tile::number<kKPerBlock>{}),
        ck_tile::multi_index<2>{0, 0},
        dstr);
#else
    auto src_window = window::make_tile_window(
        src_view,
        mint::nd_index<2>{kMPerBlock, kKPerBlock},
        mint::nd_index<2>{0, 0},
        dstr);

    auto dst_window = window::make_tile_window(
        dst_view,
        mint::nd_index<2>{kMPerBlock, kKPerBlock},
        mint::nd_index<2>{0, 0},
        dstr);
#endif

    // Test 0: Load from src -> VGPR
    auto tile = ops::load_tile(src_window);
    results[0] = 1;

    // Test 1: Store from VGPR -> dst
    ops::store_tile(dst_window, tile);
    results[1] = 1;

    // Sync all threads before verifying
    __syncthreads();

    // Test 2: Verify data - thread 0 checks several positions
    if(threadIdx.x == 0)
    {
        bool match = true;
        // Check corners and middle of the tile
        const int check_positions[] = {
            0,                                // (0,0)
            kKPerBlock - 1,                   // (0, K-1)
            kKPerBlock,                       // (1, 0)
            (kMPerBlock / 2) * k_size,        // (M/2, 0)
            (kMPerBlock / 2) * k_size +
                kKPerBlock / 2,               // (M/2, K/2)
            (kMPerBlock - 1) * k_size +
                kKPerBlock - 1,               // (M-1, K-1)
        };

        for(int i = 0; i < 6; ++i)
        {
            int pos = check_positions[i];
            if(static_cast<float>(p_dst[pos]) !=
               static_cast<float>(p_src[pos]))
            {
                match = false;
                break;
            }
        }
        results[2] = match ? 1 : 0;
    }
}

int main()
{
    constexpr int kNumTests = 3;
    constexpr int M = 128;
    constexpr int K = 64;
    constexpr int kTotalElems = M * K;

    // Allocate device memory
    int h_results[kNumTests] = {};

    ck_tile::DeviceMem src_buf(kTotalElems * sizeof(DataType));
    ck_tile::DeviceMem dst_buf(kTotalElems * sizeof(DataType));
    ck_tile::DeviceMem result_buf(kNumTests * sizeof(int));

    // Fill source with a known pattern
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
        M, K,
        reinterpret_cast<int*>(result_buf.GetDeviceBuffer()));
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    result_buf.FromDevice(h_results);

    const char* test_names[] = {
        "Load tile from global memory",
        "Store tile to global memory",
        "Roundtrip data integrity (src == dst at 6 positions)",
    };

    int pass_count = 0;
    for(int i = 0; i < kNumTests; ++i)
    {
        bool passed = (h_results[i] == 1);
        printf("[%s] Test %d: %s\n",
               passed ? "PASS" : "FAIL", i, test_names[i]);
        if(passed)
            ++pass_count;
    }

    printf("\n%d/%d tests passed.\n", pass_count, kNumTests);
    return (pass_count == kNumTests) ? 0 : 1;
}
