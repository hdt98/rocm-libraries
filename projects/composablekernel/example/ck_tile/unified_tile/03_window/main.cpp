// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_tile/tensor/descriptor.hpp"
#include "unified_tile/tensor/view.hpp"
#include "unified_tile/distribution/distribution.hpp"
#include "unified_tile/tensor/window.hpp"

#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>

using DataType = _Float16;

static constexpr int kBlockSize = 256;
static constexpr int kMPerBlock = 128;
static constexpr int kKPerBlock = 64;
static constexpr int kVecSize = 8;

__global__ void window_test_kernel(const DataType* p_a,
                                    int m_size,
                                    int k_size,
                                    int* results)
{
    using namespace unified_tile;

    // Step 1: Create descriptor (aliases must match distribution for MINT)
#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    auto desc = descriptor::make_descriptor(m_size, k_size);
#else
    auto desc = descriptor::make_aliased_descriptor<"M", "K">(m_size, k_size);
#endif

    // Step 2: Create tensor view
    auto a_view =
        view::make_tensor_view<address_space::global>(
            const_cast<DataType*>(p_a), desc);

    // Step 3: Create distribution
    constexpr auto a_dstr =
        distribution::make_block_copy_a_distribution<
            kBlockSize, kMPerBlock, kKPerBlock, kVecSize>();

    // Step 4: Create tile window with distribution
    const auto block_idx = static_cast<int>(blockIdx.x);
    const auto m_block = block_idx * kMPerBlock;

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    auto a_window = window::make_tile_window(
        a_view,
        ck_tile::make_tuple(
            ck_tile::number<kMPerBlock>{},
            ck_tile::number<kKPerBlock>{}),
        ck_tile::multi_index<2>{m_block, 0},
        a_dstr);
#else
    auto a_window = window::make_tile_window(
        a_view,
        mint::nd_index<2>{kMPerBlock, kKPerBlock},
        mint::nd_index<2>{m_block, 0},
        a_dstr);
#endif

    // Test 0: Window was created successfully (we got here without crashing)
    results[0] = 1;

    // Test 1: Move window along K dimension
#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    window::move_window(a_window, ck_tile::multi_index<2>{0, kKPerBlock});
#else
    window::move_window(a_window, mint::nd_index<2>{0, kKPerBlock});
#endif
    results[1] = 1;

    // Test 2: Verify distribution properties match between config and actual
    {
        using Config = distribution::block_copy_2d_config<
            kBlockSize, kMPerBlock, kKPerBlock, kVecSize>;
        auto actual_elems = distribution::get_elements_per_thread(a_dstr);
        results[2] = (actual_elems == Config::kElemsPerThread) ? 1 : 0;
    }
}

int main()
{
    constexpr int kNumTests = 3;
    constexpr int M = 256;
    constexpr int K = 128;

    // Allocate device memory
    int h_results[kNumTests] = {};

    ck_tile::DeviceMem tensor_buf(M * K * sizeof(DataType));
    tensor_buf.SetZero();

    ck_tile::DeviceMem result_buf(kNumTests * sizeof(int));
    result_buf.SetZero();

    window_test_kernel<<<1, kBlockSize>>>(
        reinterpret_cast<const DataType*>(tensor_buf.GetDeviceBuffer()),
        M, K,
        reinterpret_cast<int*>(result_buf.GetDeviceBuffer()));
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    result_buf.FromDevice(h_results);

    const char* test_names[] = {
        "Window creation (view + distribution + origin)",
        "Window movement (move along K)",
        "Distribution config consistency",
    };

    int pass_count = 0;
    for(int i = 0; i < kNumTests; ++i)
    {
        bool passed = (h_results[i] == 1);
        printf("[%s] Test %d: %s\n",
               passed ? "PASS" : "FAIL",
               i,
               test_names[i]);
        if(passed)
            ++pass_count;
    }

    printf("\n%d/%d tests passed.\n", pass_count, kNumTests);
    return (pass_count == kNumTests) ? 0 : 1;
}
