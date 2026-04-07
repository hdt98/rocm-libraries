// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_tile/distribution/distribution.hpp"

#include <hip/hip_runtime.h>
#include <cstdio>

// Test parameters
static constexpr int kBlockSize = 256;
static constexpr int kMPerBlock = 128;
static constexpr int kNPerBlock = 128;
static constexpr int kKPerBlock = 64;
static constexpr int kVecSize = 8;

__global__ void distribution_test_kernel(int* results)
{
    using namespace unified_tile::distribution;

    // --- Test 0-1: Named A distribution (M x K) ---
    {
        constexpr auto dstr =
            make_block_copy_a_distribution<kBlockSize,
                                           kMPerBlock,
                                           kKPerBlock,
                                           kVecSize>();
        // Expected: 128*64/256 = 32
        results[0] = (get_elements_per_thread(dstr) == 32) ? 1 : 0;
        results[1] = (get_num_tile_dims(dstr) == 2) ? 1 : 0;
    }

    // --- Test 2-3: Named B distribution (K x N) ---
    {
        constexpr auto dstr =
            make_block_copy_b_distribution<kBlockSize,
                                           kKPerBlock,
                                           kNPerBlock,
                                           kVecSize>();
        // Expected: 64*128/256 = 32
        results[2] = (get_elements_per_thread(dstr) == 32) ? 1 : 0;
        results[3] = (get_num_tile_dims(dstr) == 2) ? 1 : 0;
    }

    // --- Test 4: Generic 2D distribution ---
    {
        constexpr auto dstr =
            make_block_copy_2d_distribution<128, 64, 32, 4>();
        // Expected: 64*32/128 = 16
        results[4] = (get_elements_per_thread(dstr) == 16) ? 1 : 0;
    }

    // --- Test 5: Config-only query (no distribution needed) ---
    {
        using Config = block_copy_2d_config<256, 256, 32, 4>;
        // Expected: 256*32/256 = 32
        results[5] = (Config::kElemsPerThread == 32) ? 1 : 0;
        // Verify decomposition: threads_inner=8, threads_outer=32
        results[6] = (Config::kThreadsInner == 8) ? 1 : 0;
        results[7] = (Config::kThreadsOuter == 32) ? 1 : 0;
    }
}

int main()
{
    constexpr int kNumTests = 8;
    int* d_results = nullptr;
    int h_results[kNumTests] = {};

    hipMalloc(&d_results, kNumTests * sizeof(int));
    hipMemset(d_results, 0, kNumTests * sizeof(int));

    distribution_test_kernel<<<1, kBlockSize>>>(d_results);
    hipDeviceSynchronize();
    hipMemcpy(h_results, d_results, kNumTests * sizeof(int),
              hipMemcpyDeviceToHost);
    hipFree(d_results);

    const char* test_names[] = {
        "A distribution: elements_per_thread == 32",
        "A distribution: num_tile_dims == 2",
        "B distribution: elements_per_thread == 32",
        "B distribution: num_tile_dims == 2",
        "Generic 2D: elements_per_thread == 16",
        "Config: elems_per_thread == 32",
        "Config: threads_inner == 8",
        "Config: threads_outer == 32",
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
