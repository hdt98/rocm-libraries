// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_tile/tensor/descriptor.hpp"
#include "unified_tile/tensor/view.hpp"
#include "unified_tile/distribution/distribution.hpp"
#include "unified_tile/tensor/window.hpp"
#include "unified_tile/ops/load.hpp"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using DataType = _Float16;

static constexpr int kBlockSize = 256;
static constexpr int kMPerBlock = 128;
static constexpr int kKPerBlock = 64;
static constexpr int kVecSize = 8;

static constexpr int kElemsPerThread =
    kMPerBlock * kKPerBlock / kBlockSize; // = 32

/// Source pattern: a[row][col] = (row + 1) as fp16.
/// Row 0 filled with 1.0, row 1 with 2.0, ..., row 127 with 128.0.
/// After loading, each thread verifies:
///   - All loaded values are in [1.0, 128.0]
///   - No zeros (would mean OOB or failed load)
///   - Per-thread sum (accumulated in float)
__global__ void load_test_kernel(const DataType* p_a,
                                  int m_size,
                                  int k_size,
                                  float* p_thread_sums,
                                  int* p_thread_valid_counts,
                                  int* p_thread_zero_counts)
{
    using namespace unified_tile;

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    auto desc = descriptor::make_descriptor(m_size, k_size);
#else
    // MINT: aliases must match distribution aliases ("M", "K")
    auto desc = descriptor::make_aliased_descriptor<"M", "K">(m_size, k_size);
#endif
    auto a_view =
        view::make_tensor_view<address_space::global>(
            const_cast<DataType*>(p_a), desc);

    constexpr auto a_dstr =
        distribution::make_block_copy_a_distribution<
            kBlockSize, kMPerBlock, kKPerBlock, kVecSize>();

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    auto a_window = window::make_tile_window(
        a_view,
        ck_tile::make_tuple(
            ck_tile::number<kMPerBlock>{},
            ck_tile::number<kKPerBlock>{}),
        ck_tile::multi_index<2>{0, 0},
        a_dstr);
#else
    auto a_window = window::make_tile_window(
        a_view,
        mint::nd_index<2>{kMPerBlock, kKPerBlock},
        mint::nd_index<2>{0, 0},
        a_dstr);
#endif

    auto a_tile = ops::load_tile(a_window);

    const int tid = threadIdx.x;
    float sum = 0.0f;
    int valid_count = 0;
    int zero_count = 0;
#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    constexpr int buf_size = kElemsPerThread;
    for(int i = 0; i < buf_size; ++i)
    {
        float val = static_cast<float>(a_tile.get_thread_buffer()[i]);
        sum += val;
        if(val >= 1.0f && val <= 128.0f)
            ++valid_count;
        if(val == 0.0f)
            ++zero_count;
    }
#else
    mint::static_for_n<kElemsPerThread>()([&](auto i) {
        float val = static_cast<float>(a_tile.memory().template at<i>());
        sum += val;
        if(val >= 1.0f && val <= 128.0f)
            ++valid_count;
        if(val == 0.0f)
            ++zero_count;
    });
#endif

    p_thread_sums[tid] = sum;
    p_thread_valid_counts[tid] = valid_count;
    p_thread_zero_counts[tid] = zero_count;
}

int main()
{
    constexpr int M = 128;
    constexpr int K = 64;
    constexpr int total = M * K;

    // Fill with row pattern: a[r][c] = (r + 1)
    DataType* h_a = new DataType[total];
    for(int r = 0; r < M; ++r)
        for(int c = 0; c < K; ++c)
            h_a[r * K + c] = static_cast<DataType>(static_cast<float>(r + 1));

    DataType* d_a = nullptr;
    float* d_sums = nullptr;
    int* d_valid = nullptr;
    int* d_zeros = nullptr;
    hipMalloc(&d_a, total * sizeof(DataType));
    hipMalloc(&d_sums, kBlockSize * sizeof(float));
    hipMalloc(&d_valid, kBlockSize * sizeof(int));
    hipMalloc(&d_zeros, kBlockSize * sizeof(int));
    hipMemcpy(d_a, h_a, total * sizeof(DataType), hipMemcpyHostToDevice);
    hipMemset(d_sums, 0, kBlockSize * sizeof(float));
    hipMemset(d_valid, 0, kBlockSize * sizeof(int));
    hipMemset(d_zeros, 0, kBlockSize * sizeof(int));

    load_test_kernel<<<1, kBlockSize>>>(d_a, M, K, d_sums, d_valid, d_zeros);
    hipDeviceSynchronize();

    float* h_sums = new float[kBlockSize];
    int* h_valid = new int[kBlockSize];
    int* h_zeros = new int[kBlockSize];
    hipMemcpy(h_sums, d_sums, kBlockSize * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_valid, d_valid, kBlockSize * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_zeros, d_zeros, kBlockSize * sizeof(int), hipMemcpyDeviceToHost);

    hipFree(d_a);
    hipFree(d_sums);
    hipFree(d_valid);
    hipFree(d_zeros);
    delete[] h_a;

    // Test 0: Every thread loaded exactly kElemsPerThread valid values [1,128]
    int valid_pass = 0;
    for(int t = 0; t < kBlockSize; ++t)
        if(h_valid[t] == kElemsPerThread)
            ++valid_pass;
    bool test0 = (valid_pass == kBlockSize);

    // Test 1: No thread loaded any zeros (would mean OOB or failed load)
    int zero_free = 0;
    for(int t = 0; t < kBlockSize; ++t)
        if(h_zeros[t] == 0)
            ++zero_free;
    bool test1 = (zero_free == kBlockSize);

    // Test 2: Global sum = K * sum(1..M) = 64 * (128*129/2) = 528384
    double global_sum = 0.0;
    for(int t = 0; t < kBlockSize; ++t)
        global_sum += static_cast<double>(h_sums[t]);
    double expected_sum = static_cast<double>(K) * (M * (M + 1) / 2);
    bool test2 = (fabs(global_sum - expected_sum) < 1.0);

    // Test 3: Per-thread sum is positive (sanity)
    int positive_pass = 0;
    for(int t = 0; t < kBlockSize; ++t)
        if(h_sums[t] > 0.0f)
            ++positive_pass;
    bool test3 = (positive_pass == kBlockSize);

    printf("[%s] Test 0: All threads loaded %d valid values in [1,128] (%d/%d)\n",
           test0 ? "PASS" : "FAIL", kElemsPerThread, valid_pass, kBlockSize);
    printf("[%s] Test 1: No thread loaded zeros (%d/%d zero-free)\n",
           test1 ? "PASS" : "FAIL", zero_free, kBlockSize);
    printf("[%s] Test 2: Global sum %.0f == expected %.0f\n",
           test2 ? "PASS" : "FAIL", global_sum, expected_sum);
    printf("[%s] Test 3: All thread sums positive (%d/%d)\n",
           test3 ? "PASS" : "FAIL", positive_pass, kBlockSize);

    int total_pass = (test0 ? 1 : 0) + (test1 ? 1 : 0) +
                     (test2 ? 1 : 0) + (test3 ? 1 : 0);
    printf("\n%d/4 tests passed.\n", total_pass);

    delete[] h_sums;
    delete[] h_valid;
    delete[] h_zeros;
    return (total_pass == 4) ? 0 : 1;
}
