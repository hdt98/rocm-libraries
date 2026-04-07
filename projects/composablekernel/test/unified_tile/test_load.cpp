// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_tile/tensor/descriptor.hpp"
#include "unified_tile/tensor/view.hpp"
#include "unified_tile/distribution/distribution.hpp"
#include "unified_tile/tensor/window.hpp"
#include "unified_tile/ops/load.hpp"

#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <vector>

using DataType = _Float16;

static constexpr int kBlockSize = 256;
static constexpr int kM = 128;
static constexpr int kK = 64;
static constexpr int kVecSize = 8;
static constexpr int kElemsPerThread = kM * kK / kBlockSize; // 32

// Result layout per thread: [sum, valid_count, zero_count]
static constexpr int kResultsPerThread = 3;
static constexpr int kTotalResults = kBlockSize * kResultsPerThread;

__global__ void load_test_kernel(const DataType* p_a,
                                  int m_size,
                                  int k_size,
                                  float* p_results)
{
    using namespace unified_tile;

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    auto desc = descriptor::make_descriptor(m_size, k_size);
#else
    auto desc = descriptor::make_aliased_descriptor<"M", "K">(m_size, k_size);
#endif
    auto a_view =
        view::make_tensor_view<address_space::global>(
            const_cast<DataType*>(p_a), desc);

    constexpr auto a_dstr =
        distribution::make_block_copy_a_distribution<
            kBlockSize, kM, kK, kVecSize>();

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    auto a_window = window::make_tile_window(
        a_view,
        ck_tile::make_tuple(ck_tile::number<kM>{}, ck_tile::number<kK>{}),
        ck_tile::multi_index<2>{0, 0},
        a_dstr);
#else
    auto a_window = window::make_tile_window(
        a_view,
        mint::nd_index<2>{kM, kK},
        mint::nd_index<2>{0, 0},
        a_dstr);
#endif

    auto a_tile = ops::load_tile(a_window);

    const int tid = threadIdx.x;
    float sum = 0.0f;
    float valid_count = 0.0f;
    float zero_count = 0.0f;
#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    constexpr int buf_size = kElemsPerThread;
    for(int i = 0; i < buf_size; ++i)
    {
        float val = static_cast<float>(a_tile.get_thread_buffer()[i]);
        sum += val;
        if(val >= 1.0f && val <= 128.0f)
            valid_count += 1.0f;
        if(val == 0.0f)
            zero_count += 1.0f;
    }
#else
    mint::static_for_n<kElemsPerThread>()([&](auto i) {
        float val = static_cast<float>(a_tile.memory().template at<i>());
        sum += val;
        if(val >= 1.0f && val <= 128.0f)
            valid_count += 1.0f;
        if(val == 0.0f)
            zero_count += 1.0f;
    });
#endif

    p_results[tid * kResultsPerThread + 0] = sum;
    p_results[tid * kResultsPerThread + 1] = valid_count;
    p_results[tid * kResultsPerThread + 2] = zero_count;
}

class TestUnifiedTileLoad : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        // Fill source: a[r][c] = (r + 1)
        std::vector<DataType> h_a(kM * kK);
        for(int r = 0; r < kM; ++r)
            for(int c = 0; c < kK; ++c)
                h_a[r * kK + c] =
                    static_cast<DataType>(static_cast<float>(r + 1));

        ck_tile::DeviceMem a_buf(kM * kK * sizeof(DataType));
        a_buf.ToDevice(h_a.data());

        ck_tile::DeviceMem result_buf(kTotalResults * sizeof(float));
        result_buf.SetZero();

        load_test_kernel<<<1, kBlockSize>>>(
            reinterpret_cast<const DataType*>(a_buf.GetDeviceBuffer()),
            kM, kK,
            reinterpret_cast<float*>(result_buf.GetDeviceBuffer()));
        HIP_CHECK_ERROR(hipDeviceSynchronize());

        h_results_.resize(kTotalResults);
        result_buf.FromDevice(h_results_.data());
    }

    std::vector<float> h_results_;
};

TEST_F(TestUnifiedTileLoad, AllValuesInRange)
{
    int pass = 0;
    for(int t = 0; t < kBlockSize; ++t)
    {
        float valid = h_results_[t * kResultsPerThread + 1];
        if(static_cast<int>(valid) == kElemsPerThread)
            ++pass;
    }
    EXPECT_EQ(pass, kBlockSize)
        << "All threads should load " << kElemsPerThread
        << " values in [1, 128]";
}

TEST_F(TestUnifiedTileLoad, NoZerosLoaded)
{
    int pass = 0;
    for(int t = 0; t < kBlockSize; ++t)
    {
        float zeros = h_results_[t * kResultsPerThread + 2];
        if(static_cast<int>(zeros) == 0)
            ++pass;
    }
    EXPECT_EQ(pass, kBlockSize)
        << "No thread should have loaded zeros";
}

TEST_F(TestUnifiedTileLoad, GlobalSumCorrect)
{
    double global_sum = 0.0;
    for(int t = 0; t < kBlockSize; ++t)
        global_sum += static_cast<double>(h_results_[t * kResultsPerThread + 0]);

    double expected = static_cast<double>(kK) * (kM * (kM + 1) / 2);
    EXPECT_NEAR(global_sum, expected, 1.0)
        << "Global sum should be K * sum(1..M)";
}

TEST_F(TestUnifiedTileLoad, AllThreadSumsPositive)
{
    int pass = 0;
    for(int t = 0; t < kBlockSize; ++t)
    {
        if(h_results_[t * kResultsPerThread + 0] > 0.0f)
            ++pass;
    }
    EXPECT_EQ(pass, kBlockSize)
        << "All per-thread sums should be positive";
}
