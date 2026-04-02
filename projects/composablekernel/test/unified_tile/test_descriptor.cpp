// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_tile/config.hpp"
#include "unified_tile/tensor/descriptor.hpp"

#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

// Device kernel: create descriptors and write results to output array
__global__ void descriptor_test_kernel(int* results)
{
    using namespace unified_tile::descriptor;

    // Test 0: 2D packed descriptor has 2 dimensions
    {
        auto desc  = make_descriptor(128, 64);
        results[0] = (get_num_dimensions(desc) == 2) ? 1 : 0;
    }

    // Test 1: 3D packed descriptor has 3 dimensions
    {
        auto desc  = make_descriptor(4, 128, 64);
        results[1] = (get_num_dimensions(desc) == 3) ? 1 : 0;
    }

    // Test 2: 2D descriptor with strides has 2 dimensions
    {
#ifdef UNIFIED_TILE_BACKEND_CK_TILE
        auto desc = make_descriptor_with_strides(
            ck_tile::make_tuple(128, 64), ck_tile::make_tuple(64, 1));
#else
        auto desc = make_descriptor_with_strides(
            mint::nd_index<2>{128, 64}, mint::nd_index<2>{64, 1});
#endif
        results[2] = (get_num_dimensions(desc) == 2) ? 1 : 0;
    }
}

class TestUnifiedTileDescriptor : public ::testing::Test
{
    protected:
    static constexpr int kNumTests = 3;
    int host_results[kNumTests]    = {0};

    void SetUp() override
    {
        ck_tile::DeviceMem result_buf(kNumTests * sizeof(int));
        result_buf.SetZero();

        descriptor_test_kernel<<<1, 1>>>(
            reinterpret_cast<int*>(result_buf.GetDeviceBuffer()));
        HIP_CHECK_ERROR(hipDeviceSynchronize());

        result_buf.FromDevice(host_results);
    }
};

TEST_F(TestUnifiedTileDescriptor, PackedDescriptor2D)
{
    EXPECT_EQ(host_results[0], 1);
}

TEST_F(TestUnifiedTileDescriptor, PackedDescriptor3D)
{
    EXPECT_EQ(host_results[1], 1);
}

TEST_F(TestUnifiedTileDescriptor, StridedDescriptor2D)
{
    EXPECT_EQ(host_results[2], 1);
}
