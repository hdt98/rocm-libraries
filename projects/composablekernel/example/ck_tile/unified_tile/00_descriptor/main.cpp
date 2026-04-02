// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_tile/config.hpp"
#include "unified_tile/tensor/descriptor.hpp"

#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>

// Device kernel that creates descriptors and validates properties
__global__ void test_descriptor_kernel(int* results)
{
    using namespace unified_tile::descriptor;

    // Test 1: 2D packed descriptor (M=128, K=64)
    {
        auto desc  = make_descriptor(128, 64);
        results[0] = (get_num_dimensions(desc) == 2) ? 1 : 0;
    }

    // Test 2: 3D packed descriptor (B=4, M=128, K=64)
    {
        auto desc  = make_descriptor(4, 128, 64);
        results[1] = (get_num_dimensions(desc) == 3) ? 1 : 0;
    }
}

int main()
{
#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    const char* backend_name = "CK_Tile";
#else
    const char* backend_name = "MINT";
#endif

    printf("[Unified Tile] Descriptor Example (Backend: %s)\n", backend_name);

    constexpr int kNumTests = 2;

    ck_tile::DeviceMem result_buf(kNumTests * sizeof(int));
    result_buf.SetZero();

    test_descriptor_kernel<<<1, 1>>>(
        reinterpret_cast<int*>(result_buf.GetDeviceBuffer()));
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    int host_results[kNumTests] = {0};
    result_buf.FromDevice(host_results);

    const char* test_names[] = {
        "2D packed descriptor created",
        "3D packed descriptor created",
    };

    int pass_count = 0;
    for(int i = 0; i < kNumTests; ++i)
    {
        if(host_results[i] == 1)
        {
            printf("PASSED: %s\n", test_names[i]);
            ++pass_count;
        }
        else
        {
            printf("FAILED: %s\n", test_names[i]);
        }
    }

    if(pass_count == kNumTests)
    {
        printf("All tests passed!\n");
        return 0;
    }
    else
    {
        printf("%d/%d tests failed!\n", kNumTests - pass_count, kNumTests);
        return 1;
    }
}
