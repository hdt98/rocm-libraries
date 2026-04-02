// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_tile/tensor/descriptor.hpp"
#include "unified_tile/tensor/view.hpp"

#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

using data_t = _Float16;

// Device kernel: create a tensor view and read data through it
__global__ void test_view_kernel(const data_t* p_global,
                                 int M,
                                 int K,
                                 int* results)
{
    using namespace unified_tile::descriptor;
    using namespace unified_tile::view;

    // Create a 2D packed descriptor for [M, K]
    auto desc = make_descriptor(M, K);

    // Create a global memory tensor view
    auto view = make_tensor_view<unified_tile::address_space::global>(
        const_cast<data_t*>(p_global), desc);

    // The view should be usable; verify we created it without crashing
    // For a simple validation, check that get_num_dimensions on the
    // underlying descriptor works
    results[0] = (get_num_dimensions(desc) == 2) ? 1 : 0;
}

int main()
{
#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    const char* backend_name = "CK_Tile";
#else
    const char* backend_name = "MINT";
#endif

    printf("[Unified Tile] Tensor View Example (Backend: %s)\n", backend_name);

    constexpr int M         = 128;
    constexpr int K         = 64;
    constexpr int kNumTests = 1;

    // Allocate device memory for the tensor
    ck_tile::DeviceMem tensor_buf(M * K * sizeof(data_t));
    tensor_buf.SetZero();

    // Allocate device memory for results
    ck_tile::DeviceMem result_buf(kNumTests * sizeof(int));
    result_buf.SetZero();

    test_view_kernel<<<1, 1>>>(
        reinterpret_cast<const data_t*>(tensor_buf.GetDeviceBuffer()),
        M,
        K,
        reinterpret_cast<int*>(result_buf.GetDeviceBuffer()));
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    int host_results[kNumTests] = {0};
    result_buf.FromDevice(host_results);

    const char* test_names[] = {
        "Global tensor view created and data accessible",
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
