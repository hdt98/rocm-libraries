// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-side loader for the rocm_ck vector add example. Loads multiple kernel
// variants from a kpack archive and runs each one, verifying correctness.
//
// This file includes only rocm_ck_vector_add_args.hpp (no CK Tile dependency).
// The consteval variant table ensures host and device agree on configuration.

#include "rocm_ck_vector_add_args.hpp"

#include <hip/hip_runtime.h>

#include <rocm_kpack/kpack.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(call)                                  \
    do                                                   \
    {                                                    \
        hipError_t err = (call);                         \
        if(err != hipSuccess)                            \
        {                                                \
            std::fprintf(stderr,                         \
                         "HIP error %d (%s) at %s:%d\n", \
                         err,                            \
                         hipGetErrorString(err),         \
                         __FILE__,                       \
                         __LINE__);                      \
            std::exit(1);                                \
        }                                                \
    } while(0)

struct variant_info
{
    const char* name;
    rocm_ck::vector_add_kernel_info kernel;
};

static constexpr variant_info VARIANTS[] = {
    {"vector_add_block256", rocm_ck::make_vector_add_kernel({.block_size = 256})},
    {"vector_add_block512", rocm_ck::make_vector_add_kernel({.block_size = 512})},
    {"vector_add_block1024", rocm_ck::make_vector_add_kernel({.block_size = 1024})},
};

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::fprintf(stderr, "Usage: %s <path-to-kernels.kpack>\n", argv[0]);
        return 1;
    }

    // --- Open the kpack archive ---
    kpack_archive_t archive = nullptr;
    kpack_error_t kerr      = kpack_open(argv[1], &archive);
    if(kerr != KPACK_SUCCESS)
    {
        std::fprintf(stderr, "Failed to open archive '%s' (error %d)\n", argv[1], kerr);
        return 1;
    }

    size_t arch_count = 0;
    kpack_get_architecture_count(archive, &arch_count);
    std::printf("Opened %s — architectures:", argv[1]);
    for(size_t i = 0; i < arch_count; ++i)
    {
        const char* arch_name = nullptr;
        kpack_get_architecture(archive, i, &arch_name);
        std::printf("%s %s", (i > 0 ? "," : ""), arch_name);
    }
    std::printf("\n");

    // --- Detect current GPU architecture ---
    hipDeviceProp_t device_props;
    HIP_CHECK(hipGetDeviceProperties(&device_props, 0));

    std::string gpu_arch = device_props.gcnArchName;
    size_t colon_pos     = gpu_arch.find(':');
    if(colon_pos != std::string::npos)
    {
        gpu_arch = gpu_arch.substr(0, colon_pos);
    }
    std::printf("Detected GPU: %s\n", gpu_arch.c_str());

    // --- Allocate device memory (shared across all variants) ---
    const int NUM_ELEMENTS = 4096;

    std::vector<float> host_a(NUM_ELEMENTS);
    std::vector<float> host_b(NUM_ELEMENTS);
    std::vector<float> host_result(NUM_ELEMENTS);
    for(int i = 0; i < NUM_ELEMENTS; ++i)
    {
        host_a[i] = static_cast<float>(i);
        host_b[i] = static_cast<float>(i * 2);
    }

    float* device_a      = nullptr;
    float* device_b      = nullptr;
    float* device_result = nullptr;
    HIP_CHECK(hipMalloc(&device_a, NUM_ELEMENTS * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_b, NUM_ELEMENTS * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_result, NUM_ELEMENTS * sizeof(float)));

    HIP_CHECK(
        hipMemcpy(device_a, host_a.data(), NUM_ELEMENTS * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(device_b, host_b.data(), NUM_ELEMENTS * sizeof(float), hipMemcpyHostToDevice));

    // --- Run each variant ---
    bool all_passed = true;

    for(const auto& variant : VARIANTS)
    {
        // Load kernel code object
        void* kernel_code_object       = nullptr;
        size_t kernel_code_object_size = 0;
        kerr                           = kpack_get_kernel(
            archive, variant.name, gpu_arch.c_str(), &kernel_code_object, &kernel_code_object_size);
        if(kerr != KPACK_SUCCESS)
        {
            std::fprintf(stderr,
                         "  %s: no kernel for '%s' (error %d)\n",
                         variant.name,
                         gpu_arch.c_str(),
                         kerr);
            all_passed = false;
            continue;
        }

        // Create HIP module and get function
        hipModule_t module            = nullptr;
        hipFunction_t kernel_function = nullptr;
        HIP_CHECK(hipModuleLoadData(&module, kernel_code_object));
        HIP_CHECK(hipModuleGetFunction(&kernel_function, module, variant.name));

        // Launch
        const int grid_size =
            (NUM_ELEMENTS + variant.kernel.block_size - 1) / variant.kernel.block_size;
        const int block_size = variant.kernel.thread_block_size;

        // Clear result buffer
        HIP_CHECK(hipMemset(device_result, 0, NUM_ELEMENTS * sizeof(float)));

        rocm_ck::VectorAddArgs kernel_args = {NUM_ELEMENTS, device_a, device_b, device_result};
        size_t kernel_args_size            = sizeof(kernel_args);
        void* launch_config[]              = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                              &kernel_args,
                                              HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                              &kernel_args_size,
                                              HIP_LAUNCH_PARAM_END};

        HIP_CHECK(hipModuleLaunchKernel(kernel_function,
                                        grid_size,
                                        1,
                                        1,
                                        block_size,
                                        1,
                                        1,
                                        0,
                                        nullptr,
                                        nullptr,
                                        launch_config));
        HIP_CHECK(hipDeviceSynchronize());

        // Verify
        HIP_CHECK(hipMemcpy(host_result.data(),
                            device_result,
                            NUM_ELEMENTS * sizeof(float),
                            hipMemcpyDeviceToHost));

        bool passed = true;
        for(int i = 0; i < NUM_ELEMENTS; ++i)
        {
            float expected = static_cast<float>(i) + static_cast<float>(i * 2);
            if(std::fabs(host_result[i] - expected) > 1e-5f)
            {
                std::fprintf(stderr,
                             "  %s: MISMATCH at index %d: got %f, expected %f\n",
                             variant.name,
                             i,
                             host_result[i],
                             expected);
                passed = false;
                break;
            }
        }

        std::printf("  %s (grid=%d, block=%d): %s\n",
                    variant.name,
                    grid_size,
                    block_size,
                    passed ? "PASSED" : "FAILED");

        if(!passed)
            all_passed = false;

        HIP_CHECK(hipModuleUnload(module));
        kpack_free_kernel(kernel_code_object);
    }

    // --- Cleanup ---
    HIP_CHECK(hipFree(device_a));
    HIP_CHECK(hipFree(device_b));
    HIP_CHECK(hipFree(device_result));
    kpack_close(archive);

    return all_passed ? 0 : 1;
}
