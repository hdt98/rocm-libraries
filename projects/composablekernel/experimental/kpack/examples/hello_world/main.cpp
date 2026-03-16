// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Kpack Hello World — demonstrates the full kpack consumer pipeline:
//   Open archive → detect GPU → load kernel → launch → verify

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

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::fprintf(stderr, "Usage: %s <path-to-kernels.kpack>\n", argv[0]);
        return 1;
    }

    // --- Open the kpack archive and list its contents ---
    // A kpack archive bundles per-architecture .hsaco code objects with a
    // MessagePack table of contents. The runtime maps it and parses the TOC.
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

    // --- Detect the current GPU architecture ---
    // HIP reports the full ISA string (e.g. "gfx942:sramecc+:xnack-").
    // The archive uses the base name ("gfx942"), so strip the feature flags.
    hipDeviceProp_t device_props;
    HIP_CHECK(hipGetDeviceProperties(&device_props, 0));

    std::string gpu_arch = device_props.gcnArchName;
    size_t colon_pos     = gpu_arch.find(':');
    if(colon_pos != std::string::npos)
    {
        gpu_arch = gpu_arch.substr(0, colon_pos);
    }
    std::printf("Detected GPU: %s\n", gpu_arch.c_str());

    // --- Load the matching kernel code object from the archive ---
    // kpack_get_kernel returns a pointer to the raw .hsaco bytes for the
    // requested binary name + architecture. The caller owns the memory.
    void* kernel_code_object       = nullptr;
    size_t kernel_code_object_size = 0;
    kerr                           = kpack_get_kernel(archive,
                            "vector_add_kernel",
                            gpu_arch.c_str(),
                            &kernel_code_object,
                            &kernel_code_object_size);
    if(kerr != KPACK_SUCCESS)
    {
        std::fprintf(stderr, "No kernel for '%s' in archive (error %d)\n", gpu_arch.c_str(), kerr);
        kpack_close(archive);
        return 1;
    }
    std::printf("Loaded kernel: %zu bytes\n", kernel_code_object_size);

    // --- Create a HIP module from the code object ---
    // hipModuleLoadData takes the raw .hsaco bytes and creates a module handle.
    // hipModuleGetFunction extracts the named kernel entry point.
    hipModule_t module            = nullptr;
    hipFunction_t kernel_function = nullptr;
    HIP_CHECK(hipModuleLoadData(&module, kernel_code_object));
    HIP_CHECK(hipModuleGetFunction(&kernel_function, module, "vectorAdd"));

    // --- Allocate memory and run the kernel ---
    const int NUM_ELEMENTS = 1024;
    std::printf("Running vectorAdd (N=%d)...\n", NUM_ELEMENTS);

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

    // Launch via the module API — pack kernel arguments into a struct and pass
    // through HIP_LAUNCH_PARAM to hipModuleLaunchKernel.
    struct
    {
        const float* a;
        const float* b;
        float* c;
        int n;
    } kernel_args           = {device_a, device_b, device_result, NUM_ELEMENTS};
    size_t kernel_args_size = sizeof(kernel_args);
    void* launch_config[]   = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                               &kernel_args,
                               HIP_LAUNCH_PARAM_BUFFER_SIZE,
                               &kernel_args_size,
                               HIP_LAUNCH_PARAM_END};

    int block_size = 256;
    int grid_size  = (NUM_ELEMENTS + block_size - 1) / block_size;
    HIP_CHECK(hipModuleLaunchKernel(
        kernel_function, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, nullptr, launch_config));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(
        host_result.data(), device_result, NUM_ELEMENTS * sizeof(float), hipMemcpyDeviceToHost));

    // --- Verify results ---
    bool passed = true;
    for(int i = 0; i < NUM_ELEMENTS; ++i)
    {
        float expected = static_cast<float>(i) + static_cast<float>(i * 2);
        if(std::fabs(host_result[i] - expected) > 1e-5f)
        {
            std::fprintf(
                stderr, "MISMATCH at index %d: got %f, expected %f\n", i, host_result[i], expected);
            passed = false;
            break;
        }
    }
    std::printf(passed ? "Verification PASSED!\n" : "Verification FAILED!\n");

    // --- Cleanup ---
    HIP_CHECK(hipFree(device_a));
    HIP_CHECK(hipFree(device_b));
    HIP_CHECK(hipFree(device_result));
    HIP_CHECK(hipModuleUnload(module));
    kpack_free_kernel(kernel_code_object);
    kpack_close(archive);

    return passed ? 0 : 1;
}