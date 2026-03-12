// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Kpack Hello World -- demonstrates the full kpack consumer workflow:
// 1. Open a .kpack archive
// 2. Enumerate its contents
// 3. Detect the current GPU architecture
// 4. Load the matching kernel code object
// 5. Create a HIP module and launch the kernel
// 6. Verify results

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

static const char* kpack_error_string(kpack_error_t err)
{
    switch(err)
    {
    case KPACK_SUCCESS: return "success";
    case KPACK_ERROR_INVALID_ARGUMENT: return "invalid argument";
    case KPACK_ERROR_FILE_NOT_FOUND: return "file not found";
    case KPACK_ERROR_INVALID_FORMAT: return "invalid format";
    case KPACK_ERROR_UNSUPPORTED_VERSION: return "unsupported version";
    case KPACK_ERROR_KERNEL_NOT_FOUND: return "kernel not found";
    case KPACK_ERROR_DECOMPRESSION_FAILED: return "decompression failed";
    case KPACK_ERROR_OUT_OF_MEMORY: return "out of memory";
    case KPACK_ERROR_NOT_IMPLEMENTED: return "not implemented";
    case KPACK_ERROR_IO_ERROR: return "I/O error";
    case KPACK_ERROR_MSGPACK_PARSE_FAILED: return "msgpack parse failed";
    case KPACK_ERROR_ARCH_NOT_FOUND: return "architecture not found";
    default: return "unknown error";
    }
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::fprintf(stderr, "Usage: %s <path-to-kernels.kpack>\n", argv[0]);
        return 1;
    }

    const char* kpack_path = argv[1];

    std::printf("=== Kpack Hello World ===\n");

    // --- Step 1: Open archive ---
    kpack_archive_t archive = nullptr;
    kpack_error_t kerr      = kpack_open(kpack_path, &archive);
    if(kerr != KPACK_SUCCESS)
    {
        std::fprintf(
            stderr, "Failed to open archive '%s': %s\n", kpack_path, kpack_error_string(kerr));
        return 1;
    }
    std::printf("Opened archive: %s\n", kpack_path);

    // --- Step 2: Enumerate contents ---
    size_t arch_count = 0;
    kpack_get_architecture_count(archive, &arch_count);
    std::printf("  Architectures:");
    for(size_t i = 0; i < arch_count; ++i)
    {
        const char* arch = nullptr;
        kpack_get_architecture(archive, i, &arch);
        std::printf("%s %s", (i > 0 ? "," : ""), arch);
    }
    std::printf("\n");

    size_t bin_count = 0;
    kpack_get_binary_count(archive, &bin_count);
    std::printf("  Binaries:");
    for(size_t i = 0; i < bin_count; ++i)
    {
        const char* bin = nullptr;
        kpack_get_binary(archive, i, &bin);
        std::printf("%s %s", (i > 0 ? "," : ""), bin);
    }
    std::printf("\n\n");

    // --- Step 3: Detect GPU ---
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));

    // gcnArchName gives the full ISA string (e.g., "gfx942:sramecc+:xnack-")
    // We need the base architecture name for kpack lookup
    std::string full_arch = props.gcnArchName;
    std::string base_arch = full_arch;
    auto colon_pos        = base_arch.find(':');
    if(colon_pos != std::string::npos)
    {
        base_arch = base_arch.substr(0, colon_pos);
    }
    std::printf("Detected GPU: %s (base: %s)\n", full_arch.c_str(), base_arch.c_str());

    // --- Step 4: Load kernel ---
    void* kernel_data  = nullptr;
    size_t kernel_size = 0;

    // Try full arch first, then base arch
    kerr = kpack_get_kernel(
        archive, "vector_add_kernel", full_arch.c_str(), &kernel_data, &kernel_size);
    if(kerr == KPACK_ERROR_KERNEL_NOT_FOUND)
    {
        kerr = kpack_get_kernel(
            archive, "vector_add_kernel", base_arch.c_str(), &kernel_data, &kernel_size);
    }

    if(kerr != KPACK_SUCCESS)
    {
        std::fprintf(stderr,
                     "No kernel found for architecture '%s' (or '%s'): %s\n",
                     full_arch.c_str(),
                     base_arch.c_str(),
                     kpack_error_string(kerr));
        std::fprintf(stderr, "Available architectures in archive:");
        for(size_t i = 0; i < arch_count; ++i)
        {
            const char* arch = nullptr;
            kpack_get_architecture(archive, i, &arch);
            std::fprintf(stderr, " %s", arch);
        }
        std::fprintf(stderr, "\n");
        kpack_close(archive);
        return 1;
    }

    std::printf("Loading kernel for %s... OK (%zu bytes)\n", base_arch.c_str(), kernel_size);

    // --- Step 5: Create HIP module ---
    hipModule_t module = nullptr;
    HIP_CHECK(hipModuleLoadData(&module, kernel_data));
    std::printf("Loading HIP module... OK\n");

    hipFunction_t func = nullptr;
    HIP_CHECK(hipModuleGetFunction(&func, module, "vectorAdd"));
    std::printf("Got function handle: vectorAdd\n\n");

    // --- Step 6: Run kernel ---
    const int N = 1024;
    std::printf("Running vectorAdd (N=%d)...\n", N);

    // Allocate and initialize host vectors
    std::vector<float> h_a(N), h_b(N), h_c(N);
    for(int i = 0; i < N; ++i)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    HIP_CHECK(hipMalloc(&d_a, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_b, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_c, N * sizeof(float)));

    // Copy to device
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // Launch kernel via module API
    int n = N;
    struct
    {
        const float* a;
        const float* b;
        float* c;
        int n;
    } args = {d_a, d_b, d_c, n};

    size_t args_size = sizeof(args);
    void* config[]   = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                        &args,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE,
                        &args_size,
                        HIP_LAUNCH_PARAM_END};

    int block_size = 256;
    int grid_size  = (N + block_size - 1) / block_size;

    HIP_CHECK(hipModuleLaunchKernel(
        func, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, nullptr, config));
    HIP_CHECK(hipDeviceSynchronize());

    // Copy back
    HIP_CHECK(hipMemcpy(h_c.data(), d_c, N * sizeof(float), hipMemcpyDeviceToHost));

    // Verify
    bool passed = true;
    for(int i = 0; i < N; ++i)
    {
        float expected = static_cast<float>(i) + static_cast<float>(i * 2);
        if(std::fabs(h_c[i] - expected) > 1e-5f)
        {
            std::fprintf(
                stderr, "MISMATCH at index %d: got %f, expected %f\n", i, h_c[i], expected);
            passed = false;
            break;
        }
    }

    if(passed)
    {
        std::printf("Verification PASSED!\n");
    }
    else
    {
        std::printf("Verification FAILED!\n");
    }

    // --- Step 7: Cleanup ---
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    HIP_CHECK(hipModuleUnload(module));
    kpack_free_kernel(kernel_data);
    kpack_close(archive);

    return passed ? 0 : 1;
}
