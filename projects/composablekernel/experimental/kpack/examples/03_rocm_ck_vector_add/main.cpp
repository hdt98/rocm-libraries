// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-side loader for the rocm_ck vector add example. Loads multiple kernel
// variants from a kpack archive and runs each one, verifying correctness.
//
// Supports fp32, fp16, and bf16 variants. Each variant gets typed device
// buffers with host-side conversion for upload/download/verification.

#include "rocm_vector_add_api.hpp"

#include <hip/hip_runtime.h>

#include <rocm_kpack/kpack.h>

#include <cmath>
#include <cstdint>
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
    rocm_ck::vector_add_struct kernel;
};

static constexpr variant_info VARIANTS[] = {
    // Existing variants using backward-compatible vector_add_config
    {"vector_add_fp32_b256",
     rocm_ck::make_kernel({.block_size = 256, .compute_type = rocm_ck::DataType::FP32})},
    {"vector_add_fp32_b512",
     rocm_ck::make_kernel({.block_size = 512, .compute_type = rocm_ck::DataType::FP32})},
    {"vector_add_fp32_b1024",
     rocm_ck::make_kernel({.block_size = 1024, .compute_type = rocm_ck::DataType::FP32})},
    {"vector_add_fp16_b512",
     rocm_ck::make_kernel({.block_size = 512, .compute_type = rocm_ck::DataType::FP16})},
    {"vector_add_fp16_b1024",
     rocm_ck::make_kernel({.block_size = 1024, .compute_type = rocm_ck::DataType::FP16})},
    {"vector_add_bf16_b512",
     rocm_ck::make_kernel({.block_size = 512, .compute_type = rocm_ck::DataType::BF16})},
    // Explicit sig+algo form — same config as fp32_b256, proves new API path
    {"vector_add_fp32_b256_sa",
     rocm_ck::make_kernel(rocm_ck::elementwise_signature{rocm_ck::DataType::FP32},
                          rocm_ck::elementwise_algorithm{256, 1, 256, true})},
};

// --- Host-side type conversion utilities ---
// main.cpp is compiled with GCC (not hipcc), so hip's half/bfloat16 operator
// overloads are not available. We use _Float16 (GCC built-in) for fp16 and
// raw bit manipulation for bf16.

static std::uint16_t float_to_bf16_bits(float f)
{
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    // Round to nearest even: add rounding bias based on LSB + trailing bits
    u += 0x7FFF + ((u >> 16) & 1);
    return static_cast<std::uint16_t>(u >> 16);
}

static float bf16_bits_to_float(std::uint16_t bits)
{
    std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

/// Convert a float to the device type and store into a byte buffer.
static void float_to_typed(rocm_ck::DataType dt, float value, void* dst)
{
    switch(dt)
    {
    case rocm_ck::DataType::FP32: *static_cast<float*>(dst) = value; break;
    case rocm_ck::DataType::FP16:
        *static_cast<_Float16*>(dst) = static_cast<_Float16>(value);
        break;
    case rocm_ck::DataType::BF16:
        *static_cast<std::uint16_t*>(dst) = float_to_bf16_bits(value);
        break;
    case rocm_ck::DataType::FP8: break; // not used
    }
}

/// Read a typed value from a byte buffer and convert to float.
static float typed_to_float(rocm_ck::DataType dt, const void* src)
{
    switch(dt)
    {
    case rocm_ck::DataType::FP32: return *static_cast<const float*>(src);
    case rocm_ck::DataType::FP16: return static_cast<float>(*static_cast<const _Float16*>(src));
    case rocm_ck::DataType::BF16:
        return bf16_bits_to_float(*static_cast<const std::uint16_t*>(src));
    case rocm_ck::DataType::FP8: return 0.0f;
    }
    return 0.0f;
}

/// Tolerance for verification based on data type.
static float tolerance_for(rocm_ck::DataType dt)
{
    switch(dt)
    {
    case rocm_ck::DataType::FP32: return 1e-5f;
    case rocm_ck::DataType::FP16: return 1e-2f;
    case rocm_ck::DataType::BF16: return 1e-1f;
    case rocm_ck::DataType::FP8: return 1.0f;
    }
    return 1e-5f;
}

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

    // --- Test data (small integers exactly representable in fp16 and bf16) ---
    // bf16 has 7 mantissa bits, so only integers ≤ 128 are exact.
    // Keep sums ≤ 62 to stay well within range.
    const int NUM_ELEMENTS = 4096;

    std::vector<float> host_a(NUM_ELEMENTS);
    std::vector<float> host_b(NUM_ELEMENTS);
    for(int i = 0; i < NUM_ELEMENTS; ++i)
    {
        host_a[i] = static_cast<float>(i % 32);
        host_b[i] = static_cast<float>((i * 2) % 32);
    }

    // --- Run each variant ---
    bool all_passed = true;

    for(const auto& variant : VARIANTS)
    {
        const auto dt         = variant.kernel.compute_type;
        const int type_bytes  = rocm_ck::data_type_bits(dt) / 8;
        const size_t buf_size = static_cast<size_t>(NUM_ELEMENTS) * type_bytes;

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

        // Allocate typed device buffers
        void* device_a      = nullptr;
        void* device_b      = nullptr;
        void* device_result = nullptr;
        HIP_CHECK(hipMalloc(&device_a, buf_size));
        HIP_CHECK(hipMalloc(&device_b, buf_size));
        HIP_CHECK(hipMalloc(&device_result, buf_size));

        // Convert float test data to typed host buffers and upload
        std::vector<char> typed_a(buf_size);
        std::vector<char> typed_b(buf_size);
        for(int i = 0; i < NUM_ELEMENTS; ++i)
        {
            float_to_typed(dt, host_a[i], typed_a.data() + i * type_bytes);
            float_to_typed(dt, host_b[i], typed_b.data() + i * type_bytes);
        }

        HIP_CHECK(hipMemcpy(device_a, typed_a.data(), buf_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(device_b, typed_b.data(), buf_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(device_result, 0, buf_size));

        // Launch
        const int grid_size =
            (NUM_ELEMENTS + variant.kernel.block_tile - 1) / variant.kernel.block_tile;
        const int block_size = variant.kernel.thread_block_size;

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

        // Download and verify
        std::vector<char> typed_result(buf_size);
        HIP_CHECK(hipMemcpy(typed_result.data(), device_result, buf_size, hipMemcpyDeviceToHost));

        const float tol = tolerance_for(dt);
        bool passed     = true;
        for(int i = 0; i < NUM_ELEMENTS; ++i)
        {
            float got      = typed_to_float(dt, typed_result.data() + i * type_bytes);
            float expected = host_a[i] + host_b[i];
            if(std::fabs(got - expected) > tol)
            {
                std::fprintf(stderr,
                             "  %s: MISMATCH at index %d: got %f, expected %f\n",
                             variant.name,
                             i,
                             got,
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

        HIP_CHECK(hipFree(device_a));
        HIP_CHECK(hipFree(device_b));
        HIP_CHECK(hipFree(device_result));
        HIP_CHECK(hipModuleUnload(module));
        kpack_free_kernel(kernel_code_object);
    }

    // --- Cleanup ---
    kpack_close(archive);

    return all_passed ? 0 : 1;
}
