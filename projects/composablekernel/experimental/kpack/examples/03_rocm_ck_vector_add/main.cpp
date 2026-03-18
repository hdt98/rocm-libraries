// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-side loader for the rocm_ck vector add example. Loads multiple kernel
// variants from a kpack archive and runs each one, verifying correctness.
//
// Supports same-type (fp32, fp16, bf16) and mixed-type (fp16->fp32, fp32->fp16)
// variants. Each variant gets typed device buffers with host-side conversion
// for upload/download/verification. Kernels compute c = alpha * a + beta * b.

#include "rocm_vector_add_registry.hpp"

#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/gpu_arch.hpp>
#include <rocm_ck/hip_check.hpp>

#include <hip/hip_runtime.h>

#include <rocm_kpack/kpack.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using rocm_ck::ALL_VARIANTS;
using rocm_ck::ALL_VARIANTS_COUNT;

/// Run a single variant: load from archive, launch kernel, verify results.
/// Returns true if the variant passed verification.
static bool runVariant(const rocm_ck::VariantDescriptor& variant,
                       kpack_archive_t archive,
                       const char* gpu_arch,
                       const std::vector<float>& host_a,
                       const std::vector<float>& host_b,
                       float alpha,
                       float beta)
{
    const int num_elements    = static_cast<int>(host_a.size());
    const auto in_type        = variant.kernel.in_type;
    const auto out_type       = variant.kernel.out_type;
    const int in_type_bytes   = rocm_ck::data_type_bits(in_type) / 8;
    const int out_type_bytes  = rocm_ck::data_type_bits(out_type) / 8;
    const size_t in_buf_size  = static_cast<size_t>(num_elements) * in_type_bytes;
    const size_t out_buf_size = static_cast<size_t>(num_elements) * out_type_bytes;

    // Load kernel code object
    void* kernel_code_object       = nullptr;
    size_t kernel_code_object_size = 0;
    kpack_error_t kerr             = kpack_get_kernel(
        archive, variant.name, gpu_arch, &kernel_code_object, &kernel_code_object_size);
    if(kerr != KPACK_SUCCESS)
    {
        std::fprintf(stderr, "  %s: no kernel for '%s' (error %d)\n", variant.name, gpu_arch, kerr);
        return false;
    }

    // Create HIP module and get function
    hipModule_t module            = nullptr;
    hipFunction_t kernel_function = nullptr;
    HIP_CHECK(hipModuleLoadData(&module, kernel_code_object));
    HIP_CHECK(hipModuleGetFunction(&kernel_function, module, variant.name));

    // Allocate device buffers (inputs use in_type size, output uses out_type size)
    void* device_a      = nullptr;
    void* device_b      = nullptr;
    void* device_result = nullptr;
    HIP_CHECK(hipMalloc(&device_a, in_buf_size));
    HIP_CHECK(hipMalloc(&device_b, in_buf_size));
    HIP_CHECK(hipMalloc(&device_result, out_buf_size));

    // Convert float test data to typed host buffers and upload
    std::vector<char> typed_a(in_buf_size);
    std::vector<char> typed_b(in_buf_size);
    for(int i = 0; i < num_elements; ++i)
    {
        rocm_ck::float_to_typed(in_type, host_a[i], typed_a.data() + i * in_type_bytes);
        rocm_ck::float_to_typed(in_type, host_b[i], typed_b.data() + i * in_type_bytes);
    }

    HIP_CHECK(hipMemcpy(device_a, typed_a.data(), in_buf_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_b, typed_b.data(), in_buf_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(device_result, 0, out_buf_size));

    // Launch
    const int grid_size =
        (num_elements + variant.kernel.block_tile - 1) / variant.kernel.block_tile;
    const int block_size = variant.kernel.thread_block_size;
    const bool aligned   = rocm_ck::isAligned(variant.kernel, num_elements);
    std::printf("  %s: tile=%d, warps=%d, threads=%d, N=%d %s\n",
                variant.name,
                variant.kernel.block_tile,
                variant.kernel.block_warps,
                block_size,
                num_elements,
                aligned ? "(aligned)" : "(padded)");

    rocm_ck::VectorAddArgs kernel_args = {.n     = num_elements,
                                          .alpha = alpha,
                                          .beta  = beta,
                                          .a     = device_a,
                                          .b     = device_b,
                                          .c     = device_result};
    size_t kernel_args_size            = sizeof(kernel_args);
    void* launch_config[]              = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                          &kernel_args,
                                          HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                          &kernel_args_size,
                                          HIP_LAUNCH_PARAM_END};

    HIP_CHECK(hipModuleLaunchKernel(
        kernel_function, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, nullptr, launch_config));
    HIP_CHECK(hipDeviceSynchronize());

    // Download and verify (output uses out_type)
    std::vector<char> typed_result(out_buf_size);
    HIP_CHECK(hipMemcpy(typed_result.data(), device_result, out_buf_size, hipMemcpyDeviceToHost));

    const float tol = rocm_ck::tolerance_for(out_type);
    bool passed     = true;
    for(int i = 0; i < num_elements; ++i)
    {
        float got = rocm_ck::typed_to_float(out_type, typed_result.data() + i * out_type_bytes);
        float expected = alpha * host_a[i] + beta * host_b[i];
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

    HIP_CHECK(hipFree(device_a));
    HIP_CHECK(hipFree(device_b));
    HIP_CHECK(hipFree(device_result));
    HIP_CHECK(hipModuleUnload(module));
    kpack_free_kernel(kernel_code_object);

    return passed;
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
    std::string gpu_arch = rocm_ck::get_gpu_arch();
    if(gpu_arch.empty())
    {
        std::fprintf(stderr, "Failed to detect GPU architecture\n");
        kpack_close(archive);
        return 1;
    }
    std::printf("Detected GPU: %s\n", gpu_arch.c_str());

    // --- Test data (small integers exactly representable in fp16 and bf16) ---
    // bf16 has 7 mantissa bits, so only integers <= 128 are exact.
    // Keep sums <= 62 to stay well within range.
    const int NUM_ELEMENTS = 4096;

    std::vector<float> host_a(NUM_ELEMENTS);
    std::vector<float> host_b(NUM_ELEMENTS);
    for(int i = 0; i < NUM_ELEMENTS; ++i)
    {
        host_a[i] = static_cast<float>(i % 32);
        host_b[i] = static_cast<float>((i * 2) % 32);
    }

    // --- Demonstrate findVariant (same-type lookups) ---
    std::printf("\nVariant selection for N=%d:\n", NUM_ELEMENTS);
    for(auto dt : {rocm_ck::DataType::FP32, rocm_ck::DataType::FP16, rocm_ck::DataType::BF16})
    {
        const auto* best = rocm_ck::findVariant(dt, dt, NUM_ELEMENTS);
        if(best)
            std::printf("  %s -> %s (tile=%d, warps=%d)\n",
                        rocm_ck::data_type_name(dt),
                        best->name,
                        best->kernel.block_tile,
                        best->kernel.block_warps);
    }

    // --- Verify all variants with plain add (alpha=1, beta=1) ---
    std::printf("\nRunning all %d variants (alpha=1, beta=1):\n", ALL_VARIANTS_COUNT);
    bool all_passed = true;

    for(const auto& variant : ALL_VARIANTS)
    {
        if(!runVariant(variant, archive, gpu_arch.c_str(), host_a, host_b, 1.0f, 1.0f))
            all_passed = false;
    }

    // --- Scaled-add test (alpha=2, beta=0.5) ---
    std::printf("\nScaled-add test (alpha=2, beta=0.5):\n");
    const auto* scaled_variant =
        rocm_ck::findVariant(rocm_ck::DataType::FP32, rocm_ck::DataType::FP32, NUM_ELEMENTS);
    if(scaled_variant)
    {
        if(!runVariant(*scaled_variant, archive, gpu_arch.c_str(), host_a, host_b, 2.0f, 0.5f))
            all_passed = false;
    }

    // --- Cleanup ---
    kpack_close(archive);

    return all_passed ? 0 : 1;
}
