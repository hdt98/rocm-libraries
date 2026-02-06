// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_bf16_common.hpp"
#include <cmath>
#include <chrono>
#include <hip/hip_runtime.h>
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/device_prop.hpp"

using namespace ck_tile;
using namespace ck_tile_test;

// Device kernel to test native bf16 operations
__global__ void test_native_conversion_kernel(const float* input, bf16_t* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        // This will use either software or hardware conversion based on compile flags
        output[idx] = float_to_bf16(input[idx]);
    }
}

// Device kernel for FMA accumulation test (bf16 * bf16 + fp32 accumulator)
__global__ void
test_fma_accumulate_kernel(const bf16_t* a, const bf16_t* b, float* acc, size_t n, int iterations)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        float result = acc[idx];
        bf16_t val_a = a[idx];
        bf16_t val_b = b[idx];
        // Simulate GEMM inner loop: repeated bf16 * bf16 accumulated in fp32
        for(int i = 0; i < iterations; i++)
        {
            result = fma(bf16_to_float(val_a), bf16_to_float(val_b), result);
        }
        acc[idx] = result;
    }
}

// Device kernel to test arithmetic performance
__global__ void
test_arithmetic_performance_kernel(const bf16_t* a, const bf16_t* b, bf16_t* c, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
#if CK_TILE_USE_CUSTOM_DATA_TYPE
        // Perform some arithmetic operations
        bf16_t val_a  = a[idx];
        bf16_t val_b  = b[idx];
        bf16_t result = val_a + val_b;
        result        = result * val_a;
        result        = result - val_b;
        c[idx]        = result;
#else
        // When custom data type is not used, convert to float, compute, convert back
        float val_a  = static_cast<float>(a[idx]);
        float val_b  = static_cast<float>(b[idx]);
        float result = val_a + val_b;
        result       = result * val_a;
        result       = result - val_b;
        c[idx]       = float_to_bf16(result);
#endif
    }
}

class Bf16PlatformTest : public Bf16TestBase
{
    protected:
    std::string device_name;

    void SetUp() override
    {
        Bf16TestBase::SetUp();
        device_name = get_device_name();
    }
};

// Test compile-time flags
TEST_F(Bf16PlatformTest, CompileTimeFlags)
{
    std::cout << "=== BF16 Platform Configuration ===" << std::endl;
    std::cout << "Device: " << device_name << std::endl;

#ifdef CK_TILE_USE_LLVM_BUILTIN_BF16
    std::cout << "CK_TILE_USE_LLVM_BUILTIN_BF16: " << CK_TILE_USE_LLVM_BUILTIN_BF16 << std::endl;
#else
    std::cout << "CK_TILE_USE_LLVM_BUILTIN_BF16: undefined (defaults to 0)" << std::endl;
#endif

#ifdef CK_TILE_USE_CUSTOM_DATA_TYPE
    std::cout << "CK_TILE_USE_CUSTOM_DATA_TYPE: " << CK_TILE_USE_CUSTOM_DATA_TYPE << std::endl;
#else
    std::cout << "CK_TILE_USE_CUSTOM_DATA_TYPE: undefined (defaults to 0)" << std::endl;
#endif

#ifdef __gfx950__
    std::cout << "__gfx950__ is defined" << std::endl;
#else
    std::cout << "__gfx950__ is NOT defined" << std::endl;
#endif

#ifdef CK_GFX950_SUPPORT
    std::cout << "CK_GFX950_SUPPORT is defined" << std::endl;
#else
    std::cout << "CK_GFX950_SUPPORT is NOT defined" << std::endl;
#endif

    std::cout << "===================================" << std::endl;
}

// Test type identification
TEST_F(Bf16PlatformTest, TypeIdentification)
{
    // Check what type bf16_t actually is
    std::cout << "sizeof(bf16_t): " << sizeof(bf16_t) << " bytes" << std::endl;
    std::cout << "sizeof(bfloat16_t): " << sizeof(bfloat16_t) << " bytes" << std::endl;

#if CK_TILE_USE_CUSTOM_DATA_TYPE
    std::cout << "Using custom bf16 struct implementation" << std::endl;
    EXPECT_TRUE((std::is_same<bf16_t, bfloat16_t>::value));
    EXPECT_TRUE((std::is_class<bf16_t>::value));
#else
#if CK_TILE_USE_LLVM_BUILTIN_BF16
    std::cout << "Using LLVM __bf16 builtin type" << std::endl;
    EXPECT_TRUE((std::is_same<bfloat16_t, __bf16>::value));
#else
    std::cout << "Using ushort as bf16 type" << std::endl;
    EXPECT_TRUE((std::is_same<bfloat16_t, ushort>::value));
#endif
    EXPECT_TRUE((std::is_same<bf16_t, bfloat16_t>::value));
#endif

    // Always 2 bytes regardless of implementation
    EXPECT_EQ(sizeof(bf16_t), 2);
}

// Test native hardware conversion on gfx950
TEST_F(Bf16PlatformTest, NativeHardwareConversion)
{
#if CK_TILE_USE_LLVM_BUILTIN_BF16 && (defined(__gfx950__) || defined(CK_GFX950_SUPPORT))
    std::cout << "Testing native hardware bf16 conversion on " << device_name << std::endl;

    // Test that native conversion is being used
    {
        float f      = 3.14159f;
        bf16_t b     = float_to_bf16(f);
        float f_back = static_cast<float>(b);

        // The conversion should still work correctly
        EXPECT_NEAR(f_back, f, 0.01f);

        // Check that we're using the expected conversion path
        // When using native __bf16, the conversion should be a simple cast
        bf16_t b_native = static_cast<bf16_t>(f);
        EXPECT_EQ(bf16_to_bits(b), bf16_to_bits(b_native));
    }
#else
    std::cout << "Native hardware bf16 conversion not available on this platform" << std::endl;
    std::cout << "Using software conversion implementation" << std::endl;
#endif
}

// Test conversion accuracy across implementations
TEST_F(Bf16PlatformTest, ConversionAccuracyComparison)
{
    // Test values that stress different aspects of conversion
    std::vector<float> test_values = {1.0f,
                                      -1.0f,
                                      0.5f,
                                      -0.5f,
                                      3.14159f,
                                      -2.71828f,
                                      1.001953125f, // Requires rounding
                                      std::numeric_limits<float>::max(),
                                      std::numeric_limits<float>::min(),
                                      std::numeric_limits<float>::infinity(),
                                      -std::numeric_limits<float>::infinity(),
                                      std::numeric_limits<float>::quiet_NaN(),
                                      0.0f,
                                      -0.0f};

    std::cout << "\nConversion accuracy test:" << std::endl;

    for(float f : test_values)
    {
        // Standard conversion
        bf16_t b_standard = float_to_bf16(f, constant<bf16_rounding_mode::standard>{});

        // Truncation
        bf16_t b_truncate = float_to_bf16(f, constant<bf16_rounding_mode::truncate>{});

        if(!std::isnan(f))
        {
            // For non-NaN values, standard rounding should be more accurate
            float f_standard = static_cast<float>(b_standard);
            float f_truncate = static_cast<float>(b_truncate);

            if(!std::isinf(f) && !std::isinf(f_standard))
            {
                // Skip comparison when standard rounding overflows to infinity
                // (this can happen at boundaries like float_max where rounding
                // causes mantissa overflow that propagates to exponent)
                float err_standard = std::abs(f - f_standard);
                float err_truncate = std::abs(f - f_truncate);

                // Standard rounding should never be worse than truncation
                EXPECT_LE(err_standard, err_truncate + 1e-10f)
                    << "For value " << f << ", standard rounding error (" << err_standard
                    << ") should not exceed truncation error (" << err_truncate << ")";
            }
        }
    }
}

// Test device-side performance characteristics
TEST_F(Bf16PlatformTest, DevicePerformance)
{
    const size_t n       = 1024 * 1024; // 1M elements
    const int iterations = 100;

    float* d_float;
    bf16_t* d_bf16_a;
    bf16_t* d_bf16_b;
    bf16_t* d_bf16_c;

    hip_check_error(hipMalloc(&d_float, n * sizeof(float)));
    hip_check_error(hipMalloc(&d_bf16_a, n * sizeof(bf16_t)));
    hip_check_error(hipMalloc(&d_bf16_b, n * sizeof(bf16_t)));
    hip_check_error(hipMalloc(&d_bf16_c, n * sizeof(bf16_t)));

    // Initialize data
    std::vector<float> h_float(n);
    for(size_t i = 0; i < n; i++)
    {
        h_float[i] = static_cast<float>(i % 1000) / 1000.0f;
    }

    hip_check_error(hipMemcpy(d_float, h_float.data(), n * sizeof(float), hipMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    // Warm up
    for(int i = 0; i < 10; i++)
    {
        test_native_conversion_kernel<<<grid, block>>>(d_float, d_bf16_a, n);
    }
    hip_check_error(hipDeviceSynchronize());

    // Time float to bf16 conversion
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++)
    {
        test_native_conversion_kernel<<<grid, block>>>(d_float, d_bf16_a, n);
    }
    hip_check_error(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    auto duration     = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double avg_time   = static_cast<double>(duration) / iterations;
    double throughput = (n * sizeof(float) + n * sizeof(bf16_t)) / (avg_time * 1e6); // GB/s

    std::cout << "\n=== Performance Results ===" << std::endl;
    std::cout << "Float to BF16 conversion:" << std::endl;
    std::cout << "  Average time: " << avg_time << " μs" << std::endl;
    std::cout << "  Throughput: " << throughput << " GB/s" << std::endl;

    // Initialize bf16 data for arithmetic test
    test_native_conversion_kernel<<<grid, block>>>(d_float, d_bf16_b, n);
    hip_check_error(hipDeviceSynchronize());

    // Time bf16 arithmetic operations
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++)
    {
        test_arithmetic_performance_kernel<<<grid, block>>>(d_bf16_a, d_bf16_b, d_bf16_c, n);
    }
    hip_check_error(hipDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();

    duration   = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    avg_time   = static_cast<double>(duration) / iterations;
    throughput = (3 * n * sizeof(bf16_t)) / (avg_time * 1e6); // GB/s

    std::cout << "\nBF16 arithmetic operations:" << std::endl;
    std::cout << "  Average time: " << avg_time << " μs" << std::endl;
    std::cout << "  Throughput: " << throughput << " GB/s" << std::endl;
    std::cout << "===========================" << std::endl;

    hip_check_error(hipFree(d_float));
    hip_check_error(hipFree(d_bf16_a));
    hip_check_error(hipFree(d_bf16_b));
    hip_check_error(hipFree(d_bf16_c));
}

// Test platform-specific edge cases
TEST_F(Bf16PlatformTest, PlatformEdgeCases)
{
    // Test that the implementation handles architecture-specific quirks correctly

    // Test subnormal handling
    {
        float subnormal = std::numeric_limits<float>::denorm_min();
        bf16_t b        = float_to_bf16(subnormal);

        // bf16 doesn't support subnormals, should flush to zero
        EXPECT_EQ(bf16_to_bits(b), 0x0000);
    }

    // Test NaN propagation with different NaN patterns
    {
        // Create different NaN bit patterns
        std::vector<uint32_t> nan_patterns = {
            0x7FC00000, // Quiet NaN (standard)
            0x7F800001, // Signaling NaN (smallest)
            0x7FFFFFFF, // All mantissa bits set
            0xFFC00000, // Negative quiet NaN
            0xFF800001, // Negative signaling NaN
        };

        for(uint32_t pattern : nan_patterns)
        {
            float f = bit_cast<float>(pattern);
            EXPECT_TRUE(std::isnan(f));

            // Use explicit standard rounding mode which preserves sNaN
            // The standard rounding implementation sets bit 16 when converting
            // NaN patterns with mantissa bits only in lower 16 bits
            bf16_t b =
                bit_cast<bf16_t>(float_to_bf16_raw(f, constant<bf16_rounding_mode::standard>{}));
            EXPECT_TRUE(isnan(b)) << "Pattern 0x" << std::hex << pattern
                                  << " should convert to bf16 NaN";
        }
    }

    // Test boundary values
    // Note: BF16 has the same 8-bit exponent as float32, so they have the same range.
    // BF16 max is approximately 3.39e38 (0x7F7F = 2^127 * (1 + 127/128))
    {
        // Value within bf16 range
        float large = 3.38953139e38f; // Close to bf16 max
        bf16_t b    = float_to_bf16(large);
        EXPECT_FALSE(std::isinf(bf16_to_float(b)));

        // Float infinity should convert to bf16 infinity
        float inf = std::numeric_limits<float>::infinity();
        b         = float_to_bf16(inf);
        EXPECT_TRUE(std::isinf(bf16_to_float(b)));

        // Float max with truncation should NOT overflow (same exponent range)
        // Note: Standard rounding CAN overflow float_max to infinity due to
        // mantissa rounding propagating to the exponent. Use truncation mode
        // to verify that the range is preserved without rounding effects.
        float float_max = std::numeric_limits<float>::max();
        b               = bit_cast<bf16_t>(
            float_to_bf16_raw(float_max, constant<bf16_rounding_mode::truncate>{}));
        EXPECT_FALSE(std::isinf(bf16_to_float(b)))
            << "Float max with truncation should NOT overflow to bf16 infinity";
    }
}

// Test FMA accumulation precision (critical for GEMM kernels)
TEST_F(Bf16PlatformTest, FmaAccumulationPrecision)
{
    // This test verifies that bf16 * bf16 accumulated in fp32 maintains precision.
    // This is the standard pattern used in matrix multiplication kernels.

    const size_t n       = 256;
    const int iterations = 1000;

    bf16_t* d_a;
    bf16_t* d_b;
    float* d_acc;

    hip_check_error(hipMalloc(&d_a, n * sizeof(bf16_t)));
    hip_check_error(hipMalloc(&d_b, n * sizeof(bf16_t)));
    hip_check_error(hipMalloc(&d_acc, n * sizeof(float)));

    std::vector<bf16_t> h_a(n);
    std::vector<bf16_t> h_b(n);
    std::vector<float> h_acc(n, 0.0f);

    // Test case 1: Small values that would underflow if accumulated in bf16
    // 0.001 * 0.001 = 0.000001, which is below bf16 precision
    // But accumulated 1000 times in fp32: 1000 * 0.000001 = 0.001
    {
        for(size_t i = 0; i < n; i++)
        {
            h_a[i]   = float_to_bf16(0.001f);
            h_b[i]   = float_to_bf16(0.001f);
            h_acc[i] = 0.0f;
        }

        hip_check_error(hipMemcpy(d_a, h_a.data(), n * sizeof(bf16_t), hipMemcpyHostToDevice));
        hip_check_error(hipMemcpy(d_b, h_b.data(), n * sizeof(bf16_t), hipMemcpyHostToDevice));
        hip_check_error(hipMemcpy(d_acc, h_acc.data(), n * sizeof(float), hipMemcpyHostToDevice));

        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        test_fma_accumulate_kernel<<<grid, block>>>(d_a, d_b, d_acc, n, iterations);
        hip_check_error(hipDeviceSynchronize());

        hip_check_error(hipMemcpy(h_acc.data(), d_acc, n * sizeof(float), hipMemcpyDeviceToHost));

        // Expected: iterations * (0.001 * 0.001) = 1000 * 0.000001 = 0.001
        // Note: bf16(0.001) is approximately 0.0009765625 due to rounding
        float bf16_val = bf16_to_float(float_to_bf16(0.001f));
        float expected = static_cast<float>(iterations) * bf16_val * bf16_val;

        for(size_t i = 0; i < n; i++)
        {
            EXPECT_NEAR(h_acc[i], expected, expected * 0.01f)
                << "FMA accumulation failed at index " << i;
        }

        std::cout << "FMA small value accumulation: " << iterations << " iterations of " << bf16_val
                  << " * " << bf16_val << " = " << h_acc[0] << " (expected: " << expected << ")"
                  << std::endl;
    }

    // Test case 2: Mixed signs to test catastrophic cancellation
    // Alternating +1 and -1 should sum to 0 (or close to it)
    {
        for(size_t i = 0; i < n; i++)
        {
            h_a[i]   = float_to_bf16(1.0f);
            h_b[i]   = float_to_bf16((i % 2 == 0) ? 1.0f : -1.0f);
            h_acc[i] = 0.0f;
        }

        hip_check_error(hipMemcpy(d_a, h_a.data(), n * sizeof(bf16_t), hipMemcpyHostToDevice));
        hip_check_error(hipMemcpy(d_b, h_b.data(), n * sizeof(bf16_t), hipMemcpyHostToDevice));
        hip_check_error(hipMemcpy(d_acc, h_acc.data(), n * sizeof(float), hipMemcpyHostToDevice));

        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        test_fma_accumulate_kernel<<<grid, block>>>(d_a, d_b, d_acc, n, iterations);
        hip_check_error(hipDeviceSynchronize());

        hip_check_error(hipMemcpy(h_acc.data(), d_acc, n * sizeof(float), hipMemcpyDeviceToHost));

        // Even indices: 1000 * (1.0 * 1.0) = 1000
        // Odd indices: 1000 * (1.0 * -1.0) = -1000
        for(size_t i = 0; i < n; i++)
        {
            float expected =
                (i % 2 == 0) ? static_cast<float>(iterations) : -static_cast<float>(iterations);
            EXPECT_EQ(h_acc[i], expected) << "FMA mixed sign accumulation failed at index " << i;
        }
    }

    // Test case 3: Large values near overflow boundary
    {
        // Use values that when squared approach bf16 max but don't overflow
        float large_val = 100.0f; // 100 * 100 = 10000, well within range
        for(size_t i = 0; i < n; i++)
        {
            h_a[i]   = float_to_bf16(large_val);
            h_b[i]   = float_to_bf16(large_val);
            h_acc[i] = 0.0f;
        }

        hip_check_error(hipMemcpy(d_a, h_a.data(), n * sizeof(bf16_t), hipMemcpyHostToDevice));
        hip_check_error(hipMemcpy(d_b, h_b.data(), n * sizeof(bf16_t), hipMemcpyHostToDevice));
        hip_check_error(hipMemcpy(d_acc, h_acc.data(), n * sizeof(float), hipMemcpyHostToDevice));

        // Only 10 iterations to avoid overflow: 10 * 10000 = 100000
        const int large_iterations = 10;
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        test_fma_accumulate_kernel<<<grid, block>>>(d_a, d_b, d_acc, n, large_iterations);
        hip_check_error(hipDeviceSynchronize());

        hip_check_error(hipMemcpy(h_acc.data(), d_acc, n * sizeof(float), hipMemcpyDeviceToHost));

        float bf16_large = bf16_to_float(float_to_bf16(large_val));
        float expected   = static_cast<float>(large_iterations) * bf16_large * bf16_large;

        for(size_t i = 0; i < n; i++)
        {
            EXPECT_NEAR(h_acc[i], expected, expected * 0.001f)
                << "FMA large value accumulation failed at index " << i;
        }

        std::cout << "FMA large value accumulation: " << large_iterations << " iterations of "
                  << bf16_large << " * " << bf16_large << " = " << h_acc[0]
                  << " (expected: " << expected << ")" << std::endl;
    }

    hip_check_error(hipFree(d_a));
    hip_check_error(hipFree(d_b));
    hip_check_error(hipFree(d_acc));
}

// Summary test
TEST_F(Bf16PlatformTest, PlatformSummary)
{
    std::cout << "\n=== BF16 Implementation Summary ===" << std::endl;
    std::cout << "Device: " << device_name << std::endl;

#if CK_TILE_USE_CUSTOM_DATA_TYPE
    std::cout << "Implementation: Custom BF16 struct with software arithmetic" << std::endl;
#elif CK_TILE_USE_LLVM_BUILTIN_BF16
#if defined(__gfx950__) || defined(CK_GFX950_SUPPORT)
    std::cout << "Implementation: Hardware __bf16 with native conversion (gfx950)" << std::endl;
#else
    std::cout << "Implementation: LLVM __bf16 builtin type" << std::endl;
#endif
#else
    std::cout << "Implementation: ushort with software conversion" << std::endl;
#endif

    std::cout << "Vector types supported: bf16x2_t, bf16x4_t, bf16x8_t, etc." << std::endl;
    std::cout << "Arithmetic operators: "
              << (CK_TILE_USE_CUSTOM_DATA_TYPE ? "Available" : "Not available") << std::endl;
    std::cout << "===================================" << std::endl;
}
