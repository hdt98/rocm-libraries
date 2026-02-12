// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_bf16_common.hpp"
#include <cmath>
#include <type_traits>
#include <hip/hip_runtime.h>
#include "ck_tile/host/hip_check_error.hpp"

using namespace ck_tile;
using namespace ck_tile_test;

// Device kernel for testing vector operations
__global__ void test_vector_conversion_kernel(const float* input, bf16_t* output, size_t n)
{
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if(idx + 1 < n)
    {
        fp32x2_t f32_vec;
        f32_vec.x = input[idx];
        f32_vec.y = input[idx + 1];

        bf16x2_t bf16_vec = fp32x2_to_bf16x2(f32_vec);
        output[idx]       = bf16_vec.x;
        output[idx + 1]   = bf16_vec.y;
    }
}

__global__ void test_vector_element_access_kernel(bf16_t* data, size_t n)
{
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if(idx + 3 < n)
    {
        // Test bf16x4_t element access
        bf16x4_t vec4;
        vec4.x = data[idx];
        vec4.y = data[idx + 1];
        vec4.z = data[idx + 2];
        vec4.w = data[idx + 3];

        // Test lo/hi access
        bf16x2_t lo = vec4.lo;
        bf16x2_t hi = vec4.hi;

        // Write back swapped
        data[idx]     = hi.x; // was vec4.z
        data[idx + 1] = hi.y; // was vec4.w
        data[idx + 2] = lo.x; // was vec4.x
        data[idx + 3] = lo.y; // was vec4.y
    }
}

class Bf16VectorTest : public Bf16TestBase
{
};

// Test vector type sizes and alignment
TEST_F(Bf16VectorTest, VectorTypeSizes)
{
    // Verify sizes
    EXPECT_EQ(sizeof(bf16x2_t), 4);    // 2 * 2 bytes
    EXPECT_EQ(sizeof(bf16x4_t), 8);    // 4 * 2 bytes
    EXPECT_EQ(sizeof(bf16x8_t), 16);   // 8 * 2 bytes
    EXPECT_EQ(sizeof(bf16x16_t), 32);  // 16 * 2 bytes
    EXPECT_EQ(sizeof(bf16x32_t), 64);  // 32 * 2 bytes
    EXPECT_EQ(sizeof(bf16x64_t), 128); // 64 * 2 bytes

    // Verify alignment
    EXPECT_EQ(alignof(bf16x2_t), 4);
    EXPECT_EQ(alignof(bf16x4_t), 8);
    EXPECT_EQ(alignof(bf16x8_t), 16);
    EXPECT_EQ(alignof(bf16x16_t), 32);
    EXPECT_EQ(alignof(bf16x32_t), 64);
    EXPECT_EQ(alignof(bf16x64_t), 128);
}

// Test fp32x2_to_bf16x2 conversion
TEST_F(Bf16VectorTest, Fp32x2ToBf16x2Conversion)
{
    // Basic conversion
    {
        fp32x2_t f32_vec;
        f32_vec.x = 1.0f;
        f32_vec.y = 2.0f;

        bf16x2_t bf16_vec = fp32x2_to_bf16x2(f32_vec);

        EXPECT_EQ(static_cast<float>(bf16_vec.x), 1.0f);
        EXPECT_EQ(static_cast<float>(bf16_vec.y), 2.0f);
    }

    // Special values
    {
        fp32x2_t f32_vec;
        f32_vec.x = std::numeric_limits<float>::infinity();
        f32_vec.y = -std::numeric_limits<float>::infinity();

        bf16x2_t bf16_vec = fp32x2_to_bf16x2(f32_vec);

        EXPECT_TRUE(std::isinf(static_cast<float>(bf16_vec.x)));
        EXPECT_GT(static_cast<float>(bf16_vec.x), 0.0f);
        EXPECT_TRUE(std::isinf(static_cast<float>(bf16_vec.y)));
        EXPECT_LT(static_cast<float>(bf16_vec.y), 0.0f);
    }

    // NaN values
    {
        fp32x2_t f32_vec;
        f32_vec.x = std::numeric_limits<float>::quiet_NaN();
        f32_vec.y = 3.14159f;

        bf16x2_t bf16_vec = fp32x2_to_bf16x2(f32_vec);

        EXPECT_TRUE(isnan(bf16_vec.x));
        EXPECT_NEAR(bf16_to_float(bf16_vec.y), 3.14159f, 0.01f);
    }
}

// Test different rounding modes for vector conversion
TEST_F(Bf16VectorTest, Fp32x2ToBf16x2RoundingModes)
{
    fp32x2_t f32_vec;
    f32_vec.x = 1.001953125f; // Requires rounding
    f32_vec.y = -1.001953125f;

    // Standard rounding
    {
        bf16x2_t bf16_vec = fp32x2_to_bf16x2<bf16_rounding_mode::standard>(f32_vec);
        float x_result    = static_cast<float>(bf16_vec.x);
        float y_result    = static_cast<float>(bf16_vec.y);
        EXPECT_NEAR(x_result, 1.001953125f, 0.01f);
        EXPECT_NEAR(y_result, -1.001953125f, 0.01f);
    }

    // Truncation mode
    {
        bf16x2_t bf16_vec = fp32x2_to_bf16x2<bf16_rounding_mode::truncate>(f32_vec);
        float x_result    = static_cast<float>(bf16_vec.x);
        float y_result    = static_cast<float>(bf16_vec.y);
        EXPECT_LE(x_result, 1.001953125f);
        EXPECT_GE(y_result, -1.001953125f);
    }
}

// Test vector element access
TEST_F(Bf16VectorTest, VectorElementAccess)
{
    // bf16x2_t element access
    {
        bf16x2_t vec2;
        vec2.x = float_to_bf16(1.0f);
        vec2.y = float_to_bf16(2.0f);

        EXPECT_EQ(static_cast<float>(vec2.x), 1.0f);
        EXPECT_EQ(static_cast<float>(vec2.y), 2.0f);

        // Modify elements
        vec2.x = float_to_bf16(3.0f);
        vec2.y = float_to_bf16(4.0f);

        EXPECT_EQ(static_cast<float>(vec2.x), 3.0f);
        EXPECT_EQ(static_cast<float>(vec2.y), 4.0f);
    }

    // bf16x4_t element access
    {
        bf16x4_t vec4;
        vec4.x = float_to_bf16(1.0f);
        vec4.y = float_to_bf16(2.0f);
        vec4.z = float_to_bf16(3.0f);
        vec4.w = float_to_bf16(4.0f);

        EXPECT_EQ(static_cast<float>(vec4.x), 1.0f);
        EXPECT_EQ(static_cast<float>(vec4.y), 2.0f);
        EXPECT_EQ(static_cast<float>(vec4.z), 3.0f);
        EXPECT_EQ(static_cast<float>(vec4.w), 4.0f);

        // Test lo/hi access
        bf16x2_t lo = vec4.lo;
        bf16x2_t hi = vec4.hi;

        EXPECT_EQ(static_cast<float>(lo.x), 1.0f);
        EXPECT_EQ(static_cast<float>(lo.y), 2.0f);
        EXPECT_EQ(static_cast<float>(hi.x), 3.0f);
        EXPECT_EQ(static_cast<float>(hi.y), 4.0f);
    }
}

// Test vector initialization patterns
TEST_F(Bf16VectorTest, VectorInitialization)
{
    // Default initialization
    {
        bf16x2_t vec2;
        bf16x4_t vec4;
        bf16x8_t vec8;
        // Default initialized vectors have undefined values, so we just check they exist
        EXPECT_EQ(sizeof(vec2), 4);
        EXPECT_EQ(sizeof(vec4), 8);
        EXPECT_EQ(sizeof(vec8), 16);
    }

    // Brace initialization
    {
        bf16x2_t vec2 = {float_to_bf16(1.0f), float_to_bf16(2.0f)};
        EXPECT_EQ(static_cast<float>(vec2.x), 1.0f);
        EXPECT_EQ(static_cast<float>(vec2.y), 2.0f);

        bf16x4_t vec4 = {
            float_to_bf16(1.0f), float_to_bf16(2.0f), float_to_bf16(3.0f), float_to_bf16(4.0f)};
        EXPECT_EQ(static_cast<float>(vec4.x), 1.0f);
        EXPECT_EQ(static_cast<float>(vec4.y), 2.0f);
        EXPECT_EQ(static_cast<float>(vec4.z), 3.0f);
        EXPECT_EQ(static_cast<float>(vec4.w), 4.0f);
    }
}

// Test vector operations on device
TEST_F(Bf16VectorTest, VectorOperationsDevice)
{
    const size_t n = 256;
    float* d_float_input;
    bf16_t* d_bf16_output;

    hip_check_error(hipMalloc(&d_float_input, n * sizeof(float)));
    hip_check_error(hipMalloc(&d_bf16_output, n * sizeof(bf16_t)));

    // Generate test data
    std::vector<float> h_float_input(n);
    std::vector<bf16_t> h_bf16_output(n);

    for(size_t i = 0; i < n; i++)
    {
        h_float_input[i] = static_cast<float>(i) * 0.1f - 12.8f;
    }

    // Test vector conversion on device
    {
        hip_check_error(hipMemcpy(
            d_float_input, h_float_input.data(), n * sizeof(float), hipMemcpyHostToDevice));

        dim3 block(128);
        dim3 grid((n / 2 + block.x - 1) / block.x);
        test_vector_conversion_kernel<<<grid, block>>>(d_float_input, d_bf16_output, n);

        hip_check_error(hipMemcpy(
            h_bf16_output.data(), d_bf16_output, n * sizeof(bf16_t), hipMemcpyDeviceToHost));

        // Verify results
        for(size_t i = 0; i < n; i++)
        {
            float expected = h_float_input[i];
            float actual   = static_cast<float>(h_bf16_output[i]);
            EXPECT_NEAR(actual, expected, std::abs(expected) * 0.01f + 0.01f)
                << "Mismatch at index " << i;
        }
    }

    // Test element access on device
    {
        // Initialize with pattern
        for(size_t i = 0; i < n; i++)
        {
            h_bf16_output[i] = float_to_bf16(static_cast<float>(i % 4));
        }

        hip_check_error(hipMemcpy(
            d_bf16_output, h_bf16_output.data(), n * sizeof(bf16_t), hipMemcpyHostToDevice));

        dim3 block(64);
        dim3 grid((n / 4 + block.x - 1) / block.x);
        test_vector_element_access_kernel<<<grid, block>>>(d_bf16_output, n);

        hip_check_error(hipMemcpy(
            h_bf16_output.data(), d_bf16_output, n * sizeof(bf16_t), hipMemcpyDeviceToHost));

        // Verify swapping worked correctly
        for(size_t i = 0; i < n; i += 4)
        {
            // Original: [0, 1, 2, 3]
            // After swap: [2, 3, 0, 1]
            EXPECT_EQ(static_cast<float>(h_bf16_output[i]), 2.0f);
            EXPECT_EQ(static_cast<float>(h_bf16_output[i + 1]), 3.0f);
            EXPECT_EQ(static_cast<float>(h_bf16_output[i + 2]), 0.0f);
            EXPECT_EQ(static_cast<float>(h_bf16_output[i + 3]), 1.0f);
        }
    }

    hip_check_error(hipFree(d_float_input));
    hip_check_error(hipFree(d_bf16_output));
}

// Test vector type traits
TEST_F(Bf16VectorTest, VectorTypeTraits)
{
    // Verify vector types are trivially copyable
    EXPECT_TRUE(std::is_trivially_copyable<bf16x2_t>::value);
    EXPECT_TRUE(std::is_trivially_copyable<bf16x4_t>::value);
    EXPECT_TRUE(std::is_trivially_copyable<bf16x8_t>::value);
    EXPECT_TRUE(std::is_trivially_copyable<bf16x16_t>::value);
    EXPECT_TRUE(std::is_trivially_copyable<bf16x32_t>::value);
    EXPECT_TRUE(std::is_trivially_copyable<bf16x64_t>::value);

    // Verify POD nature
    EXPECT_TRUE(std::is_standard_layout<bf16x2_t>::value);
    EXPECT_TRUE(std::is_standard_layout<bf16x4_t>::value);
    EXPECT_TRUE(std::is_standard_layout<bf16x8_t>::value);
}

// Test edge cases with vector operations
TEST_F(Bf16VectorTest, VectorEdgeCases)
{
    // Vector with all special values
    {
        bf16x4_t vec;
        vec.x = numeric<bf16_t>::infinity();
        vec.y = bits_to_bf16(0xFF80); // -infinity
        vec.z = numeric<bf16_t>::quiet_NaN();
        vec.w = float_to_bf16(0.0f);

        // Verify each element
        EXPECT_TRUE(std::isinf(static_cast<float>(vec.x)) && static_cast<float>(vec.x) > 0);
        EXPECT_TRUE(std::isinf(static_cast<float>(vec.y)) && static_cast<float>(vec.y) < 0);
        EXPECT_TRUE(isnan(vec.z));
        EXPECT_EQ(static_cast<float>(vec.w), 0.0f);
    }

    // Vector conversion with float max (IEEE RTN rounding to infinity)
    {
        fp32x2_t f32_vec;
        f32_vec.x = std::numeric_limits<float>::max();
        f32_vec.y = -std::numeric_limits<float>::max();

        bf16x2_t bf16_vec = fp32x2_to_bf16x2(f32_vec);

        // BF16 has same 8-bit exponent but only 7 mantissa bits (vs 23 for float32).
        // So bf16::max < float::max.
        // Hardware behavior differs by architecture:
        // - gfx950: RTN rounding -> float::max rounds to infinity (IEEE-754 compliant)
        // - gfx9 (gfx90a, gfx908, gfx942): Saturates -> float::max clamps to bf16::max
        // - gfx12/gfx1250: Saturates -> float::max clamps to bf16::max (faster, non-IEEE)
        float result_x = bf16_to_float(bf16_vec.x);
        float result_y = bf16_to_float(bf16_vec.y);

#ifdef CK_TILE_BF16_OVERFLOW_SATURATES
        // gfx9/gfx11/gfx12: Hardware saturates to bf16::max
        EXPECT_FALSE(std::isinf(result_x))
            << "gfx9/gfx11/gfx12: float::max should saturate to bf16::max";
        EXPECT_FALSE(std::isinf(result_y))
            << "gfx9/gfx11/gfx12: -float::max should saturate to -bf16::max";
#else
        // gfx950 and software: RTN rounding to infinity (IEEE-754 behavior)
        EXPECT_TRUE(std::isinf(result_x) && result_x > 0)
            << "Float max should overflow to bf16 +infinity with RTN rounding";
        EXPECT_TRUE(std::isinf(result_y) && result_y < 0)
            << "Negative float max should overflow to bf16 -infinity with RTN rounding";
#endif
    }

    // Vector conversion with denormals
    {
        fp32x2_t f32_vec;
        f32_vec.x = std::numeric_limits<float>::denorm_min();
        f32_vec.y = -std::numeric_limits<float>::denorm_min();

        bf16x2_t bf16_vec = fp32x2_to_bf16x2(f32_vec);

        // bf16 doesn't support denormals, should flush to zero
        EXPECT_EQ(static_cast<float>(bf16_vec.x), 0.0f);
        EXPECT_EQ(static_cast<float>(bf16_vec.y), -0.0f);
    }
}
