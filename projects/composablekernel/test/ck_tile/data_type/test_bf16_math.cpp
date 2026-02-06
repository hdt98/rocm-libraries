// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_bf16_common.hpp"
#include <cmath>
#include <hip/hip_runtime.h>
#include "ck_tile/host/hip_check_error.hpp"

using namespace ck_tile;
using namespace ck_tile_test;

// Device kernels for testing math functions
__global__ void test_abs_kernel(const bf16_t* input, bf16_t* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        output[idx] = abs(input[idx]);
    }
}

__global__ void test_sqrt_kernel(const bf16_t* input, bf16_t* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        output[idx] = sqrt(input[idx]);
    }
}

__global__ void test_exp_kernel(const bf16_t* input, bf16_t* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        output[idx] = exp(input[idx]);
    }
}

__global__ void test_exp2_kernel(const bf16_t* input, bf16_t* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        output[idx] = exp2(input[idx]);
    }
}

__global__ void test_log_kernel(const bf16_t* input, bf16_t* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        output[idx] = log(input[idx]);
    }
}

class Bf16MathTest : public Bf16TestBase
{
    protected:
    void* d_input                     = nullptr;
    void* d_output                    = nullptr;
    static constexpr size_t test_size = 256;

    void SetUp() override
    {
        Bf16TestBase::SetUp();
        hip_check_error(hipMalloc(&d_input, test_size * sizeof(bf16_t)));
        hip_check_error(hipMalloc(&d_output, test_size * sizeof(bf16_t)));
    }

    void TearDown() override
    {
        if(d_input)
            hip_check_error(hipFree(d_input));
        if(d_output)
            hip_check_error(hipFree(d_output));
        Bf16TestBase::TearDown();
    }
};

// Test abs() function on host
TEST_F(Bf16MathTest, AbsHost)
{
    // Positive value
    {
        bf16_t x      = float_to_bf16(3.14159f);
        bf16_t result = abs(x);
        EXPECT_EQ(bf16_to_bits(result), bf16_to_bits(x));
    }

    // Negative value
    {
        bf16_t x      = float_to_bf16(-3.14159f);
        bf16_t result = abs(x);
        // abs() should clear the sign bit, giving the positive value
        bf16_t expected = float_to_bf16(3.14159f);
        EXPECT_NEAR(bf16_to_float(result), 3.14159f, 0.01f);
        EXPECT_EQ(bf16_to_bits(result), bf16_to_bits(expected)); // Should match positive conversion
    }

    // Zero
    {
        bf16_t x      = float_to_bf16(0.0f);
        bf16_t result = abs(x);
        EXPECT_EQ(bf16_to_bits(result), 0x0000);
    }

    // Negative zero
    {
        bf16_t x      = bits_to_bf16(0x8000);
        bf16_t result = abs(x);
        EXPECT_EQ(bf16_to_bits(result), 0x0000); // Should become positive zero
    }

    // Infinity
    {
        bf16_t x      = numeric<bf16_t>::infinity();
        bf16_t result = abs(x);
        EXPECT_EQ(bf16_to_bits(result), 0x7F80);
    }

    // Negative infinity
    {
        bf16_t x      = bits_to_bf16(0xFF80);
        bf16_t result = abs(x);
        EXPECT_EQ(bf16_to_bits(result), 0x7F80); // Should become positive infinity
    }

    // NaN
    {
        bf16_t x      = numeric<bf16_t>::quiet_NaN();
        bf16_t result = abs(x);
        // abs() should clear sign bit but preserve NaN
        EXPECT_TRUE(isnan(result));
        EXPECT_FALSE(bf16_to_bits(result) & 0x8000); // Sign bit should be clear
    }
}

// Test isnan() predicate
TEST_F(Bf16MathTest, IsNanPredicate)
{
    // Normal values - should not be NaN
    EXPECT_FALSE(isnan(float_to_bf16(1.0f)));
    EXPECT_FALSE(isnan(float_to_bf16(-1.0f)));
    EXPECT_FALSE(isnan(float_to_bf16(0.0f)));
    EXPECT_FALSE(isnan(bits_to_bf16(0x8000))); // -0.0

    // Infinity - should not be NaN
    EXPECT_FALSE(isnan(numeric<bf16_t>::infinity()));
    EXPECT_FALSE(isnan(bits_to_bf16(0xFF80))); // -infinity

    // Various NaN patterns - should be NaN
    EXPECT_TRUE(isnan(numeric<bf16_t>::quiet_NaN()));
    EXPECT_TRUE(isnan(numeric<bf16_t>::signaling_NaN()));

    // Test various NaN bit patterns
    for(uint16_t mant = 1; mant <= 0x7F; mant++)
    {
        // Positive NaN
        bf16_t pos_nan = bits_to_bf16(0x7F80 | mant);
        EXPECT_TRUE(isnan(pos_nan)) << "Bits 0x" << std::hex << (0x7F80 | mant) << " should be NaN";

        // Negative NaN
        bf16_t neg_nan = bits_to_bf16(0xFF80 | mant);
        EXPECT_TRUE(isnan(neg_nan)) << "Bits 0x" << std::hex << (0xFF80 | mant) << " should be NaN";
    }
}

// Test abs() function on device
TEST_F(Bf16MathTest, AbsDevice)
{
    std::vector<bf16_t> h_input;
    std::vector<bf16_t> h_output(test_size);

    // Generate test values
    h_input.push_back(float_to_bf16(1.0f));
    h_input.push_back(float_to_bf16(-1.0f));
    h_input.push_back(float_to_bf16(3.14159f));
    h_input.push_back(float_to_bf16(-3.14159f));
    h_input.push_back(float_to_bf16(0.0f));
    h_input.push_back(bits_to_bf16(0x8000)); // -0.0
    h_input.push_back(numeric<bf16_t>::infinity());
    h_input.push_back(bits_to_bf16(0xFF80)); // -infinity
    h_input.push_back(numeric<bf16_t>::quiet_NaN());
    h_input.push_back(numeric<bf16_t>::max());
    h_input.push_back(numeric<bf16_t>::lowest());

    // Fill remaining with random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    while(h_input.size() < test_size)
    {
        h_input.push_back(float_to_bf16(dist(gen)));
    }

    // Copy to device
    hip_check_error(
        hipMemcpy(d_input, h_input.data(), test_size * sizeof(bf16_t), hipMemcpyHostToDevice));

    // Launch kernel
    dim3 block(256);
    dim3 grid((test_size + block.x - 1) / block.x);
    test_abs_kernel<<<grid, block>>>(
        static_cast<bf16_t*>(d_input), static_cast<bf16_t*>(d_output), test_size);

    // Copy back
    hip_check_error(
        hipMemcpy(h_output.data(), d_output, test_size * sizeof(bf16_t), hipMemcpyDeviceToHost));

    // Verify results
    for(size_t i = 0; i < test_size; i++)
    {
        float input_val  = static_cast<float>(h_input[i]);
        float output_val = static_cast<float>(h_output[i]);

        if(isnan(h_input[i]))
        {
            EXPECT_TRUE(isnan(h_output[i])) << "abs(NaN) should be NaN";
            // Sign bit should be cleared
            EXPECT_FALSE(bf16_to_bits(h_output[i]) & 0x8000);
        }
        else
        {
            EXPECT_EQ(output_val, std::abs(input_val))
                << "abs(" << input_val << ") = " << output_val << " at index " << i;
        }
    }
}

// Test sqrt() function on device
TEST_F(Bf16MathTest, SqrtDevice)
{
    std::vector<bf16_t> h_input;
    std::vector<bf16_t> h_output(test_size);

    // Generate test values
    h_input.push_back(float_to_bf16(0.0f));
    h_input.push_back(float_to_bf16(1.0f));
    h_input.push_back(float_to_bf16(4.0f));
    h_input.push_back(float_to_bf16(9.0f));
    h_input.push_back(float_to_bf16(16.0f));
    h_input.push_back(float_to_bf16(0.25f));
    h_input.push_back(float_to_bf16(0.5f));
    h_input.push_back(float_to_bf16(2.0f));
    h_input.push_back(numeric<bf16_t>::infinity());
    h_input.push_back(float_to_bf16(-1.0f)); // Should produce NaN
    h_input.push_back(bits_to_bf16(0xFF80)); // -infinity, should produce NaN
    h_input.push_back(numeric<bf16_t>::quiet_NaN());

    // Fill remaining with positive values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    while(h_input.size() < test_size)
    {
        h_input.push_back(float_to_bf16(dist(gen)));
    }

    // Copy to device
    hip_check_error(
        hipMemcpy(d_input, h_input.data(), test_size * sizeof(bf16_t), hipMemcpyHostToDevice));

    // Launch kernel
    dim3 block(256);
    dim3 grid((test_size + block.x - 1) / block.x);
    test_sqrt_kernel<<<grid, block>>>(
        static_cast<bf16_t*>(d_input), static_cast<bf16_t*>(d_output), test_size);

    // Copy back
    hip_check_error(
        hipMemcpy(h_output.data(), d_output, test_size * sizeof(bf16_t), hipMemcpyDeviceToHost));

    // Verify results
    for(size_t i = 0; i < test_size; i++)
    {
        float input_val  = static_cast<float>(h_input[i]);
        float output_val = static_cast<float>(h_output[i]);

        if(isnan(h_input[i]))
        {
            EXPECT_TRUE(isnan(h_output[i])) << "sqrt(NaN) should be NaN";
        }
        else if(input_val < 0.0f)
        {
            EXPECT_TRUE(isnan(h_output[i])) << "sqrt(negative) should be NaN";
        }
        else if(std::isinf(input_val))
        {
            EXPECT_TRUE(std::isinf(output_val) && output_val > 0) << "sqrt(+inf) should be +inf";
        }
        else
        {
            float expected = std::sqrt(input_val);
            // Allow for some error due to bf16 precision
            EXPECT_NEAR(output_val, expected, expected * 0.01f)
                << "sqrt(" << input_val << ") = " << output_val << " at index " << i;
        }
    }
}

// Test exp() function on device
TEST_F(Bf16MathTest, ExpDevice)
{
    std::vector<bf16_t> h_input;
    std::vector<bf16_t> h_output(test_size);

    // Generate test values
    h_input.push_back(float_to_bf16(0.0f));           // exp(0) = 1
    h_input.push_back(float_to_bf16(1.0f));           // exp(1) = e
    h_input.push_back(float_to_bf16(-1.0f));          // exp(-1) = 1/e
    h_input.push_back(float_to_bf16(2.0f));           // exp(2) = e^2
    h_input.push_back(float_to_bf16(std::log(2.0f))); // exp(ln(2)) = 2
    h_input.push_back(numeric<bf16_t>::infinity());
    h_input.push_back(bits_to_bf16(0xFF80)); // -infinity
    h_input.push_back(numeric<bf16_t>::quiet_NaN());

    // Add values that will overflow/underflow
    h_input.push_back(float_to_bf16(100.0f));  // Will overflow to infinity
    h_input.push_back(float_to_bf16(-100.0f)); // Will underflow to zero

    // Fill remaining with reasonable values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    while(h_input.size() < test_size)
    {
        h_input.push_back(float_to_bf16(dist(gen)));
    }

    // Copy to device
    hip_check_error(
        hipMemcpy(d_input, h_input.data(), test_size * sizeof(bf16_t), hipMemcpyHostToDevice));

    // Launch kernel
    dim3 block(256);
    dim3 grid((test_size + block.x - 1) / block.x);
    test_exp_kernel<<<grid, block>>>(
        static_cast<bf16_t*>(d_input), static_cast<bf16_t*>(d_output), test_size);

    // Copy back
    hip_check_error(
        hipMemcpy(h_output.data(), d_output, test_size * sizeof(bf16_t), hipMemcpyDeviceToHost));

    // Verify results
    for(size_t i = 0; i < test_size; i++)
    {
        float input_val  = static_cast<float>(h_input[i]);
        float output_val = static_cast<float>(h_output[i]);

        if(isnan(h_input[i]))
        {
            EXPECT_TRUE(isnan(h_output[i])) << "exp(NaN) should be NaN";
        }
        else if(input_val == std::numeric_limits<float>::infinity())
        {
            EXPECT_TRUE(std::isinf(output_val) && output_val > 0) << "exp(+inf) should be +inf";
        }
        else if(input_val == -std::numeric_limits<float>::infinity())
        {
            EXPECT_EQ(output_val, 0.0f) << "exp(-inf) should be 0";
        }
        else if(input_val > 80.0f)
        { // Will overflow in bf16
            EXPECT_TRUE(std::isinf(output_val) && output_val > 0)
                << "exp(large) should overflow to +inf";
        }
        else if(input_val < -80.0f)
        { // Will underflow in bf16
            EXPECT_EQ(output_val, 0.0f) << "exp(very negative) should underflow to 0";
        }
        else
        {
            float expected = std::exp(input_val);
            // Allow for significant error due to bf16 precision
            float rel_error = std::abs(output_val - expected) / (expected + 1e-10f);
            EXPECT_LT(rel_error, 0.02f) << "exp(" << input_val << ") = " << output_val
                                        << ", expected " << expected << " at index " << i;
        }
    }
}

// Test exp2() function on device
TEST_F(Bf16MathTest, Exp2Device)
{
    std::vector<bf16_t> h_input;
    std::vector<bf16_t> h_output(test_size);

    // Generate test values
    h_input.push_back(float_to_bf16(0.0f));  // 2^0 = 1
    h_input.push_back(float_to_bf16(1.0f));  // 2^1 = 2
    h_input.push_back(float_to_bf16(2.0f));  // 2^2 = 4
    h_input.push_back(float_to_bf16(-1.0f)); // 2^(-1) = 0.5
    h_input.push_back(float_to_bf16(10.0f)); // 2^10 = 1024
    h_input.push_back(numeric<bf16_t>::infinity());
    h_input.push_back(bits_to_bf16(0xFF80)); // -infinity
    h_input.push_back(numeric<bf16_t>::quiet_NaN());

    // Fill remaining with reasonable values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    while(h_input.size() < test_size)
    {
        h_input.push_back(float_to_bf16(dist(gen)));
    }

    // Copy to device
    hip_check_error(
        hipMemcpy(d_input, h_input.data(), test_size * sizeof(bf16_t), hipMemcpyHostToDevice));

    // Launch kernel
    dim3 block(256);
    dim3 grid((test_size + block.x - 1) / block.x);
    test_exp2_kernel<<<grid, block>>>(
        static_cast<bf16_t*>(d_input), static_cast<bf16_t*>(d_output), test_size);

    // Copy back
    hip_check_error(
        hipMemcpy(h_output.data(), d_output, test_size * sizeof(bf16_t), hipMemcpyDeviceToHost));

    // Verify results
    for(size_t i = 0; i < test_size; i++)
    {
        float input_val  = static_cast<float>(h_input[i]);
        float output_val = static_cast<float>(h_output[i]);

        if(isnan(h_input[i]))
        {
            EXPECT_TRUE(isnan(h_output[i])) << "exp2(NaN) should be NaN";
        }
        else if(input_val == std::numeric_limits<float>::infinity())
        {
            EXPECT_TRUE(std::isinf(output_val) && output_val > 0) << "exp2(+inf) should be +inf";
        }
        else if(input_val == -std::numeric_limits<float>::infinity())
        {
            EXPECT_EQ(output_val, 0.0f) << "exp2(-inf) should be 0";
        }
        else if(input_val > 120.0f)
        { // Will overflow in bf16
            EXPECT_TRUE(std::isinf(output_val) && output_val > 0)
                << "exp2(large) should overflow to +inf";
        }
        else if(input_val < -120.0f)
        { // Will underflow in bf16
            EXPECT_EQ(output_val, 0.0f) << "exp2(very negative) should underflow to 0";
        }
        else
        {
            float expected = std::exp2(input_val);
            // Allow for significant error due to bf16 precision
            float rel_error = std::abs(output_val - expected) / (expected + 1e-10f);
            EXPECT_LT(rel_error, 0.02f) << "exp2(" << input_val << ") = " << output_val
                                        << ", expected " << expected << " at index " << i;
        }
    }
}

// Test log() function on device
TEST_F(Bf16MathTest, LogDevice)
{
    std::vector<bf16_t> h_input;
    std::vector<bf16_t> h_output(test_size);

    // Generate test values
    h_input.push_back(float_to_bf16(1.0f));           // log(1) = 0
    h_input.push_back(float_to_bf16(std::exp(1.0f))); // log(e) = 1
    h_input.push_back(float_to_bf16(2.0f));           // log(2) = ln(2)
    h_input.push_back(float_to_bf16(10.0f));          // log(10) = ln(10)
    h_input.push_back(float_to_bf16(0.5f));           // log(0.5) = -ln(2)
    h_input.push_back(float_to_bf16(0.0f));           // log(0) = -inf
    h_input.push_back(float_to_bf16(-1.0f));          // log(negative) = NaN
    h_input.push_back(numeric<bf16_t>::infinity());
    h_input.push_back(bits_to_bf16(0xFF80)); // -infinity
    h_input.push_back(numeric<bf16_t>::quiet_NaN());

    // Fill remaining with positive values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.01f, 100.0f);
    while(h_input.size() < test_size)
    {
        h_input.push_back(float_to_bf16(dist(gen)));
    }

    // Copy to device
    hip_check_error(
        hipMemcpy(d_input, h_input.data(), test_size * sizeof(bf16_t), hipMemcpyHostToDevice));

    // Launch kernel
    dim3 block(256);
    dim3 grid((test_size + block.x - 1) / block.x);
    test_log_kernel<<<grid, block>>>(
        static_cast<bf16_t*>(d_input), static_cast<bf16_t*>(d_output), test_size);

    // Copy back
    hip_check_error(
        hipMemcpy(h_output.data(), d_output, test_size * sizeof(bf16_t), hipMemcpyDeviceToHost));

    // Verify results
    for(size_t i = 0; i < test_size; i++)
    {
        float input_val  = static_cast<float>(h_input[i]);
        float output_val = static_cast<float>(h_output[i]);

        if(isnan(h_input[i]))
        {
            EXPECT_TRUE(isnan(h_output[i])) << "log(NaN) should be NaN";
        }
        else if(input_val < 0.0f)
        {
            EXPECT_TRUE(isnan(h_output[i])) << "log(negative) should be NaN";
        }
        else if(input_val == 0.0f)
        {
            EXPECT_TRUE(std::isinf(output_val) && output_val < 0) << "log(0) should be -inf";
        }
        else if(std::isinf(input_val) && input_val > 0)
        {
            EXPECT_TRUE(std::isinf(output_val) && output_val > 0) << "log(+inf) should be +inf";
        }
        else if(std::isinf(input_val) && input_val < 0)
        {
            EXPECT_TRUE(isnan(h_output[i])) << "log(-inf) should be NaN";
        }
        else
        {
            float expected = std::log(input_val);
            // Allow for significant error due to bf16 precision
            float abs_error = std::abs(output_val - expected);
            float rel_error = abs_error / (std::abs(expected) + 1e-10f);
            // Use absolute error for values close to zero
            if(std::abs(expected) < 0.1f)
            {
                EXPECT_LT(abs_error, 0.01f) << "log(" << input_val << ") = " << output_val
                                            << ", expected " << expected << " at index " << i;
            }
            else
            {
                EXPECT_LT(rel_error, 0.02f) << "log(" << input_val << ") = " << output_val
                                            << ", expected " << expected << " at index " << i;
            }
        }
    }
}
