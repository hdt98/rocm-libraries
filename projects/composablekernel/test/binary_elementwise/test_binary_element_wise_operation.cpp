// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include <hip/hip_runtime.h>

#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/host_utility/hip_check_error.hpp"

#include <cmath>

using ::ck::hip_check_error;

using ck::bhalf_t;
using ck::half_t;
using ck::type_convert;

using namespace ck::tensor_operation::element_wise;

// Helper to check if two floats are close enough
template <typename T>
bool is_close(T a, T b, float rtol = 1e-5f, float atol = 1e-5f)
{
    float fa = type_convert<float>(a);
    float fb = type_convert<float>(b);
    return std::abs(fa - fb) <= atol + rtol * std::abs(fb);
}

// =============================================================================
// Add Tests
// =============================================================================

TEST(BinaryElementWiseOp_Host, Add_float_float_float)
{
    Add op;
    float y;
    op(y, 1.5f, 2.5f);
    EXPECT_FLOAT_EQ(y, 4.0f);
}

TEST(BinaryElementWiseOp_Host, Add_double_double_double)
{
    Add op;
    double y;
    op(y, 1.5, 2.5);
    EXPECT_DOUBLE_EQ(y, 4.0);
}

TEST(BinaryElementWiseOp_Host, Add_float_float_half)
{
    // This tests the fix: should convert half_t to float, not the other way
    Add op;
    float y;
    float x0       = 1.0f + 1e-6f; // Value with precision beyond half_t
    half_t x1      = type_convert<half_t>(2.0f);
    // Reference: x1 promoted to float (not x0 demoted to half_t)
    float expected = x0 + type_convert<float>(x1);

    op(y, x0, x1);

    EXPECT_FLOAT_EQ(y, expected);
}

TEST(BinaryElementWiseOp_Host, Add_half_float_half)
{
    // This tests the fix: should compute in float, then convert to half_t
    Add op;
    half_t y;
    float x0  = 1000.0f;
    half_t x1 = type_convert<half_t>(0.001f);

    // Reference: compute in float, then convert
    float ref_f = x0 + type_convert<float>(x1);
    half_t ref  = type_convert<half_t>(ref_f);

    op(y, x0, x1);

    EXPECT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Add_half_half_half)
{
    Add op;
    half_t y;
    half_t x0 = type_convert<half_t>(1.5f);
    half_t x1 = type_convert<half_t>(2.5f);

    op(y, x0, x1);

    EXPECT_TRUE(is_close(y, type_convert<half_t>(4.0f)));
}

TEST(BinaryElementWiseOp_Host, Add_bhalf_bhalf_bhalf)
{
    Add op;
    bhalf_t y;
    bhalf_t x0 = type_convert<bhalf_t>(1.5f);
    bhalf_t x1 = type_convert<bhalf_t>(2.5f);

    // Reference: bhalf_t version promotes to float
    float ref_f = type_convert<float>(x0) + type_convert<float>(x1);
    bhalf_t ref = type_convert<bhalf_t>(ref_f);

    op(y, x0, x1);

    EXPECT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Add_bhalf_float_bhalf)
{
    Add op;
    bhalf_t y;
    float x0   = 100.0f;
    bhalf_t x1 = type_convert<bhalf_t>(0.5f);

    float ref_f = x0 + type_convert<float>(x1);
    bhalf_t ref = type_convert<bhalf_t>(ref_f);

    op(y, x0, x1);

    EXPECT_EQ(y, ref);
}

// =============================================================================
// Multiply Tests
// =============================================================================

TEST(BinaryElementWiseOp_Host, Multiply_float_float_float)
{
    Multiply op;
    float y;
    op(y, 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(y, 6.0f);
}

TEST(BinaryElementWiseOp_Host, Multiply_double_double_double)
{
    Multiply op;
    double y;
    op(y, 2.0, 3.0);
    EXPECT_DOUBLE_EQ(y, 6.0);
}

TEST(BinaryElementWiseOp_Host, Multiply_float_float_half)
{
    // This tests the fix: should convert half_t to float
    Multiply op;
    float y;
    float x0       = 1.0f + 1e-6f;
    half_t x1      = type_convert<half_t>(2.0f);
    // Reference: x1 promoted to float (not x0 demoted to half_t)
    float expected = x0 * type_convert<float>(x1);

    op(y, x0, x1);

    EXPECT_FLOAT_EQ(y, expected);
}

TEST(BinaryElementWiseOp_Host, Multiply_half_float_half)
{
    // This tests the fix: should compute in float, then convert
    Multiply op;
    half_t y;
    float x0  = 100.0f;
    half_t x1 = type_convert<half_t>(0.01f);

    float ref_f = x0 * type_convert<float>(x1);
    half_t ref  = type_convert<half_t>(ref_f);

    op(y, x0, x1);

    EXPECT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Multiply_half_half_half)
{
    Multiply op;
    half_t y;
    half_t x0 = type_convert<half_t>(2.0f);
    half_t x1 = type_convert<half_t>(3.0f);

    op(y, x0, x1);

    EXPECT_TRUE(is_close(y, type_convert<half_t>(6.0f)));
}

TEST(BinaryElementWiseOp_Host, Multiply_bhalf_bhalf_bhalf)
{
    Multiply op;
    bhalf_t y;
    bhalf_t x0 = type_convert<bhalf_t>(2.0f);
    bhalf_t x1 = type_convert<bhalf_t>(3.0f);

    float ref_f = type_convert<float>(x0) * type_convert<float>(x1);
    bhalf_t ref = type_convert<bhalf_t>(ref_f);

    op(y, x0, x1);

    EXPECT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Multiply_bhalf_float_bhalf)
{
    Multiply op;
    bhalf_t y;
    float x0   = 10.0f;
    bhalf_t x1 = type_convert<bhalf_t>(0.5f);

    float ref_f = x0 * type_convert<float>(x1);
    bhalf_t ref = type_convert<bhalf_t>(ref_f);

    op(y, x0, x1);

    EXPECT_EQ(y, ref);
}

// =============================================================================
// Bilinear Tests
// =============================================================================

TEST(BinaryElementWiseOp_Host, Bilinear_float_float_float)
{
    Bilinear op(2.0f, 3.0f);
    float y;
    op(y, 1.0f, 2.0f);
    // y = alpha * x0 + beta * x1 = 2.0 * 1.0 + 3.0 * 2.0 = 8.0
    EXPECT_FLOAT_EQ(y, 8.0f);
}

TEST(BinaryElementWiseOp_Host, Bilinear_double_double_double)
{
    Bilinear op(2.0f, 3.0f);
    double y;
    op(y, 1.0, 2.0);
    // y = alpha * x0 + beta * x1 = 2.0 * 1.0 + 3.0 * 2.0 = 8.0
    EXPECT_DOUBLE_EQ(y, 8.0);
}

TEST(BinaryElementWiseOp_Host, Bilinear_half_half_half)
{
    // This tests the fix: should compute in float precision
    Bilinear op(2.0f, 3.0f);
    half_t y;
    half_t x0 = type_convert<half_t>(1.0f);
    half_t x1 = type_convert<half_t>(2.0f);

    // Reference: compute in float
    float ref_f = 2.0f * type_convert<float>(x0) + 3.0f * type_convert<float>(x1);
    half_t ref  = type_convert<half_t>(ref_f);

    op(y, x0, x1);

    EXPECT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Bilinear_half_half_half_precision)
{
    // Test that precision is maintained with small alpha/beta
    Bilinear op(0.001f, 0.002f);
    half_t y;
    half_t x0 = type_convert<half_t>(100.0f);
    half_t x1 = type_convert<half_t>(200.0f);

    // Reference in float
    float ref_f = 0.001f * type_convert<float>(x0) + 0.002f * type_convert<float>(x1);
    half_t ref  = type_convert<half_t>(ref_f);

    op(y, x0, x1);

    EXPECT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Bilinear_half_float_half)
{
    Bilinear op(2.0f, 3.0f);
    half_t y;
    float x0  = 1.5f;
    half_t x1 = type_convert<half_t>(2.5f);

    float ref_f = 2.0f * x0 + 3.0f * type_convert<float>(x1);
    half_t ref  = type_convert<half_t>(ref_f);

    op(y, x0, x1);

    EXPECT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Bilinear_bhalf_bhalf_bhalf)
{
    Bilinear op(2.0f, 3.0f);
    bhalf_t y;
    bhalf_t x0 = type_convert<bhalf_t>(1.0f);
    bhalf_t x1 = type_convert<bhalf_t>(2.0f);

    float ref_f = 2.0f * type_convert<float>(x0) + 3.0f * type_convert<float>(x1);
    bhalf_t ref = type_convert<bhalf_t>(ref_f);

    op(y, x0, x1);

    EXPECT_EQ(y, ref);
}

// =============================================================================
// Subtract Tests
// =============================================================================

TEST(BinaryElementWiseOp_Host, Subtract_float_float_float)
{
    Subtract op;
    float y;
    op(y, 5.0f, 3.0f);
    EXPECT_FLOAT_EQ(y, 2.0f);
}

TEST(BinaryElementWiseOp_Host, Subtract_double_double_double)
{
    Subtract op;
    double y;
    op(y, 5.0, 3.0);
    EXPECT_DOUBLE_EQ(y, 2.0);
}

TEST(BinaryElementWiseOp_Host, Subtract_half_half_half)
{
    Subtract op;
    half_t y;
    half_t x0 = type_convert<half_t>(5.0f);
    half_t x1 = type_convert<half_t>(3.0f);

    op(y, x0, x1);

    EXPECT_TRUE(is_close(y, type_convert<half_t>(2.0f)));
}

TEST(BinaryElementWiseOp_Host, Subtract_bhalf_bhalf_bhalf)
{
    Subtract op;
    bhalf_t y;
    bhalf_t x0 = type_convert<bhalf_t>(5.0f);
    bhalf_t x1 = type_convert<bhalf_t>(3.0f);

    // bhalf_t version promotes to float
    float ref_f = type_convert<float>(x0) - type_convert<float>(x1);
    bhalf_t ref = type_convert<bhalf_t>(ref_f);

    op(y, x0, x1);

    EXPECT_EQ(y, ref);
}

// =============================================================================
// ScaleAdd Tests
// =============================================================================

TEST(BinaryElementWiseOp_Host, ScaleAdd_float_float_half)
{
    ScaleAdd op(2.0f);
    float y;
    float x0  = 3.0f;
    half_t x1 = type_convert<half_t>(1.0f);

    // y = scale * x0 + x1 = 2.0 * 3.0 + 1.0 = 7.0
    op(y, x0, x1);

    EXPECT_FLOAT_EQ(y, 7.0f);
}

TEST(BinaryElementWiseOp_Host, ScaleAdd_float_float_bhalf)
{
    ScaleAdd op(2.0f);
    float y;
    float x0   = 3.0f;
    bhalf_t x1 = type_convert<bhalf_t>(1.0f);

    op(y, x0, x1);

    EXPECT_FLOAT_EQ(y, 7.0f);
}

// =============================================================================
// Max/Min Tests
// =============================================================================

TEST(BinaryElementWiseOp_Host, Max_float)
{
    Max op;
    float y;
    op(y, 3.0f, 5.0f);
    EXPECT_FLOAT_EQ(y, 5.0f);

    op(y, 7.0f, 2.0f);
    EXPECT_FLOAT_EQ(y, 7.0f);
}

TEST(BinaryElementWiseOp_Host, Max_half)
{
    Max op;
    half_t y;
    half_t x0 = type_convert<half_t>(3.0f);
    half_t x1 = type_convert<half_t>(5.0f);

    op(y, x0, x1);
    EXPECT_TRUE(is_close(y, type_convert<half_t>(5.0f)));
}

TEST(BinaryElementWiseOp_Host, Min_float)
{
    Min op;
    float y;
    op(y, 3.0f, 5.0f);
    EXPECT_FLOAT_EQ(y, 3.0f);

    op(y, 7.0f, 2.0f);
    EXPECT_FLOAT_EQ(y, 2.0f);
}

TEST(BinaryElementWiseOp_Host, Min_half)
{
    Min op;
    half_t y;
    half_t x0 = type_convert<half_t>(3.0f);
    half_t x1 = type_convert<half_t>(5.0f);

    op(y, x0, x1);
    EXPECT_TRUE(is_close(y, type_convert<half_t>(3.0f)));
}

// =============================================================================
// AddClamp Tests
// =============================================================================

TEST(BinaryElementWiseOp_Host, AddClamp_float)
{
    AddClamp op(0.0f, 10.0f);
    float y;

    // Normal case
    op(y, 3.0f, 4.0f);
    EXPECT_FLOAT_EQ(y, 7.0f);

    // Clamp to floor
    op(y, -5.0f, 2.0f);
    EXPECT_FLOAT_EQ(y, 0.0f);

    // Clamp to ceil
    op(y, 8.0f, 5.0f);
    EXPECT_FLOAT_EQ(y, 10.0f);
}

TEST(BinaryElementWiseOp_Host, AddClamp_double)
{
    AddClamp op(0.0f, 10.0f);
    double y;

    // Normal case
    op(y, 3.0, 4.0);
    EXPECT_DOUBLE_EQ(y, 7.0);

    // Clamp to floor
    op(y, -5.0, 2.0);
    EXPECT_DOUBLE_EQ(y, 0.0);

    // Clamp to ceil
    op(y, 8.0, 5.0);
    EXPECT_DOUBLE_EQ(y, 10.0);
}

TEST(BinaryElementWiseOp_Host, AddClamp_half_float_half)
{
    AddClamp op(0.0f, 10.0f);
    half_t y;
    float x0  = 3.0f;
    half_t x1 = type_convert<half_t>(4.0f);

    op(y, x0, x1);

    EXPECT_TRUE(is_close(y, type_convert<half_t>(7.0f)));
}

TEST(BinaryElementWiseOp_Host, AddClamp_half_half_half)
{
    AddClamp op(0.0f, 10.0f);
    half_t y;
    half_t x0 = type_convert<half_t>(3.0f);
    half_t x1 = type_convert<half_t>(4.0f);

    op(y, x0, x1);

    EXPECT_TRUE(is_close(y, type_convert<half_t>(7.0f)));
}

TEST(BinaryElementWiseOp_Host, AddClamp_bhalf_float_bhalf)
{
    AddClamp op(0.0f, 10.0f);
    bhalf_t y;
    float x0   = 3.0f;
    bhalf_t x1 = type_convert<bhalf_t>(4.0f);

    float ref_f = 3.0f + type_convert<float>(x1);
    bhalf_t ref = type_convert<bhalf_t>(ref_f);

    op(y, x0, x1);

    EXPECT_EQ(y, ref);
}

// =============================================================================
// AddRelu Tests
// =============================================================================

TEST(BinaryElementWiseOp_Host, AddRelu_float)
{
    AddRelu op;
    float y;

    // Positive result
    op(y, 3.0f, 4.0f);
    EXPECT_FLOAT_EQ(y, 7.0f);

    // Negative sum -> relu to 0
    op(y, -5.0f, 2.0f);
    EXPECT_FLOAT_EQ(y, 0.0f);
}

TEST(BinaryElementWiseOp_Host, AddRelu_double)
{
    AddRelu op;
    double y;

    // Positive result
    op(y, 3.0, 4.0);
    EXPECT_DOUBLE_EQ(y, 7.0);

    // Negative sum -> relu to 0
    op(y, -5.0, 2.0);
    EXPECT_DOUBLE_EQ(y, 0.0);
}

TEST(BinaryElementWiseOp_Host, AddRelu_half_float_half)
{
    AddRelu op;
    half_t y;
    float x0  = 3.0f;
    half_t x1 = type_convert<half_t>(4.0f);

    // Reference: compute in float
    float sum   = x0 + type_convert<float>(x1);
    float ref_f = sum > 0.0f ? sum : 0.0f;
    half_t ref  = type_convert<half_t>(ref_f);

    op(y, x0, x1);

    EXPECT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, AddRelu_half_half_half)
{
    AddRelu op;
    half_t y;
    half_t x0 = type_convert<half_t>(3.0f);
    half_t x1 = type_convert<half_t>(4.0f);

    op(y, x0, x1);

    EXPECT_TRUE(is_close(y, type_convert<half_t>(7.0f)));
}

TEST(BinaryElementWiseOp_Host, AddRelu_half_half_half_negative)
{
    AddRelu op;
    half_t y;
    half_t x0 = type_convert<half_t>(-5.0f);
    half_t x1 = type_convert<half_t>(2.0f);

    op(y, x0, x1);

    EXPECT_TRUE(is_close(y, type_convert<half_t>(0.0f)));
}

TEST(BinaryElementWiseOp_Host, AddRelu_bhalf_float_bhalf)
{
    AddRelu op;
    bhalf_t y;
    float x0   = 3.0f;
    bhalf_t x1 = type_convert<bhalf_t>(4.0f);

    float sum   = x0 + type_convert<float>(x1);
    float ref_f = sum > 0.0f ? sum : 0.0f;
    bhalf_t ref = type_convert<bhalf_t>(ref_f);

    op(y, x0, x1);

    EXPECT_EQ(y, ref);
}

// =============================================================================
// AddHardswish Tests
// =============================================================================

TEST(BinaryElementWiseOp_Host, AddHardswish_float)
{
    AddHardswish op;
    float y;

    // x = 0 -> hardswish(0) = 0
    op(y, 0.0f, 0.0f);
    EXPECT_NEAR(y, 0.0f, 1e-5f);

    // x = 6 -> hardswish(6) = 6
    op(y, 3.0f, 3.0f);
    EXPECT_NEAR(y, 6.0f, 1e-5f);

    // x = -3 -> hardswish(-3) = 0 (since b = x + 3 = 0)
    op(y, -1.5f, -1.5f);
    EXPECT_NEAR(y, 0.0f, 1e-5f);
}

TEST(BinaryElementWiseOp_Host, AddHardswish_double)
{
    AddHardswish op;
    double y;

    // x = 0 -> hardswish(0) = 0
    op(y, 0.0, 0.0);
    EXPECT_NEAR(y, 0.0, 1e-10);

    // x = 6 -> hardswish(6) = 6
    op(y, 3.0, 3.0);
    EXPECT_NEAR(y, 6.0, 1e-10);

    // x = -3 -> hardswish(-3) = 0 (since b = x + 3 = 0)
    op(y, -1.5, -1.5);
    EXPECT_NEAR(y, 0.0, 1e-10);
}

TEST(BinaryElementWiseOp_Host, AddHardswish_half)
{
    AddHardswish op;
    half_t y;
    half_t x0 = type_convert<half_t>(3.0f);
    half_t x1 = type_convert<half_t>(3.0f);

    op(y, x0, x1);

    // x = 6, hardswish(6) = 6
    EXPECT_TRUE(is_close(y, type_convert<half_t>(6.0f), 1e-2f, 1e-2f));
}

// =============================================================================
// AddFastGelu Tests
// Note: These test current behavior. half_t version has precision TODO.
// =============================================================================

TEST(BinaryElementWiseOp_Host, AddFastGelu_float)
{
    AddFastGelu op;
    float y;

    // x = 0 -> gelu(0) = 0
    op(y, 0.0f, 0.0f);
    EXPECT_NEAR(y, 0.0f, 1e-5f);

    // x = 2 -> gelu(2) ~= 1.9545
    op(y, 1.0f, 1.0f);
    EXPECT_NEAR(y, 1.9545f, 0.01f);
}

TEST(BinaryElementWiseOp_Host, AddFastGelu_half_float_half)
{
    AddFastGelu op;
    half_t y;
    float x0  = 1.0f;
    half_t x1 = type_convert<half_t>(1.0f);

    // This version computes in float - should be accurate
    op(y, x0, x1);

    EXPECT_TRUE(is_close(y, type_convert<half_t>(1.9545f), 0.01f, 0.01f));
}

TEST(BinaryElementWiseOp_Host, AddFastGelu_half_half_half)
{
    // TODO: This version computes in half_t - may have precision issues
    AddFastGelu op;
    half_t y;
    half_t x0 = type_convert<half_t>(1.0f);
    half_t x1 = type_convert<half_t>(1.0f);

    op(y, x0, x1);

    // Use looser tolerance due to half_t precision
    EXPECT_TRUE(is_close(y, type_convert<half_t>(1.9545f), 0.05f, 0.05f));
}

TEST(BinaryElementWiseOp_Host, AddFastGelu_bhalf_bhalf_bhalf)
{
    AddFastGelu op;
    bhalf_t y;
    bhalf_t x0 = type_convert<bhalf_t>(1.0f);
    bhalf_t x1 = type_convert<bhalf_t>(1.0f);

    // bhalf_t version computes in float
    op(y, x0, x1);

    EXPECT_TRUE(is_close(y, type_convert<bhalf_t>(1.9545f), 0.02f, 0.02f));
}

// =============================================================================
// MultiplyFastGelu Tests
// =============================================================================

TEST(BinaryElementWiseOp_Host, MultiplyFastGelu_float)
{
    MultiplyFastGelu op;
    float y;

    // x = 1 * 2 = 2 -> gelu(2) ~= 1.9545
    op(y, 1.0f, 2.0f);
    EXPECT_NEAR(y, 1.9545f, 0.01f);
}

TEST(BinaryElementWiseOp_Host, MultiplyFastGelu_half_float_half)
{
    MultiplyFastGelu op;
    half_t y;
    float x0  = 1.0f;
    half_t x1 = type_convert<half_t>(2.0f);

    op(y, x0, x1);

    EXPECT_TRUE(is_close(y, type_convert<half_t>(1.9545f), 0.01f, 0.01f));
}

TEST(BinaryElementWiseOp_Host, MultiplyFastGelu_half_half_half)
{
    // TODO: This version computes in half_t - may have precision issues
    MultiplyFastGelu op;
    half_t y;
    half_t x0 = type_convert<half_t>(1.0f);
    half_t x1 = type_convert<half_t>(2.0f);

    op(y, x0, x1);

    EXPECT_TRUE(is_close(y, type_convert<half_t>(1.9545f), 0.05f, 0.05f));
}

// =============================================================================
// AddSilu Tests
// =============================================================================

TEST(BinaryElementWiseOp_Host, AddSilu_float)
{
    AddSilu op;
    float y;

    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    // silu(0) = 0
    op(y, 0.0f, 0.0f);
    EXPECT_NEAR(y, 0.0f, 1e-5f);

    // silu(2) ~= 1.7616
    op(y, 1.0f, 1.0f);
    EXPECT_NEAR(y, 1.7616f, 0.01f);
}

TEST(BinaryElementWiseOp_Host, AddSilu_half_float_half)
{
    AddSilu op;
    half_t y;
    float x0  = 1.0f;
    half_t x1 = type_convert<half_t>(1.0f);

    op(y, x0, x1);

    EXPECT_TRUE(is_close(y, type_convert<half_t>(1.7616f), 0.01f, 0.01f));
}

TEST(BinaryElementWiseOp_Host, AddSilu_half_half_half)
{
    // TODO: This version computes in half_t - may have precision issues
    AddSilu op;
    half_t y;
    half_t x0 = type_convert<half_t>(1.0f);
    half_t x1 = type_convert<half_t>(1.0f);

    op(y, x0, x1);

    EXPECT_TRUE(is_close(y, type_convert<half_t>(1.7616f), 0.05f, 0.05f));
}

TEST(BinaryElementWiseOp_Host, AddSilu_bhalf_float_bhalf)
{
    AddSilu op;
    bhalf_t y;
    float x0   = 1.0f;
    bhalf_t x1 = type_convert<bhalf_t>(1.0f);

    op(y, x0, x1);

    EXPECT_TRUE(is_close(y, type_convert<bhalf_t>(1.7616f), 0.02f, 0.02f));
}

// =============================================================================
// Precision Regression Tests
// These tests specifically verify the fixes we made don't regress
// =============================================================================

TEST(BinaryElementWiseOp_Host, Add_float_float_half_precision_regression)
{
    // Before fix: y = x0 + type_convert<half_t>(x1) would lose x0 precision
    // After fix:  y = x0 + type_convert<float>(x1) preserves x0 precision
    Add op;
    float y;

    // Use a value that requires float precision
    float x0  = 1.0000001f;
    half_t x1 = type_convert<half_t>(0.0f);

    op(y, x0, x1);

    // Should preserve the full float precision of x0
    EXPECT_FLOAT_EQ(y, x0);
}

TEST(BinaryElementWiseOp_Host, Multiply_half_float_half_precision_regression)
{
    // Before fix: y = type_convert<half_t>(x0) * x1 lost precision in x0
    // After fix:  y = type_convert<half_t>(x0 * type_convert<float>(x1))
    Multiply op;
    half_t y;

    float x0  = 1000.0f;
    half_t x1 = type_convert<half_t>(0.001f);

    // Reference computation in float
    float ref_f = x0 * type_convert<float>(x1);
    half_t ref  = type_convert<half_t>(ref_f);

    op(y, x0, x1);

    EXPECT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Bilinear_half_half_half_precision_regression)
{
    // Before fix: computed entirely in half_t including alpha/beta
    // After fix:  promotes to float for computation
    Bilinear op(0.1f, 0.2f);
    half_t y;

    half_t x0 = type_convert<half_t>(10.0f);
    half_t x1 = type_convert<half_t>(20.0f);

    // Reference in float
    float ref_f = 0.1f * type_convert<float>(x0) + 0.2f * type_convert<float>(x1);
    half_t ref  = type_convert<half_t>(ref_f);

    op(y, x0, x1);

    EXPECT_EQ(y, ref);
}

// =============================================================================
// Device-Side Tests
// These tests verify the operations work correctly when executed on GPU
// =============================================================================

// Kernel templates for testing binary operations on device
template <typename Op, typename Y, typename X0, typename X1>
__global__ void test_binary_op_kernel(Y* y, X0 x0, X1 x1)
{
    Op op;
    op(*y, x0, x1);
}

template <typename Y, typename X0, typename X1>
__global__ void test_bilinear_kernel(Y* y, X0 x0, X1 x1, Bilinear op)
{
    op(*y, x0, x1);
}

// Helper to run a binary op kernel and return the result
template <typename Op, typename Y, typename X0, typename X1>
Y run_binary_op_on_device(X0 x0, X1 x1)
{
    Y* d_y;
    Y h_y;
    hip_check_error(hipMalloc(&d_y, sizeof(Y)));
    test_binary_op_kernel<Op, Y, X0, X1><<<1, 1>>>(d_y, x0, x1);
    hip_check_error(hipDeviceSynchronize());
    hip_check_error(hipMemcpy(&h_y, d_y, sizeof(Y), hipMemcpyDeviceToHost));
    hip_check_error(hipFree(d_y));
    return h_y;
}

// Helper to run bilinear kernel and return the result
template <typename Y, typename X0, typename X1>
Y run_bilinear_on_device(X0 x0, X1 x1, Bilinear op)
{
    Y* d_y;
    Y h_y;
    hip_check_error(hipMalloc(&d_y, sizeof(Y)));
    test_bilinear_kernel<Y, X0, X1><<<1, 1>>>(d_y, x0, x1, op);
    hip_check_error(hipDeviceSynchronize());
    hip_check_error(hipMemcpy(&h_y, d_y, sizeof(Y), hipMemcpyDeviceToHost));
    hip_check_error(hipFree(d_y));
    return h_y;
}

// -----------------------------------------------------------------------------
// Device Add Tests
// -----------------------------------------------------------------------------

TEST(BinaryElementWiseOp_Device, Add_float_float_float)
{
    float y = run_binary_op_on_device<Add, float>(1.5f, 2.5f);
    EXPECT_FLOAT_EQ(y, 4.0f);
}

TEST(BinaryElementWiseOp_Device, Add_float_float_half)
{
    // Test precision fix on device: should convert half_t to float, not the other way
    float x0       = 1.0f + 1e-6f; // Value with precision beyond half_t
    half_t x1      = type_convert<half_t>(2.0f);
    float y        = run_binary_op_on_device<Add, float>(x0, x1);
    // Reference: x1 promoted to float (not x0 demoted to half_t)
    float expected = x0 + type_convert<float>(x1);
    EXPECT_FLOAT_EQ(y, expected);
}

TEST(BinaryElementWiseOp_Device, Add_half_float_half)
{
    // Test precision fix on device: should compute in float, then convert to half_t
    float x0    = 1000.0f;
    half_t x1   = type_convert<half_t>(0.001f);
    half_t y    = run_binary_op_on_device<Add, half_t>(x0, x1);
    float ref_f = x0 + type_convert<float>(x1);
    half_t ref  = type_convert<half_t>(ref_f);
    EXPECT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Device, Add_half_half_half)
{
    half_t x0 = type_convert<half_t>(1.5f);
    half_t x1 = type_convert<half_t>(2.5f);
    half_t y  = run_binary_op_on_device<Add, half_t>(x0, x1);
    EXPECT_TRUE(is_close(y, type_convert<half_t>(4.0f)));
}

// -----------------------------------------------------------------------------
// Device Multiply Tests
// -----------------------------------------------------------------------------

TEST(BinaryElementWiseOp_Device, Multiply_float_float_float)
{
    float y = run_binary_op_on_device<Multiply, float>(2.0f, 3.0f);
    EXPECT_FLOAT_EQ(y, 6.0f);
}

TEST(BinaryElementWiseOp_Device, Multiply_float_float_half)
{
    // Test precision fix on device: should convert half_t to float
    float x0       = 1.0f + 1e-6f;
    half_t x1      = type_convert<half_t>(2.0f);
    float y        = run_binary_op_on_device<Multiply, float>(x0, x1);
    // Reference: x1 promoted to float (not x0 demoted to half_t)
    float expected = x0 * type_convert<float>(x1);
    EXPECT_FLOAT_EQ(y, expected);
}

TEST(BinaryElementWiseOp_Device, Multiply_half_float_half)
{
    // Test precision fix on device: should compute in float, then convert
    float x0    = 100.0f;
    half_t x1   = type_convert<half_t>(0.01f);
    half_t y    = run_binary_op_on_device<Multiply, half_t>(x0, x1);
    float ref_f = x0 * type_convert<float>(x1);
    half_t ref  = type_convert<half_t>(ref_f);
    EXPECT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Device, Multiply_half_half_half)
{
    half_t x0 = type_convert<half_t>(2.0f);
    half_t x1 = type_convert<half_t>(3.0f);
    half_t y  = run_binary_op_on_device<Multiply, half_t>(x0, x1);
    EXPECT_TRUE(is_close(y, type_convert<half_t>(6.0f)));
}

// -----------------------------------------------------------------------------
// Device Bilinear Tests
// -----------------------------------------------------------------------------

TEST(BinaryElementWiseOp_Device, Bilinear_float_float_float)
{
    // y = alpha * x0 + beta * x1 = 2.0 * 1.0 + 3.0 * 2.0 = 8.0
    float y = run_bilinear_on_device<float>(1.0f, 2.0f, Bilinear(2.0f, 3.0f));
    EXPECT_FLOAT_EQ(y, 8.0f);
}

TEST(BinaryElementWiseOp_Device, Bilinear_half_half_half)
{
    // Test precision fix on device: should compute in float precision
    half_t x0   = type_convert<half_t>(1.0f);
    half_t x1   = type_convert<half_t>(2.0f);
    half_t y    = run_bilinear_on_device<half_t>(x0, x1, Bilinear(2.0f, 3.0f));
    float ref_f = 2.0f * type_convert<float>(x0) + 3.0f * type_convert<float>(x1);
    half_t ref  = type_convert<half_t>(ref_f);
    EXPECT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Device, Bilinear_half_half_half_precision)
{
    // Test precision with small alpha/beta on device
    half_t x0   = type_convert<half_t>(100.0f);
    half_t x1   = type_convert<half_t>(200.0f);
    half_t y    = run_bilinear_on_device<half_t>(x0, x1, Bilinear(0.001f, 0.002f));
    float ref_f = 0.001f * type_convert<float>(x0) + 0.002f * type_convert<float>(x1);
    half_t ref  = type_convert<half_t>(ref_f);
    EXPECT_EQ(y, ref);
}

// -----------------------------------------------------------------------------
// Device Precision Regression Tests
// These tests specifically verify the fixes work correctly on GPU
// -----------------------------------------------------------------------------

TEST(BinaryElementWiseOp_Device, Add_float_float_half_precision_regression)
{
    // Before fix: y = x0 + type_convert<half_t>(x1) would lose x0 precision
    // After fix:  y = x0 + type_convert<float>(x1) preserves x0 precision
    float x0  = 1.0000001f;
    half_t x1 = type_convert<half_t>(0.0f);
    float y   = run_binary_op_on_device<Add, float>(x0, x1);
    EXPECT_FLOAT_EQ(y, x0);
}

TEST(BinaryElementWiseOp_Device, Multiply_half_float_half_precision_regression)
{
    // Before fix: y = type_convert<half_t>(x0) * x1 lost precision in x0
    // After fix:  y = type_convert<half_t>(x0 * type_convert<float>(x1))
    float x0    = 1000.0f;
    half_t x1   = type_convert<half_t>(0.001f);
    half_t y    = run_binary_op_on_device<Multiply, half_t>(x0, x1);
    float ref_f = x0 * type_convert<float>(x1);
    half_t ref  = type_convert<half_t>(ref_f);
    EXPECT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Device, Bilinear_half_half_half_precision_regression)
{
    // Before fix: computed entirely in half_t including alpha/beta
    // After fix:  promotes to float for computation
    half_t x0   = type_convert<half_t>(10.0f);
    half_t x1   = type_convert<half_t>(20.0f);
    half_t y    = run_bilinear_on_device<half_t>(x0, x1, Bilinear(0.1f, 0.2f));
    float ref_f = 0.1f * type_convert<float>(x0) + 0.2f * type_convert<float>(x1);
    half_t ref  = type_convert<half_t>(ref_f);
    EXPECT_EQ(y, ref);
}
