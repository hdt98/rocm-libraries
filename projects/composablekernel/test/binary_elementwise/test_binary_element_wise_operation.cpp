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

// Helper to check if two floats are close enough (used by FastGelu/Silu tests
// where the operator is an approximation and bit-exact comparison isn't feasible).
template <typename T>
bool is_close(T a, T b, float rtol = 1e-5f, float atol = 1e-5f)
{
    float fa = type_convert<float>(a);
    float fb = type_convert<float>(b);
    return std::abs(fa - fb) <= atol + rtol * std::abs(fb);
}

// Run a default-constructed binary op and verify against a float-precision reference.
// `ref_fn(float, float) -> float` computes the expected value in float; the result is
// then type_convert'd to Y for bit-exact comparison.
template <typename Op, typename Y, typename X0, typename X1, typename RefFn>
void check_binary_op_default(X0 x0, X1 x1, RefFn ref_fn)
{
    Op op;
    Y y;
    op(y, x0, x1);
    Y ref = type_convert<Y>(ref_fn(type_convert<float>(x0), type_convert<float>(x1)));
    EXPECT_EQ(y, ref);
}

// Variant for ops with constructor args (Bilinear, AddClamp, ScaleAdd).
template <typename Y, typename Op, typename X0, typename X1, typename RefFn>
void check_binary_op(Op op, X0 x0, X1 x1, RefFn ref_fn)
{
    Y y;
    op(y, x0, x1);
    Y ref = type_convert<Y>(ref_fn(type_convert<float>(x0), type_convert<float>(x1)));
    EXPECT_EQ(y, ref);
}

// Verify a precomputed device result against a float-precision reference.
// (Used by device tests where op invocation is via run_binary_op_on_device etc.)
template <typename Y, typename X0, typename X1, typename RefFn>
void check_device_result(Y y, X0 x0, X1 x1, RefFn ref_fn)
{
    Y ref = type_convert<Y>(ref_fn(type_convert<float>(x0), type_convert<float>(x1)));
    EXPECT_EQ(y, ref);
}

// =============================================================================
// Add Tests
// =============================================================================

TEST(BinaryElementWiseOp_Host, Add_float_float_float)
{
    Add op;
    float y;
    float x0  = 1.5f;
    float x1  = 2.5f;
    float ref = x0 + x1;
    op(y, x0, x1);
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Add_double_double_double)
{
    Add op;
    double y;
    double x0  = 1.5;
    double x1  = 2.5;
    double ref = x0 + x1;
    op(y, x0, x1);
    EXPECT_DOUBLE_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Add_float_float_half)
{
    // Tests the fix: x1 promoted to float, not x0 demoted to half_t.
    // x0 carries precision beyond half_t to detect a wrong demotion.
    check_binary_op_default<Add, float>(
        1.0f + 1e-6f, type_convert<half_t>(2.0f), [](float a, float b) { return a + b; });
}

TEST(BinaryElementWiseOp_Host, Add_half_float_half)
{
    // Tests the fix: x0 promoted to float, computed, then demoted to half_t.
    check_binary_op_default<Add, half_t>(
        1000.0f, type_convert<half_t>(0.001f), [](float a, float b) { return a + b; });
}

TEST(BinaryElementWiseOp_Host, Add_half_half_half)
{
    check_binary_op_default<Add, half_t>(type_convert<half_t>(1.5f),
                                         type_convert<half_t>(2.5f),
                                         [](float a, float b) { return a + b; });
}

TEST(BinaryElementWiseOp_Host, Add_bhalf_bhalf_bhalf)
{
    check_binary_op_default<Add, bhalf_t>(type_convert<bhalf_t>(1.5f),
                                          type_convert<bhalf_t>(2.5f),
                                          [](float a, float b) { return a + b; });
}

TEST(BinaryElementWiseOp_Host, Add_bhalf_float_bhalf)
{
    check_binary_op_default<Add, bhalf_t>(
        100.0f, type_convert<bhalf_t>(0.5f), [](float a, float b) { return a + b; });
}

// =============================================================================
// Multiply Tests
// =============================================================================

TEST(BinaryElementWiseOp_Host, Multiply_float_float_float)
{
    Multiply op;
    float y;
    float x0  = 2.0f;
    float x1  = 3.0f;
    float ref = x0 * x1;
    op(y, x0, x1);
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Multiply_double_double_double)
{
    Multiply op;
    double y;
    double x0  = 2.0;
    double x1  = 3.0;
    double ref = x0 * x1;
    op(y, x0, x1);
    EXPECT_DOUBLE_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Multiply_float_float_half)
{
    // Tests the fix: x1 promoted to float, not x0 demoted to half_t.
    check_binary_op_default<Multiply, float>(
        1.0f + 1e-6f, type_convert<half_t>(2.0f), [](float a, float b) { return a * b; });
}

TEST(BinaryElementWiseOp_Host, Multiply_half_float_half)
{
    // Tests the fix: x0 promoted to float, computed, then demoted to half_t.
    check_binary_op_default<Multiply, half_t>(
        100.0f, type_convert<half_t>(0.01f), [](float a, float b) { return a * b; });
}

TEST(BinaryElementWiseOp_Host, Multiply_half_half_half)
{
    check_binary_op_default<Multiply, half_t>(type_convert<half_t>(2.0f),
                                              type_convert<half_t>(3.0f),
                                              [](float a, float b) { return a * b; });
}

TEST(BinaryElementWiseOp_Host, Multiply_bhalf_bhalf_bhalf)
{
    check_binary_op_default<Multiply, bhalf_t>(type_convert<bhalf_t>(2.0f),
                                               type_convert<bhalf_t>(3.0f),
                                               [](float a, float b) { return a * b; });
}

TEST(BinaryElementWiseOp_Host, Multiply_bhalf_float_bhalf)
{
    check_binary_op_default<Multiply, bhalf_t>(
        10.0f, type_convert<bhalf_t>(0.5f), [](float a, float b) { return a * b; });
}

// =============================================================================
// Bilinear Tests
// =============================================================================

TEST(BinaryElementWiseOp_Host, Bilinear_float_float_float)
{
    const float alpha = 2.0f;
    const float beta  = 3.0f;
    Bilinear op(alpha, beta);
    float y;
    float x0  = 1.0f;
    float x1  = 2.0f;
    float ref = alpha * x0 + beta * x1;
    op(y, x0, x1);
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Bilinear_double_double_double)
{
    const float alpha = 2.0f;
    const float beta  = 3.0f;
    Bilinear op(alpha, beta);
    double y;
    double x0  = 1.0;
    double x1  = 2.0;
    double ref = alpha * x0 + beta * x1;
    op(y, x0, x1);
    EXPECT_DOUBLE_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Bilinear_half_half_half)
{
    // Tests the fix: compute in float precision.
    const float alpha = 2.0f;
    const float beta  = 3.0f;
    check_binary_op<half_t>(Bilinear(alpha, beta),
                            type_convert<half_t>(1.0f),
                            type_convert<half_t>(2.0f),
                            [=](float a, float b) { return alpha * a + beta * b; });
}

TEST(BinaryElementWiseOp_Host, Bilinear_half_half_half_precision_regression_small_coeffs)
{
    // Before fix: alpha/beta demoted to half_t, losing precision (e.g. 0.001f -> 0.0009994f).
    // After fix:  alpha/beta stay float, computation done in float, demoted only at the end.
    const float alpha = 0.001f;
    const float beta  = 0.002f;
    check_binary_op<half_t>(Bilinear(alpha, beta),
                            type_convert<half_t>(100.0f),
                            type_convert<half_t>(200.0f),
                            [=](float a, float b) { return alpha * a + beta * b; });
}

TEST(BinaryElementWiseOp_Host, Bilinear_half_float_half)
{
    const float alpha = 2.0f;
    const float beta  = 3.0f;
    check_binary_op<half_t>(Bilinear(alpha, beta),
                            1.5f,
                            type_convert<half_t>(2.5f),
                            [=](float a, float b) { return alpha * a + beta * b; });
}

TEST(BinaryElementWiseOp_Host, Bilinear_bhalf_bhalf_bhalf)
{
    const float alpha = 2.0f;
    const float beta  = 3.0f;
    check_binary_op<bhalf_t>(Bilinear(alpha, beta),
                             type_convert<bhalf_t>(1.0f),
                             type_convert<bhalf_t>(2.0f),
                             [=](float a, float b) { return alpha * a + beta * b; });
}

// =============================================================================
// Subtract Tests
// =============================================================================

TEST(BinaryElementWiseOp_Host, Subtract_float_float_float)
{
    Subtract op;
    float y;
    float x0  = 5.0f;
    float x1  = 3.0f;
    float ref = x0 - x1;
    op(y, x0, x1);
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Subtract_double_double_double)
{
    Subtract op;
    double y;
    double x0  = 5.0;
    double x1  = 3.0;
    double ref = x0 - x1;
    op(y, x0, x1);
    EXPECT_DOUBLE_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Subtract_half_half_half)
{
    check_binary_op_default<Subtract, half_t>(type_convert<half_t>(5.0f),
                                              type_convert<half_t>(3.0f),
                                              [](float a, float b) { return a - b; });
}

TEST(BinaryElementWiseOp_Host, Subtract_bhalf_bhalf_bhalf)
{
    check_binary_op_default<Subtract, bhalf_t>(type_convert<bhalf_t>(5.0f),
                                               type_convert<bhalf_t>(3.0f),
                                               [](float a, float b) { return a - b; });
}

// =============================================================================
// ScaleAdd Tests
// =============================================================================

TEST(BinaryElementWiseOp_Host, ScaleAdd_float_float_half)
{
    const float scale = 2.0f;
    ScaleAdd op(scale);
    float y;
    float x0  = 3.0f;
    half_t x1 = type_convert<half_t>(1.0f);
    float ref = scale * x0 + type_convert<float>(x1);
    op(y, x0, x1);
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, ScaleAdd_float_float_bhalf)
{
    const float scale = 2.0f;
    ScaleAdd op(scale);
    float y;
    float x0   = 3.0f;
    bhalf_t x1 = type_convert<bhalf_t>(1.0f);
    float ref  = scale * x0 + type_convert<float>(x1);
    op(y, x0, x1);
    EXPECT_FLOAT_EQ(y, ref);
}

// =============================================================================
// Max/Min Tests
// =============================================================================

TEST(BinaryElementWiseOp_Host, Max_float_second_larger)
{
    Max op;
    float y;
    float x0  = 3.0f;
    float x1  = 5.0f;
    float ref = x0 > x1 ? x0 : x1;
    op(y, x0, x1);
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Max_float_first_larger)
{
    Max op;
    float y;
    float x0  = 7.0f;
    float x1  = 2.0f;
    float ref = x0 > x1 ? x0 : x1;
    op(y, x0, x1);
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Max_half)
{
    check_binary_op_default<Max, half_t>(type_convert<half_t>(3.0f),
                                         type_convert<half_t>(5.0f),
                                         [](float a, float b) { return a > b ? a : b; });
}

TEST(BinaryElementWiseOp_Host, Min_float_first_smaller)
{
    Min op;
    float y;
    float x0  = 3.0f;
    float x1  = 5.0f;
    float ref = x0 < x1 ? x0 : x1;
    op(y, x0, x1);
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Min_float_second_smaller)
{
    Min op;
    float y;
    float x0  = 7.0f;
    float x1  = 2.0f;
    float ref = x0 < x1 ? x0 : x1;
    op(y, x0, x1);
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, Min_half)
{
    check_binary_op_default<Min, half_t>(type_convert<half_t>(3.0f),
                                         type_convert<half_t>(5.0f),
                                         [](float a, float b) { return a < b ? a : b; });
}

// =============================================================================
// AddClamp Tests
// =============================================================================

// AddClamp test bounds (shared by float and double tests below).
static constexpr float clamp_lo = 0.0f;
static constexpr float clamp_hi = 10.0f;

TEST(BinaryElementWiseOp_Host, AddClamp_float_normal)
{
    AddClamp op(clamp_lo, clamp_hi);
    float y;
    float x0  = 3.0f;
    float x1  = 4.0f;
    float sum = x0 + x1;
    float ref = sum < clamp_lo ? clamp_lo : (sum > clamp_hi ? clamp_hi : sum);
    op(y, x0, x1);
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, AddClamp_float_floor)
{
    AddClamp op(clamp_lo, clamp_hi);
    float y;
    float x0  = -5.0f;
    float x1  = 2.0f;
    float sum = x0 + x1;
    float ref = sum < clamp_lo ? clamp_lo : (sum > clamp_hi ? clamp_hi : sum);
    op(y, x0, x1);
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, AddClamp_float_ceil)
{
    AddClamp op(clamp_lo, clamp_hi);
    float y;
    float x0  = 8.0f;
    float x1  = 5.0f;
    float sum = x0 + x1;
    float ref = sum < clamp_lo ? clamp_lo : (sum > clamp_hi ? clamp_hi : sum);
    op(y, x0, x1);
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, AddClamp_double_normal)
{
    AddClamp op(clamp_lo, clamp_hi);
    double y;
    double x0  = 3.0;
    double x1  = 4.0;
    double sum = x0 + x1;
    double ref = sum < double{clamp_lo} ? double{clamp_lo}
                                        : (sum > double{clamp_hi} ? double{clamp_hi} : sum);
    op(y, x0, x1);
    EXPECT_DOUBLE_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, AddClamp_double_floor)
{
    AddClamp op(clamp_lo, clamp_hi);
    double y;
    double x0  = -5.0;
    double x1  = 2.0;
    double sum = x0 + x1;
    double ref = sum < double{clamp_lo} ? double{clamp_lo}
                                        : (sum > double{clamp_hi} ? double{clamp_hi} : sum);
    op(y, x0, x1);
    EXPECT_DOUBLE_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, AddClamp_double_ceil)
{
    AddClamp op(clamp_lo, clamp_hi);
    double y;
    double x0  = 8.0;
    double x1  = 5.0;
    double sum = x0 + x1;
    double ref = sum < double{clamp_lo} ? double{clamp_lo}
                                        : (sum > double{clamp_hi} ? double{clamp_hi} : sum);
    op(y, x0, x1);
    EXPECT_DOUBLE_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, AddClamp_half_float_half)
{
    const float lo = 0.0f;
    const float hi = 10.0f;
    check_binary_op<half_t>(
        AddClamp(lo, hi), 3.0f, type_convert<half_t>(4.0f), [=](float a, float b) {
            float s = a + b;
            return s < lo ? lo : (s > hi ? hi : s);
        });
}

TEST(BinaryElementWiseOp_Host, AddClamp_half_half_half)
{
    const float lo = 0.0f;
    const float hi = 10.0f;
    check_binary_op<half_t>(AddClamp(lo, hi),
                            type_convert<half_t>(3.0f),
                            type_convert<half_t>(4.0f),
                            [=](float a, float b) {
                                float s = a + b;
                                return s < lo ? lo : (s > hi ? hi : s);
                            });
}

TEST(BinaryElementWiseOp_Host, AddClamp_bhalf_float_bhalf)
{
    const float lo = 0.0f;
    const float hi = 10.0f;
    check_binary_op<bhalf_t>(
        AddClamp(lo, hi), 3.0f, type_convert<bhalf_t>(4.0f), [=](float a, float b) {
            float s = a + b;
            return s < lo ? lo : (s > hi ? hi : s);
        });
}

// =============================================================================
// AddRelu Tests
// =============================================================================

TEST(BinaryElementWiseOp_Host, AddRelu_float_positive)
{
    AddRelu op;
    float y;
    float x0  = 3.0f;
    float x1  = 4.0f;
    float sum = x0 + x1;
    float ref = sum > 0.0f ? sum : 0.0f;
    op(y, x0, x1);
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, AddRelu_float_negative)
{
    AddRelu op;
    float y;
    float x0  = -5.0f;
    float x1  = 2.0f;
    float sum = x0 + x1;
    float ref = sum > 0.0f ? sum : 0.0f;
    op(y, x0, x1);
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, AddRelu_double_positive)
{
    AddRelu op;
    double y;
    double x0  = 3.0;
    double x1  = 4.0;
    double sum = x0 + x1;
    double ref = sum > 0.0 ? sum : 0.0;
    op(y, x0, x1);
    EXPECT_DOUBLE_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, AddRelu_double_negative)
{
    AddRelu op;
    double y;
    double x0  = -5.0;
    double x1  = 2.0;
    double sum = x0 + x1;
    double ref = sum > 0.0 ? sum : 0.0;
    op(y, x0, x1);
    EXPECT_DOUBLE_EQ(y, ref);
}

// Reference for AddRelu: compute sum in float, clamp negatives to 0.
static auto add_relu_ref = [](float a, float b) {
    float s = a + b;
    return s > 0.0f ? s : 0.0f;
};

TEST(BinaryElementWiseOp_Host, AddRelu_half_float_half)
{
    check_binary_op_default<AddRelu, half_t>(3.0f, type_convert<half_t>(4.0f), add_relu_ref);
}

TEST(BinaryElementWiseOp_Host, AddRelu_half_half_half)
{
    check_binary_op_default<AddRelu, half_t>(
        type_convert<half_t>(3.0f), type_convert<half_t>(4.0f), add_relu_ref);
}

TEST(BinaryElementWiseOp_Host, AddRelu_half_half_half_negative)
{
    check_binary_op_default<AddRelu, half_t>(
        type_convert<half_t>(-5.0f), type_convert<half_t>(2.0f), add_relu_ref);
}

TEST(BinaryElementWiseOp_Host, AddRelu_bhalf_float_bhalf)
{
    check_binary_op_default<AddRelu, bhalf_t>(3.0f, type_convert<bhalf_t>(4.0f), add_relu_ref);
}

// =============================================================================
// AddHardswish Tests
// =============================================================================

// hardswish in float matches the operator's formula (with precise 1/6).
static float hardswish_f(float a)
{
    float b = a + 3.0f;
    return (b > 0.0f) * (b > 6.0f ? 6.0f : b) * a * (1.0f / 6.0f);
}

static double hardswish_d(double a)
{
    double b = a + 3.0;
    return (b > 0.0) * (b > 6.0 ? 6.0 : b) * a * (1.0 / 6.0);
}

TEST(BinaryElementWiseOp_Host, AddHardswish_float_zero)
{
    AddHardswish op;
    float y;
    float x0  = 0.0f;
    float x1  = 0.0f;
    float ref = hardswish_f(x0 + x1);
    op(y, x0, x1);
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, AddHardswish_float_six)
{
    AddHardswish op;
    float y;
    float x0  = 3.0f;
    float x1  = 3.0f;
    float ref = hardswish_f(x0 + x1);
    op(y, x0, x1);
    EXPECT_FLOAT_EQ(y, ref);
}

// b = x + 3 = 0 boundary case.
TEST(BinaryElementWiseOp_Host, AddHardswish_float_neg_three)
{
    AddHardswish op;
    float y;
    float x0  = -1.5f;
    float x1  = -1.5f;
    float ref = hardswish_f(x0 + x1);
    op(y, x0, x1);
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, AddHardswish_double_zero)
{
    AddHardswish op;
    double y;
    double x0  = 0.0;
    double x1  = 0.0;
    double ref = hardswish_d(x0 + x1);
    op(y, x0, x1);
    EXPECT_DOUBLE_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, AddHardswish_double_six)
{
    AddHardswish op;
    double y;
    double x0  = 3.0;
    double x1  = 3.0;
    double ref = hardswish_d(x0 + x1);
    op(y, x0, x1);
    EXPECT_DOUBLE_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, AddHardswish_double_neg_three)
{
    AddHardswish op;
    double y;
    double x0  = -1.5;
    double x1  = -1.5;
    double ref = hardswish_d(x0 + x1);
    op(y, x0, x1);
    EXPECT_DOUBLE_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Host, AddHardswish_half)
{
    check_binary_op_default<AddHardswish, half_t>(
        type_convert<half_t>(3.0f), type_convert<half_t>(3.0f), [](float a, float b) {
            return hardswish_f(a + b);
        });
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
    check_binary_op_default<Multiply, half_t>(
        1000.0f, type_convert<half_t>(0.001f), [](float a, float b) { return a * b; });
}

TEST(BinaryElementWiseOp_Host, Bilinear_half_half_half_precision_regression)
{
    // Before fix: computed entirely in half_t including alpha/beta
    // After fix:  promotes to float for computation
    const float alpha = 0.1f;
    const float beta  = 0.2f;
    check_binary_op<half_t>(Bilinear(alpha, beta),
                            type_convert<half_t>(10.0f),
                            type_convert<half_t>(20.0f),
                            [=](float a, float b) { return alpha * a + beta * b; });
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
    float x0  = 1.5f;
    float x1  = 2.5f;
    float y   = run_binary_op_on_device<Add, float>(x0, x1);
    float ref = x0 + x1;
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Device, Add_float_float_half)
{
    // Tests the fix on device: x1 promoted to float, not x0 demoted to half_t.
    float x0  = 1.0f + 1e-6f; // Value with precision beyond half_t
    half_t x1 = type_convert<half_t>(2.0f);
    float y   = run_binary_op_on_device<Add, float>(x0, x1);
    check_device_result(y, x0, x1, [](float a, float b) { return a + b; });
}

TEST(BinaryElementWiseOp_Device, Add_half_float_half)
{
    // Test precision fix on device: should compute in float, then convert to half_t
    float x0  = 1000.0f;
    half_t x1 = type_convert<half_t>(0.001f);
    half_t y  = run_binary_op_on_device<Add, half_t>(x0, x1);
    check_device_result(y, x0, x1, [](float a, float b) { return a + b; });
}

TEST(BinaryElementWiseOp_Device, Add_half_half_half)
{
    half_t x0 = type_convert<half_t>(1.5f);
    half_t x1 = type_convert<half_t>(2.5f);
    half_t y  = run_binary_op_on_device<Add, half_t>(x0, x1);
    check_device_result(y, x0, x1, [](float a, float b) { return a + b; });
}

// -----------------------------------------------------------------------------
// Device Multiply Tests
// -----------------------------------------------------------------------------

TEST(BinaryElementWiseOp_Device, Multiply_float_float_float)
{
    float x0  = 2.0f;
    float x1  = 3.0f;
    float y   = run_binary_op_on_device<Multiply, float>(x0, x1);
    float ref = x0 * x1;
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Device, Multiply_float_float_half)
{
    // Tests the fix on device: x1 promoted to float, not x0 demoted to half_t.
    float x0  = 1.0f + 1e-6f;
    half_t x1 = type_convert<half_t>(2.0f);
    float y   = run_binary_op_on_device<Multiply, float>(x0, x1);
    check_device_result(y, x0, x1, [](float a, float b) { return a * b; });
}

TEST(BinaryElementWiseOp_Device, Multiply_half_float_half)
{
    // Test precision fix on device: should compute in float, then convert
    float x0  = 100.0f;
    half_t x1 = type_convert<half_t>(0.01f);
    half_t y  = run_binary_op_on_device<Multiply, half_t>(x0, x1);
    check_device_result(y, x0, x1, [](float a, float b) { return a * b; });
}

TEST(BinaryElementWiseOp_Device, Multiply_half_half_half)
{
    half_t x0 = type_convert<half_t>(2.0f);
    half_t x1 = type_convert<half_t>(3.0f);
    half_t y  = run_binary_op_on_device<Multiply, half_t>(x0, x1);
    check_device_result(y, x0, x1, [](float a, float b) { return a * b; });
}

// -----------------------------------------------------------------------------
// Device Bilinear Tests
// -----------------------------------------------------------------------------

TEST(BinaryElementWiseOp_Device, Bilinear_float_float_float)
{
    const float alpha = 2.0f;
    const float beta  = 3.0f;
    float x0          = 1.0f;
    float x1          = 2.0f;
    float y           = run_bilinear_on_device<float>(x0, x1, Bilinear(alpha, beta));
    float ref         = alpha * x0 + beta * x1;
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryElementWiseOp_Device, Bilinear_half_half_half)
{
    // Test precision fix on device: should compute in float precision
    const float alpha = 2.0f;
    const float beta  = 3.0f;
    half_t x0         = type_convert<half_t>(1.0f);
    half_t x1         = type_convert<half_t>(2.0f);
    half_t y          = run_bilinear_on_device<half_t>(x0, x1, Bilinear(alpha, beta));
    check_device_result(y, x0, x1, [=](float a, float b) { return alpha * a + beta * b; });
}

TEST(BinaryElementWiseOp_Device, Bilinear_half_half_half_precision_regression_small_coeffs)
{
    // Before fix: alpha/beta demoted to half_t, losing precision.
    // After fix:  alpha/beta stay float, computation done in float, demoted only at the end.
    const float alpha = 0.001f;
    const float beta  = 0.002f;
    half_t x0         = type_convert<half_t>(100.0f);
    half_t x1         = type_convert<half_t>(200.0f);
    half_t y          = run_bilinear_on_device<half_t>(x0, x1, Bilinear(alpha, beta));
    check_device_result(y, x0, x1, [=](float a, float b) { return alpha * a + beta * b; });
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
    float x0  = 1000.0f;
    half_t x1 = type_convert<half_t>(0.001f);
    half_t y  = run_binary_op_on_device<Multiply, half_t>(x0, x1);
    check_device_result(y, x0, x1, [](float a, float b) { return a * b; });
}

TEST(BinaryElementWiseOp_Device, Bilinear_half_half_half_precision_regression)
{
    // Before fix: computed entirely in half_t including alpha/beta
    // After fix:  promotes to float for computation
    const float alpha = 0.1f;
    const float beta  = 0.2f;
    half_t x0         = type_convert<half_t>(10.0f);
    half_t x1         = type_convert<half_t>(20.0f);
    half_t y          = run_bilinear_on_device<half_t>(x0, x1, Bilinear(alpha, beta));
    check_device_result(y, x0, x1, [=](float a, float b) { return alpha * a + beta * b; });
}
