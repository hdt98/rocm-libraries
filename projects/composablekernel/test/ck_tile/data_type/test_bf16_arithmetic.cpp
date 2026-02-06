// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_bf16_common.hpp"
#include <cmath>

using namespace ck_tile;
using namespace ck_tile_test;

// This test file is only compiled when CK_TILE_USE_CUSTOM_DATA_TYPE=1
// because arithmetic operators are only defined in that case

class Bf16ArithmeticTest : public Bf16TestBase
{
};

#if CK_TILE_USE_CUSTOM_DATA_TYPE

// Test comparison operators
TEST_F(Bf16ArithmeticTest, ComparisonOperators)
{
    // Equal values
    {
        bf16_t a = float_to_bf16(1.0f);
        bf16_t b = float_to_bf16(1.0f);
        EXPECT_TRUE(a == b);
        EXPECT_FALSE(a != b);
        EXPECT_FALSE(a < b);
        EXPECT_TRUE(a <= b);
        EXPECT_FALSE(a > b);
        EXPECT_TRUE(a >= b);
    }

    // Different values
    {
        bf16_t a = float_to_bf16(1.0f);
        bf16_t b = float_to_bf16(2.0f);
        EXPECT_FALSE(a == b);
        EXPECT_TRUE(a != b);
        EXPECT_TRUE(a < b);
        EXPECT_TRUE(a <= b);
        EXPECT_FALSE(a > b);
        EXPECT_FALSE(a >= b);
    }

    // Negative values
    {
        bf16_t a = float_to_bf16(-2.0f);
        bf16_t b = float_to_bf16(-1.0f);
        EXPECT_TRUE(a < b);
        EXPECT_TRUE(a <= b);
        EXPECT_FALSE(a > b);
        EXPECT_FALSE(a >= b);
    }

    // Zero comparisons
    {
        bf16_t zero     = float_to_bf16(0.0f);
        bf16_t neg_zero = bits_to_bf16(0x8000);
        bf16_t pos      = float_to_bf16(1.0f);
        bf16_t neg      = float_to_bf16(-1.0f);

        // 0.0 == -0.0 due to epsilon comparison
        EXPECT_TRUE(zero == neg_zero);

        EXPECT_TRUE(neg < zero);
        EXPECT_TRUE(zero < pos);
        EXPECT_TRUE(neg < pos);
    }

    // Infinity comparisons
    {
        bf16_t inf     = numeric<bf16_t>::infinity();
        bf16_t neg_inf = bits_to_bf16(0xFF80);
        bf16_t max_val = numeric<bf16_t>::max();

        EXPECT_TRUE(max_val < inf);
        EXPECT_TRUE(neg_inf < max_val);
        EXPECT_TRUE(neg_inf < inf);
        EXPECT_FALSE(inf < inf);
        EXPECT_TRUE(inf == inf);
    }

    // NaN comparisons - all should return false except !=
    {
        bf16_t nan = numeric<bf16_t>::quiet_NaN();
        bf16_t val = float_to_bf16(1.0f);

        EXPECT_FALSE(nan == nan);
        EXPECT_TRUE(nan != nan);
        EXPECT_FALSE(nan < val);
        EXPECT_FALSE(nan <= val);
        EXPECT_FALSE(nan > val);
        EXPECT_FALSE(nan >= val);
        EXPECT_FALSE(val < nan);
        EXPECT_FALSE(val > nan);
    }
}

// Test addition operator
TEST_F(Bf16ArithmeticTest, AdditionOperator)
{
    // Basic addition
    {
        bf16_t a = float_to_bf16(1.0f);
        bf16_t b = float_to_bf16(2.0f);
        bf16_t c = a + b;
        EXPECT_EQ(static_cast<float>(c), 3.0f);
    }

    // Negative values
    {
        bf16_t a = float_to_bf16(-1.5f);
        bf16_t b = float_to_bf16(2.5f);
        bf16_t c = a + b;
        EXPECT_EQ(static_cast<float>(c), 1.0f);
    }

    // Zero addition
    {
        bf16_t a    = float_to_bf16(3.14f);
        bf16_t zero = float_to_bf16(0.0f);
        bf16_t c    = a + zero;
        EXPECT_EQ(static_cast<float>(c), 3.14f);
    }

    // Overflow to infinity
    {
        bf16_t max_val = numeric<bf16_t>::max();
        bf16_t c       = max_val + max_val;
        EXPECT_TRUE(std::isinf(static_cast<float>(c)));
    }

    // Infinity arithmetic
    {
        bf16_t inf = numeric<bf16_t>::infinity();
        bf16_t val = float_to_bf16(1.0f);
        bf16_t c   = inf + val;
        EXPECT_TRUE(std::isinf(static_cast<float>(c)));
    }

    // NaN propagation
    {
        bf16_t nan = numeric<bf16_t>::quiet_NaN();
        bf16_t val = float_to_bf16(1.0f);
        bf16_t c   = nan + val;
        EXPECT_TRUE(isnan(c));
    }
}

// Test subtraction operator
TEST_F(Bf16ArithmeticTest, SubtractionOperator)
{
    // Basic subtraction
    {
        bf16_t a = float_to_bf16(3.0f);
        bf16_t b = float_to_bf16(1.0f);
        bf16_t c = a - b;
        EXPECT_EQ(static_cast<float>(c), 2.0f);
    }

    // Negative result
    {
        bf16_t a = float_to_bf16(1.0f);
        bf16_t b = float_to_bf16(3.0f);
        bf16_t c = a - b;
        EXPECT_EQ(static_cast<float>(c), -2.0f);
    }

    // Subtracting negative
    {
        bf16_t a = float_to_bf16(1.0f);
        bf16_t b = float_to_bf16(-2.0f);
        bf16_t c = a - b;
        EXPECT_EQ(static_cast<float>(c), 3.0f);
    }

    // Cancellation
    {
        bf16_t a = float_to_bf16(3.14159f);
        bf16_t c = a - a;
        EXPECT_EQ(static_cast<float>(c), 0.0f);
    }
}

// Test unary negation
TEST_F(Bf16ArithmeticTest, UnaryNegation)
{
    // Positive to negative
    {
        bf16_t a = float_to_bf16(3.14159f);
        bf16_t b = -a;
        EXPECT_EQ(static_cast<float>(b), -3.14159f);
        EXPECT_EQ(bf16_to_bits(b), bf16_to_bits(a) ^ 0x8000);
    }

    // Negative to positive
    {
        bf16_t a = float_to_bf16(-2.71828f);
        bf16_t b = -a;
        EXPECT_EQ(static_cast<float>(b), 2.71828f);
    }

    // Zero negation
    {
        bf16_t zero     = float_to_bf16(0.0f);
        bf16_t neg_zero = -zero;
        EXPECT_EQ(bf16_to_bits(neg_zero), 0x8000);
    }

    // Infinity negation
    {
        bf16_t inf     = numeric<bf16_t>::infinity();
        bf16_t neg_inf = -inf;
        EXPECT_TRUE(std::isinf(static_cast<float>(neg_inf)));
        EXPECT_LT(static_cast<float>(neg_inf), 0.0f);
    }

    // NaN negation
    {
        bf16_t nan     = numeric<bf16_t>::quiet_NaN();
        bf16_t neg_nan = -nan;
        EXPECT_TRUE(isnan(neg_nan));
        // Sign bit should be flipped
        EXPECT_EQ(bf16_to_bits(neg_nan), bf16_to_bits(nan) ^ 0x8000);
    }
}

// Test multiplication operator
TEST_F(Bf16ArithmeticTest, MultiplicationOperator)
{
    // Basic multiplication
    {
        bf16_t a = float_to_bf16(2.0f);
        bf16_t b = float_to_bf16(3.0f);
        bf16_t c = a * b;
        EXPECT_EQ(static_cast<float>(c), 6.0f);
    }

    // Negative multiplication
    {
        bf16_t a = float_to_bf16(-2.0f);
        bf16_t b = float_to_bf16(3.0f);
        bf16_t c = a * b;
        EXPECT_EQ(static_cast<float>(c), -6.0f);
    }

    // Multiplication by zero
    {
        bf16_t a    = float_to_bf16(3.14159f);
        bf16_t zero = float_to_bf16(0.0f);
        bf16_t c    = a * zero;
        EXPECT_EQ(static_cast<float>(c), 0.0f);
    }

    // Multiplication by one
    {
        bf16_t a   = float_to_bf16(3.14159f);
        bf16_t one = float_to_bf16(1.0f);
        bf16_t c   = a * one;
        EXPECT_EQ(static_cast<float>(c), 3.14159f);
    }

    // Overflow
    {
        bf16_t large = float_to_bf16(1e20f);
        bf16_t c     = large * large;
        EXPECT_TRUE(std::isinf(static_cast<float>(c)));
    }

    // Underflow
    {
        bf16_t small = float_to_bf16(1e-20f);
        bf16_t c     = small * small;
        EXPECT_EQ(static_cast<float>(c), 0.0f);
    }
}

// Test division operator
TEST_F(Bf16ArithmeticTest, DivisionOperator)
{
    // Basic division
    {
        bf16_t a = float_to_bf16(6.0f);
        bf16_t b = float_to_bf16(2.0f);
        bf16_t c = a / b;
        EXPECT_EQ(static_cast<float>(c), 3.0f);
    }

    // Division resulting in fraction
    {
        bf16_t a = float_to_bf16(1.0f);
        bf16_t b = float_to_bf16(3.0f);
        bf16_t c = a / b;
        EXPECT_NEAR(static_cast<float>(c), 0.333333f, 0.01f);
    }

    // Division by zero
    {
        bf16_t a    = float_to_bf16(1.0f);
        bf16_t zero = float_to_bf16(0.0f);
        bf16_t c    = a / zero;
        EXPECT_TRUE(std::isinf(static_cast<float>(c)));
        EXPECT_GT(static_cast<float>(c), 0.0f);
    }

    // Negative division by zero
    {
        bf16_t a    = float_to_bf16(-1.0f);
        bf16_t zero = float_to_bf16(0.0f);
        bf16_t c    = a / zero;
        EXPECT_TRUE(std::isinf(static_cast<float>(c)));
        EXPECT_LT(static_cast<float>(c), 0.0f);
    }

    // Zero divided by zero
    {
        bf16_t zero1 = float_to_bf16(0.0f);
        bf16_t zero2 = float_to_bf16(0.0f);
        bf16_t c     = zero1 / zero2;
        EXPECT_TRUE(isnan(c));
    }
}

// Test compound assignment operators
TEST_F(Bf16ArithmeticTest, CompoundAssignment)
{
    // Addition assignment
    {
        bf16_t a = float_to_bf16(1.0f);
        bf16_t b = float_to_bf16(2.0f);
        a += b;
        EXPECT_EQ(static_cast<float>(a), 3.0f);
    }

    // Subtraction assignment
    {
        bf16_t a = float_to_bf16(3.0f);
        bf16_t b = float_to_bf16(1.0f);
        a -= b;
        EXPECT_EQ(static_cast<float>(a), 2.0f);
    }

    // Multiplication assignment
    {
        bf16_t a = float_to_bf16(2.0f);
        bf16_t b = float_to_bf16(3.0f);
        a *= b;
        EXPECT_EQ(static_cast<float>(a), 6.0f);
    }

    // Division assignment
    {
        bf16_t a = float_to_bf16(6.0f);
        bf16_t b = float_to_bf16(2.0f);
        a /= b;
        EXPECT_EQ(static_cast<float>(a), 3.0f);
    }

    // Chain assignments
    {
        bf16_t a = float_to_bf16(1.0f);
        bf16_t b = float_to_bf16(2.0f);
        bf16_t c = float_to_bf16(3.0f);
        a += b += c; // b becomes 5, then a becomes 6
        EXPECT_EQ(static_cast<float>(b), 5.0f);
        EXPECT_EQ(static_cast<float>(a), 6.0f);
    }
}

// Test increment/decrement operators
TEST_F(Bf16ArithmeticTest, IncrementDecrement)
{
    // Pre-increment
    {
        bf16_t a = float_to_bf16(1.0f);
        bf16_t b = ++a;
        EXPECT_EQ(static_cast<float>(a), 2.0f);
        EXPECT_EQ(static_cast<float>(b), 2.0f);
    }

    // Post-increment
    {
        bf16_t a = float_to_bf16(1.0f);
        bf16_t b = a++;
        EXPECT_EQ(static_cast<float>(a), 2.0f);
        EXPECT_EQ(static_cast<float>(b), 1.0f);
    }

    // Pre-decrement
    {
        bf16_t a = float_to_bf16(2.0f);
        bf16_t b = --a;
        EXPECT_EQ(static_cast<float>(a), 1.0f);
        EXPECT_EQ(static_cast<float>(b), 1.0f);
    }

    // Post-decrement
    {
        bf16_t a = float_to_bf16(2.0f);
        bf16_t b = a--;
        EXPECT_EQ(static_cast<float>(a), 1.0f);
        EXPECT_EQ(static_cast<float>(b), 2.0f);
    }

    // Increment zero
    {
        bf16_t a = float_to_bf16(0.0f);
        ++a;
        EXPECT_EQ(static_cast<float>(a), 1.0f);
    }

    // Decrement zero
    {
        bf16_t a = float_to_bf16(0.0f);
        --a;
        EXPECT_EQ(static_cast<float>(a), -1.0f);
    }

    // Multiple increments
    {
        bf16_t a = float_to_bf16(0.0f);
        ++(++a); // Should increment twice
        EXPECT_EQ(static_cast<float>(a), 2.0f);
    }
}

// Test overflow/underflow behavior
TEST_F(Bf16ArithmeticTest, OverflowUnderflow)
{
    // Addition overflow
    {
        bf16_t max_val = numeric<bf16_t>::max();
        bf16_t result  = max_val + max_val;
        EXPECT_TRUE(std::isinf(static_cast<float>(result)));
        EXPECT_GT(static_cast<float>(result), 0.0f);
    }

    // Subtraction overflow (to negative infinity)
    {
        bf16_t max_val = numeric<bf16_t>::max();
        bf16_t lowest  = numeric<bf16_t>::lowest();
        bf16_t result  = lowest - max_val;
        EXPECT_TRUE(std::isinf(static_cast<float>(result)));
        EXPECT_LT(static_cast<float>(result), 0.0f);
    }

    // Multiplication overflow
    {
        bf16_t large  = float_to_bf16(1e10f);
        bf16_t result = large * large;
        EXPECT_TRUE(std::isinf(static_cast<float>(result)));
    }

    // Multiplication underflow
    {
        bf16_t tiny   = numeric<bf16_t>::min();
        bf16_t half   = float_to_bf16(0.5f);
        bf16_t result = tiny * half * half * half; // Should underflow to zero
        // Note: bf16 doesn't have denormals, so it flushes to zero
        EXPECT_EQ(static_cast<float>(result), 0.0f);
    }
}

// Test special cases with infinity and NaN
TEST_F(Bf16ArithmeticTest, SpecialCases)
{
    bf16_t inf     = numeric<bf16_t>::infinity();
    bf16_t neg_inf = bits_to_bf16(0xFF80);
    bf16_t nan     = numeric<bf16_t>::quiet_NaN();
    bf16_t one     = float_to_bf16(1.0f);
    bf16_t zero    = float_to_bf16(0.0f);

    // Infinity + Infinity = Infinity
    EXPECT_TRUE(std::isinf(static_cast<float>(inf + inf)));

    // Infinity - Infinity = NaN
    EXPECT_TRUE(isnan(inf - inf));

    // Infinity * 0 = NaN
    EXPECT_TRUE(isnan(inf * zero));

    // Infinity / Infinity = NaN
    EXPECT_TRUE(isnan(inf / inf));

    // 1 / 0 = Infinity
    EXPECT_TRUE(std::isinf(static_cast<float>(one / zero)));

    // NaN with any operation = NaN
    EXPECT_TRUE(isnan(nan + one));
    EXPECT_TRUE(isnan(nan - one));
    EXPECT_TRUE(isnan(nan * one));
    EXPECT_TRUE(isnan(nan / one));
    EXPECT_TRUE(isnan(one + nan));
    EXPECT_TRUE(isnan(one - nan));
    EXPECT_TRUE(isnan(one * nan));
    EXPECT_TRUE(isnan(one / nan));
}

#else // !CK_TILE_USE_CUSTOM_DATA_TYPE

// When custom data type is not used, arithmetic operators are not available
TEST_F(Bf16ArithmeticTest, ArithmeticNotAvailable)
{
    GTEST_SKIP() << "Arithmetic operators are only available when CK_TILE_USE_CUSTOM_DATA_TYPE=1";
}

#endif // CK_TILE_USE_CUSTOM_DATA_TYPE
