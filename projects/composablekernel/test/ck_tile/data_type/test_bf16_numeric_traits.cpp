// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_bf16_common.hpp"
#include <cmath>

using namespace ck_tile;
using namespace ck_tile_test;

class Bf16NumericTraitsTest : public Bf16TestBase
{
};

// Test numeric_traits structure
TEST_F(Bf16NumericTraitsTest, NumericTraitsValues)
{
    // bf16 has 8-bit exponent and 7-bit mantissa
    EXPECT_EQ(numeric_traits<bf16_t>::exp, 8);
    EXPECT_EQ(numeric_traits<bf16_t>::mant, 7);
    EXPECT_EQ(numeric_traits<bf16_t>::PackedSize, 1);
}

// Test numeric<bf16_t>::min()
TEST_F(Bf16NumericTraitsTest, MinValue)
{
    bf16_t min_val = numeric<bf16_t>::min();
    uint16_t bits  = bf16_to_bits(min_val);

    // bf16 min normal: sign=0, exp=00000001, mant=0000000
    EXPECT_EQ(bits, 0x0080);

    // Should be smallest positive normal value
    float f = static_cast<float>(min_val);
    EXPECT_GT(f, 0.0f);
    EXPECT_TRUE(std::isnormal(f));

    // Verify it's approximately 2^-126 * (1 + 0/128) = 2^-126
    EXPECT_NEAR(f, std::ldexp(1.0f, -126), 1e-45f);
}

// Test numeric<bf16_t>::max()
TEST_F(Bf16NumericTraitsTest, MaxValue)
{
    bf16_t max_val = numeric<bf16_t>::max();
    uint16_t bits  = bf16_to_bits(max_val);

    // bf16 max normal: sign=0, exp=11111110, mant=1111111
    EXPECT_EQ(bits, 0x7F7F);

    // Should be largest finite value
    float f = static_cast<float>(max_val);
    EXPECT_GT(f, 0.0f);
    EXPECT_TRUE(std::isfinite(f));
    EXPECT_FALSE(std::isinf(f));

    // Verify it's approximately 2^127 * (1 + 127/128)
    float expected = std::ldexp(1.0f + 127.0f / 128.0f, 127);
    EXPECT_NEAR(f, expected, expected * 1e-6f);
}

// Test numeric<bf16_t>::lowest()
TEST_F(Bf16NumericTraitsTest, LowestValue)
{
    bf16_t lowest_val = numeric<bf16_t>::lowest();
    uint16_t bits     = bf16_to_bits(lowest_val);

    // bf16 lowest (most negative): sign=1, exp=11111110, mant=1111111
    EXPECT_EQ(bits, 0xFF7F);

    // Should be most negative finite value
    float f = static_cast<float>(lowest_val);
    EXPECT_LT(f, 0.0f);
    EXPECT_TRUE(std::isfinite(f));

    // Should be negative of max
    EXPECT_EQ(f, -static_cast<float>(numeric<bf16_t>::max()));
}

// Test numeric<bf16_t>::epsilon()
TEST_F(Bf16NumericTraitsTest, EpsilonValue)
{
    bf16_t epsilon_val = numeric<bf16_t>::epsilon();
    uint16_t bits      = bf16_to_bits(epsilon_val);

    // bf16 epsilon: 2^-7 (smallest increment from 1.0)
    // sign=0, exp=01111000, mant=0000000
    EXPECT_EQ(bits, 0x3C00);

    float f = static_cast<float>(epsilon_val);
    EXPECT_EQ(f, std::ldexp(1.0f, -7)); // 2^-7 = 1/128
    EXPECT_EQ(f, 0.0078125f);

    // Verify it's the difference between 1.0 and the next larger value
    bf16_t one          = float_to_bf16(1.0f);
    bf16_t one_plus_eps = float_to_bf16(1.0f + static_cast<float>(epsilon_val));
    EXPECT_NE(bf16_to_bits(one), bf16_to_bits(one_plus_eps));
}

// // Test numeric<bf16_t>::round_error()
TEST_F(Bf16NumericTraitsTest, RoundErrorValue)
{
    bf16_t round_error_val = numeric<bf16_t>::round_error();
    uint16_t bits          = bf16_to_bits(round_error_val);

    // bf16 round error: 0.5
    // sign=0, exp=01111110, mant=0000000
    EXPECT_EQ(bits, 0x3F00);

    float f = static_cast<float>(round_error_val);
    EXPECT_EQ(f, 0.5f);
}

// // Test numeric<bf16_t>::infinity()
TEST_F(Bf16NumericTraitsTest, InfinityValue)
{
    bf16_t inf_val = numeric<bf16_t>::infinity();
    uint16_t bits  = bf16_to_bits(inf_val);

    // bf16 infinity: sign=0, exp=11111111, mant=0000000
    EXPECT_EQ(bits, 0x7F80);

    float f = static_cast<float>(inf_val);
    EXPECT_TRUE(std::isinf(f));
    EXPECT_GT(f, 0.0f);
}

// // Test numeric<bf16_t>::quiet_NaN()
TEST_F(Bf16NumericTraitsTest, QuietNaNValue)
{
    bf16_t qnan_val = numeric<bf16_t>::quiet_NaN();
    uint16_t bits   = bf16_to_bits(qnan_val);

    // bf16 quiet NaN: sign=0, exp=11111111, mant=non-zero
    EXPECT_EQ(bits, 0x7FFF);
    EXPECT_EQ((bits & 0x7F80), 0x7F80); // All exponent bits set
    EXPECT_NE((bits & 0x007F), 0);      // Mantissa non-zero

    EXPECT_TRUE(isnan(qnan_val));
}

// // Test numeric<bf16_t>::signaling_NaN()
TEST_F(Bf16NumericTraitsTest, SignalingNaNValue)
{
    bf16_t snan_val = numeric<bf16_t>::signaling_NaN();
    uint16_t bits   = bf16_to_bits(snan_val);

    // bf16 signaling NaN: sign=0, exp=11111111, mant=non-zero
    // Note: The implementation returns the same bit pattern as quiet NaN
    EXPECT_EQ(bits, 0x7FFF);
    EXPECT_EQ((bits & 0x7F80), 0x7F80); // All exponent bits set
    EXPECT_NE((bits & 0x007F), 0);      // Mantissa non-zero

    EXPECT_TRUE(isnan(snan_val));
}

// // Test numeric<bf16_t>::denorm_min()
TEST_F(Bf16NumericTraitsTest, DenormMinValue)
{
    bf16_t denorm_min_val = numeric<bf16_t>::denorm_min();
    uint16_t bits         = bf16_to_bits(denorm_min_val);

    // bf16 smallest positive subnormal: sign=0, exp=00000000, mant=0000001
    EXPECT_EQ(bits, 0x0001);

    float f = bf16_to_float(denorm_min_val);
    EXPECT_GT(f, 0.0f);

    // bf16 subnormal with exponent=0, mantissa=1:
    // Value = 2^(1-127) * (0 + 1/128) = 2^-126 * 2^-7 = 2^-133
    // Note: For subnormals, the implicit leading bit is 0, not 1
    EXPECT_NEAR(f, std::ldexp(1.0f, -133), 1e-45f);
}

// // Test numeric<bf16_t>::zero()
// Test numeric<bf16_t>::zero() - verifies zero value has all bits zero and converts correctly
TEST_F(Bf16NumericTraitsTest, ZeroValue)
{
    bf16_t zero_val = numeric<bf16_t>::zero();
    uint16_t bits   = bf16_to_bits(zero_val);

    // bf16 zero: all bits zero
    EXPECT_EQ(bits, 0x0000);

    float f = static_cast<float>(zero_val);
    EXPECT_EQ(f, 0.0f);
    EXPECT_FALSE(std::signbit(f)); // Positive zero
}

// Test special value bit patterns - verifies IEEE 754 special values (zero, infinity, NaN) are
// correctly represented
TEST_F(Bf16NumericTraitsTest, SpecialValueBitPatterns)
{
    // Positive zero: sign=0, exp=0, mant=0 - verifies positive zero bit pattern and conversion
    {
        bf16_t val = bits_to_bf16(0x0000);
        EXPECT_EQ(static_cast<float>(val), 0.0f);
        EXPECT_FALSE(std::signbit(static_cast<float>(val)));
    }

    // Negative zero: sign=1, exp=0, mant=0 - verifies negative zero preserves sign bit
    {
        bf16_t val = bits_to_bf16(0x8000);
        EXPECT_EQ(static_cast<float>(val), -0.0f);
        EXPECT_TRUE(std::signbit(static_cast<float>(val)));
    }

    // Positive infinity: sign=0, exp=all 1s, mant=0 - verifies infinity representation
    {
        bf16_t val = bits_to_bf16(0x7F80);
        EXPECT_TRUE(std::isinf(static_cast<float>(val)));
        EXPECT_GT(static_cast<float>(val), 0.0f);
    }

    // Negative infinity: sign=1, exp=all 1s, mant=0 - verifies negative infinity representation
    {
        bf16_t val = bits_to_bf16(0xFF80);
        EXPECT_TRUE(std::isinf(static_cast<float>(val)));
        EXPECT_LT(static_cast<float>(val), 0.0f);
    }

    // Various NaN patterns: exp=all 1s, mant=non-zero - verifies all valid NaN bit patterns are
    // detected
    {
        // Quiet NaN with different mantissa bits - tests all positive NaN patterns (0x7F81 to
        // 0x7FFF)
        for(uint16_t mant = 1; mant <= 0x7F; mant++)
        {
            uint16_t bits = 0x7F80 | mant;
            bf16_t val    = bits_to_bf16(bits);
            EXPECT_TRUE(isnan(val)) << "Bits 0x" << std::hex << bits << " should be NaN";
        }

        // Negative NaN - tests all negative NaN patterns (0xFF81 to 0xFFFF)
        for(uint16_t mant = 1; mant <= 0x7F; mant++)
        {
            uint16_t bits = 0xFF80 | mant;
            bf16_t val    = bits_to_bf16(bits);
            EXPECT_TRUE(isnan(val)) << "Bits 0x" << std::hex << bits << " should be NaN";
        }
    }
}

// // Test relationships between special values
TEST_F(Bf16NumericTraitsTest, SpecialValueRelationships)
{
    // min < max
    EXPECT_LT(static_cast<float>(numeric<bf16_t>::min()),
              static_cast<float>(numeric<bf16_t>::max()));

    // lowest < max
    EXPECT_LT(static_cast<float>(numeric<bf16_t>::lowest()),
              static_cast<float>(numeric<bf16_t>::max()));

    // lowest == -max
    EXPECT_EQ(static_cast<float>(numeric<bf16_t>::lowest()),
              -static_cast<float>(numeric<bf16_t>::max()));

    // denorm_min < min
    EXPECT_LT(static_cast<float>(numeric<bf16_t>::denorm_min()),
              static_cast<float>(numeric<bf16_t>::min()));

    // zero < denorm_min < min < 1.0 < max < infinity
    EXPECT_EQ(static_cast<float>(numeric<bf16_t>::zero()), 0.0f);
    EXPECT_LT(static_cast<float>(numeric<bf16_t>::zero()),
              static_cast<float>(numeric<bf16_t>::denorm_min()));
    EXPECT_LT(static_cast<float>(numeric<bf16_t>::denorm_min()),
              static_cast<float>(numeric<bf16_t>::min()));
    EXPECT_LT(static_cast<float>(numeric<bf16_t>::min()), 1.0f);
    EXPECT_LT(1.0f, static_cast<float>(numeric<bf16_t>::max()));
    EXPECT_LT(static_cast<float>(numeric<bf16_t>::max()),
              static_cast<float>(numeric<bf16_t>::infinity()));
}

// // Test edge cases and boundary values
TEST_F(Bf16NumericTraitsTest, EdgeCases)
{
    // Test values just above and below special boundaries

    // Value just above min
    {
        uint16_t bits = bf16_to_bits(numeric<bf16_t>::min()) + 1;
        bf16_t val    = bits_to_bf16(bits);
        EXPECT_GT(static_cast<float>(val), static_cast<float>(numeric<bf16_t>::min()));
        EXPECT_TRUE(std::isnormal(static_cast<float>(val)));
    }

    // Value just below max (not infinity)
    {
        uint16_t bits = bf16_to_bits(numeric<bf16_t>::max()) - 1;
        bf16_t val    = bits_to_bf16(bits);
        EXPECT_LT(static_cast<float>(val), static_cast<float>(numeric<bf16_t>::max()));
        EXPECT_TRUE(std::isfinite(static_cast<float>(val)));
    }

    // Largest value that's not infinity (max normal)
    {
        uint16_t bits = 0x7F7F; // Exponent = 254, mantissa = all 1s
        bf16_t val    = bits_to_bf16(bits);
        EXPECT_TRUE(std::isfinite(static_cast<float>(val)));
        EXPECT_FALSE(std::isinf(static_cast<float>(val)));
    }

    // Smallest value that is infinity
    {
        uint16_t bits = 0x7F80; // Exponent = 255, mantissa = 0
        bf16_t val    = bits_to_bf16(bits);
        EXPECT_TRUE(std::isinf(static_cast<float>(val)));
    }
}
