// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_bf16_common.hpp"
#include <cmath>

using namespace ck_tile;
using namespace ck_tile_test;

class Bf16ConversionTest : public Bf16TestBase
{
};

// Test float to bf16 conversion with default rounding mode
TEST_F(Bf16ConversionTest, FloatToBf16Basic)
{
    // Test exact representable values
    {
        float f  = 1.0f;
        bf16_t b = float_to_bf16(f);
        EXPECT_EQ(bf16_to_bits(b), 0x3F80);
        EXPECT_EQ(static_cast<float>(b), 1.0f);
    }

    {
        float f  = -1.0f;
        bf16_t b = float_to_bf16(f);
        EXPECT_EQ(bf16_to_bits(b), 0xBF80);
        EXPECT_EQ(static_cast<float>(b), -1.0f);
    }

    {
        float f  = 2.0f;
        bf16_t b = float_to_bf16(f);
        EXPECT_EQ(bf16_to_bits(b), 0x4000);
        EXPECT_EQ(static_cast<float>(b), 2.0f);
    }

    {
        float f  = 0.5f;
        bf16_t b = float_to_bf16(f);
        EXPECT_EQ(bf16_to_bits(b), 0x3F00);
        EXPECT_EQ(static_cast<float>(b), 0.5f);
    }
}

// Test special values
TEST_F(Bf16ConversionTest, FloatToBf16SpecialValues)
{
    // Zero
    {
        bf16_t b = float_to_bf16(0.0f);
        EXPECT_EQ(bf16_to_bits(b), 0x0000);
        EXPECT_EQ(static_cast<float>(b), 0.0f);
    }

    // Negative zero
    {
        bf16_t b = float_to_bf16(-0.0f);
        EXPECT_EQ(bf16_to_bits(b), 0x8000);
        EXPECT_EQ(static_cast<float>(b), -0.0f);
    }

    // Infinity
    {
        bf16_t b = float_to_bf16(std::numeric_limits<float>::infinity());
        EXPECT_EQ(bf16_to_bits(b), 0x7F80);
        EXPECT_TRUE(std::isinf(static_cast<float>(b)));
        EXPECT_TRUE(static_cast<float>(b) > 0);
    }

    // Negative infinity
    {
        bf16_t b = float_to_bf16(-std::numeric_limits<float>::infinity());
        EXPECT_EQ(bf16_to_bits(b), 0xFF80);
        EXPECT_TRUE(std::isinf(static_cast<float>(b)));
        EXPECT_TRUE(static_cast<float>(b) < 0);
    }

    // NaN
    {
        bf16_t b = float_to_bf16(std::numeric_limits<float>::quiet_NaN());
        EXPECT_TRUE(isnan(b));
        EXPECT_TRUE((bf16_to_bits(b) & 0x7F80) == 0x7F80); // Exponent all 1s
        EXPECT_TRUE((bf16_to_bits(b) & 0x007F) != 0);      // Mantissa not zero
    }
}

// Test rounding behavior
TEST_F(Bf16ConversionTest, FloatToBf16Rounding)
{
    // Test round-to-nearest-even (default mode)
    {
        // Value that requires rounding, should round to nearest bf16 value
        float f      = 1.001953125f; // Between 1.0 and 1.0078125, closer to 1.0
        bf16_t b     = float_to_bf16(f);
        float result = static_cast<float>(b);
        // Should round to nearest: 1.0f (since 1.001953125 is closer to 1.0 than 1.0078125)
        EXPECT_EQ(result, 1.0f);
    }

    // Test values that require rounding
    {
        float f      = 1.0009765625f; // Not exactly representable in bf16, closer to 1.0
        bf16_t b     = float_to_bf16(f);
        float result = static_cast<float>(b);
        // Should round to nearest: 1.0f (since 1.0009765625 is closer to 1.0 than 1.0078125)
        EXPECT_EQ(result, 1.0f);
    }
}

// Test different rounding modes
TEST_F(Bf16ConversionTest, FloatToBf16RoundingModes)
{
    // Standard rounding (round-to-nearest-even)
    {
        bf16_t b = float_to_bf16(1.001953125f, constant<bf16_rounding_mode::standard>{});
        // Verify exact bit pattern to validate rounding mode semantics
        // 1.001953125f (0x3f804000) should round to 1.0f (0x3F80) with round-to-nearest
        EXPECT_EQ(bf16_to_bits(b), 0x3F80);
        float result = static_cast<float>(b);
        EXPECT_EQ(result, 1.0f);
    }

    // Truncation mode (round-toward-zero, no rounding)
    {
        bf16_t b = float_to_bf16(1.001953125f, constant<bf16_rounding_mode::truncate>{});
        // Verify exact bit pattern: truncate just shifts right by 16 bits
        // 1.001953125f (0x3f804000) >> 16 = 0x3f80 = 1.0f
        EXPECT_EQ(bf16_to_bits(b), 0x3F80);
        float result = static_cast<float>(b);
        EXPECT_EQ(result, 1.0f);
        // Truncation should not increase value (rounds toward zero)
        EXPECT_LE(result, 1.001953125f);
    }

    // Truncation with NaN preservation
    {
        bf16_t b = float_to_bf16(std::numeric_limits<float>::quiet_NaN(),
                                 constant<bf16_rounding_mode::truncate_with_nan>{});
        EXPECT_TRUE(isnan(b));
    }
}

// Test double to bf16 conversion
TEST_F(Bf16ConversionTest, DoubleToBf16)
{
    {
        double d = 1.0;
        bf16_t b = double_to_bf16(d);
        EXPECT_EQ(bf16_to_bits(b), 0x3F80);
        EXPECT_EQ(static_cast<float>(b), 1.0f);
    }

    {
        double d     = -3.141592653589793;
        bf16_t b     = double_to_bf16(d);
        float result = static_cast<float>(b);
        EXPECT_NEAR(result, -3.141592653589793, 0.01);
    }

    // Large double value
    {
        double d = 1e100; // Much larger than bf16 can represent
        bf16_t b = double_to_bf16(d);
        EXPECT_TRUE(std::isinf(static_cast<float>(b)));
    }
}

// Test integer to bf16 conversion
TEST_F(Bf16ConversionTest, IntToBf16)
{
#if CK_TILE_USE_CUSTOM_DATA_TYPE
    {
        int i = 42;
        bf16_t b(i);
        EXPECT_EQ(static_cast<float>(b), 42.0f);
    }

    {
        int i = -100;
        bf16_t b(i);
        EXPECT_EQ(static_cast<float>(b), -100.0f);
    }

    {
        int i = 0;
        bf16_t b(i);
        EXPECT_EQ(static_cast<float>(b), 0.0f);
    }

    // Large int that requires rounding in bf16
    {
        int i = 16777217; // 2^24 + 1, not exactly representable in float
        bf16_t b(i);
        float result = static_cast<float>(b);
        EXPECT_NEAR(result, static_cast<float>(i), 256.0f);
    }
#endif
}

// Test bf16 to float conversion
TEST_F(Bf16ConversionTest, Bf16ToFloat)
{
    // Test all special bf16 values
    auto special_values = generate_special_bf16_values();
    for(const auto& bf16_val : special_values)
    {
        uint16_t bits = bf16_to_bits(bf16_val);
        float f       = bf16_to_float(bf16_val);

        if(isnan(bf16_val))
        {
            // Debug: Check bit pattern and float value
            uint32_t f_bits = bit_cast<uint32_t>(f);
            EXPECT_TRUE(std::isnan(f))
                << "bf16 NaN (bits=0x" << std::hex << bits << std::dec
                << ") should convert to float NaN, but got float with bits=0x" << std::hex << f_bits
                << std::dec << " value=" << f;
        }
        else if(bits == 0x7F80)
        {
            EXPECT_TRUE(std::isinf(f) && f > 0) << "bf16 +inf should convert to float +inf";
        }
        else if(bits == 0xFF80)
        {
            EXPECT_TRUE(std::isinf(f) && f < 0) << "bf16 -inf should convert to float -inf";
        }
        else
        {
            // For normal values, conversion should be exact
            bf16_t b_back = float_to_bf16(f);
            EXPECT_EQ(bf16_to_bits(bf16_val), bf16_to_bits(b_back))
                << "Round-trip conversion should preserve bf16 value";
        }
    }
}

// Test bf16 to double conversion
TEST_F(Bf16ConversionTest, Bf16ToDouble)
{
    {
        bf16_t b = float_to_bf16(1.0f);
        double d = bf16_to_double(b);
        EXPECT_EQ(d, 1.0);
    }

    {
        bf16_t b = numeric<bf16_t>::infinity();
        double d = bf16_to_double(b);
        EXPECT_TRUE(std::isinf(d) && d > 0);
    }

    {
        bf16_t b = numeric<bf16_t>::quiet_NaN();
        double d = bf16_to_double(b);
        EXPECT_TRUE(std::isnan(d));
    }
}

// Test bf16 to int conversion
TEST_F(Bf16ConversionTest, Bf16ToInt)
{
#if CK_TILE_USE_CUSTOM_DATA_TYPE
    {
        bf16_t b = float_to_bf16(42.0f);
        int i    = static_cast<int>(b);
        EXPECT_EQ(i, 42);
    }

    {
        bf16_t b = float_to_bf16(-100.0f);
        int i    = static_cast<int>(b);
        EXPECT_EQ(i, -100);
    }

    {
        bf16_t b = float_to_bf16(0.0f);
        int i    = static_cast<int>(b);
        EXPECT_EQ(i, 0);
    }

    // Test rounding behavior
    {
        bf16_t b = float_to_bf16(42.7f);
        int i    = static_cast<int>(b);
        EXPECT_EQ(i, 42); // Should truncate
    }

    {
        bf16_t b = float_to_bf16(-42.7f);
        int i    = static_cast<int>(b);
        EXPECT_EQ(i, -42); // Should truncate towards zero
    }
#endif
}

// // Test fp16 to bf16 conversion
TEST_F(Bf16ConversionTest, Fp16ToBf16)
{
    {
        fp16_t h = static_cast<fp16_t>(1.0f);
        bf16_t b = fp16_to_bf16(h);
        EXPECT_EQ(static_cast<float>(b), 1.0f);
    }

    {
        fp16_t h = static_cast<fp16_t>(-0.5f);
        bf16_t b = fp16_to_bf16(h);
        EXPECT_EQ(static_cast<float>(b), -0.5f);
    }

    // fp16 infinity
    {
        fp16_t h = numeric<fp16_t>::infinity();
        bf16_t b = fp16_to_bf16(h);
        EXPECT_TRUE(std::isinf(static_cast<float>(b)));
    }

    // fp16 NaN
    {
        fp16_t h = numeric<fp16_t>::quiet_NaN();
        bf16_t b = fp16_to_bf16(h);
        EXPECT_TRUE(isnan(b));
    }
}

// // Test bf16 to fp16 conversion
TEST_F(Bf16ConversionTest, Bf16ToFp16)
{
    {
        bf16_t b = float_to_bf16(1.0f);
        fp16_t h = bf16_to_fp16(b);
        EXPECT_EQ(static_cast<float>(h), 1.0f);
    }

    // Test value that's representable in bf16 but may lose precision in fp16
    {
        bf16_t b = float_to_bf16(131072.0f); // 2^17
        fp16_t h = bf16_to_fp16(b);
        // fp16 max is 65504, so this should overflow to infinity
        EXPECT_TRUE(std::isinf(static_cast<float>(h)));
    }
}

// // Test round-trip conversions
TEST_F(Bf16ConversionTest, RoundTripConversions)
{
    // Generate test values
    auto test_floats = generate_test_floats();

    for(float f : test_floats)
    {
        // Skip if the float is too large for bf16
        if(std::abs(f) > 3.38953139e38f && !std::isinf(f) && !std::isnan(f))
        {
            continue;
        }

        // float -> bf16 -> float
        bf16_t b     = float_to_bf16(f);
        float f_back = static_cast<float>(b);

        if(std::isnan(f))
        {
            EXPECT_TRUE(std::isnan(f_back)) << "NaN should be preserved";
        }
        else if(std::isinf(f))
        {
            EXPECT_TRUE(std::isinf(f_back)) << "Infinity should be preserved";
            EXPECT_EQ(std::signbit(f), std::signbit(f_back)) << "Sign should be preserved";
        }
        else
        {
            // For normal values, check if round-trip preserves the bf16 value
            bf16_t b_back = float_to_bf16(f_back);
            EXPECT_EQ(bf16_to_bits(b), bf16_to_bits(b_back))
                << "Round-trip should preserve bf16 representation for " << f;
        }
    }
}

// // Test denormal handling
TEST_F(Bf16ConversionTest, DenormalHandling)
{
    // Float denormals are much smaller than bf16 denormals (float has 23 mantissa bits,
    // bf16 has 7), so float denormals flush to zero when converted to bf16.
    // Note: bf16 does support denormals (see numeric<bf16_t>::denorm_min() = 0x0001),
    // but float denormals are below the smallest representable bf16 value.
    {
        float f  = std::numeric_limits<float>::denorm_min();
        bf16_t b = float_to_bf16(f);
        EXPECT_EQ(bf16_to_bits(b), 0x0000) << "Float denormal should flush to zero in bf16";
    }

    {
        float f  = -std::numeric_limits<float>::denorm_min();
        bf16_t b = float_to_bf16(f);
        EXPECT_EQ(bf16_to_bits(b), 0x8000)
            << "Negative float denormal should flush to negative zero in bf16";
    }

    // Test smallest normal bf16 value
    {
        bf16_t b = numeric<bf16_t>::min();
        float f  = static_cast<float>(b);
        EXPECT_GT(f, 0.0f);
        EXPECT_TRUE(std::isnormal(f)) << "bf16 min should convert to normal float";
    }
}

// Test overflow handling
TEST_F(Bf16ConversionTest, OverflowHandling)
{
    // Note: BF16 has the same 8-bit exponent as float32, but only 7 mantissa bits vs 23.
    // This means bf16::max (0x7F7F) ≈ 3.39e38 is LESS than float::max ≈ 3.40e38.
    //
    // Hardware behavior differs by architecture:
    // - gfx950: RTN rounding -> float::max rounds to infinity (IEEE-754 compliant)
    // - gfx12/gfx1250: Saturates -> float::max clamps to bf16::max (faster, non-IEEE)

    // Test float max overflow behavior (architecture-dependent)
    {
        float f            = std::numeric_limits<float>::max();
        bf16_t b           = float_to_bf16(f);
        float result       = bf16_to_float(b);
        uint16_t bf16_bits = bf16_to_bits(b);

#ifdef CK_TILE_BF16_OVERFLOW_SATURATES
        // gfx12/gfx1250: Hardware saturates to bf16::max
        EXPECT_FALSE(std::isinf(result))
            << "gfx12/gfx1250: float::max should saturate to bf16::max (0x7f7f). Got bf16=0x"
            << std::hex << bf16_bits << std::dec << " result=" << result;
        EXPECT_EQ(bf16_bits, 0x7f7f)
            << "gfx12/gfx1250: Expected saturation to bf16::max (0x7f7f), got 0x" << std::hex
            << bf16_bits << std::dec;
#else
        // gfx950 and software: RTN rounding to infinity (IEEE-754 behavior)
        EXPECT_TRUE(std::isinf(result) && result > 0)
            << "gfx950/software: float::max should overflow to +infinity with RTN rounding. Got "
               "bf16=0x"
            << std::hex << bf16_bits << std::dec << " result=" << result;
        EXPECT_EQ(bf16_bits, 0x7f80)
            << "Expected +infinity (0x7f80), got 0x" << std::hex << bf16_bits << std::dec;
#endif
    }

    {
        float f            = -std::numeric_limits<float>::max();
        bf16_t b           = float_to_bf16(f);
        float result       = bf16_to_float(b);
        uint16_t bf16_bits = bf16_to_bits(b);

#ifdef CK_TILE_BF16_OVERFLOW_SATURATES
        // gfx12/gfx1250: Hardware saturates to -bf16::max
        EXPECT_FALSE(std::isinf(result))
            << "gfx12/gfx1250: -float::max should saturate to -bf16::max (0xff7f)";
        EXPECT_EQ(bf16_bits, 0xff7f)
            << "gfx12/gfx1250: Expected saturation to -bf16::max (0xff7f), got 0x" << std::hex
            << bf16_bits << std::dec;
#else
        // gfx950 and software: RTN rounding to -infinity (IEEE-754 behavior)
        EXPECT_TRUE(std::isinf(result) && result < 0)
            << "gfx950/software: -float::max should overflow to -infinity with RTN rounding";
        EXPECT_EQ(bf16_bits, 0xff80)
            << "Expected -infinity (0xff80), got 0x" << std::hex << bf16_bits << std::dec;
#endif
    }

    // Test infinity passthrough
    {
        float f      = std::numeric_limits<float>::infinity();
        bf16_t b     = float_to_bf16(f);
        float result = bf16_to_float(b);
        EXPECT_TRUE(std::isinf(result) && result > 0)
            << "Float +infinity should convert to bf16 +infinity";
    }

    {
        float f      = -std::numeric_limits<float>::infinity();
        bf16_t b     = float_to_bf16(f);
        float result = bf16_to_float(b);
        EXPECT_TRUE(std::isinf(result) && result < 0)
            << "Float -infinity should convert to bf16 -infinity";
    }
}
