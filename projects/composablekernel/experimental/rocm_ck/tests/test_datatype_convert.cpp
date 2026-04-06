// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/datatype_convert.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>

using ::rocm_ck::bf16BitsToFloat;
using ::rocm_ck::DataType;
using ::rocm_ck::floatToBf16Bits;
using ::rocm_ck::floatToFp8e4m3Fnuz;
using ::rocm_ck::floatToFp8e4m3Ocp;
using ::rocm_ck::floatToFp8e5m2Fnuz;
using ::rocm_ck::floatToFp8e5m2Ocp;
using ::rocm_ck::floatToTyped;
using ::rocm_ck::fp8e4m3FnuzToFloat;
using ::rocm_ck::fp8e4m3OcpToFloat;
using ::rocm_ck::fp8e5m2FnuzToFloat;
using ::rocm_ck::fp8e5m2OcpToFloat;
using ::rocm_ck::toleranceFor;
using ::rocm_ck::typedToFloat;

// ============================================================================
// BF16 conversions
// ============================================================================

TEST(BF16Convert, RoundtripExactValues)
{
    EXPECT_FLOAT_EQ(bf16BitsToFloat(floatToBf16Bits(1.0f)), 1.0f);
    EXPECT_FLOAT_EQ(bf16BitsToFloat(floatToBf16Bits(0.0f)), 0.0f);
    EXPECT_FLOAT_EQ(bf16BitsToFloat(floatToBf16Bits(-1.0f)), -1.0f);
}

TEST(BF16Convert, PreservesSign)
{
    std::uint16_t pos_bits = floatToBf16Bits(1.0f);
    std::uint16_t neg_bits = floatToBf16Bits(-1.0f);
    EXPECT_EQ(pos_bits & 0x8000, 0);
    EXPECT_NE(neg_bits & 0x8000, 0);
}

TEST(BF16Convert, RoundsToNearestEven)
{
    // Value that requires rounding — test tie-breaking
    float f            = 1.0f + 0x1.004p-7f; // Between two BF16 representable values
    std::uint16_t bits = floatToBf16Bits(f);
    float recovered    = bf16BitsToFloat(bits);
    EXPECT_NEAR(recovered, f, 1e-2f);
}

TEST(BF16Convert, HandlesInfinity)
{
    float pos_inf = std::numeric_limits<float>::infinity();
    float neg_inf = -std::numeric_limits<float>::infinity();

    std::uint16_t pos_bits = floatToBf16Bits(pos_inf);
    std::uint16_t neg_bits = floatToBf16Bits(neg_inf);

    // BF16 infinity: exp=0xFF, mant=0 → bits = 0x7F80 or 0xFF80
    EXPECT_EQ(pos_bits, 0x7F80);
    EXPECT_EQ(neg_bits, 0xFF80);
}

TEST(BF16Convert, PreservesZero)
{
    EXPECT_EQ(floatToBf16Bits(0.0f), 0);
    EXPECT_EQ(floatToBf16Bits(-0.0f), 0x8000);
}

// ============================================================================
// FP8 E4M3 FNUZ conversions
// ============================================================================

TEST(FP8_FNUZ, RoundtripSmallValues)
{
    float f           = 1.0f;
    std::uint8_t bits = floatToFp8e4m3Fnuz(f);
    float recovered   = fp8e4m3FnuzToFloat(bits);
    EXPECT_FLOAT_EQ(recovered, f);
}

TEST(FP8_FNUZ, SaturatesAtMax)
{
    // FP8 E4M3 FNUZ max ≈ 240.0f
    float overflow    = 300.0f;
    std::uint8_t bits = floatToFp8e4m3Fnuz(overflow);
    float recovered   = fp8e4m3FnuzToFloat(bits);
    EXPECT_LE(recovered, 240.0f);
}

TEST(FP8_FNUZ, HandlesZero)
{
    EXPECT_EQ(floatToFp8e4m3Fnuz(0.0f), 0);
    EXPECT_EQ(floatToFp8e4m3Fnuz(-0.0f), 0); // FNUZ: no negative zero
    EXPECT_FLOAT_EQ(fp8e4m3FnuzToFloat(0), 0.0f);
}

TEST(FP8_FNUZ, HandlesNaN)
{
    float nan         = std::numeric_limits<float>::quiet_NaN();
    std::uint8_t bits = floatToFp8e4m3Fnuz(nan);
    EXPECT_EQ(bits, 0x80); // FNUZ NaN encoding
    EXPECT_TRUE(std::isnan(fp8e4m3FnuzToFloat(0x80)));
}

TEST(FP8_FNUZ, RoundtripNegativeValues)
{
    float f           = -1.0f;
    std::uint8_t bits = floatToFp8e4m3Fnuz(f);
    float recovered   = fp8e4m3FnuzToFloat(bits);
    EXPECT_FLOAT_EQ(recovered, f);
}

// ============================================================================
// BF8 E5M2 FNUZ conversions
// ============================================================================

TEST(BF8_FNUZ, RoundtripSmallValues)
{
    float f           = 1.0f;
    std::uint8_t bits = floatToFp8e5m2Fnuz(f);
    float recovered   = fp8e5m2FnuzToFloat(bits);
    EXPECT_FLOAT_EQ(recovered, f);
}

TEST(BF8_FNUZ, SaturatesAtMax)
{
    // BF8 E5M2 FNUZ max = 57344.0f
    float overflow    = 100000.0f;
    std::uint8_t bits = floatToFp8e5m2Fnuz(overflow);
    float recovered   = fp8e5m2FnuzToFloat(bits);
    EXPECT_LE(recovered, 57344.0f);
}

TEST(BF8_FNUZ, HandlesZero)
{
    EXPECT_EQ(floatToFp8e5m2Fnuz(0.0f), 0);
    EXPECT_EQ(floatToFp8e5m2Fnuz(-0.0f), 0); // FNUZ: no negative zero
    EXPECT_FLOAT_EQ(fp8e5m2FnuzToFloat(0), 0.0f);
}

TEST(BF8_FNUZ, HandlesNaN)
{
    float nan         = std::numeric_limits<float>::quiet_NaN();
    std::uint8_t bits = floatToFp8e5m2Fnuz(nan);
    EXPECT_EQ(bits, 0x80); // FNUZ NaN encoding
    EXPECT_TRUE(std::isnan(fp8e5m2FnuzToFloat(0x80)));
}

// ============================================================================
// FP8 E4M3 OCP conversions
// ============================================================================

TEST(FP8_OCP, RoundtripSmallValues)
{
    float f           = 1.0f;
    std::uint8_t bits = floatToFp8e4m3Ocp(f);
    float recovered   = fp8e4m3OcpToFloat(bits);
    EXPECT_FLOAT_EQ(recovered, f);
}

TEST(FP8_OCP, SaturatesAtMax)
{
    // FP8 E4M3 OCP max = 448.0f
    float overflow    = 500.0f;
    std::uint8_t bits = floatToFp8e4m3Ocp(overflow);
    float recovered   = fp8e4m3OcpToFloat(bits);
    EXPECT_LE(recovered, 448.0f);
}

TEST(FP8_OCP, PreservesNegativeZero)
{
    EXPECT_EQ(floatToFp8e4m3Ocp(0.0f), 0);
    EXPECT_EQ(floatToFp8e4m3Ocp(-0.0f), 0x80); // OCP: has negative zero
    EXPECT_FLOAT_EQ(fp8e4m3OcpToFloat(0x80), -0.0f);
}

TEST(FP8_OCP, HandlesNaN)
{
    float nan         = std::numeric_limits<float>::quiet_NaN();
    std::uint8_t bits = floatToFp8e4m3Ocp(nan);
    // OCP E4M3 NaN: exp=15, mant=7 → 0x7F (positive) or 0xFF (negative)
    EXPECT_TRUE((bits & 0x7F) == 0x7F);
    EXPECT_TRUE(std::isnan(fp8e4m3OcpToFloat(0x7F)));
}

// ============================================================================
// BF8 E5M2 OCP conversions
// ============================================================================

TEST(BF8_OCP, RoundtripSmallValues)
{
    float f           = 1.0f;
    std::uint8_t bits = floatToFp8e5m2Ocp(f);
    float recovered   = fp8e5m2OcpToFloat(bits);
    EXPECT_FLOAT_EQ(recovered, f);
}

TEST(BF8_OCP, SaturatesAtMax)
{
    // BF8 E5M2 OCP max = 57344.0f
    float overflow    = 100000.0f;
    std::uint8_t bits = floatToFp8e5m2Ocp(overflow);
    float recovered   = fp8e5m2OcpToFloat(bits);
    EXPECT_LE(recovered, 57344.0f);
}

TEST(BF8_OCP, PreservesNegativeZero)
{
    EXPECT_EQ(floatToFp8e5m2Ocp(0.0f), 0);
    EXPECT_EQ(floatToFp8e5m2Ocp(-0.0f), 0x80); // OCP: has negative zero
    EXPECT_FLOAT_EQ(fp8e5m2OcpToFloat(0x80), -0.0f);
}

TEST(BF8_OCP, HandlesInfinity)
{
    float pos_inf     = std::numeric_limits<float>::infinity();
    std::uint8_t bits = floatToFp8e5m2Ocp(pos_inf);
    // OCP E5M2 Inf: exp=31, mant=0 → 0x7C
    EXPECT_TRUE(std::isinf(fp8e5m2OcpToFloat(0x7C)));
}

TEST(BF8_OCP, HandlesNaN)
{
    float nan         = std::numeric_limits<float>::quiet_NaN();
    std::uint8_t bits = floatToFp8e5m2Ocp(nan);
    // OCP E5M2 NaN: exp=31, mant≠0 → 0x7D, 0x7E, 0x7F
    EXPECT_TRUE((bits & 0x7C) == 0x7C && (bits & 0x03) != 0);
    EXPECT_TRUE(std::isnan(fp8e5m2OcpToFloat(0x7F)));
}

// ============================================================================
// floatToTyped / typedToFloat dispatch
// ============================================================================

TEST(TypedConvert, DispatchesFP32)
{
    float buf;
    floatToTyped(DataType::FP32, 3.14f, &buf);
    EXPECT_FLOAT_EQ(buf, 3.14f);
    EXPECT_FLOAT_EQ(typedToFloat(DataType::FP32, &buf), 3.14f);
}

TEST(TypedConvert, DispatchesFP64)
{
    double buf;
    floatToTyped(DataType::FP64, 2.718f, &buf);
    EXPECT_NEAR(buf, 2.718, 1e-3);
    EXPECT_FLOAT_EQ(typedToFloat(DataType::FP64, &buf), 2.718f);
}

TEST(TypedConvert, DispatchesFP16)
{
    _Float16 buf;
    floatToTyped(DataType::FP16, 1.5f, &buf);
    EXPECT_FLOAT_EQ(typedToFloat(DataType::FP16, &buf), 1.5f);
}

TEST(TypedConvert, DispatchesBF16)
{
    std::uint16_t buf;
    floatToTyped(DataType::BF16, 1.0f, &buf);
    EXPECT_EQ(buf, floatToBf16Bits(1.0f));
    EXPECT_FLOAT_EQ(typedToFloat(DataType::BF16, &buf), 1.0f);
}

TEST(TypedConvert, DispatchesFP8_FNUZ)
{
    std::uint8_t buf;
    floatToTyped(DataType::FP8_FNUZ, 1.0f, &buf);
    EXPECT_EQ(buf, floatToFp8e4m3Fnuz(1.0f));
    EXPECT_FLOAT_EQ(typedToFloat(DataType::FP8_FNUZ, &buf), 1.0f);
}

TEST(TypedConvert, DispatchesBF8_FNUZ)
{
    std::uint8_t buf;
    floatToTyped(DataType::BF8_FNUZ, 1.0f, &buf);
    EXPECT_EQ(buf, floatToFp8e5m2Fnuz(1.0f));
    EXPECT_FLOAT_EQ(typedToFloat(DataType::BF8_FNUZ, &buf), 1.0f);
}

TEST(TypedConvert, DispatchesFP8_OCP)
{
    std::uint8_t buf;
    floatToTyped(DataType::FP8_OCP, 1.0f, &buf);
    EXPECT_EQ(buf, floatToFp8e4m3Ocp(1.0f));
    EXPECT_FLOAT_EQ(typedToFloat(DataType::FP8_OCP, &buf), 1.0f);
}

TEST(TypedConvert, DispatchesBF8_OCP)
{
    std::uint8_t buf;
    floatToTyped(DataType::BF8_OCP, 1.0f, &buf);
    EXPECT_EQ(buf, floatToFp8e5m2Ocp(1.0f));
    EXPECT_FLOAT_EQ(typedToFloat(DataType::BF8_OCP, &buf), 1.0f);
}

TEST(TypedConvert, DispatchesI8)
{
    std::int8_t buf;
    floatToTyped(DataType::I8, 42.7f, &buf);
    EXPECT_EQ(buf, 43); // rounds to nearest
    EXPECT_FLOAT_EQ(typedToFloat(DataType::I8, &buf), 43.0f);
}

TEST(TypedConvert, DispatchesI32)
{
    std::int32_t buf;
    floatToTyped(DataType::I32, -123.4f, &buf);
    EXPECT_EQ(buf, -123);
    EXPECT_FLOAT_EQ(typedToFloat(DataType::I32, &buf), -123.0f);
}

TEST(TypedConvert, ClampsI8Overflow)
{
    std::int8_t buf;
    floatToTyped(DataType::I8, 200.0f, &buf);
    EXPECT_EQ(buf, 127); // clamps to max
}

TEST(TypedConvert, ClampsI8Underflow)
{
    std::int8_t buf;
    floatToTyped(DataType::I8, -200.0f, &buf);
    EXPECT_EQ(buf, -128); // clamps to min
}

// ============================================================================
// toleranceFor
// ============================================================================

TEST(ToleranceFor, ReturnsSensibleValuesForFloatingPoint)
{
    EXPECT_GT(toleranceFor(DataType::FP64), 0.0f);
    EXPECT_GT(toleranceFor(DataType::FP32), 0.0f);
    EXPECT_GT(toleranceFor(DataType::FP16), 0.0f);
    EXPECT_GT(toleranceFor(DataType::BF16), 0.0f);
}

TEST(ToleranceFor, ReturnsCalibratedValuesForFP8)
{
    // E4M3: 3 mantissa bits → ULP at 1.0 = 2^-3 = 0.125
    EXPECT_FLOAT_EQ(toleranceFor(DataType::FP8_FNUZ), 0.125f);
    EXPECT_FLOAT_EQ(toleranceFor(DataType::FP8_OCP), 0.125f);
}

TEST(ToleranceFor, ReturnsCalibratedValuesForBF8)
{
    // E5M2: 2 mantissa bits → ULP at 1.0 = 2^-2 = 0.25
    EXPECT_FLOAT_EQ(toleranceFor(DataType::BF8_FNUZ), 0.25f);
    EXPECT_FLOAT_EQ(toleranceFor(DataType::BF8_OCP), 0.25f);
}

TEST(ToleranceFor, ReturnsZeroForIntegers)
{
    EXPECT_FLOAT_EQ(toleranceFor(DataType::I8), 0.0f);
    EXPECT_FLOAT_EQ(toleranceFor(DataType::I32), 0.0f);
}

TEST(ToleranceFor, OrdersTolerancesByPrecision)
{
    // More precise types should have tighter tolerances
    EXPECT_LT(toleranceFor(DataType::FP64), toleranceFor(DataType::FP32));
    EXPECT_LT(toleranceFor(DataType::FP32), toleranceFor(DataType::FP16));
    EXPECT_LT(toleranceFor(DataType::FP16), toleranceFor(DataType::BF16));
    EXPECT_LT(toleranceFor(DataType::BF16), toleranceFor(DataType::FP8_FNUZ));
    EXPECT_LT(toleranceFor(DataType::FP8_FNUZ), toleranceFor(DataType::BF8_FNUZ));
}

// ============================================================================
// detail::clz (if publicly testable via namespace access)
// ============================================================================

TEST(DetailClz, CountsLeadingZeros)
{
    EXPECT_EQ(rocm_ck::detail::clz(0), 32);
    EXPECT_EQ(rocm_ck::detail::clz(1), 31);
    EXPECT_EQ(rocm_ck::detail::clz(0x80000000u), 0);
    EXPECT_EQ(rocm_ck::detail::clz(0x00FF0000u), 8);
    EXPECT_EQ(rocm_ck::detail::clz(0x00000001u), 31);
}
