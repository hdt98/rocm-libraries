// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file TestFp8E4M3.cpp
/// @brief Type-specific tests for fp8_e4m3 (OCP E4M3 format).
///
/// Common tests (type properties, construction, math functions, numeric_limits) are in
/// TestPortableTypes.cpp. This file covers exact-value round-trip, exhaustive 256-bit-pattern
/// round-trip, OCP E4M3-essential semantics (dual NaN encoding at 0x7F/0xFF, no infinity,
/// saturate-to-MAX, signed zero), rounding (parametrized), saturation, underflow, and
/// numeric_limits checks.

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace hipdnn_data_sdk::types;

// ============================================================================
// fp8_e4m3 Type-Specific Tests
// ============================================================================

TEST(TestFp8E4M3, RoundTripExactValues)
{
    // These values are exactly representable in fp8_e4m3 (OCP E4M3 format)
    const std::vector<float> exactValues = {
        0.0f,          1.0f,  -1.0f, 2.0f,  -2.0f, 4.0f,   -4.0f,
        8.0f,          -8.0f, 0.5f,  -0.5f, 0.25f, -0.25f,
        448.0f, // MAX (0x7E)
        -448.0f, // LOWEST (0xFE)
        0.015625f, // MIN_NORMAL = 2^-6 (0x08)
        -0.015625f,
        0.001953125f, // denorm_min = 2^-9 (0x01)
        -0.001953125f,
    };

    for(const float val : exactValues)
    {
        const fp8_e4m3 v(val);
        EXPECT_EQ(static_cast<float>(v), val) << "Round-trip failed for " << val;
    }
}

// ============================================================================
// Rounding Tests
// ============================================================================

namespace
{
struct RoundingTestCase
{
    float input;
    float expected;

    friend std::ostream& operator<<(std::ostream& os, const RoundingTestCase& tc)
    {
        return os << tc.input << " -> " << tc.expected;
    }
};
} // namespace

class TestFp8E4M3Rounding : public ::testing::TestWithParam<RoundingTestCase>
{
};

TEST_P(TestFp8E4M3Rounding, Rounding)
{
    auto [input, expected] = GetParam();
    const fp8_e4m3 val(input);
    EXPECT_EQ(static_cast<float>(val), expected);
}

// Midpoint values that are exactly halfway between two representable fp8_e4m3
// encodings; round-to-nearest-even selects the encoding with an even mantissa.
// OCP E4M3 normal-range mantissa step: for exp-field e, step = 2^(e-7) / 8.
// Midpoint between value(e,m) and value(e,m+1) = value(e,m) + step/2.
//
// Lookup table selected examples:
//   0x40=(1.0, mant=0 even), 0x41=(1.125, mant=1 odd), 0x42=(1.25, mant=2 even)
//   0x48=(2.0, mant=0 even), 0x49=(2.25, mant=1 odd), 0x4A=(2.5, mant=2 even)
//   0x7E=(448.0, mant=6 even) is MAX; midpoint to phantom 512 is 480 -> round to even (448)
INSTANTIATE_TEST_SUITE_P(
    MidpointRounding,
    TestFp8E4M3Rounding,
    ::testing::Values(
        // 0x40 (mant=0, even) vs 0x41 (mant=1, odd): midpoint = 1.0625 -> round down -> 1.0
        RoundingTestCase{1.0625f, 1.0f},
        // 0x41 (mant=1, odd) vs 0x42 (mant=2, even): midpoint = 1.1875 -> round up -> 1.25
        RoundingTestCase{1.1875f, 1.25f},
        // 0x48 (mant=0, even) vs 0x49 (mant=1, odd): midpoint = 2.125 -> round down -> 2.0
        RoundingTestCase{2.125f, 2.0f},
        // 0x49 (mant=1, odd) vs 0x4A (mant=2, even): midpoint = 2.375 -> round up -> 2.5
        RoundingTestCase{2.375f, 2.5f},
        // 0x7E (mant=6, even) vs phantom 0x80-space: midpoint = 480.0 -> round down to MAX=448.0
        // Midpoint formula: MAX + (phantom_next - MAX)/2 = 448 + (512-448)/2 = 448 + 32 = 480
        RoundingTestCase{480.0f, 448.0f}));

// Values in the subnormal / underflow range.
// OCP E4M3 denorms: value = mant * 2^-9 (mant=1..7). denorm_min = 2^-9 ≈ 0.001953125.
// Underflow threshold = half of denorm_min = 2^-10 ≈ 9.765625e-4.
// At exactly 2^-10 the result rounds to 0 (even mantissa tie-break).
// fp32 subnormal inputs (fp32Exp==0) flush directly to zero regardless of value.
INSTANTIATE_TEST_SUITE_P(
    SubnormalRounding,
    TestFp8E4M3Rounding,
    ::testing::Values(
        // Well below underflow threshold (2^-10): flushes to zero
        RoundingTestCase{1e-4f, 0.0f},
        // At exactly the underflow threshold 2^-10: round to even -> zero
        RoundingTestCase{9.765625e-4f, 0.0f},
        // Slightly above 2^-10: rounds to denorm_min=0x01=2^-9
        RoundingTestCase{1.0e-3f, 0.001953125f},
        // Exactly denorm_min = 2^-9 = 0.001953125: encodes exactly as 0x01
        RoundingTestCase{0.001953125f, 0.001953125f},
        // Midpoint between denorm 0x01 (mant=1 odd) and 0x02 (mant=2 even): rounds up to 0x02
        // Midpoint = (0.001953125 + 0.00390625) / 2 = 0.0029296875
        RoundingTestCase{0.0029296875f, 0.00390625f},
        // Very small: underflows to zero
        RoundingTestCase{1e-10f, 0.0f},
        // Negative: same underflow to zero
        RoundingTestCase{-1e-4f, 0.0f},
        // Negative subnormal: -0.001953125 encodes to -0x01
        RoundingTestCase{-0.001953125f, -0.001953125f},
        // [Gap 2] Even-LSB subnormal midpoint: 0x02 (mant=2, even) vs 0x03 (mant=3, odd).
        // Midpoint = (0.00390625 + 0.005859375) / 2 = 0.0048828125.
        // RNE ties-to-even: lower candidate 0x02 has even LSB → round DOWN to 0x02=0.00390625.
        // Code path: shift=22, halfPoint=0x200000, remainder=0x200000, fp8Mant=2 (even) → no-op.
        RoundingTestCase{0.0048828125f, 0.00390625f},
        // [Gap 3] Denormal-to-normal carry: bracket below the carry midpoint.
        // 0x07=0.013671875 (mant=7, odd), 0x08=0.015625 (smallest normal, exp=1, mant=0).
        // Midpoint = 0.0146484375. Below midpoint: rounds down to 0x07=0.013671875.
        // Input 0.01416015625 = 29*2^-11 = 1.8125*2^-7: fp8Mant=7, remainder=0x080000 < halfPoint
        // 0x100000 → no rounding → 0x07=0.013671875.
        RoundingTestCase{0.01416015625f, 0.013671875f},
        // [Gap 3] Exact carry midpoint: 0.0146484375 = 7.5*2^-9 = 1.111*2^-7.
        // subnormal path: shift=21, halfPoint=0x100000, remainder=0x100000=halfPoint, fp8Mant=7
        // (odd) → round up → fp8Mant=8 > MANT_MASK → carry into smallest normal 0x08=0.015625.
        RoundingTestCase{0.0146484375f, 0.015625f},
        // [Gap 3] Above carry midpoint: definitively rounds up to smallest normal.
        // Input 0.01513671875 = 31*2^-11 = 1.9375*2^-7: fp8Mant=7, remainder=0x180000 > halfPoint
        // 0x100000 → round up → carry → 0x08=0.015625.
        RoundingTestCase{0.01513671875f, 0.015625f}));

// Values at the boundary between two adjacent encodings (not at exact midpoints).
// Verifies correct rounding direction in the non-tie case.
INSTANTIATE_TEST_SUITE_P(BoundaryRounding,
                         TestFp8E4M3Rounding,
                         ::testing::Values(
                             // Slightly above 1.0 (0x40): closer to 1.0 than 1.125 -> 1.0
                             RoundingTestCase{1.06f, 1.0f},
                             // Slightly above midpoint 1.0625: closer to 1.125 -> 1.125
                             RoundingTestCase{1.07f, 1.125f},
                             // Between 416.0 (0x7D) and 448.0 (0x7E): closer to 416 -> 416
                             RoundingTestCase{420.0f, 416.0f},
                             // Between 416.0 and 448.0: closer to 448 -> 448
                             RoundingTestCase{440.0f, 448.0f},
                             // Negative boundary
                             RoundingTestCase{-1.06f, -1.0f}));

// Rounding where mantissa overflow from the round-up cascades into the exponent,
// or where the final exponent overflows into saturation.
//
// OCP E4M3 mant bits = 3; when fp8Mant rounds from 7 to 8 (overflow), exp increments.
// If exp reaches > 15 (MAX_EXP), the value saturates.
// The saturation midpoint: MAX=448 (exp=15, mant=6), phantom next = 512 (exp=16 = overflow).
// Midpoint = (448+512)/2 = 480.0; mant at MAX is 6 (even) -> round down to MAX=448.
//
// For the odd-mantissa saturation case: 480.0 is the midpoint; a value of 480.0 rounds
// to 448 (even). But 480.0 + eps would round to 448 as well (still below 512 by more
// than half). Values >= 480.0 that round up: the midpoint with odd mant is handled
// via the carry-into-saturation path.
INSTANTIATE_TEST_SUITE_P(
    RoundingOverflow,
    TestFp8E4M3Rounding,
    ::testing::Values(
        // Mantissa overflow: exp increments, stays in range
        // 0x47 (mant=7, odd) = 1.875; midpoint to 0x48=2.0 -> mant overflow -> 2.0
        RoundingTestCase{1.9375f, 2.0f},
        // 0x4F (mant=7, odd) = 3.75; midpoint to 0x50=4.0 -> mant overflow -> 4.0
        RoundingTestCase{3.875f, 4.0f},
        // Exponent overflow into saturation:
        // 480.0 is the midpoint between MAX=448 and phantom 512 (mant=6, even -> round down)
        RoundingTestCase{480.0f, 448.0f},
        // Just below 480.0: rounds to MAX
        RoundingTestCase{449.0f, 448.0f},
        // Negative counterparts
        RoundingTestCase{-1.9375f, -2.0f},
        RoundingTestCase{-3.875f, -4.0f},
        RoundingTestCase{-480.0f, -448.0f},
        // NaN-zone collision (Fp8E4M3.hpp:199-210): inputs inside [480, 488) encode normally to
        // (exp=15, mant=7) = bit pattern 0x7F, which is the OCP E4M3 NaN reserved encoding.
        // The post-encoding check detects this and saturates to MAX=448.0 (0x7E) instead.
        // 481.0f: fp32Exp=135 → E4M3 exp=15, fp8Mant=(0x708000>>20)&7=7 → NaN zone → 448.0
        RoundingTestCase{481.0f, 448.0f},
        // Negative mirror: -481.0f → bit pattern would be 0xFF (negative NaN) → saturate to -448.0
        RoundingTestCase{-481.0f, -448.0f}));

// NaN-zone collision with saturate=false: inputs in [480, 488) that encode to (exp=15, mant=7)
// must return NaN (0x7F / 0xFF) rather than a finite value, since saturate=false means "do not
// clamp to MAX; use the natural NaN encoding instead".
// 481.0f has fp8Mant=7 and exp=15 → float_to_fp8_e4m3_bits returns 0x7F → isnan is true.
TEST(TestFp8E4M3, NanZoneCollisionSaturateFalse)
{
    // Positive input in the NaN zone: saturate=false returns 0x7F (positive NaN)
    const uint8_t posBits = detail::float_to_fp8_e4m3_bits(481.0f, /*saturate=*/false);
    EXPECT_EQ(posBits, static_cast<uint8_t>(0x7F));
    EXPECT_TRUE(std::isnan(detail::fp8_e4m3_bits_to_float(posBits)));

    // Negative input in the NaN zone: saturate=false returns 0xFF (negative NaN)
    const uint8_t negBits = detail::float_to_fp8_e4m3_bits(-481.0f, /*saturate=*/false);
    EXPECT_EQ(negBits, static_cast<uint8_t>(0xFF));
    EXPECT_TRUE(std::isnan(detail::fp8_e4m3_bits_to_float(negBits)));
}

// ============================================================================
// Saturation Tests
// ============================================================================

TEST(TestFp8E4M3, SaturationPositive)
{
    // Values beyond 448 saturate to MAX=0x7E=448.0 when saturate=true (default)
    const fp8_e4m3 val1(1e6f);
    EXPECT_EQ(static_cast<float>(val1), 448.0f);

    const fp8_e4m3 val2(449.0f);
    EXPECT_EQ(static_cast<float>(val2), 448.0f);

    const fp8_e4m3 val3(896.0f);
    EXPECT_EQ(static_cast<float>(val3), 448.0f);

    const fp8_e4m3 val4(std::nextafter(448.0f, 1e9f));
    EXPECT_EQ(static_cast<float>(val4), 448.0f);

    // Rounding-into-saturation: 480.0 midpoint with mant=6 (even) -> round down to MAX=448
    const fp8_e4m3 val5(480.0f);
    EXPECT_EQ(static_cast<float>(val5), 448.0f);

    // saturate=false: values beyond 448 return NaN (0x7F), since E4M3 OCP has no Inf
    EXPECT_EQ(detail::float_to_fp8_e4m3_bits(1000.0f, /*saturate=*/false),
              static_cast<uint8_t>(0x7F));
    EXPECT_EQ(detail::float_to_fp8_e4m3_bits(-1000.0f, /*saturate=*/false),
              static_cast<uint8_t>(0xFF));
}

TEST(TestFp8E4M3, SaturationNegative)
{
    const fp8_e4m3 val1(-1e6f);
    EXPECT_EQ(static_cast<float>(val1), -448.0f);

    const fp8_e4m3 val2(-449.0f);
    EXPECT_EQ(static_cast<float>(val2), -448.0f);

    const fp8_e4m3 val3(-896.0f);
    EXPECT_EQ(static_cast<float>(val3), -448.0f);

    // Rounding-into-saturation: -480.0 mirrors the positive case above
    const fp8_e4m3 val4(-480.0f);
    EXPECT_EQ(static_cast<float>(val4), -448.0f);
}

TEST(TestFp8E4M3, SaturationInfinity)
{
    // +Inf with saturate=true (default) -> MAX=448.0
    const fp8_e4m3 posInf(std::numeric_limits<float>::infinity());
    EXPECT_EQ(static_cast<float>(posInf), 448.0f);

    // -Inf with saturate=true -> LOWEST=-448.0
    const fp8_e4m3 negInf(-std::numeric_limits<float>::infinity());
    EXPECT_EQ(static_cast<float>(negInf), -448.0f);

    // +Inf with saturate=false -> NaN (0x7F)
    EXPECT_EQ(detail::float_to_fp8_e4m3_bits(std::numeric_limits<float>::infinity(),
                                             /*saturate=*/false),
              static_cast<uint8_t>(0x7F));

    // -Inf with saturate=false -> negative NaN (0xFF)
    EXPECT_EQ(detail::float_to_fp8_e4m3_bits(-std::numeric_limits<float>::infinity(),
                                             /*saturate=*/false),
              static_cast<uint8_t>(0xFF));
}

// ============================================================================
// Underflow Tests
// ============================================================================

TEST(TestFp8E4M3, Underflow)
{
    // Very small positive values flush to positive zero
    const fp8_e4m3 val1(1e-10f);
    EXPECT_EQ(static_cast<float>(val1), 0.0f);

    const fp8_e4m3 val2(-1e-10f);
    EXPECT_EQ(static_cast<float>(val2), 0.0f);

    // fp32 denormal inputs flush to zero (fp32Exp==0 path)
    const fp8_e4m3 val3(std::numeric_limits<float>::denorm_min());
    EXPECT_EQ(static_cast<float>(val3), 0.0f);

    const fp8_e4m3 val4(-std::numeric_limits<float>::denorm_min());
    EXPECT_EQ(static_cast<float>(val4), 0.0f);

    // The underflow threshold for OCP E4M3 is 2^-10 (half of denorm_min=2^-9).
    // At exactly 2^-10: rounds to zero (even mantissa tie-break, fp8Mant=0 is even)
    const fp8_e4m3 val5(9.765625e-4f); // 2^-10
    EXPECT_EQ(static_cast<float>(val5), 0.0f);

    // Negative signed zero: OCP E4M3 supports -0.0
    const fp8_e4m3 negZero(-0.0f);
    EXPECT_EQ(negZero.data, static_cast<uint8_t>(0x80)); // negative zero bit pattern
    EXPECT_EQ(static_cast<float>(negZero), -0.0f);
}

// ============================================================================
// E4M3 OCP-specific: NaN, Infinity, Bit-Pattern Round-Trip
// ============================================================================

TEST(TestFp8E4M3, RoundTripAllPatterns)
{
    // For every bit pattern: decode to float via the lookup table, re-encode back,
    // and verify that the resulting bit pattern is identical.
    // OCP E4M3 NaN encodings: 0x7F (positive NaN) and 0xFF (negative NaN).
    // Re-encoding a float NaN returns 0x7F or 0xFF by sign preservation.
    for(int bits = 0; bits <= 0xFF; ++bits)
    {
        const auto pattern = static_cast<uint8_t>(bits);
        const auto decoded = fp8_e4m3::from_bits(pattern);
        const auto f = static_cast<float>(decoded);

        // NaN round-trip: 0x7F decodes to NaN; re-encoding any NaN returns 0x7F or 0xFF
        if(pattern == detail::FP8_E4M3_NAN || pattern == static_cast<uint8_t>(0xFF))
        {
            EXPECT_TRUE(std::isnan(f))
                << "Pattern 0x" << std::hex << bits << " should decode to NaN";
            // Re-encode NaN: sign bit is preserved (0x7F -> 0x7F, 0xFF -> 0xFF)
            const fp8_e4m3 reencoded(f);
            EXPECT_EQ(reencoded.data, pattern)
                << "NaN round-trip failed for 0x" << std::hex << bits;
            continue;
        }

        // All non-NaN patterns must survive the round-trip exactly.
        const fp8_e4m3 reencoded(f);
        EXPECT_EQ(reencoded.data, pattern) << "Round-trip failed for bit pattern 0x" << std::hex
                                           << bits << " (float value " << std::dec << f << ")";
    }
}

TEST(TestFp8E4M3, IsinfAlwaysFalse)
{
    // No bit pattern in fp8_e4m3 represents infinity (OCP E4M3 has no Inf)
    for(int bits = 0; bits <= 0xFF; ++bits)
    {
        EXPECT_FALSE(isinf(fp8_e4m3::from_bits(static_cast<uint8_t>(bits))))
            << "Bit pattern 0x" << std::hex << bits << " should not be infinity";
    }
}

TEST(TestFp8E4M3, MathFunctions)
{
    // isfinite: all values except NaN are finite
    EXPECT_TRUE(isfinite(fp8_e4m3(1.0f)));
    EXPECT_TRUE(isfinite(fp8_e4m3(0.0f)));
    EXPECT_TRUE(isfinite(fp8_e4m3(448.0f)));
    EXPECT_FALSE(isfinite(fp8_e4m3::from_bits(detail::FP8_E4M3_NAN))); // 0x7F = NaN

    // isnan: OCP E4M3 has NaN at 0x7F and 0xFF
    EXPECT_TRUE(isnan(fp8_e4m3::from_bits(static_cast<uint8_t>(0x7F))));
    EXPECT_TRUE(isnan(fp8_e4m3::from_bits(static_cast<uint8_t>(0xFF))));
    EXPECT_FALSE(isnan(fp8_e4m3(1.0f)));
    EXPECT_FALSE(isnan(fp8_e4m3(0.0f)));

    // isinf: always false for OCP E4M3
    EXPECT_FALSE(isinf(fp8_e4m3(1.0f)));
    EXPECT_FALSE(isinf(fp8_e4m3::from_bits(static_cast<uint8_t>(0x7F))));

    // signbit: positive values have sign bit 0, negative have sign bit 1
    EXPECT_FALSE(signbit(fp8_e4m3(1.0f)));
    EXPECT_TRUE(signbit(fp8_e4m3(-1.0f)));
    EXPECT_FALSE(signbit(fp8_e4m3(0.0f)));
    EXPECT_TRUE(signbit(fp8_e4m3(-0.0f)));
    // OCP negative NaN (0xFF) has sign bit set
    EXPECT_TRUE(signbit(fp8_e4m3::from_bits(static_cast<uint8_t>(0xFF))));
    // OCP positive NaN (0x7F) does not have sign bit set
    EXPECT_FALSE(signbit(fp8_e4m3::from_bits(static_cast<uint8_t>(0x7F))));
}

// ============================================================================
// Specific numeric_limits Value Tests
// ============================================================================

TEST(TestFp8E4M3, NumericLimitsSpecificValues)
{
    using L = std::numeric_limits<fp8_e4m3>;

    // max() = 448.0 (0x7E)
    EXPECT_EQ(static_cast<float>(L::max()), 448.0f);
    EXPECT_EQ(L::max().data, static_cast<uint8_t>(0x7E));

    // min() = 2^-6 = 0.015625 (0x08, smallest positive normal)
    EXPECT_EQ(static_cast<float>(L::min()), 0.015625f);
    EXPECT_EQ(L::min().data, static_cast<uint8_t>(0x08));

    // lowest() = -448.0 (0xFE)
    EXPECT_EQ(static_cast<float>(L::lowest()), -448.0f);
    EXPECT_EQ(L::lowest().data, static_cast<uint8_t>(0xFE));

    // epsilon() = 2^-3 = 0.125 (0x20, 3 mantissa bits)
    EXPECT_EQ(static_cast<float>(L::epsilon()), 0.125f);
    EXPECT_EQ(L::epsilon().data, static_cast<uint8_t>(0x20));

    // round_error() = 0.5 (0x30)
    EXPECT_EQ(static_cast<float>(L::round_error()), 0.5f);
    EXPECT_EQ(L::round_error().data, static_cast<uint8_t>(0x30));

    // denorm_min() = 2^-9 ≈ 0.001953125 (0x01, smallest positive subnormal)
    EXPECT_EQ(L::denorm_min().data, static_cast<uint8_t>(0x01));
    EXPECT_EQ(static_cast<float>(L::denorm_min()), 0.001953125f);

    // infinity() returns max() since OCP E4M3 has no infinity
    EXPECT_EQ(L::infinity().data, L::max().data);

    // quiet_NaN() and signaling_NaN() both return 0x7F (positive NaN)
    EXPECT_EQ(L::quiet_NaN().data, static_cast<uint8_t>(0x7F));
    EXPECT_EQ(L::signaling_NaN().data, static_cast<uint8_t>(0x7F));

    // Boolean fields
    EXPECT_FALSE(L::has_infinity);
    EXPECT_TRUE(L::has_quiet_NaN);
    EXPECT_FALSE(L::has_signaling_NaN);
    EXPECT_EQ(L::has_denorm, std::denorm_present);
    EXPECT_EQ(L::round_style, std::round_to_nearest);
    EXPECT_EQ(L::radix, 2);
    EXPECT_EQ(L::digits, 4); // 3 mantissa + 1 implicit
    EXPECT_EQ(L::min_exponent, -5); // min_normal = 2^(min_exponent-1) = 2^-6
    EXPECT_EQ(L::max_exponent, 8); // max_normal = (2-2^-3) * 2^7 < 2^8
}

// ============================================================================
// Named Constants Tests
// ============================================================================

TEST(TestFp8E4M3, NamedConstants)
{
    // OCP E4M3 NaN encodings: 0x7F and 0xFF
    EXPECT_EQ(detail::FP8_E4M3_NAN, static_cast<uint8_t>(0x7F));
    EXPECT_EQ(detail::FP8_E4M3_MAX, static_cast<uint8_t>(0x7E));
    EXPECT_EQ(detail::FP8_E4M3_LOWEST, static_cast<uint8_t>(0xFE));
    EXPECT_EQ(detail::FP8_E4M3_MIN_NORMAL, static_cast<uint8_t>(0x08));
    EXPECT_EQ(detail::FP8_E4M3_DENORM_MIN, static_cast<uint8_t>(0x01));
    EXPECT_EQ(detail::FP8_E4M3_EPSILON, static_cast<uint8_t>(0x20));
    EXPECT_EQ(detail::FP8_E4M3_ROUND_ERROR, static_cast<uint8_t>(0x30));

    // Verify values decode to expected float values
    EXPECT_EQ(static_cast<float>(fp8_e4m3::from_bits(detail::FP8_E4M3_MAX)), 448.0f);
    EXPECT_EQ(static_cast<float>(fp8_e4m3::from_bits(detail::FP8_E4M3_MIN_NORMAL)), 0.015625f);
    EXPECT_EQ(static_cast<float>(fp8_e4m3::from_bits(detail::FP8_E4M3_DENORM_MIN)), 0.001953125f);
    EXPECT_EQ(static_cast<float>(fp8_e4m3::from_bits(detail::FP8_E4M3_EPSILON)), 0.125f);
    EXPECT_EQ(static_cast<float>(fp8_e4m3::from_bits(detail::FP8_E4M3_ROUND_ERROR)), 0.5f);

    // OCP E4M3 has no infinity; infinity() returns max()
    EXPECT_EQ(std::numeric_limits<fp8_e4m3>::infinity().data,
              std::numeric_limits<fp8_e4m3>::max().data);
    EXPECT_FALSE(std::numeric_limits<fp8_e4m3>::has_infinity);

    // NaN assertions
    EXPECT_TRUE(isnan(std::numeric_limits<fp8_e4m3>::quiet_NaN()));
    EXPECT_EQ(std::numeric_limits<fp8_e4m3>::quiet_NaN().data, static_cast<uint8_t>(0x7F));
}
