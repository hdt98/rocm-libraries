// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file TestFp8E5M2.cpp
/// @brief Type-specific tests for fp8_e5m2 (OCP E5M2 format).
///
/// Common tests (type properties, construction, math functions, numeric_limits) are in
/// TestPortableTypes.cpp. This file covers exact-value round-trip, exhaustive 256-bit-pattern
/// round-trip, OCP E5M2-essential semantics (Inf at 0x7C/0xFC, NaN when exp=0x1F&&mant!=0,
/// saturate-to-MAX vs Inf), rounding (parametrized), saturation, underflow, and
/// numeric_limits checks.

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace hipdnn_data_sdk::types;

// ============================================================================
// fp8_e5m2 Type-Specific Tests
// ============================================================================

TEST(TestFp8E5M2, RoundTripExactValues)
{
    // These values are exactly representable in fp8_e5m2 (OCP E5M2 format)
    const std::vector<float> exactValues = {
        0.0f,
        1.0f,
        -1.0f,
        2.0f,
        -2.0f,
        4.0f,
        -4.0f,
        8.0f,
        -8.0f,
        0.5f,
        -0.5f,
        0.25f,
        -0.25f,
        57344.0f, // MAX (0x7B)
        -57344.0f, // LOWEST (0xFB)
        6.103515625e-5f, // MIN_NORMAL = 2^-14 (0x04)
        -6.103515625e-5f,
        1.52587890625e-5f, // denorm_min = 2^-16 (0x01)
        -1.52587890625e-5f,
    };

    for(const float val : exactValues)
    {
        const fp8_e5m2 v(val);
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

class TestFp8E5M2Rounding : public ::testing::TestWithParam<RoundingTestCase>
{
};

TEST_P(TestFp8E5M2Rounding, Rounding)
{
    auto [input, expected] = GetParam();
    const fp8_e5m2 val(input);
    EXPECT_EQ(static_cast<float>(val), expected);
}

// Midpoint values that are exactly halfway between two representable fp8_e5m2
// encodings; round-to-nearest-even selects the encoding with an even mantissa.
// OCP E5M2 normal-range mantissa step: for exp-field e, step = 2^(e-15) / 4.
//
// Lookup table selected examples:
//   0x40=(1.0, mant=0 even), 0x41=(1.25, mant=1 odd), 0x42=(1.5, mant=2 even)
//   0x44=(2.0, mant=0 even), 0x45=(2.5, mant=1 odd)
//   0x7B=(57344.0=MAX, mant=3 odd): midpoint to Inf territory (61440) -> odd -> round up to Inf
INSTANTIATE_TEST_SUITE_P(
    MidpointRounding,
    TestFp8E5M2Rounding,
    ::testing::Values(
        // 0x40 (mant=0, even) vs 0x41 (mant=1, odd): midpoint = 1.125 -> round down -> 1.0
        RoundingTestCase{1.125f, 1.0f},
        // 0x41 (mant=1, odd) vs 0x42 (mant=2, even): midpoint = 1.375 -> round up -> 1.5
        RoundingTestCase{1.375f, 1.5f},
        // 0x44 (mant=0, even) vs 0x45 (mant=1, odd): midpoint = 2.25 -> round down -> 2.0
        RoundingTestCase{2.25f, 2.0f},
        // 0x7A (mant=2, even) vs 0x7B (mant=3, odd): midpoint = (49152+57344)/2 = 53248
        // round down to 0x7A (mant=2 even) = 49152
        RoundingTestCase{53248.0f, 49152.0f}));

// Values in the subnormal / underflow range.
// OCP E5M2 denorms: value = mant * 2^-16 (mant=1..3). denorm_min = 2^-16 = 1.52587890625e-5.
// Underflow threshold = half of denorm_min = 2^-17 = 7.62939453125e-6.
// At exactly 2^-17 the result rounds to 0 (even mantissa tie-break, fp8Mant=0 is even).
INSTANTIATE_TEST_SUITE_P(
    SubnormalRounding,
    TestFp8E5M2Rounding,
    ::testing::Values(
        // Well below underflow threshold (2^-17): flushes to zero
        RoundingTestCase{1e-6f, 0.0f},
        // At exactly the underflow threshold 2^-17: round to even -> zero
        RoundingTestCase{7.62939453125e-6f, 0.0f},
        // Slightly above 2^-17: rounds to denorm_min=0x01=2^-16
        RoundingTestCase{8e-6f, 1.52587890625e-5f},
        // Exactly denorm_min = 2^-16: encodes exactly as 0x01
        RoundingTestCase{1.52587890625e-5f, 1.52587890625e-5f},
        // Midpoint between 0x01 (mant=1 odd) and 0x02 (mant=2 even):
        // midpoint = (1.52587890625e-5 + 3.0517578125e-5) / 2 = 2.288818359375e-5
        // round up to 0x02 (mant=2 even) = 3.0517578125e-5
        RoundingTestCase{2.288818359375e-5f, 3.0517578125e-5f},
        // Very small: underflows to zero
        RoundingTestCase{1e-10f, 0.0f},
        // Negative: same underflow to zero
        RoundingTestCase{-1e-6f, 0.0f},
        // Negative subnormal: -1.52587890625e-5 encodes to -0x01
        RoundingTestCase{-1.52587890625e-5f, -1.52587890625e-5f},
        // [Gap 2] Even-LSB subnormal midpoint: 0x02 (mant=2, even) vs 0x03 (mant=3, odd).
        // Denormal step = 2^-16. Midpoint = (2*2^-16 + 3*2^-16) / 2 = 2.5*2^-16 = 3.814697265625e-5.
        // RNE ties-to-even: lower candidate 0x02 has even LSB → round DOWN to 0x02=3.0517578125e-5.
        // Code path: shift=22, halfPoint=0x200000, remainder=0x200000, fp8Mant=2 (even) → no-op.
        RoundingTestCase{3.814697265625e-5f, 3.0517578125e-5f},
        // [Gap 3] Denormal-to-normal carry: bracket below the carry midpoint.
        // 0x03=4.57763671875e-5 (mant=3, odd), 0x04=6.103515625e-5 (smallest normal, exp=1, mant=0).
        // Midpoint = 5.340576171875e-5 = 3.5*2^-16. Below midpoint: rounds to 0x03.
        // value strictly below the carry midpoint 3.5·2^-16: subnormal, shift=22, fp8Mant=3, remainder=0x100000 <
        // halfPoint 0x200000 → no rounding → 0x03=4.57763671875e-5.
        RoundingTestCase{4.96246337890625e-5f, 4.57763671875e-5f},
        // [Gap 3] Exact carry midpoint: 5.340576171875e-5 = 3.5*2^-16 = 1.75*2^-15.
        // subnormal path: shift=22, halfPoint=0x200000, remainder=0x200000=halfPoint, fp8Mant=3
        // (odd) → round up → fp8Mant=4 > MANT_MASK → carry into smallest normal 0x04=6.103515625e-5.
        RoundingTestCase{5.340576171875e-5f, 6.103515625e-5f},
        // [Gap 3] Above carry midpoint: definitively rounds up to smallest normal.
        // Input 15*2^-18 = 1.875*2^-15: subnormal, fp8Mant=3, remainder=0x300000 > halfPoint
        // 0x200000 → round up → carry → 0x04=6.103515625e-5.
        RoundingTestCase{5.7220458984375e-5f, 6.103515625e-5f}));

// Values at the boundary between two adjacent encodings (not at exact midpoints).
// Verifies correct rounding direction in the non-tie case.
INSTANTIATE_TEST_SUITE_P(BoundaryRounding,
                         TestFp8E5M2Rounding,
                         ::testing::Values(
                             // Slightly above 1.0 (0x40): closer to 1.0 than 1.25 -> 1.0
                             RoundingTestCase{1.1f, 1.0f},
                             // Slightly above midpoint 1.125: closer to 1.25 -> 1.25
                             RoundingTestCase{1.15f, 1.25f},
                             // Between 49152.0 (0x7A) and 57344.0 (0x7B): closer to 49152 -> 49152
                             RoundingTestCase{50000.0f, 49152.0f},
                             // Between 49152.0 and 57344.0: closer to 57344 -> 57344
                             RoundingTestCase{56000.0f, 57344.0f},
                             // Negative boundary
                             RoundingTestCase{-1.1f, -1.0f}));

// Rounding where mantissa overflow from the round-up cascades into the exponent,
// or where the final exponent overflows into saturation or Inf.
//
// OCP E5M2 mant bits = 2; when fp8Mant rounds from 3 to 4 (overflow), exp increments.
// If exp reaches >= 31 (MAX_EXP), the value produces Inf (saturate=false) or MAX (saturate=true).
// Saturation midpoint: MAX=57344 (exp=30, mant=3=odd), phantom next = 65536 (Inf territory).
// Midpoint = (57344+65536)/2 = 61440; since mant=3 is odd, round UP -> Inf or MAX.
INSTANTIATE_TEST_SUITE_P(
    RoundingOverflow,
    TestFp8E5M2Rounding,
    ::testing::Values(
        // Mantissa overflow: exp increments, stays in range
        // 0x43 (mant=3, odd) = 1.75; midpoint to 0x44=2.0 -> mant overflow -> 2.0
        RoundingTestCase{1.875f, 2.0f},
        // 0x47 (mant=3, odd) = 3.5; midpoint to 0x48=4.0 -> mant overflow -> 4.0
        RoundingTestCase{3.75f, 4.0f},
        // Exponent overflow into saturation (saturate=true by default):
        // 61440 is midpoint between MAX=57344 (mant=3 odd) and phantom 65536 -> round up -> MAX
        RoundingTestCase{61440.0f, 57344.0f},
        // Negative counterparts
        RoundingTestCase{-1.875f, -2.0f},
        RoundingTestCase{-3.75f, -4.0f},
        RoundingTestCase{-61440.0f, -57344.0f}));

// ============================================================================
// Saturation Tests
// ============================================================================

TEST(TestFp8E5M2, SaturationPositive)
{
    // Values beyond 57344 saturate to MAX=0x7B with saturate=true (default)
    const fp8_e5m2 val1(1e10f);
    EXPECT_EQ(static_cast<float>(val1), 57344.0f);

    const fp8_e5m2 val2(60000.0f);
    EXPECT_EQ(static_cast<float>(val2), 57344.0f);

    const fp8_e5m2 val3(57345.0f);
    EXPECT_EQ(static_cast<float>(val3), 57344.0f);

    const fp8_e5m2 val4(std::nextafter(57344.0f, 1e9f));
    EXPECT_EQ(static_cast<float>(val4), 57344.0f);

    // Rounding-into-saturation: 61440.0 (midpoint, mant=3 odd) -> rounds up -> MAX (saturate=true)
    const fp8_e5m2 val5(61440.0f);
    EXPECT_EQ(static_cast<float>(val5), 57344.0f);

    // saturate=false: values beyond 57344 return +Inf (0x7C)
    EXPECT_EQ(detail::float_to_fp8_e5m2_bits(65536.0f, /*saturate=*/false),
              static_cast<uint8_t>(0x7C));
    // saturate=false with rounding-into-overflow: 61440.0 rounds to Inf
    EXPECT_EQ(detail::float_to_fp8_e5m2_bits(61440.0f, /*saturate=*/false),
              static_cast<uint8_t>(0x7C));
}

TEST(TestFp8E5M2, SaturationNegative)
{
    const fp8_e5m2 val1(-1e10f);
    EXPECT_EQ(static_cast<float>(val1), -57344.0f);

    const fp8_e5m2 val2(-60000.0f);
    EXPECT_EQ(static_cast<float>(val2), -57344.0f);

    const fp8_e5m2 val3(-57345.0f);
    EXPECT_EQ(static_cast<float>(val3), -57344.0f);

    // Rounding-into-saturation mirrors positive case
    const fp8_e5m2 val4(-61440.0f);
    EXPECT_EQ(static_cast<float>(val4), -57344.0f);

    // saturate=false: returns -Inf (0xFC)
    EXPECT_EQ(detail::float_to_fp8_e5m2_bits(-65536.0f, /*saturate=*/false),
              static_cast<uint8_t>(0xFC));
}

TEST(TestFp8E5M2, SaturationInfinity)
{
    // +Inf with saturate=true (default) -> MAX=57344.0
    const fp8_e5m2 posInf(std::numeric_limits<float>::infinity());
    EXPECT_EQ(static_cast<float>(posInf), 57344.0f);

    // -Inf with saturate=true -> LOWEST=-57344.0
    const fp8_e5m2 negInf(-std::numeric_limits<float>::infinity());
    EXPECT_EQ(static_cast<float>(negInf), -57344.0f);

    // +Inf with saturate=false -> +Inf (0x7C)
    EXPECT_EQ(detail::float_to_fp8_e5m2_bits(std::numeric_limits<float>::infinity(),
                                             /*saturate=*/false),
              static_cast<uint8_t>(detail::FP8_E5M2_POS_INF));

    // -Inf with saturate=false -> -Inf (0xFC)
    EXPECT_EQ(detail::float_to_fp8_e5m2_bits(-std::numeric_limits<float>::infinity(),
                                             /*saturate=*/false),
              static_cast<uint8_t>(detail::FP8_E5M2_NEG_INF));
}

// ============================================================================
// Underflow Tests
// ============================================================================

TEST(TestFp8E5M2, Underflow)
{
    // Very small positive values flush to positive zero
    const fp8_e5m2 val1(1e-10f);
    EXPECT_EQ(static_cast<float>(val1), 0.0f);

    const fp8_e5m2 val2(-1e-10f);
    EXPECT_EQ(static_cast<float>(val2), 0.0f);

    // fp32 denormal inputs flush to zero (fp32Exp==0 path)
    const fp8_e5m2 val3(std::numeric_limits<float>::denorm_min());
    EXPECT_EQ(static_cast<float>(val3), 0.0f);

    const fp8_e5m2 val4(-std::numeric_limits<float>::denorm_min());
    EXPECT_EQ(static_cast<float>(val4), 0.0f);

    const fp8_e5m2 val5(1e-6f);
    EXPECT_EQ(static_cast<float>(val5), 0.0f);

    // The underflow threshold for OCP E5M2 is 2^-17.
    // At exactly 2^-17: rounds to zero (even mantissa tie-break, fp8Mant=0 is even)
    const fp8_e5m2 val6(7.62939453125e-6f); // 2^-17
    EXPECT_EQ(static_cast<float>(val6), 0.0f);

    // OCP E5M2 supports -0.0
    const fp8_e5m2 negZero(-0.0f);
    EXPECT_EQ(negZero.data, static_cast<uint8_t>(0x80));
    EXPECT_EQ(static_cast<float>(negZero), -0.0f);
}

// ============================================================================
// E5M2 OCP-specific: Infinity, NaN, Bit-Pattern Round-Trip
// ============================================================================

TEST(TestFp8E5M2, OcpInfinityTruthTable)
{
    // OCP E5M2 has infinity at 0x7C (+Inf) and 0xFC (-Inf).
    // All other bit patterns must NOT be infinity.
    for(int bits = 0; bits <= 0xFF; ++bits)
    {
        const bool expectedInf = (bits == 0x7C || bits == 0xFC);
        EXPECT_EQ(isinf(fp8_e5m2::from_bits(static_cast<uint8_t>(bits))), expectedInf)
            << "isinf wrong for bit pattern 0x" << std::hex << bits;
    }
}

TEST(TestFp8E5M2, OcpNanTruthTable)
{
    // OCP E5M2: isnan is true when exp field = 11111 = 31 AND mant != 0.
    // Bit pattern structure: [S|EEEEE|MM]. NaN condition: (bits & 0x7C) == 0x7C && (bits & 0x03) != 0
    for(int bits = 0; bits <= 0xFF; ++bits)
    {
        const auto pattern = static_cast<uint8_t>(bits);
        const bool expectedNan = ((pattern & 0x7Cu) == 0x7Cu) && ((pattern & 0x03u) != 0u);
        EXPECT_EQ(isnan(fp8_e5m2::from_bits(pattern)), expectedNan)
            << "isnan wrong for bit pattern 0x" << std::hex << bits;
    }
}

TEST(TestFp8E5M2, RoundTripAllPatterns)
{
    // For every bit pattern: decode to float via the lookup table, re-encode back,
    // and verify that the resulting bit pattern is identical.
    // OCP E5M2 NaN: any pattern with (bits & 0x7C) == 0x7C && (bits & 0x03) != 0.
    // OCP E5M2 Inf: 0x7C and 0xFC.
    // NaN round-trips: a float NaN re-encodes with its sign bit preserved as 0x7F or 0xFF.
    for(int bits = 0; bits <= 0xFF; ++bits)
    {
        const auto pattern = static_cast<uint8_t>(bits);
        const auto decoded = fp8_e5m2::from_bits(pattern);
        const auto f = static_cast<float>(decoded);

        // NaN patterns: OCP E5M2 NaN has exp=31 and mant != 0
        if(((pattern & 0x7Cu) == 0x7Cu) && ((pattern & 0x03u) != 0u))
        {
            EXPECT_TRUE(std::isnan(f))
                << "NaN pattern 0x" << std::hex << bits << " should decode to float NaN";
            // NaN re-encodes: sign is preserved; the canonical NaN encoding has mant != 0
            // float_to_fp8_e5m2_bits maps any float NaN to 0x7F (positive) or 0xFF (negative)
            const fp8_e5m2 reencoded(f);
            EXPECT_TRUE(isnan(reencoded))
                << "Re-encoded NaN for pattern 0x" << std::hex << bits << " should still be NaN";
            continue;
        }

        // Inf patterns (0x7C, 0xFC): decode to ±Inf, re-encode to ±Inf with saturate=false
        if(pattern == 0x7C || pattern == 0xFC)
        {
            EXPECT_TRUE(std::isinf(f))
                << "Inf pattern 0x" << std::hex << bits << " should decode to float Inf";
            // With default saturate=true, Inf->MAX. We must test round-trip of the bit pattern
            // via the float_to_fp8_e5m2_bits with saturate=false to preserve Inf.
            const uint8_t reencoded = detail::float_to_fp8_e5m2_bits(f, /*saturate=*/false);
            EXPECT_EQ(reencoded, pattern) << "Inf round-trip failed for 0x" << std::hex << bits;
            continue;
        }

        // All finite, non-NaN patterns must survive the round-trip exactly.
        const fp8_e5m2 reencoded(f);
        EXPECT_EQ(reencoded.data, pattern) << "Round-trip failed for bit pattern 0x" << std::hex
                                           << bits << " (float value " << std::dec << f << ")";
    }
}

TEST(TestFp8E5M2, MathFunctions)
{
    // isfinite: finite for all non-Inf, non-NaN
    EXPECT_TRUE(isfinite(fp8_e5m2(1.0f)));
    EXPECT_TRUE(isfinite(fp8_e5m2(0.0f)));
    EXPECT_TRUE(isfinite(fp8_e5m2(57344.0f)));
    EXPECT_FALSE(isfinite(fp8_e5m2::from_bits(static_cast<uint8_t>(0x7C)))); // +Inf
    EXPECT_FALSE(isfinite(fp8_e5m2::from_bits(static_cast<uint8_t>(0x7F)))); // NaN

    // isinf: true only for 0x7C and 0xFC
    EXPECT_TRUE(isinf(fp8_e5m2::from_bits(static_cast<uint8_t>(0x7C))));
    EXPECT_TRUE(isinf(fp8_e5m2::from_bits(static_cast<uint8_t>(0xFC))));
    EXPECT_FALSE(isinf(fp8_e5m2(1.0f)));
    EXPECT_FALSE(isinf(fp8_e5m2::from_bits(static_cast<uint8_t>(0x7F)))); // NaN not Inf

    // isnan: true for exp=31 && mant != 0 (OCP E5M2 does not distinguish QNAN/SNAN)
    EXPECT_TRUE(isnan(fp8_e5m2::from_bits(static_cast<uint8_t>(0x7F)))); // NaN (canonical)
    EXPECT_TRUE(isnan(fp8_e5m2::from_bits(static_cast<uint8_t>(0x7D)))); // NaN (alternate encoding)
    EXPECT_FALSE(isnan(fp8_e5m2(1.0f)));
    EXPECT_FALSE(isnan(fp8_e5m2::from_bits(static_cast<uint8_t>(0x7C)))); // Inf not NaN

    // signbit
    EXPECT_FALSE(signbit(fp8_e5m2(1.0f)));
    EXPECT_TRUE(signbit(fp8_e5m2(-1.0f)));
    EXPECT_FALSE(signbit(fp8_e5m2(0.0f)));
    EXPECT_TRUE(signbit(fp8_e5m2(-0.0f)));
    EXPECT_FALSE(signbit(fp8_e5m2::from_bits(static_cast<uint8_t>(0x7C)))); // +Inf
    EXPECT_TRUE(signbit(fp8_e5m2::from_bits(static_cast<uint8_t>(0xFC)))); // -Inf
}

// ============================================================================
// Specific numeric_limits Value Tests
// ============================================================================

TEST(TestFp8E5M2, NumericLimitsSpecificValues)
{
    using L = std::numeric_limits<fp8_e5m2>;

    // max() = 57344.0 (0x7B)
    EXPECT_EQ(static_cast<float>(L::max()), 57344.0f);
    EXPECT_EQ(L::max().data, static_cast<uint8_t>(0x7B));

    // min() = 2^-14 = 6.103515625e-5 (0x04, smallest positive normal)
    EXPECT_EQ(static_cast<float>(L::min()), 6.103515625e-5f);
    EXPECT_EQ(L::min().data, static_cast<uint8_t>(0x04));

    // lowest() = -57344.0 (0xFB)
    EXPECT_EQ(static_cast<float>(L::lowest()), -57344.0f);
    EXPECT_EQ(L::lowest().data, static_cast<uint8_t>(0xFB));

    // epsilon() = 2^-2 = 0.25 (0x34, 2 mantissa bits)
    EXPECT_EQ(static_cast<float>(L::epsilon()), 0.25f);
    EXPECT_EQ(L::epsilon().data, static_cast<uint8_t>(0x34));

    // round_error() = 0.5 (0x38)
    EXPECT_EQ(static_cast<float>(L::round_error()), 0.5f);
    EXPECT_EQ(L::round_error().data, static_cast<uint8_t>(0x38));

    // denorm_min() = 2^-16 = 1.52587890625e-5 (0x01, smallest positive subnormal)
    EXPECT_EQ(L::denorm_min().data, static_cast<uint8_t>(0x01));
    EXPECT_EQ(static_cast<float>(L::denorm_min()), 1.52587890625e-5f);

    // infinity() = 0x7C (OCP E5M2 has infinity)
    EXPECT_EQ(L::infinity().data, static_cast<uint8_t>(detail::FP8_E5M2_POS_INF));
    EXPECT_TRUE(isinf(L::infinity()));
    EXPECT_FALSE(signbit(L::infinity()));

    // quiet_NaN() = 0x7F (canonical NaN)
    EXPECT_EQ(L::quiet_NaN().data, static_cast<uint8_t>(detail::FP8_E5M2_NAN));
    EXPECT_TRUE(isnan(L::quiet_NaN()));

    // signaling_NaN() returns the same as quiet_NaN() for OCP E5M2
    // (OCP E5M2 does not distinguish between signaling and quiet NaN)
    EXPECT_EQ(L::signaling_NaN().data, static_cast<uint8_t>(detail::FP8_E5M2_NAN));
    EXPECT_TRUE(isnan(L::signaling_NaN()));

    // Boolean fields
    EXPECT_TRUE(L::has_infinity);
    EXPECT_TRUE(L::has_quiet_NaN);
    EXPECT_FALSE(L::has_signaling_NaN); // OCP E5M2 does not distinguish signaling NaN
    EXPECT_EQ(L::has_denorm, std::denorm_present);
    EXPECT_EQ(L::round_style, std::round_to_nearest);
    EXPECT_EQ(L::radix, 2);
    EXPECT_EQ(L::digits, 3); // 2 mantissa + 1 implicit
    EXPECT_EQ(L::min_exponent, -13); // min_normal = 2^(min_exponent-1) = 2^-14
    EXPECT_EQ(L::max_exponent, 16); // max_normal = (2-2^-2) * 2^15 < 2^16
}

// ============================================================================
// Named Constants Tests
// ============================================================================

TEST(TestFp8E5M2, NamedConstants)
{
    // Infinity encodings
    EXPECT_EQ(detail::FP8_E5M2_POS_INF, static_cast<uint8_t>(0x7C));
    EXPECT_EQ(detail::FP8_E5M2_NEG_INF, static_cast<uint8_t>(0xFC));

    // NaN encoding (OCP E5M2 does not distinguish signaling vs quiet NaN)
    EXPECT_EQ(detail::FP8_E5M2_NAN, static_cast<uint8_t>(0x7F));

    // Range constants
    EXPECT_EQ(detail::FP8_E5M2_MAX, static_cast<uint8_t>(0x7B));
    EXPECT_EQ(detail::FP8_E5M2_LOWEST, static_cast<uint8_t>(0xFB));
    EXPECT_EQ(detail::FP8_E5M2_MIN_NORMAL, static_cast<uint8_t>(0x04));
    EXPECT_EQ(detail::FP8_E5M2_DENORM_MIN, static_cast<uint8_t>(0x01));

    // Verify values decode to expected float values
    EXPECT_EQ(static_cast<float>(fp8_e5m2::from_bits(detail::FP8_E5M2_MAX)), 57344.0f);
    EXPECT_EQ(static_cast<float>(fp8_e5m2::from_bits(detail::FP8_E5M2_MIN_NORMAL)),
              6.103515625e-5f);
    EXPECT_EQ(static_cast<float>(fp8_e5m2::from_bits(detail::FP8_E5M2_DENORM_MIN)),
              1.52587890625e-5f);

    // OCP E5M2 has infinity
    EXPECT_TRUE(std::numeric_limits<fp8_e5m2>::has_infinity);
    EXPECT_TRUE(isinf(std::numeric_limits<fp8_e5m2>::infinity()));
    EXPECT_EQ(std::numeric_limits<fp8_e5m2>::infinity().data,
              static_cast<uint8_t>(detail::FP8_E5M2_POS_INF));

    // NaN assertions (OCP E5M2 does not distinguish signaling vs quiet NaN)
    EXPECT_TRUE(isnan(std::numeric_limits<fp8_e5m2>::quiet_NaN()));
    EXPECT_TRUE(isnan(std::numeric_limits<fp8_e5m2>::signaling_NaN()));
    EXPECT_FALSE(std::numeric_limits<fp8_e5m2>::has_signaling_NaN);
}
