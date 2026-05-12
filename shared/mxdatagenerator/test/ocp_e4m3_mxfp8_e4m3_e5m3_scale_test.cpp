// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Regression tests for the gfx1250 MXFP8 (E4M3 data) extension variants
// `ocp_e4m3_mxfp8_e5m3` and `ocp_e4m3_mxfp8_e4m3`. The OCP MXFP8 spec only
// defines the UE8M0-scaled variant; gfx1250 hardware also supports E5M3 and
// E4M3 8-bit-float scales via the v_wmma_scale_f32_*_f8f6f4 instruction's
// matrix_*_scale_fmt modifiers. These tests mirror the existing FP4 scale
// variant tests in `ocp_e2m1_mxfp4_e4m3_e5m3_scale_test.cpp` to keep coverage
// symmetric across data formats.

#include <mxDataGenerator/dataTypeInfo.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace
{
using namespace DGen;

template <typename DT>
constexpr ScaleType scaleTypeFor()
{
    if constexpr(DT::scaleInfo.exponentBits == 5 && DT::scaleInfo.mantissaBits == 3)
        return ScaleType::E5M3;
    else if constexpr(DT::scaleInfo.exponentBits == 4 && DT::scaleInfo.mantissaBits == 3)
        return ScaleType::E4M3;
    else
        static_assert(std::is_same_v<DT, void>, "Unexpected scale format for DT.");
}

template <typename DT>
uint8_t getScaleOneByte()
{
    constexpr auto st = scaleTypeFor<DT>();
    if constexpr(st == ScaleType::E5M3)
        return static_cast<uint8_t>(getScaleOne<ScaleType::E5M3>());
    else
        return static_cast<uint8_t>(getScaleOne<ScaleType::E4M3>());
}

template <typename DT>
uint8_t getScaleNanByte()
{
    constexpr auto st = scaleTypeFor<DT>();
    if constexpr(st == ScaleType::E5M3)
        return static_cast<uint8_t>(getScaleNan<ScaleType::E5M3>());
    else
        return static_cast<uint8_t>(getScaleNan<ScaleType::E4M3>());
}

template <typename DT>
double scaleFactorFromByte(uint8_t scaleByte)
{
    const int scaleExp = getExponentValue<uint8_t>(
        scaleByte, DT::scaleInfo.mantissaBits, DT::scaleInfo.exponentBits);
    return std::pow(2.0, static_cast<double>(scaleExp - static_cast<int>(DT::scaleInfo.bias)));
}

template <typename DT>
class OcpE4M3MxFp8AltScaleTest : public ::testing::Test
{
};

using OcpE4M3AltScaleTypes
    = ::testing::Types<DGen::ocp_e4m3_mxfp8_e5m3, DGen::ocp_e4m3_mxfp8_e4m3>;
TYPED_TEST_SUITE(OcpE4M3MxFp8AltScaleTest, OcpE4M3AltScaleTypes);

TYPED_TEST(OcpE4M3MxFp8AltScaleTest, ScaleOneHasExpectedExponent)
{
    const uint8_t one = getScaleOneByte<TypeParam>();

    const int oneExp = getExponentValue<uint8_t>(
        one, TypeParam::scaleInfo.mantissaBits, TypeParam::scaleInfo.exponentBits);

    EXPECT_EQ(oneExp, static_cast<int>(TypeParam::scaleInfo.bias));
}

TYPED_TEST(OcpE4M3MxFp8AltScaleTest, SetOneCreatesOneInDouble)
{
    uint8_t scale[1] = {0};
    uint8_t data[1]  = {0};

    setOne<TypeParam>(scale, data, 0, 0, false);
    EXPECT_DOUBLE_EQ(toDouble<TypeParam>(scale, data, 0, 0), 1.0);

    // Subnormal-encoded 1.0 is unrepresentable for these scale formats (see
    // setOne notes); `subNormal=true` falls back to the same normal encoding.
    setOne<TypeParam>(scale, data, 0, 0, true);
    EXPECT_DOUBLE_EQ(toDouble<TypeParam>(scale, data, 0, 0), 1.0);
}

TYPED_TEST(OcpE4M3MxFp8AltScaleTest, SetNaNSetsScaleNaNAndPropagates)
{
    uint8_t scale[1] = {0};
    uint8_t data[1]  = {DGen::ocp_e4m3_mxfp8::oneMask};

    setNaN<TypeParam>(scale, data, 0, 0);
    EXPECT_TRUE(isNaN<TypeParam>(scale, data, 0, 0));
    EXPECT_TRUE(std::isnan(toDouble<TypeParam>(scale, data, 0, 0)));
}

TYPED_TEST(OcpE4M3MxFp8AltScaleTest, ScaleZeroForcesZero)
{
    uint8_t scale[1] = {0};
    uint8_t data[1]  = {DGen::ocp_e4m3_mxfp8::oneMask};

    EXPECT_TRUE(isZero<TypeParam>(scale, data, 0, 0));
    EXPECT_DOUBLE_EQ(toDouble<TypeParam>(scale, data, 0, 0), 0.0);
}

TYPED_TEST(OcpE4M3MxFp8AltScaleTest, IsInfAlwaysFalseForE4M3Data)
{
    // E4M3 data has no infinity representation, so isInf must be false for
    // every (scale, data) byte pair.
    for(int s = 0; s < 256; ++s)
    {
        for(int d = 0; d < 256; ++d)
        {
            const uint8_t scale[1] = {static_cast<uint8_t>(s)};
            const uint8_t data[1]  = {static_cast<uint8_t>(d)};
            EXPECT_FALSE(isInf<TypeParam>(scale, data, 0, 0))
                << "scale=" << s << " data=" << d;
        }
    }
}

TYPED_TEST(OcpE4M3MxFp8AltScaleTest, ToDoubleMatchesBaseTypeWithPowerOfTwoScaling)
{
    // Use the spec UE8M0=1 scale to read the raw data byte's value, then
    // multiply by the scale-factor we expect from the alternative scale type's
    // encoding. The two should match exactly (data path is identical; only
    // the scale exponent decoding differs).
    const uint8_t baseScale[1] = {DGen::Constants::E8M0_1};

    // Test data bytes that survive the scale change cleanly: zero, one,
    // dataMaxNormal, dataMaxNegativeNormal, smallest positive subnormal, and
    // the corresponding negative.
    const std::vector<uint8_t> dataBytes = {
        DGen::ocp_e4m3_mxfp8::positiveZeroMask,
        DGen::ocp_e4m3_mxfp8::oneMask,
        DGen::ocp_e4m3_mxfp8::dataMaxPositiveNormalMask,
        DGen::ocp_e4m3_mxfp8::dataMaxNegativeNormalMask,
        DGen::ocp_e4m3_mxfp8::dataMaxPositiveSubNormalMask,
        DGen::ocp_e4m3_mxfp8::dataMaxNegativeSubNormalMask,
    };

    const uint8_t nanScale = getScaleNanByte<TypeParam>();

    for(int s = 0; s < 256; ++s)
    {
        const uint8_t scaleByte = static_cast<uint8_t>(s);
        const uint8_t scale[1]  = {scaleByte};

        for(const uint8_t dataByte : dataBytes)
        {
            const uint8_t data[1] = {dataByte};
            const double  actual  = toDouble<TypeParam>(scale, data, 0, 0);

            if(scaleByte == nanScale)
            {
                EXPECT_TRUE(std::isnan(actual));
                continue;
            }

            // Skip raw E4M3 data NaN bytes (positive and negative); the data
            // already encodes NaN regardless of scale.
            const uint8_t dataNoSign = dataByte & ~ocp_e4m3_mxfp8::signBitMask;
            if(dataNoSign == DGen::ocp_e4m3_mxfp8::dataNaNMasks[0])
            {
                EXPECT_TRUE(std::isnan(actual));
                continue;
            }

            if(scaleByte == 0
               || dataByte == DGen::ocp_e4m3_mxfp8::positiveZeroMask
               || dataByte == DGen::ocp_e4m3_mxfp8::negativeZeroMask)
            {
                EXPECT_DOUBLE_EQ(actual, 0.0);
                continue;
            }

            const double base     = toDouble<DGen::ocp_e4m3_mxfp8>(baseScale, data, 0, 0);
            const double expected = base * scaleFactorFromByte<TypeParam>(scaleByte);
            EXPECT_DOUBLE_EQ(actual, expected)
                << "scaleByte=" << int(scaleByte) << " dataByte=" << int(dataByte);
        }
    }
}
} // namespace
