// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Regression tests for the gfx1250 MXFP8 (E5M2 data) extension variants
// `ocp_e5m2_mxfp8_e5m3` and `ocp_e5m2_mxfp8_e4m3`. See the header comment in
// `ocp_e4m3_mxfp8_e4m3_e5m3_scale_test.cpp` for context. The tests below add
// E5M2-specific coverage for the data-type's infinity representation, which
// E4M3 lacks.

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
class OcpE5M2MxFp8AltScaleTest : public ::testing::Test
{
};

using OcpE5M2AltScaleTypes
    = ::testing::Types<DGen::ocp_e5m2_mxfp8_e5m3, DGen::ocp_e5m2_mxfp8_e4m3>;
TYPED_TEST_SUITE(OcpE5M2MxFp8AltScaleTest, OcpE5M2AltScaleTypes);

TYPED_TEST(OcpE5M2MxFp8AltScaleTest, ScaleOneHasExpectedExponent)
{
    const uint8_t one    = getScaleOneByte<TypeParam>();
    const int     oneExp = getExponentValue<uint8_t>(
        one, TypeParam::scaleInfo.mantissaBits, TypeParam::scaleInfo.exponentBits);

    EXPECT_EQ(oneExp, static_cast<int>(TypeParam::scaleInfo.bias));
}

TYPED_TEST(OcpE5M2MxFp8AltScaleTest, SetOneCreatesOneInDouble)
{
    uint8_t scale[1] = {0};
    uint8_t data[1]  = {0};

    setOne<TypeParam>(scale, data, 0, 0, false);
    EXPECT_DOUBLE_EQ(toDouble<TypeParam>(scale, data, 0, 0), 1.0);

    setOne<TypeParam>(scale, data, 0, 0, true);
    EXPECT_DOUBLE_EQ(toDouble<TypeParam>(scale, data, 0, 0), 1.0);
}

TYPED_TEST(OcpE5M2MxFp8AltScaleTest, SetNaNSetsScaleNaNAndPropagates)
{
    uint8_t scale[1] = {0};
    uint8_t data[1]  = {DGen::ocp_e5m2_mxfp8::oneMask};

    setNaN<TypeParam>(scale, data, 0, 0);
    EXPECT_TRUE(isNaN<TypeParam>(scale, data, 0, 0));
    EXPECT_TRUE(std::isnan(toDouble<TypeParam>(scale, data, 0, 0)));
}

TYPED_TEST(OcpE5M2MxFp8AltScaleTest, ScaleZeroForcesZero)
{
    uint8_t scale[1] = {0};
    uint8_t data[1]  = {DGen::ocp_e5m2_mxfp8::oneMask};

    EXPECT_TRUE(isZero<TypeParam>(scale, data, 0, 0));
    EXPECT_DOUBLE_EQ(toDouble<TypeParam>(scale, data, 0, 0), 0.0);
}

TYPED_TEST(OcpE5M2MxFp8AltScaleTest, SetInfRoundtripsThroughToFloat)
{
    // E5M2 has its own infinity bit pattern in the data byte; setInf should
    // make toFloat return ±inf regardless of the scale variant.
    uint8_t scale[1] = {getScaleOneByte<TypeParam>()};
    uint8_t data[1]  = {0};

    setInf<TypeParam>(scale, data, 0, 0);
    EXPECT_TRUE(isInf<TypeParam>(scale, data, 0, 0));
    EXPECT_TRUE(std::isinf(toFloat<TypeParam>(scale, data, 0, 0)));
}

TYPED_TEST(OcpE5M2MxFp8AltScaleTest, ToDoubleMatchesBaseTypeWithPowerOfTwoScaling)
{
    const uint8_t baseScale[1] = {DGen::Constants::E8M0_1};

    const std::vector<uint8_t> dataBytes = {
        DGen::ocp_e5m2_mxfp8::positiveZeroMask,
        DGen::ocp_e5m2_mxfp8::oneMask,
        DGen::ocp_e5m2_mxfp8::dataMaxPositiveNormalMask,
        DGen::ocp_e5m2_mxfp8::dataMaxNegativeNormalMask,
        DGen::ocp_e5m2_mxfp8::dataMaxPositiveSubNormalMask,
        DGen::ocp_e5m2_mxfp8::dataMaxNegativeSubNormalMask,
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

            if(scaleByte == 0
               || dataByte == DGen::ocp_e5m2_mxfp8::positiveZeroMask
               || dataByte == DGen::ocp_e5m2_mxfp8::negativeZeroMask)
            {
                EXPECT_DOUBLE_EQ(actual, 0.0);
                continue;
            }

            const double base     = toDouble<DGen::ocp_e5m2_mxfp8>(baseScale, data, 0, 0);
            const double expected = base * scaleFactorFromByte<TypeParam>(scaleByte);
            EXPECT_DOUBLE_EQ(actual, expected)
                << "scaleByte=" << int(scaleByte) << " dataByte=" << int(dataByte);
        }
    }
}
} // namespace
