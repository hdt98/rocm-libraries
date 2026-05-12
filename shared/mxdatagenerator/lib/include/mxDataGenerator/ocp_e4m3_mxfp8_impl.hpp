// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "dataTypeInfo.hpp"

//return true iff XN = NAN
template <>
inline bool isNaN<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes,
                                  uint8_t const* dataBytes,
                                  index_t         scaleIndex,
                                  index_t         dataIndex)
{
    uint8_t data  = *(dataBytes + dataIndex);
    uint8_t scale = *(scaleBytes + scaleIndex);

    if(scale == getScaleNan<ScaleType::E8M0>())
        return true;

    // set sign bit to 0
    data &= (~ocp_e4m3_mxfp8::signBitMask);

    auto& nanValues{ocp_e4m3_mxfp8::dataNaNMasks};
    return std::count(nanValues.begin(), nanValues.end(), data) > 0;
}

template <>
inline bool isNaNPacked<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes,
                                        uint8_t const* dataBytes,
                                        index_t         scaleIndex,
                                        index_t         dataIndex)
{
    return isNaN<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

// return true iff XN = 0
template <>
inline bool isZero<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes,
                                   uint8_t const* dataBytes,
                                   index_t         scaleIndex,
                                   index_t         dataIndex)
{
    if(isNaN<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return false;

    // keep bits 0-7 same, set sign bit to 0
    // no need to check for scale as it doesn't have a zero representation
    uint8_t data = *(dataBytes + dataIndex) & (~ocp_e4m3_mxfp8::signBitMask);

    return data == ocp_e4m3_mxfp8::positiveZeroMask;
}

template <>
inline bool isZeroPacked<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes,
                                         uint8_t const* dataBytes,
                                         index_t         scaleIndex,
                                         index_t         dataIndex)
{
    return isZero<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

// no infinity representation in ocp_e4m3_mxfp8 will always return false
template <>
inline bool isInf<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes [[maybe_unused]],
                                  uint8_t const* dataBytes [[maybe_unused]],
                                  index_t         scaleIndex [[maybe_unused]],
                                  index_t         dataIndex [[maybe_unused]])
{
    return false;
}

template <>
inline bool isInfPacked<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes,
                                        uint8_t const* dataBytes,
                                        index_t         scaleIndex,
                                        index_t         dataIndex)
{
    return isInf<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline double toDouble<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes,
                                       uint8_t const* dataBytes,
                                       index_t         scaleIndex,
                                       index_t         dataIndex)
{
    if(isNaN<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<double>::quiet_NaN();

    if(isZero<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint8_t data     = *(dataBytes + dataIndex);
    int     scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e4m3_mxfp8::scaleInfo.mantissaBits,
                                             ocp_e4m3_mxfp8::scaleInfo.exponentBits);

    return convertToDouble<uint8_t, OCP_E4M3_MXFP8_DATA, ScaleInfo<ScaleType::E8M0>>(data, scaleExp);
}

template <>
inline double toDoublePacked<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes,
                                             uint8_t const* dataBytes,
                                             index_t         scaleIndex,
                                             index_t         dataIndex)
{
    return toDouble<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline float toFloat<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes,
                                     uint8_t const* dataBytes,
                                     index_t         scaleIndex,
                                     index_t         dataIndex)
{
    if(isNaN<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<float>::quiet_NaN();

    if(isZero<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint8_t data     = *(dataBytes + dataIndex);
    int     scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e4m3_mxfp8::scaleInfo.mantissaBits,
                                             ocp_e4m3_mxfp8::scaleInfo.exponentBits);

    return convertToFloat<uint8_t, OCP_E4M3_MXFP8_DATA, ScaleInfo<ScaleType::E8M0>>(data, scaleExp);
}

template <>
inline float toFloatPacked<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes,
                                           uint8_t const* dataBytes,
                                           index_t         scaleIndex,
                                           index_t         dataIndex)
{
    return toFloat<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline bool isSubnorm<ocp_e4m3_mxfp8>(uint8_t const* dataBytes, index_t dataIndex)
{
    uint8_t data = *(dataBytes + dataIndex);
    return isSubNormal<uint16_t>(
        data, ocp_e4m3_mxfp8::dataInfo.mantissaBits, ocp_e4m3_mxfp8::dataInfo.exponentBits);
}

template <>
inline bool isSubnormPacked<ocp_e4m3_mxfp8>(uint8_t const* dataBytes, index_t dataIndex)
{
    return isSubnorm<ocp_e4m3_mxfp8>(dataBytes, dataIndex);
}

template <>
inline void setOne<ocp_e4m3_mxfp8>(
    uint8_t* scaleBytes, uint8_t* dataBytes, index_t scaleIndex, index_t dataIndex, bool subNormal)
{
    //if its sub normal, 0b00000010 * E8M0_135 will equal to 1
    *(scaleBytes + scaleIndex) = subNormal ? Constants::E8M0_135 : Constants::E8M0_1;
    *(dataBytes + dataIndex)
        = subNormal ? ocp_e4m3_mxfp8::dataSubNormalOneMask : ocp_e4m3_mxfp8::oneMask;
}

//set XN = 0, scale X will not be changed
template <>
inline void setZero<ocp_e4m3_mxfp8>(uint8_t* scaleBytes [[maybe_unused]],
                                    uint8_t* dataBytes,
                                    index_t   scaleIndex [[maybe_unused]],
                                    index_t   dataIndex)
{
    *(dataBytes + dataIndex) = ocp_e4m3_mxfp8::positiveZeroMask;
}

//set XN = NAN, scale X will not be changed
template <>
inline void setNaN<ocp_e4m3_mxfp8>(uint8_t* scaleBytes [[maybe_unused]],
                                   uint8_t* dataBytes,
                                   index_t   scaleIndex [[maybe_unused]],
                                   index_t   dataIndex)
{
    *(dataBytes + dataIndex) = ocp_e4m3_mxfp8::dataNaNMasks[0];
}

//ocp_e4m3_mxfp8 does not have an infinity representation, method will just return
template <>
inline void setInf<ocp_e4m3_mxfp8>(uint8_t* scaleBytes [[maybe_unused]],
                                   uint8_t* dataBytes [[maybe_unused]],
                                   index_t   scaleIndex [[maybe_unused]],
                                   index_t   dataIndex [[maybe_unused]])
{
    return;
}

template <>
inline void
    setDataMax<ocp_e4m3_mxfp8>(uint8_t* dataBytes, index_t dataIndex, bool subNormal, bool positive)
{
    if(subNormal)
        *(dataBytes + dataIndex) = positive ? ocp_e4m3_mxfp8::dataMaxPositiveSubNormalMask
                                            : ocp_e4m3_mxfp8::dataMaxNegativeSubNormalMask;
    else
        *(dataBytes + dataIndex) = positive ? ocp_e4m3_mxfp8::dataMaxPositiveNormalMask
                                            : ocp_e4m3_mxfp8::dataMaxNegativeNormalMask;
}

template <>
inline uint64_t satConvertToType<ocp_e4m3_mxfp8>(float value)
{

    cvt t;
    t.num      = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
    {

        return sign << 15 | ocp_e4m3_mxfp8::dataNaNMasks[0];
    }

    uint8_t res = convertToType<uint8_t, ocp_e4m3_mxfp8>(value);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {Constants::E8M0_1};

    float resVal = toFloat<ocp_e4m3_mxfp8>(tScale, tData, 0, 0);

    if(std::abs(resVal) > ocp_e4m3_mxfp8::dataMaxNormalNumber
       || std::isnan(resVal)) // covers inf case too
        return value < 0 ? ocp_e4m3_mxfp8::dataMaxNegativeNormalMask
                         : ocp_e4m3_mxfp8::dataMaxPositiveNormalMask;

    if(std::abs(resVal) < ocp_e4m3_mxfp8::dataMinSubnormalNumber)
        return value < 0 ? ocp_e4m3_mxfp8::negativeZeroMask : ocp_e4m3_mxfp8::positiveZeroMask;

    return res;
}

template <>
inline uint64_t nonSatConvertToType<ocp_e4m3_mxfp8>(float value)
{

    uint8_t res = convertToType<uint8_t, ocp_e4m3_mxfp8>(value);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {Constants::E8M0_1};

    float resVal = toFloat<ocp_e4m3_mxfp8>(tScale, tData, 0, 0);

    cvt t;
    t.num      = value;
    uint sign = t.bRep >> 31;

    //std::abs(value) > dataMaxNornal covers inf case as well
    if(std::abs(resVal) > ocp_e4m3_mxfp8::dataMaxNormalNumber || std::isnan(value))
        return sign << 15 | ocp_e4m3_mxfp8::dataNaNMasks[0];

    if(std::abs(resVal) < ocp_e4m3_mxfp8::dataMinSubnormalNumber)
        return value < 0 ? ocp_e4m3_mxfp8::negativeZeroMask : ocp_e4m3_mxfp8::positiveZeroMask;

    return res;
}

template <>
inline uint64_t satConvertToTypeSR<ocp_e4m3_mxfp8>(float value, uint seed)
{
    cvt t;
    t.num      = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
        return sign << 7 | ocp_e4m3_mxfp8::dataNaNMasks[0];
    else if(value > ocp_e4m3_mxfp8::dataMaxRoundedRange)
        return ocp_e4m3_mxfp8::dataMaxPositiveNormalMask;
    else if(value < -ocp_e4m3_mxfp8::dataMaxRoundedRange)
        return ocp_e4m3_mxfp8::dataMaxNegativeNormalMask;

    uint8_t res = convertToTypeSR<uint8_t, ocp_e4m3_mxfp8>(value, seed);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {Constants::E8M0_1};

    float resVal = toFloat<ocp_e4m3_mxfp8>(tScale, tData, 0, 0);

    if(std::abs(resVal) > ocp_e4m3_mxfp8::dataMaxNormalNumber
       || std::isnan(resVal)) // covers inf case too
        return value < 0 ? ocp_e4m3_mxfp8::dataMaxNegativeNormalMask
                         : ocp_e4m3_mxfp8::dataMaxPositiveNormalMask;

    if(std::abs(resVal) < ocp_e4m3_mxfp8::dataMinSubnormalNumber)
        return value < 0 ? ocp_e4m3_mxfp8::negativeZeroMask : ocp_e4m3_mxfp8::positiveZeroMask;

    return res;
}

template <>
inline uint64_t nonSatConvertToTypeSR<ocp_e4m3_mxfp8>(float value, uint seed)
{

    cvt t;
    t.num      = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
        return sign << 7 | ocp_e4m3_mxfp8::dataNaNMasks[0];
    else if(value > ocp_e4m3_mxfp8::dataMaxRoundedRange)
        return ocp_e4m3_mxfp8::dataNaNMasks[0];
    else if(value < -ocp_e4m3_mxfp8::dataMaxRoundedRange)
        return ocp_e4m3_mxfp8::dataNaNMasks[0];

    uint8_t res = convertToTypeSR<uint8_t, ocp_e4m3_mxfp8>(value, seed);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {Constants::E8M0_1};

    float resVal = toFloat<ocp_e4m3_mxfp8>(tScale, tData, 0, 0);

    //std::abs(value) > dataMaxNornal covers inf case as well
    if(std::abs(resVal) > ocp_e4m3_mxfp8::dataMaxNormalNumber)
        return sign << 7 | ocp_e4m3_mxfp8::dataNaNMasks[0];

    if(std::abs(resVal) < ocp_e4m3_mxfp8::dataMinSubnormalNumber)
        return value < 0 ? ocp_e4m3_mxfp8::negativeZeroMask : ocp_e4m3_mxfp8::positiveZeroMask;

    return res;
}

// ============================================================================
// gfx1250 extension: ocp_e4m3_mxfp8_e5m3 (E4M3 data + E5M3 scale)
// ----------------------------------------------------------------------------
// These specializations mirror the E8M0-scale ones above but route scale-related
// helpers to ScaleType::E5M3. E5M3 differs from E8M0 in two important ways
// the impl has to honour:
//   * E5M3 has a zero representation (E8M0 does not), so isZero/isZeroPacked
//     short-circuit on a zero scale byte.
//   * E5M3 has a different NaN bit pattern, so isNaN consults
//     getScaleNan<ScaleType::E5M3>().
// All data-side bit patterns and ranges come from ocp_e4m3_mxfp8_base, so
// they're shared with the E8M0 variant verbatim.
// ============================================================================

template <>
inline bool isNaN<ocp_e4m3_mxfp8_e5m3>(uint8_t const* scaleBytes,
                                       uint8_t const* dataBytes,
                                       index_t        scaleIndex,
                                       index_t        dataIndex)
{
    uint8_t data  = *(dataBytes + dataIndex);
    uint8_t scale = *(scaleBytes + scaleIndex);

    if(scale == getScaleNan<ScaleType::E5M3>())
        return true;

    data &= (~ocp_e4m3_mxfp8_e5m3::signBitMask);
    auto& nanValues{ocp_e4m3_mxfp8_e5m3::dataNaNMasks};
    return std::count(nanValues.begin(), nanValues.end(), data) > 0;
}

template <>
inline bool isNaNPacked<ocp_e4m3_mxfp8_e5m3>(uint8_t const* scaleBytes,
                                             uint8_t const* dataBytes,
                                             index_t        scaleIndex,
                                             index_t        dataIndex)
{
    return isNaN<ocp_e4m3_mxfp8_e5m3>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline bool isZero<ocp_e4m3_mxfp8_e5m3>(uint8_t const* scaleBytes,
                                        uint8_t const* dataBytes,
                                        index_t        scaleIndex,
                                        index_t        dataIndex)
{
    if(isNaN<ocp_e4m3_mxfp8_e5m3>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return false;

    // E5M3 has a zero representation; a zero scale forces the product to zero.
    if(*(scaleBytes + scaleIndex) == 0)
        return true;

    uint8_t data = *(dataBytes + dataIndex) & (~ocp_e4m3_mxfp8_e5m3::signBitMask);
    return data == ocp_e4m3_mxfp8_e5m3::positiveZeroMask;
}

template <>
inline bool isZeroPacked<ocp_e4m3_mxfp8_e5m3>(uint8_t const* scaleBytes,
                                              uint8_t const* dataBytes,
                                              index_t        scaleIndex,
                                              index_t        dataIndex)
{
    return isZero<ocp_e4m3_mxfp8_e5m3>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

// E4M3 data has no infinity representation; this is a property of the data
// element format, so it's the same regardless of the scale type.
template <>
inline bool isInf<ocp_e4m3_mxfp8_e5m3>(uint8_t const* scaleBytes [[maybe_unused]],
                                       uint8_t const* dataBytes [[maybe_unused]],
                                       index_t        scaleIndex [[maybe_unused]],
                                       index_t        dataIndex [[maybe_unused]])
{
    return false;
}

template <>
inline bool isInfPacked<ocp_e4m3_mxfp8_e5m3>(uint8_t const* scaleBytes,
                                             uint8_t const* dataBytes,
                                             index_t        scaleIndex,
                                             index_t        dataIndex)
{
    return isInf<ocp_e4m3_mxfp8_e5m3>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline double toDouble<ocp_e4m3_mxfp8_e5m3>(uint8_t const* scaleBytes,
                                            uint8_t const* dataBytes,
                                            index_t        scaleIndex,
                                            index_t        dataIndex)
{
    if(isNaN<ocp_e4m3_mxfp8_e5m3>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<double>::quiet_NaN();

    if(isZero<ocp_e4m3_mxfp8_e5m3>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0;

    uint8_t data     = *(dataBytes + dataIndex);
    int     scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e4m3_mxfp8_e5m3::scaleInfo.mantissaBits,
                                             ocp_e4m3_mxfp8_e5m3::scaleInfo.exponentBits);

    return convertToDouble<uint8_t, OCP_E4M3_MXFP8_DATA, ScaleInfo<ScaleType::E5M3>>(data,
                                                                                     scaleExp);
}

template <>
inline double toDoublePacked<ocp_e4m3_mxfp8_e5m3>(uint8_t const* scaleBytes,
                                                  uint8_t const* dataBytes,
                                                  index_t        scaleIndex,
                                                  index_t        dataIndex)
{
    return toDouble<ocp_e4m3_mxfp8_e5m3>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline float toFloat<ocp_e4m3_mxfp8_e5m3>(uint8_t const* scaleBytes,
                                          uint8_t const* dataBytes,
                                          index_t        scaleIndex,
                                          index_t        dataIndex)
{
    if(isNaN<ocp_e4m3_mxfp8_e5m3>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<float>::quiet_NaN();

    if(isZero<ocp_e4m3_mxfp8_e5m3>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint8_t data     = *(dataBytes + dataIndex);
    int     scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e4m3_mxfp8_e5m3::scaleInfo.mantissaBits,
                                             ocp_e4m3_mxfp8_e5m3::scaleInfo.exponentBits);

    return convertToFloat<uint8_t, OCP_E4M3_MXFP8_DATA, ScaleInfo<ScaleType::E5M3>>(data, scaleExp);
}

template <>
inline float toFloatPacked<ocp_e4m3_mxfp8_e5m3>(uint8_t const* scaleBytes,
                                                uint8_t const* dataBytes,
                                                index_t        scaleIndex,
                                                index_t        dataIndex)
{
    return toFloat<ocp_e4m3_mxfp8_e5m3>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline bool isSubnorm<ocp_e4m3_mxfp8_e5m3>(uint8_t const* dataBytes, index_t dataIndex)
{
    uint8_t data = *(dataBytes + dataIndex);
    return isSubNormal<uint16_t>(data,
                                 ocp_e4m3_mxfp8_e5m3::dataInfo.mantissaBits,
                                 ocp_e4m3_mxfp8_e5m3::dataInfo.exponentBits);
}

template <>
inline bool isSubnormPacked<ocp_e4m3_mxfp8_e5m3>(uint8_t const* dataBytes, index_t dataIndex)
{
    return isSubnorm<ocp_e4m3_mxfp8_e5m3>(dataBytes, dataIndex);
}

// `subNormal` is intentionally ignored: producing exactly 1.0 from a subnormal
// E4M3 data byte (smallest subnormal = 2^-8) would need a scale of 2^8 = 256,
// which is *just barely* representable in E5M3 but not in E4M3. To keep all
// gfx1250 (data, scale) variants behaving consistently we always fall back to
// the non-subnormal encoding here. The CPU reference and the kernels both
// interpret scale=1.0 * data=oneMask as 1.0 regardless, so this only loses
// the denorm-stress signal that the E8M0-scale path provides.
template <>
inline void setOne<ocp_e4m3_mxfp8_e5m3>(uint8_t* scaleBytes,
                                        uint8_t* dataBytes,
                                        index_t  scaleIndex,
                                        index_t  dataIndex,
                                        bool     subNormal [[maybe_unused]])
{
    *(scaleBytes + scaleIndex) = getScaleOne<ScaleType::E5M3>();
    *(dataBytes + dataIndex)   = ocp_e4m3_mxfp8_e5m3::oneMask;
}

template <>
inline void setZero<ocp_e4m3_mxfp8_e5m3>(uint8_t* scaleBytes [[maybe_unused]],
                                         uint8_t* dataBytes,
                                         index_t  scaleIndex [[maybe_unused]],
                                         index_t  dataIndex)
{
    *(dataBytes + dataIndex) = ocp_e4m3_mxfp8_e5m3::positiveZeroMask;
}

template <>
inline void setNaN<ocp_e4m3_mxfp8_e5m3>(uint8_t* scaleBytes,
                                        uint8_t* dataBytes [[maybe_unused]],
                                        index_t  scaleIndex,
                                        index_t  dataIndex [[maybe_unused]])
{
    *(scaleBytes + scaleIndex) = getScaleNan<ScaleType::E5M3>();
}

template <>
inline void setInf<ocp_e4m3_mxfp8_e5m3>(uint8_t* scaleBytes [[maybe_unused]],
                                        uint8_t* dataBytes [[maybe_unused]],
                                        index_t  scaleIndex [[maybe_unused]],
                                        index_t  dataIndex [[maybe_unused]])
{
    return;
}

template <>
inline void setDataMax<ocp_e4m3_mxfp8_e5m3>(uint8_t* dataBytes,
                                            index_t  dataIndex,
                                            bool     subNormal,
                                            bool     positive)
{
    if(subNormal)
        *(dataBytes + dataIndex) = positive ? ocp_e4m3_mxfp8_e5m3::dataMaxPositiveSubNormalMask
                                            : ocp_e4m3_mxfp8_e5m3::dataMaxNegativeSubNormalMask;
    else
        *(dataBytes + dataIndex) = positive ? ocp_e4m3_mxfp8_e5m3::dataMaxPositiveNormalMask
                                            : ocp_e4m3_mxfp8_e5m3::dataMaxNegativeNormalMask;
}

template <>
inline uint64_t satConvertToType<ocp_e4m3_mxfp8_e5m3>(float value)
{
    cvt t;
    t.num     = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
        return sign << 15 | ocp_e4m3_mxfp8_e5m3::dataNaNMasks[0];

    uint8_t res = convertToType<uint8_t, ocp_e4m3_mxfp8_e5m3>(value);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {getScaleOne<ScaleType::E5M3>()};

    float resVal = toFloat<ocp_e4m3_mxfp8_e5m3>(tScale, tData, 0, 0);

    if(std::abs(resVal) > ocp_e4m3_mxfp8_e5m3::dataMaxNormalNumber || std::isnan(resVal))
        return value < 0 ? ocp_e4m3_mxfp8_e5m3::dataMaxNegativeNormalMask
                         : ocp_e4m3_mxfp8_e5m3::dataMaxPositiveNormalMask;

    if(std::abs(resVal) < ocp_e4m3_mxfp8_e5m3::dataMinSubnormalNumber)
        return value < 0 ? ocp_e4m3_mxfp8_e5m3::negativeZeroMask
                         : ocp_e4m3_mxfp8_e5m3::positiveZeroMask;

    return res;
}

template <>
inline uint64_t nonSatConvertToType<ocp_e4m3_mxfp8_e5m3>(float value)
{
    uint8_t res = convertToType<uint8_t, ocp_e4m3_mxfp8_e5m3>(value);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {getScaleOne<ScaleType::E5M3>()};

    float resVal = toFloat<ocp_e4m3_mxfp8_e5m3>(tScale, tData, 0, 0);

    cvt t;
    t.num     = value;
    uint sign = t.bRep >> 31;

    if(std::abs(resVal) > ocp_e4m3_mxfp8_e5m3::dataMaxNormalNumber || std::isnan(value))
        return sign << 15 | ocp_e4m3_mxfp8_e5m3::dataNaNMasks[0];

    if(std::abs(resVal) < ocp_e4m3_mxfp8_e5m3::dataMinSubnormalNumber)
        return value < 0 ? ocp_e4m3_mxfp8_e5m3::negativeZeroMask
                         : ocp_e4m3_mxfp8_e5m3::positiveZeroMask;

    return res;
}

template <>
inline uint64_t satConvertToTypeSR<ocp_e4m3_mxfp8_e5m3>(float value, uint seed)
{
    cvt t;
    t.num     = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
        return sign << 7 | ocp_e4m3_mxfp8_e5m3::dataNaNMasks[0];
    else if(value > ocp_e4m3_mxfp8_e5m3::dataMaxRoundedRange)
        return ocp_e4m3_mxfp8_e5m3::dataMaxPositiveNormalMask;
    else if(value < -ocp_e4m3_mxfp8_e5m3::dataMaxRoundedRange)
        return ocp_e4m3_mxfp8_e5m3::dataMaxNegativeNormalMask;

    uint8_t res = convertToTypeSR<uint8_t, ocp_e4m3_mxfp8_e5m3>(value, seed);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {getScaleOne<ScaleType::E5M3>()};

    float resVal = toFloat<ocp_e4m3_mxfp8_e5m3>(tScale, tData, 0, 0);

    if(std::abs(resVal) > ocp_e4m3_mxfp8_e5m3::dataMaxNormalNumber || std::isnan(resVal))
        return value < 0 ? ocp_e4m3_mxfp8_e5m3::dataMaxNegativeNormalMask
                         : ocp_e4m3_mxfp8_e5m3::dataMaxPositiveNormalMask;

    if(std::abs(resVal) < ocp_e4m3_mxfp8_e5m3::dataMinSubnormalNumber)
        return value < 0 ? ocp_e4m3_mxfp8_e5m3::negativeZeroMask
                         : ocp_e4m3_mxfp8_e5m3::positiveZeroMask;

    return res;
}

template <>
inline uint64_t nonSatConvertToTypeSR<ocp_e4m3_mxfp8_e5m3>(float value, uint seed)
{
    cvt t;
    t.num     = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
        return sign << 7 | ocp_e4m3_mxfp8_e5m3::dataNaNMasks[0];
    else if(value > ocp_e4m3_mxfp8_e5m3::dataMaxRoundedRange)
        return ocp_e4m3_mxfp8_e5m3::dataNaNMasks[0];
    else if(value < -ocp_e4m3_mxfp8_e5m3::dataMaxRoundedRange)
        return ocp_e4m3_mxfp8_e5m3::dataNaNMasks[0];

    uint8_t res = convertToTypeSR<uint8_t, ocp_e4m3_mxfp8_e5m3>(value, seed);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {getScaleOne<ScaleType::E5M3>()};

    float resVal = toFloat<ocp_e4m3_mxfp8_e5m3>(tScale, tData, 0, 0);

    if(std::abs(resVal) > ocp_e4m3_mxfp8_e5m3::dataMaxNormalNumber)
        return sign << 7 | ocp_e4m3_mxfp8_e5m3::dataNaNMasks[0];

    if(std::abs(resVal) < ocp_e4m3_mxfp8_e5m3::dataMinSubnormalNumber)
        return value < 0 ? ocp_e4m3_mxfp8_e5m3::negativeZeroMask
                         : ocp_e4m3_mxfp8_e5m3::positiveZeroMask;

    return res;
}

// ============================================================================
// gfx1250 extension: ocp_e4m3_mxfp8_e4m3 (E4M3 data + E4M3 scale)
// ----------------------------------------------------------------------------
// Same template as the E5M3-scale variant but using ScaleType::E4M3. This is
// the combination exercised by `mxf8_gfx1250.yaml` `DataTypeMXSA: f8`.
// ============================================================================

template <>
inline bool isNaN<ocp_e4m3_mxfp8_e4m3>(uint8_t const* scaleBytes,
                                       uint8_t const* dataBytes,
                                       index_t        scaleIndex,
                                       index_t        dataIndex)
{
    uint8_t data  = *(dataBytes + dataIndex);
    uint8_t scale = *(scaleBytes + scaleIndex);

    if(scale == getScaleNan<ScaleType::E4M3>())
        return true;

    data &= (~ocp_e4m3_mxfp8_e4m3::signBitMask);
    auto& nanValues{ocp_e4m3_mxfp8_e4m3::dataNaNMasks};
    return std::count(nanValues.begin(), nanValues.end(), data) > 0;
}

template <>
inline bool isNaNPacked<ocp_e4m3_mxfp8_e4m3>(uint8_t const* scaleBytes,
                                             uint8_t const* dataBytes,
                                             index_t        scaleIndex,
                                             index_t        dataIndex)
{
    return isNaN<ocp_e4m3_mxfp8_e4m3>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline bool isZero<ocp_e4m3_mxfp8_e4m3>(uint8_t const* scaleBytes,
                                        uint8_t const* dataBytes,
                                        index_t        scaleIndex,
                                        index_t        dataIndex)
{
    if(isNaN<ocp_e4m3_mxfp8_e4m3>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return false;

    // E4M3 has a zero representation; a zero scale forces the product to zero.
    if(*(scaleBytes + scaleIndex) == 0)
        return true;

    uint8_t data = *(dataBytes + dataIndex) & (~ocp_e4m3_mxfp8_e4m3::signBitMask);
    return data == ocp_e4m3_mxfp8_e4m3::positiveZeroMask;
}

template <>
inline bool isZeroPacked<ocp_e4m3_mxfp8_e4m3>(uint8_t const* scaleBytes,
                                              uint8_t const* dataBytes,
                                              index_t        scaleIndex,
                                              index_t        dataIndex)
{
    return isZero<ocp_e4m3_mxfp8_e4m3>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline bool isInf<ocp_e4m3_mxfp8_e4m3>(uint8_t const* scaleBytes [[maybe_unused]],
                                       uint8_t const* dataBytes [[maybe_unused]],
                                       index_t        scaleIndex [[maybe_unused]],
                                       index_t        dataIndex [[maybe_unused]])
{
    return false;
}

template <>
inline bool isInfPacked<ocp_e4m3_mxfp8_e4m3>(uint8_t const* scaleBytes,
                                             uint8_t const* dataBytes,
                                             index_t        scaleIndex,
                                             index_t        dataIndex)
{
    return isInf<ocp_e4m3_mxfp8_e4m3>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline double toDouble<ocp_e4m3_mxfp8_e4m3>(uint8_t const* scaleBytes,
                                            uint8_t const* dataBytes,
                                            index_t        scaleIndex,
                                            index_t        dataIndex)
{
    if(isNaN<ocp_e4m3_mxfp8_e4m3>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<double>::quiet_NaN();

    if(isZero<ocp_e4m3_mxfp8_e4m3>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0;

    uint8_t data     = *(dataBytes + dataIndex);
    int     scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e4m3_mxfp8_e4m3::scaleInfo.mantissaBits,
                                             ocp_e4m3_mxfp8_e4m3::scaleInfo.exponentBits);

    return convertToDouble<uint8_t, OCP_E4M3_MXFP8_DATA, ScaleInfo<ScaleType::E4M3>>(data,
                                                                                     scaleExp);
}

template <>
inline double toDoublePacked<ocp_e4m3_mxfp8_e4m3>(uint8_t const* scaleBytes,
                                                  uint8_t const* dataBytes,
                                                  index_t        scaleIndex,
                                                  index_t        dataIndex)
{
    return toDouble<ocp_e4m3_mxfp8_e4m3>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline float toFloat<ocp_e4m3_mxfp8_e4m3>(uint8_t const* scaleBytes,
                                          uint8_t const* dataBytes,
                                          index_t        scaleIndex,
                                          index_t        dataIndex)
{
    if(isNaN<ocp_e4m3_mxfp8_e4m3>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<float>::quiet_NaN();

    if(isZero<ocp_e4m3_mxfp8_e4m3>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint8_t data     = *(dataBytes + dataIndex);
    int     scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e4m3_mxfp8_e4m3::scaleInfo.mantissaBits,
                                             ocp_e4m3_mxfp8_e4m3::scaleInfo.exponentBits);

    return convertToFloat<uint8_t, OCP_E4M3_MXFP8_DATA, ScaleInfo<ScaleType::E4M3>>(data, scaleExp);
}

template <>
inline float toFloatPacked<ocp_e4m3_mxfp8_e4m3>(uint8_t const* scaleBytes,
                                                uint8_t const* dataBytes,
                                                index_t        scaleIndex,
                                                index_t        dataIndex)
{
    return toFloat<ocp_e4m3_mxfp8_e4m3>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline bool isSubnorm<ocp_e4m3_mxfp8_e4m3>(uint8_t const* dataBytes, index_t dataIndex)
{
    uint8_t data = *(dataBytes + dataIndex);
    return isSubNormal<uint16_t>(data,
                                 ocp_e4m3_mxfp8_e4m3::dataInfo.mantissaBits,
                                 ocp_e4m3_mxfp8_e4m3::dataInfo.exponentBits);
}

template <>
inline bool isSubnormPacked<ocp_e4m3_mxfp8_e4m3>(uint8_t const* dataBytes, index_t dataIndex)
{
    return isSubnorm<ocp_e4m3_mxfp8_e4m3>(dataBytes, dataIndex);
}

// See the matching note on setOne<ocp_e4m3_mxfp8_e5m3>: subnormal-encoded 1.0
// requires a scale of 2^8, which exceeds the E4M3 scale's max (~240), so we
// always emit the normal-mode encoding regardless of `subNormal`.
template <>
inline void setOne<ocp_e4m3_mxfp8_e4m3>(uint8_t* scaleBytes,
                                        uint8_t* dataBytes,
                                        index_t  scaleIndex,
                                        index_t  dataIndex,
                                        bool     subNormal [[maybe_unused]])
{
    *(scaleBytes + scaleIndex) = getScaleOne<ScaleType::E4M3>();
    *(dataBytes + dataIndex)   = ocp_e4m3_mxfp8_e4m3::oneMask;
}

template <>
inline void setZero<ocp_e4m3_mxfp8_e4m3>(uint8_t* scaleBytes [[maybe_unused]],
                                         uint8_t* dataBytes,
                                         index_t  scaleIndex [[maybe_unused]],
                                         index_t  dataIndex)
{
    *(dataBytes + dataIndex) = ocp_e4m3_mxfp8_e4m3::positiveZeroMask;
}

template <>
inline void setNaN<ocp_e4m3_mxfp8_e4m3>(uint8_t* scaleBytes,
                                        uint8_t* dataBytes [[maybe_unused]],
                                        index_t  scaleIndex,
                                        index_t  dataIndex [[maybe_unused]])
{
    *(scaleBytes + scaleIndex) = getScaleNan<ScaleType::E4M3>();
}

template <>
inline void setInf<ocp_e4m3_mxfp8_e4m3>(uint8_t* scaleBytes [[maybe_unused]],
                                        uint8_t* dataBytes [[maybe_unused]],
                                        index_t  scaleIndex [[maybe_unused]],
                                        index_t  dataIndex [[maybe_unused]])
{
    return;
}

template <>
inline void setDataMax<ocp_e4m3_mxfp8_e4m3>(uint8_t* dataBytes,
                                            index_t  dataIndex,
                                            bool     subNormal,
                                            bool     positive)
{
    if(subNormal)
        *(dataBytes + dataIndex) = positive ? ocp_e4m3_mxfp8_e4m3::dataMaxPositiveSubNormalMask
                                            : ocp_e4m3_mxfp8_e4m3::dataMaxNegativeSubNormalMask;
    else
        *(dataBytes + dataIndex) = positive ? ocp_e4m3_mxfp8_e4m3::dataMaxPositiveNormalMask
                                            : ocp_e4m3_mxfp8_e4m3::dataMaxNegativeNormalMask;
}

template <>
inline uint64_t satConvertToType<ocp_e4m3_mxfp8_e4m3>(float value)
{
    cvt t;
    t.num     = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
        return sign << 15 | ocp_e4m3_mxfp8_e4m3::dataNaNMasks[0];

    uint8_t res = convertToType<uint8_t, ocp_e4m3_mxfp8_e4m3>(value);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {getScaleOne<ScaleType::E4M3>()};

    float resVal = toFloat<ocp_e4m3_mxfp8_e4m3>(tScale, tData, 0, 0);

    if(std::abs(resVal) > ocp_e4m3_mxfp8_e4m3::dataMaxNormalNumber || std::isnan(resVal))
        return value < 0 ? ocp_e4m3_mxfp8_e4m3::dataMaxNegativeNormalMask
                         : ocp_e4m3_mxfp8_e4m3::dataMaxPositiveNormalMask;

    if(std::abs(resVal) < ocp_e4m3_mxfp8_e4m3::dataMinSubnormalNumber)
        return value < 0 ? ocp_e4m3_mxfp8_e4m3::negativeZeroMask
                         : ocp_e4m3_mxfp8_e4m3::positiveZeroMask;

    return res;
}

template <>
inline uint64_t nonSatConvertToType<ocp_e4m3_mxfp8_e4m3>(float value)
{
    uint8_t res = convertToType<uint8_t, ocp_e4m3_mxfp8_e4m3>(value);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {getScaleOne<ScaleType::E4M3>()};

    float resVal = toFloat<ocp_e4m3_mxfp8_e4m3>(tScale, tData, 0, 0);

    cvt t;
    t.num     = value;
    uint sign = t.bRep >> 31;

    if(std::abs(resVal) > ocp_e4m3_mxfp8_e4m3::dataMaxNormalNumber || std::isnan(value))
        return sign << 15 | ocp_e4m3_mxfp8_e4m3::dataNaNMasks[0];

    if(std::abs(resVal) < ocp_e4m3_mxfp8_e4m3::dataMinSubnormalNumber)
        return value < 0 ? ocp_e4m3_mxfp8_e4m3::negativeZeroMask
                         : ocp_e4m3_mxfp8_e4m3::positiveZeroMask;

    return res;
}

template <>
inline uint64_t satConvertToTypeSR<ocp_e4m3_mxfp8_e4m3>(float value, uint seed)
{
    cvt t;
    t.num     = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
        return sign << 7 | ocp_e4m3_mxfp8_e4m3::dataNaNMasks[0];
    else if(value > ocp_e4m3_mxfp8_e4m3::dataMaxRoundedRange)
        return ocp_e4m3_mxfp8_e4m3::dataMaxPositiveNormalMask;
    else if(value < -ocp_e4m3_mxfp8_e4m3::dataMaxRoundedRange)
        return ocp_e4m3_mxfp8_e4m3::dataMaxNegativeNormalMask;

    uint8_t res = convertToTypeSR<uint8_t, ocp_e4m3_mxfp8_e4m3>(value, seed);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {getScaleOne<ScaleType::E4M3>()};

    float resVal = toFloat<ocp_e4m3_mxfp8_e4m3>(tScale, tData, 0, 0);

    if(std::abs(resVal) > ocp_e4m3_mxfp8_e4m3::dataMaxNormalNumber || std::isnan(resVal))
        return value < 0 ? ocp_e4m3_mxfp8_e4m3::dataMaxNegativeNormalMask
                         : ocp_e4m3_mxfp8_e4m3::dataMaxPositiveNormalMask;

    if(std::abs(resVal) < ocp_e4m3_mxfp8_e4m3::dataMinSubnormalNumber)
        return value < 0 ? ocp_e4m3_mxfp8_e4m3::negativeZeroMask
                         : ocp_e4m3_mxfp8_e4m3::positiveZeroMask;

    return res;
}

template <>
inline uint64_t nonSatConvertToTypeSR<ocp_e4m3_mxfp8_e4m3>(float value, uint seed)
{
    cvt t;
    t.num     = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
        return sign << 7 | ocp_e4m3_mxfp8_e4m3::dataNaNMasks[0];
    else if(value > ocp_e4m3_mxfp8_e4m3::dataMaxRoundedRange)
        return ocp_e4m3_mxfp8_e4m3::dataNaNMasks[0];
    else if(value < -ocp_e4m3_mxfp8_e4m3::dataMaxRoundedRange)
        return ocp_e4m3_mxfp8_e4m3::dataNaNMasks[0];

    uint8_t res = convertToTypeSR<uint8_t, ocp_e4m3_mxfp8_e4m3>(value, seed);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {getScaleOne<ScaleType::E4M3>()};

    float resVal = toFloat<ocp_e4m3_mxfp8_e4m3>(tScale, tData, 0, 0);

    if(std::abs(resVal) > ocp_e4m3_mxfp8_e4m3::dataMaxNormalNumber)
        return sign << 7 | ocp_e4m3_mxfp8_e4m3::dataNaNMasks[0];

    if(std::abs(resVal) < ocp_e4m3_mxfp8_e4m3::dataMinSubnormalNumber)
        return value < 0 ? ocp_e4m3_mxfp8_e4m3::negativeZeroMask
                         : ocp_e4m3_mxfp8_e4m3::positiveZeroMask;

    return res;
}
