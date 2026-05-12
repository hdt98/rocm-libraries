// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <algorithm>
#include <array>

namespace DGen
{
    struct OCP_E4M3_MXFP8_DATA
    {
        static constexpr uint signBits     = 1;
        static constexpr uint exponentBits = 4;
        static constexpr uint mantissaBits = 3;
        static constexpr uint bias         = 7;
        static constexpr uint srShift      = 12;

        static constexpr int unBiasedEMin = -6;
        static constexpr int unBiasedEMax = 8;
        static constexpr int biasedEMin   = 1;
        static constexpr int biasedEMax   = 15;

        static constexpr bool hasInf  = false;
        static constexpr bool hasNan  = true;
        static constexpr bool hasZero = true;
    };

    // Data-type constants shared by all ocp_e4m3_mxfp8 scale variants. Only
    // `scaleInfo` differs between the variants (E8M0 / E5M3 / E4M3); all of
    // the data-side bit patterns and ranges stay identical because the data
    // element format is E4M3 in every case.
    struct ocp_e4m3_mxfp8_base
    {
        static constexpr OCP_E4M3_MXFP8_DATA dataInfo{};

        static constexpr uint8_t oneMask     = 0b00111000;
        static constexpr uint8_t signBitMask = 0b10000000;

        static constexpr uint8_t dataMaxPositiveNormalMask = 0b01111110;
        static constexpr uint8_t dataMaxNegativeNormalMask = 0b11111110;

        static constexpr uint8_t dataMaxPositiveSubNormalMask = 0b00000111;
        static constexpr uint8_t dataMaxNegativeSubNormalMask = 0b10000111;

        static constexpr uint8_t dataSubNormalOneMask = 0b00000010;

        static constexpr float dataMaxNormalNumber    = 448;
        static constexpr float dataMinSubnormalNumber = 0.0019531250;

        static constexpr uint8_t positiveZeroMask = 0b00000000;
        static constexpr uint8_t negativeZeroMask = 0b10000000;

        static constexpr uint8_t scaleNanMask = 0b11111111;

        static constexpr float dataMaxRoundedRange = 480; // 2^8 * 1.875

        static constexpr std::array<uint8_t, 2> dataNaNMasks{0b01111111, 0b11111111};
    };

    // OCP MXFP8 (E4M3 data) with the spec-mandated UE8M0 scale.
    struct ocp_e4m3_mxfp8 : ocp_e4m3_mxfp8_base
    {
        static constexpr ScaleInfo<ScaleType::E8M0> scaleInfo{};
    };

    // gfx1250 extension: MXFP8 (E4M3 data) with an OCP E5M3 8-bit-float scale.
    // Spec MXFP8 mandates UE8M0; this combination is exercised by the
    // gfx1250 scaled-WMMA matrix_*_scale_fmt:E5M3 mode.
    struct ocp_e4m3_mxfp8_e5m3 : ocp_e4m3_mxfp8_base
    {
        static constexpr ScaleInfo<ScaleType::E5M3> scaleInfo{};
    };

    // gfx1250 extension: MXFP8 (E4M3 data) with an OCP E4M3 8-bit-float scale.
    // Spec MXFP8 mandates UE8M0; this combination is exercised by the
    // gfx1250 scaled-WMMA matrix_*_scale_fmt:E4M3 mode and is the form used
    // by `mxf8_gfx1250.yaml` `DataTypeMXSA: f8` test groups.
    struct ocp_e4m3_mxfp8_e4m3 : ocp_e4m3_mxfp8_base
    {
        static constexpr ScaleInfo<ScaleType::E4M3> scaleInfo{};
    };

#include "ocp_e4m3_mxfp8_impl.hpp"
}
