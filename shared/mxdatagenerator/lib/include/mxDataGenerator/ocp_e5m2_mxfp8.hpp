// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <algorithm>
#include <array>

namespace DGen
{
    struct OCP_E5M2_MXFP8_DATA
    {
        static constexpr uint signBits     = 1;
        static constexpr uint exponentBits = 5;
        static constexpr uint mantissaBits = 2;
        static constexpr uint bias         = 15;
        static constexpr uint srShift      = 11;

        static constexpr int unBiasedEMin = -14;
        static constexpr int unBiasedEMax = 16;
        static constexpr int biasedEMin   = 1;
        static constexpr int biasedEMax   = 31;

        static constexpr bool hasInf  = true;
        static constexpr bool hasNan  = true;
        static constexpr bool hasZero = true;
    };

    // Data-type constants shared by all ocp_e5m2_mxfp8 scale variants. Only
    // `scaleInfo` differs between the variants (E8M0 / E5M3 / E4M3); all of
    // the data-side bit patterns and ranges stay identical because the data
    // element format is E5M2 in every case.
    struct ocp_e5m2_mxfp8_base
    {
        static constexpr OCP_E5M2_MXFP8_DATA dataInfo{};

        static constexpr uint8_t oneMask         = 0b00111100;
        static constexpr uint8_t signBitMask     = 0b10000000;
        static constexpr uint8_t positiveInfMask = 0b01111100;
        static constexpr uint8_t negativeInfMask = 0b11111100;

        static constexpr uint8_t dataMaxPositiveNormalMask = 0b01111011;
        static constexpr uint8_t dataMaxNegativeNormalMask = 0b11111011;

        static constexpr uint8_t dataSubNormalOneMask = 0b00000010;

        static constexpr uint8_t dataMaxPositiveSubNormalMask = 0b00000011;
        static constexpr uint8_t dataMaxNegativeSubNormalMask = 0b10000011;

        static constexpr float dataMaxNormalNumber    = 57344;
        static constexpr float dataMinSubNormalNumber = 0.00001525878906250000;

        static constexpr uint8_t positiveZeroMask = 0b00000000;
        static constexpr uint8_t negativeZeroMask = 0b10000000;

        static constexpr float dataMaxRoundedRange = 65536; // 2^15 * 2

        static constexpr std::array<uint8_t, 6> dataNaNMasks{
            0b11111101, 0b11111110, 0b11111111, 0b01111101, 0b01111110, 0b01111111};
    };

    // OCP MXFP8 (E5M2 data) with the spec-mandated UE8M0 scale.
    struct ocp_e5m2_mxfp8 : ocp_e5m2_mxfp8_base
    {
        static constexpr ScaleInfo<ScaleType::E8M0> scaleInfo{};
    };

    // gfx1250 extension: MXFP8 (E5M2 data) with an OCP E5M3 8-bit-float scale.
    // Spec MXFP8 mandates UE8M0; this combination is exercised by the
    // gfx1250 scaled-WMMA matrix_*_scale_fmt:E5M3 mode.
    struct ocp_e5m2_mxfp8_e5m3 : ocp_e5m2_mxfp8_base
    {
        static constexpr ScaleInfo<ScaleType::E5M3> scaleInfo{};
    };

    // gfx1250 extension: MXFP8 (E5M2 data) with an OCP E4M3 8-bit-float scale.
    // Spec MXFP8 mandates UE8M0; this combination is exercised by the
    // gfx1250 scaled-WMMA matrix_*_scale_fmt:E4M3 mode.
    struct ocp_e5m2_mxfp8_e4m3 : ocp_e5m2_mxfp8_base
    {
        static constexpr ScaleInfo<ScaleType::E4M3> scaleInfo{};
    };

#include "ocp_e5m2_mxfp8_impl.hpp"
}
