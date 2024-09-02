// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/amd_sba.hpp"

namespace ck {

enum struct SbaInstr
{
    // gfx13
    sba_f32 = 0,
    sba_half,
    sba_bf16,
    sba_scatter2_half,
    sba_scatter2_bf16
};

template <SbaInstr Instr, index_t auxdata, bool scaleBiasPacked, bool uniformScale>
struct sba_type
{
    sba_type() { static_assert(false, "never called"); }
};

// scaleBiasPacked = 0, uniformScale = 0
template <index_t auxdata>
struct sba_type<SbaInstr::sba_f32, auxdata, 0, 0>
{
    template <class FloatAcc>
    __device__ void
    Run(const FloatAcc& inAcc, const float& scale, const float& bias, FloatAcc& outSba) const
    {
#if defined(__gfx13__)
        intrin_sba_f32<auxdata, 0, 0>::Run(inAcc, scale, bias, outSba);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = bias;
        ignore = outSba;
#endif
    }
};

// scaleBiasPacked = 0, uniformScale = 1
template <index_t auxdata>
struct sba_type<SbaInstr::sba_f32, auxdata, 0, 1>
{
    template <class FloatAcc>
    __device__ void Run(const FloatAcc& inAcc,
                        const float& scale,
                        const float& bias, // unused
                        FloatAcc& outSba) const
    {
#if defined(__gfx13__)
        intrin_sba_f32<auxdata, 0, 1>::Run(inAcc, scale, outSba);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = bias;
        ignore = outSba;
#endif
    }
};

// scaleBiasPacked = 1, uniformScale = 0
template <index_t auxdata>
struct sba_type<SbaInstr::sba_f32, auxdata, 1, 0>
{
    template <class FloatAcc>
    __device__ void Run(const FloatAcc& inAcc,
                        const float& scale, // unused
                        const float& scalebias,
                        FloatAcc& outSba) const
    {
#if defined(__gfx13__)
        intrin_sba_f32<auxdata, 1, 0>::Run(inAcc, scalebias, outSba);
#else
        ignore = inAcc;
        ignore = scalebias;
        ignore = outSba;
#endif
    }
};

// scaleBiasPacked = 0, uniformScale = 0
template <index_t auxdata>
struct sba_type<SbaInstr::sba_half, auxdata, 0, 0>
{
    template <class FloatAcc>
    __device__ void
    Run(const FloatAcc& inAcc, const float& scale, const half2_t& bias, FloatAcc& outSba) const
    {
#if defined(__gfx13__)
        intrin_sba_half<auxdata, 0, 0>::Run(inAcc, scale, bias, outSba);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = bias;
        ignore = outSba;
#endif
    }
};

// scaleBiasPacked = 0, uniformScale = 1
template <index_t auxdata>
struct sba_type<SbaInstr::sba_half, auxdata, 0, 1>
{
    template <class FloatAcc>
    __device__ void Run(const FloatAcc& inAcc,
                        const float& scale,
                        const half2_t& bias, // unused
                        FloatAcc& outSba) const
    {
#if defined(__gfx13__)
        intrin_sba_half<auxdata, 0, 1>::Run(inAcc, scale, outSba);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = outSba;
#endif
    }
};

// scaleBiasPacked = 1, uniformScale = 0
template <index_t auxdata>
struct sba_type<SbaInstr::sba_half, auxdata, 1, 0>
{
    template <class FloatAcc>
    __device__ void Run(const FloatAcc& inAcc,
                        const float& scale, // unused
                        const half2_t& bias,
                        FloatAcc& outSba) const
    {
#if defined(__gfx13__)
        intrin_sba_half<auxdata, 1, 0>::Run(inAcc, bias, outSba);
#else
        ignore = inAcc;
        ignore = bias;
        ignore = outSba;
#endif
    }
};

// scaleBiasPacked = 0, uniformScale = 0
template <index_t auxdata>
struct sba_type<SbaInstr::sba_bf16, auxdata, 0, 0>
{
    template <class FloatAcc>
    __device__ void
    Run(const FloatAcc& inAcc, const float& scale, const bhalf2_t& bias, FloatAcc& outSba) const
    {
#if defined(__gfx13__)
        intrin_sba_bhalf<auxdata, 0, 0>::Run(inAcc, scale, bias, outSba);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = bias;
        ignore = outSba;
#endif
    }
};

// scaleBiasPacked = 0, uniformScale = 1
template <index_t auxdata>
struct sba_type<SbaInstr::sba_bf16, auxdata, 0, 1>
{
    template <class FloatAcc>
    __device__ void Run(const FloatAcc& inAcc,
                        const float& scale,
                        const bhalf2_t& bias, // unused
                        FloatAcc& outSba) const
    {
#if defined(__gfx13__)
        intrin_sba_bhalf<auxdata, 0, 1>::Run(inAcc, scale, outSba);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = bias;
        ignore = outSba;
#endif
    }
};

// scaleBiasPacked = 1, uniformScale = 0
template <index_t auxdata>
struct sba_type<SbaInstr::sba_bf16, auxdata, 1, 0>
{
    template <class FloatAcc>
    __device__ void Run(const FloatAcc& inAcc,
                        const float& scale, // unused
                        const bhalf2_t& bias,
                        FloatAcc& outSba) const
    {
#if defined(__gfx13__)
        intrin_sba_bhalf<auxdata, 1, 0>::Run(inAcc, bias, outSba);
#else
        ignore = inAcc;
        ignore = bias;
        ignore = outSba;
#endif
    }
};

template <typename AccDataType, index_t auxdata, bool scaleBiasPacked, bool uniformScale>
struct SbaSelector
{
    template <typename AccDataType_>
    static constexpr auto GetSba();

    template <>
    static constexpr auto GetSba<float_t>()
    {
        return SbaInstr::sba_f32;
    }
    template <>
    static constexpr auto GetSba<half_t>()
    {
        return SbaInstr::sba_half;
    }
    template <>
    static constexpr auto GetSba<bhalf_t>()
    {
        return SbaInstr::sba_bf16;
    }
    // To do for scatter

    static constexpr auto selected_sba =
        sba_type<GetSba<AccDataType>(), auxdata, scaleBiasPacked, uniformScale>{};

    __host__ __device__ constexpr SbaSelector(){};
};

template <typename AccDataType,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t activeFun,
          index_t auxdata,
          bool scaleBiasPacked,
          bool uniformScale>
struct AccSba
{
    __host__ __device__ constexpr AccSba(){};

    static constexpr index_t GetNumSbaOutComponents()
    {
        return std::is_same<float_t, AccDataType>::value ? 4 : 8;
    }

    static constexpr auto sba = SbaSelector<AccDataType, auxdata, scaleBiasPacked, uniformScale>{};
    static constexpr auto sba_instr = sba.selected_sba;
};

} // namespace ck
