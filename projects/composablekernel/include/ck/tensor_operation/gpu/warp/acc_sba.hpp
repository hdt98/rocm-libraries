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
                        const float&, // unused
                        FloatAcc& outSba) const
    {
#if defined(__gfx13__)
        intrin_sba_f32<auxdata, 0, 1>::Run(inAcc, scale, outSba);
#else
        ignore = inAcc;
        ignore = scale;
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
                        const float&, // unused
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
    Run(const FloatAcc& inAcc, const half& scale, const half2_t& bias, FloatAcc& outSba) const
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
                        const half& scale,
                        const half2_t&, // unused
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
                        const half&, // unused
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
    Run(const FloatAcc& inAcc, const bhalf_t& scale, const bhalf2_t& bias, FloatAcc& outSba) const
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
                        const bhalf_t& scale,
                        const bhalf2_t&, // unused
                        FloatAcc& outSba) const
    {
#if defined(__gfx13__)
        intrin_sba_bhalf<auxdata, 0, 1>::Run(inAcc, scale, outSba);
#else
        ignore = inAcc;
        ignore = scale;
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
                        const bhalf_t&, // unused
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

// scatter=2 scaleBiasPacked = 0, uniformScale = 0
template <index_t auxdata>
struct sba_type<SbaInstr::sba_scatter2_half, auxdata, 0, 0>
{
    template <class FloatAcc>
    __device__ void Run(const FloatAcc& inAcc,
                        const half& scale,
                        const half2_t& bias,
                        half2_t& outSba_0,
                        half2_t& outSba_1) const
    {
#if defined(__gfx13__)
        intrin_sba_scatter2_half<auxdata, 0, 0>::Run(inAcc, scale, bias, outSba_0, outSba_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = bias;
        ignore = outSba_0;
        ignore = outSba_1;
#endif
    }
};

template <index_t auxdata>
struct sba_type<SbaInstr::sba_scatter2_half, auxdata, 0, 1>
{
    template <class FloatAcc>
    __device__ void Run(const FloatAcc& inAcc,
                        const half& scale,
                        const half2_t&, // unused
                        half2_t& outSba_0,
                        half2_t& outSba_1) const
    {
#if defined(__gfx13__)
        intrin_sba_scatter2_half<auxdata, 0, 1>::Run(inAcc, scale, outSba_0, outSba_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = outSba_0;
        ignore = outSba_1;
#endif
    }
};

template <index_t auxdata>
struct sba_type<SbaInstr::sba_scatter2_half, auxdata, 1, 0>
{
    template <class FloatAcc>
    __device__ void Run(const FloatAcc& inAcc,
                        const half&, // unused
                        const half2_t& bias,
                        half2_t& outSba_0,
                        half2_t& outSba_1) const
    {
#if defined(__gfx13__)
        intrin_sba_scatter2_half<auxdata, 1, 0>::Run(inAcc, bias, outSba_0, outSba_1);
#else
        ignore = inAcc;
        ignore = bias;
        ignore = outSba_0;
        ignore = outSba_1;
#endif
    }
};

// scatter=2 scaleBiasPacked = 0, uniformScale = 0
template <index_t auxdata>
struct sba_type<SbaInstr::sba_scatter2_bf16, auxdata, 0, 0>
{
    template <class FloatAcc>
    __device__ void Run(const FloatAcc& inAcc,
                        const bhalf_t& scale,
                        const bhalf2_t& bias,
                        bhalf2_t& outSba_0,
                        bhalf2_t& outSba_1) const
    {
#if defined(__gfx13__)
        intrin_sba_scatter2_bhalf<auxdata, 0, 0>::Run(inAcc, scale, bias, outSba_0, outSba_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = bias;
        ignore = outSba_0;
        ignore = outSba_1;
#endif
    }
};

template <index_t auxdata>
struct sba_type<SbaInstr::sba_scatter2_bf16, auxdata, 0, 1>
{
    template <class FloatAcc>
    __device__ void Run(const FloatAcc& inAcc,
                        const bhalf_t& scale,
                        const bhalf2_t&, // unused
                        bhalf2_t& outSba_0,
                        bhalf2_t& outSba_1) const
    {
#if defined(__gfx13__)
        intrin_sba_scatter2_bhalf<auxdata, 0, 1>::Run(inAcc, scale, outSba_0, outSba_1);
#else
        ignore = inAcc;
        ignore = scale;
        ignore = outSba_0;
        ignore = outSba_1;
#endif
    }
};

template <index_t auxdata>
struct sba_type<SbaInstr::sba_scatter2_bf16, auxdata, 1, 0>
{
    template <class FloatAcc>
    __device__ void Run(const FloatAcc& inAcc,
                        const bhalf_t&, // unused
                        const bhalf2_t& bias,
                        bhalf2_t& outSba_0,
                        bhalf2_t& outSba_1) const
    {
#if defined(__gfx13__)
        intrin_sba_scatter2_bhalf<auxdata, 1, 0>::Run(inAcc, bias, outSba_0, outSba_1);
#else
        ignore = inAcc;
        ignore = bias;
        ignore = outSba_0;
        ignore = outSba_1;
#endif
    }
};

template <typename AccDataType,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t auxdata,
          bool scaleBiasPacked,
          bool uniformScale>
struct SbaSelector
{
    template <typename AccDataType_>
    static constexpr auto GetSba();

    template <>
    constexpr auto GetSba<float_t>()
    {
        return SbaInstr::sba_f32;
    }
    template <>
    constexpr auto GetSba<half_t>()
    {
        if constexpr(HPerWconv == 4 && WPerWconv == 2)
        {
            return SbaInstr::sba_scatter2_half;
        }
        else
        {
            return SbaInstr::sba_half;
        }
    }
    template <>
    constexpr auto GetSba<bhalf_t>()
    {
        if constexpr(HPerWconv == 4 && WPerWconv == 2)
        {
            return SbaInstr::sba_scatter2_bf16;
        }
        else
        {
            return SbaInstr::sba_bf16;
        }
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
          bool scaleBiasPacked,
          bool uniformScale,
          bool Aco = false>
struct AccSba
{
    __host__ __device__ constexpr AccSba(){};

    static constexpr index_t WaveSize = 32;

    static constexpr index_t GetNumSbaOutComponents()
    {
        if constexpr(HPerWconv == 4 && WPerWconv == 2)
        {
            return 4;
        }
        else
        {
            return 8;
        }
    }

    static constexpr index_t GetNumBiasComponents()
    {
        if constexpr(!scaleBiasPacked)
        {
            if(std::is_same<ck::half_t, AccDataType>::value ||
               std::is_same<ck::bhalf_t, AccDataType>::value)
            {
                return 2;
            }
        }
        return 1;
    }

    static constexpr index_t GetNumScaleComponents()
    {
        // Keep consistent with bias even it hasn't been used in non-pack cases
        if constexpr(!scaleBiasPacked)
        {
            if(std::is_same<ck::half_t, AccDataType>::value ||
               std::is_same<ck::bhalf_t, AccDataType>::value)
            {
                return 2;
            }
        }
        return 1;
    }

    static constexpr index_t GetWaveMaxBiasNumber()
    {
        if constexpr(scaleBiasPacked)
        {
            if constexpr(std::is_same<ck::half_t, AccDataType>::value ||
                         std::is_same<ck::bhalf_t, AccDataType>::value)
            {
                return 32;
            }
            else if constexpr(std::is_same<float, AccDataType>::value)
            {
                return 16;
            }
            else
            {
                static_assert("Never called\n");
            }
        }
        else
        {
            if constexpr(std::is_same<ck::half_t, AccDataType>::value ||
                         std::is_same<ck::bhalf_t, AccDataType>::value)
            {
                return 64;
            }
            else if constexpr(std::is_same<float, AccDataType>::value)
            {
                return 32;
            }
            else
            {
                static_assert("Never called\n");
            }
        }
    }

    using BiasVec      = vector_type<AccDataType, GetNumBiasComponents()>;
    using ScaleVec     = vector_type<AccDataType, GetNumScaleComponents()>;
    using BiasScaleVec = vector_type<AccDataType, 2>;

    // k0:repeat_number KPerWconv:k_number per instruction
    static constexpr auto CalculateDsThreadOriginDataIndex(ck::index_t KPerWconv)
    {
        // ToDo for VGPR split
        index_t K_offset            = 0;
        ck::index_t perThreadNumber = 0;
        auto laneId                 = get_lane_id();
        if constexpr(scaleBiasPacked)
        {
            perThreadNumber = 1;
            if(std::is_same<ck::half_t, AccDataType>::value ||
               std::is_same<ck::bhalf_t, AccDataType>::value)
            {
                K_offset = laneId % (KPerWconv / perThreadNumber);
            }
            else if(std::is_same<float, AccDataType>::value)
            {
                // 32bit:0.5element perThread
                K_offset = (laneId % (KPerWconv * 2 / perThreadNumber)) / 2;
            }
            return make_tuple(K_offset, 0);
        }
        else
        {
            if(std::is_same<ck::half_t, AccDataType>::value ||
               std::is_same<ck::bhalf_t, AccDataType>::value)
            {
                perThreadNumber = 2;
            }
            else if(std::is_same<float, AccDataType>::value)
            {
                perThreadNumber = 1;
            }
            K_offset = laneId % (KPerWconv / perThreadNumber);
            return make_tuple(K_offset, 0);
        }
    }

    static constexpr auto GetAuxData()
    {
        // int 8X4X8  = 0;
        // int 4X4X8  = 1;
        // int 4X4X16 = 2;
        // int 4X2X16 = 3;
        // aux_data[ 5: 0] maps to instr[ 63: 58] for VOP2M-VOP6M (MOD0)
        // aux_data[17: 6] maps to instr[ 87: 76] for VOP3M-VOP5M (MOD1)
        // aux_data[13: 6] maps to instr[ 83: 76] for VOP6M       (MOD1)
        // aux_data[25:18] maps to instr[127:120] for VOP5M-VOP6M (MOD2)
        constexpr index_t AcoFlag = Aco ? (1 << (6 + 1)) : 0;
        if constexpr((HPerWconv == 8) && (WPerWconv == 4))
            return 0 | (activeFun << 8) | (scaleBiasPacked << 26) | AcoFlag;
        else if constexpr((HPerWconv == 4) && (WPerWconv == 4))
            return 2 | (activeFun << 8) | (scaleBiasPacked << 26) | AcoFlag;
        else if constexpr((HPerWconv == 4) && (WPerWconv == 2))
            return 3 | (activeFun << 8) | (scaleBiasPacked << 26) | AcoFlag;
        static_assert("unsupport shape.");
    };

    static constexpr auto auxdata = GetAuxData();

    static constexpr auto sba =
        SbaSelector<AccDataType, HPerWconv, WPerWconv, auxdata, scaleBiasPacked, uniformScale>{};
    static constexpr auto sba_instr = sba.selected_sba;
};

} // namespace ck
