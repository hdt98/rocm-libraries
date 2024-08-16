// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/amd_wconv.hpp"

namespace ck {

enum struct WconvInstr
{
    // gfx13
    wconv_f32_f16 = 0,
    wconv_f32_bf16,
    wconv_f32_f8,
    wconv_f32_bf8,
    wconv_f32_iu8,
    wconv_i32_iu8,
    wconv_f16_f16,
    wconv_bf16_bf16,
    wconv_f16_f8,
    wconv_bf16_bf8,
    wconv_f16_iu8,
    wconv_f32_iu4,
    wconv_i32_iu4,
    wconv_f16_iu4
};

template <WconvInstr Instr,
          index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false,
          bool Signed   = false>
struct wconv_type
{
    wconv_type() { static_assert(false, "never called"); }
};

// dst: f32 or i32
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct wconv_type<WconvInstr::wconv_f32_f16,
                  H,
                  W,
                  FilterSize,
                  DilationX,
                  DilationY,
                  Iters,
                  Aco,
                  Signed>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c) const
    {
        intrin_wconv_f32_f16<H, W, FilterSize, DilationX, DilationY, Iters, Aco, Signed>::Run(
            reg_wei, reg_data, reg_c);
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct wconv_type<WconvInstr::wconv_f32_bf16,
                  H,
                  W,
                  FilterSize,
                  DilationX,
                  DilationY,
                  Iters,
                  Aco,
                  Signed>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c) const
    {
        intrin_wconv_f32_bf16<H, W, FilterSize, DilationX, DilationY, Iters, Aco, Signed>::Run(
            reg_wei, reg_data, reg_c);
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct wconv_type<WconvInstr::wconv_f32_f8,
                  H,
                  W,
                  FilterSize,
                  DilationX,
                  DilationY,
                  Iters,
                  Aco,
                  Signed>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c) const
    {
        intrin_wconv_f32_f8<H, W, FilterSize, DilationX, DilationY, Iters, Aco, Signed>::Run(
            reg_wei, reg_data, reg_c);
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct wconv_type<WconvInstr::wconv_f32_bf8,
                  H,
                  W,
                  FilterSize,
                  DilationX,
                  DilationY,
                  Iters,
                  Aco,
                  Signed>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c) const
    {
        intrin_wconv_f32_bf8<H, W, FilterSize, DilationX, DilationY, Iters, Aco, Signed>::Run(
            reg_wei, reg_data, reg_c);
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct wconv_type<WconvInstr::wconv_f32_iu8,
                  H,
                  W,
                  FilterSize,
                  DilationX,
                  DilationY,
                  Iters,
                  Aco,
                  Signed>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c) const
    {
        intrin_wconv_f32_iu8<H, W, FilterSize, DilationX, DilationY, Iters, Aco, Signed>::Run(
            reg_wei, reg_data, reg_c);
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct wconv_type<WconvInstr::wconv_i32_iu8,
                  H,
                  W,
                  FilterSize,
                  DilationX,
                  DilationY,
                  Iters,
                  Aco,
                  Signed>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c) const
    {
        intrin_wconv_i32_iu8<H, W, FilterSize, DilationX, DilationY, Iters, Aco, Signed>::Run(
            reg_wei, reg_data, reg_c);
    }
};

#if CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct wconv_type<WconvInstr::wconv_f32_iu4,
                  H,
                  W,
                  FilterSize,
                  DilationX,
                  DilationY,
                  Iters,
                  Aco,
                  Signed>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c) const
    {
        intrin_wconv_f32_iu4<H, W, FilterSize, DilationX, DilationY, Iters, Aco, Signed>::Run(
            reg_wei, reg_data, reg_c);
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct wconv_type<WconvInstr::wconv_i32_iu4,
                  H,
                  W,
                  FilterSize,
                  DilationX,
                  DilationY,
                  Iters,
                  Aco,
                  Signed>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c) const
    {
        intrin_wconv_i32_iu4<H, W, FilterSize, DilationX, DilationY, Iters, Aco, Signed>::Run(
            reg_wei, reg_data, reg_c);
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct wconv_type<WconvInstr::wconv_f16_iu4,
                  H,
                  W,
                  FilterSize,
                  DilationX,
                  DilationY,
                  Iters,
                  Aco,
                  Signed>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c) const
    {
        intrin_wconv_f16_iu4<H, W, FilterSize, DilationX, DilationY, Iters, Aco, Signed>::Run(
            reg_wei, reg_data, reg_c);
    }
};
#endif

// dst: f16 or bf16
template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct wconv_type<WconvInstr::wconv_f16_f16,
                  H,
                  W,
                  FilterSize,
                  DilationX,
                  DilationY,
                  Iters,
                  Aco,
                  Signed>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c) const
    {
        intrin_wconv_f16_f16<H, W, FilterSize, DilationX, DilationY, Iters, Aco, Signed>::Run(
            reg_wei, reg_data, reg_c);
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct wconv_type<WconvInstr::wconv_bf16_bf16,
                  H,
                  W,
                  FilterSize,
                  DilationX,
                  DilationY,
                  Iters,
                  Aco,
                  Signed>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c) const
    {
        intrin_wconv_bf16_bf16<H, W, FilterSize, DilationX, DilationY, Iters, Aco, Signed>::Run(
            reg_wei, reg_data, reg_c);
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct wconv_type<WconvInstr::wconv_f16_f8,
                  H,
                  W,
                  FilterSize,
                  DilationX,
                  DilationY,
                  Iters,
                  Aco,
                  Signed>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c) const
    {
        intrin_wconv_f16_f8<H, W, FilterSize, DilationX, DilationY, Iters, Aco, Signed>::Run(
            reg_wei, reg_data, reg_c);
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct wconv_type<WconvInstr::wconv_bf16_bf8,
                  H,
                  W,
                  FilterSize,
                  DilationX,
                  DilationY,
                  Iters,
                  Aco,
                  Signed>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c) const
    {
        intrin_wconv_bf16_bf8<H, W, FilterSize, DilationX, DilationY, Iters, Aco, Signed>::Run(
            reg_wei, reg_data, reg_c);
    }
};

template <index_t H,
          index_t W,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco,
          bool Signed>
struct wconv_type<WconvInstr::wconv_f16_iu8,
                  H,
                  W,
                  FilterSize,
                  DilationX,
                  DilationY,
                  Iters,
                  Aco,
                  Signed>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& reg_wei, const FloatB reg_data, FloatC& reg_c) const
    {
        intrin_wconv_f16_iu8<H, W, FilterSize, DilationX, DilationY, Iters, Aco, Signed>::Run(
            reg_wei, reg_data, reg_c);
    }
};

template <typename WeiDataType,
          typename InDataType,
          typename AccDataType,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters,
          bool Aco>
struct WconvSelector
{
    template <typename WeiDataType_, typename InDataType_, typename AccDataType_>
    static constexpr auto GetWconv();

    template <>
    static constexpr auto GetWconv<half_t, half_t, float_t>()
    {
        return WconvInstr::wconv_f32_f16;
    }
    template <>
    static constexpr auto GetWconv<bhalf_t, bhalf_t, float_t>()
    {
        return WconvInstr::wconv_f32_bf16;
    }

    template <>
    static constexpr auto GetWconv<f8_t, f8_t, float_t>()
    {
        return WconvInstr::wconv_f32_f8;
    }

    template <>
    static constexpr auto GetWconv<bf8_t, bf8_t, float_t>()
    {
        return WconvInstr::wconv_f32_bf8;
    }

    template <>
    static constexpr auto GetWconv<int8_t, int8_t, float_t>()
    {
        return WconvInstr::wconv_f32_iu8;
    }

    template <>
    static constexpr auto GetWconv<uint8_t, uint8_t, float_t>()
    {
        return WconvInstr::wconv_f32_iu8;
    }

    template <>
    static constexpr auto GetWconv<int4_t, int4_t, float_t>()
    {
        return WconvInstr::wconv_f32_iu4;
    }

    template <>
    static constexpr auto GetWconv<uint4_t, uint4_t, float_t>()
    {
        return WconvInstr::wconv_f32_iu4;
    }

    template <>
    static constexpr auto GetWconv<int8_t, int8_t, int32_t>()
    {
        return WconvInstr::wconv_i32_iu8;
    }

    template <>
    static constexpr auto GetWconv<uint8_t, uint8_t, int32_t>()
    {
        return WconvInstr::wconv_i32_iu8;
    }

    template <>
    static constexpr auto GetWconv<int4_t, int4_t, int32_t>()
    {
        return WconvInstr::wconv_i32_iu4;
    }

    template <>
    static constexpr auto GetWconv<uint4_t, uint4_t, int32_t>()
    {
        return WconvInstr::wconv_i32_iu4;
    }

    template <>
    static constexpr auto GetWconv<half_t, half_t, half_t>()
    {
        return WconvInstr::wconv_f16_f16;
    }

    template <>
    static constexpr auto GetWconv<bhalf_t, bhalf_t, bhalf_t>()
    {
        return WconvInstr::wconv_bf16_bf16;
    }

    template <>
    static constexpr auto GetWconv<f8_t, f8_t, half_t>()
    {
        return WconvInstr::wconv_f16_f8;
    }

    template <>
    static constexpr auto GetWconv<bf8_t, bf8_t, bhalf_t>()
    {
        return WconvInstr::wconv_bf16_bf8;
    }

    template <>
    static constexpr auto GetWconv<int8_t, int8_t, half_t>()
    {
        return WconvInstr::wconv_f16_iu8;
    }

    template <>
    static constexpr auto GetWconv<uint8_t, uint8_t, half_t>()
    {
        return WconvInstr::wconv_f16_iu8;
    }

    template <>
    static constexpr auto GetWconv<int4_t, int4_t, half_t>()
    {
        return WconvInstr::wconv_f16_iu4;
    }

    template <>
    static constexpr auto GetWconv<uint4_t, uint4_t, half_t>()
    {
        return WconvInstr::wconv_f16_iu4;
    }

    static constexpr auto Signed()
    {
        return std::is_same<InDataType, int4_t>::value || std::is_same<InDataType, int8_t>::value;
    }

    static constexpr auto selected_wconv =
        wconv_type<GetWconv<WeiDataType, InDataType, AccDataType>(),
                   HPerWconv,
                   WPerWconv,
                   FilterSize,
                   DilationX,
                   DilationY,
                   Iters,
                   Aco,
                   Signed()>{};

    __host__ __device__ constexpr WconvSelector()
    {
        static_assert(FilterSize == 1 || FilterSize == 3, "WRONG! FilterSize must be 1 or 3");
    };
};

template <typename WeiDataType,
          typename InDataType,
          typename AccDataType,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Iters = 1,
          bool Aco      = false>
struct WconvConv
{
    static constexpr index_t WaveSize = 32;

    __host__ __device__ constexpr WconvConv(){};

    template <typename ScalarType>
    static constexpr index_t SizeOfBits()
    {
        return std::is_same<int4_t, ScalarType>::value ? 4 : sizeof(ScalarType) * 8;
    }

    // WCNN input channel and output channel info
    static constexpr index_t GetNumInputChannels()
    {
        return (HPerWconv == 4) && (WPerWconv == 2) ? 128 / SizeOfBits<KernelInDataType>()
                                                    : 64 / SizeOfBits<KernelInDataType>();
    }

    static constexpr index_t GetNumOutputChannels()
    {
        return (HPerWconv == 8) && (WPerWconv == 4) ? 8 : 16;
    }

    static constexpr index_t GetUnpackedNumInputChannels()
    {
        return (HPerWconv == 4) && (WPerWconv == 2) ? 128 / SizeOfBits<InDataType>()
                                                    : 64 / SizeOfBits<InDataType>();
    }

    static constexpr index_t GetUnpackedNumOutputChannels() { return GetNumOutputChannels(); }

    // WCNN weight info
    static constexpr index_t GetWeightRegSize()
    {
        constexpr index_t WeightBitsPerLane = GetNumInputChannels() * GetNumOutputChannels() *
                                              SizeOfBits<KernelWeightDataType>() / WaveSize *
                                              FilterSize * FilterSize * Iters;

        // Round bits to Reg size (uint32_t)
        return (WeightBitsPerLane + 31) / 32;
    }

    static constexpr index_t GetNumWeightComponents()
    {
        return GetWeightRegSize() * 32 / SizeOfBits<KernelWeightDataType>();
    }

    static constexpr index_t GetNumSubTilesPerWeightTape()
    {
        return ((HPerWconv == 4) && (WPerWconv == 2)) ? 2 : 1;
    }

    static constexpr index_t GetNumWeightCompPerTile()
    {
        return GetNumWeightComponents() / GetWeightRegSize();
    }

    static constexpr index_t GetNumWeightCompPerTape()
    {
        return GetNumSubTilesPerWeightTape() * GetNumWeightCompPerTile();
    }

    static constexpr index_t GetNumWeightTape()
    {
        if constexpr(FilterSize == 1)
        {
            return 1;
        }
        else if constexpr(FilterSize == 3)
        {
            return (HPerWconv == 8) && (WPerWconv == 4) ? 5 : 9;
        }
    }

    static constexpr index_t GetNumWeightTapePerWave()
    {
        return (HPerWconv == 8) && (WPerWconv == 4) ? 2 : 1;
    }

    __host__ __device__ static constexpr auto GetWeight3RemapTable()
    {
        return Sequence<2, 1, 8, 3, 0, 7, 4, 5, 6>{};
    }

    __host__ __device__ static constexpr auto GetWeightRemapTable()
    {
        if constexpr(FilterSize == 1)
        {
            return Sequence<0>{};
        }
        else
        {
            if constexpr(GetNumWeightTape() == 9)
            {
                return GetWeight3RemapTable();
            }
            else
            {
                return Sequence<1, 4, 0, 2, 3>{};
            }
        }
    }

    __host__ __device__ static constexpr auto GetWeightSecondTapeMapTable()
    {
        if constexpr(FilterSize == 1)
        {
            return Sequence<0>{};
        }
        else
        {
            if constexpr(GetNumWeightTape() == 9)
            {
                return Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{};
            }
            else
            {
                return Sequence<3, 0, -3, 1, -3>{};
            }
        }
    }

    // WCNN data info
    static constexpr index_t GetDataRegSizePerTile()
    {
        return HPerWconv * WPerWconv * GetNumInputChannels() * SizeOfBits<KernelInDataType>() /
               WaveSize / 32;
    }

    static constexpr index_t GetNumDataCompPerTile()
    {
        return GetNumDataComponents() / GetNumImageSubTilesInVertical();
    }

    static constexpr index_t GetNumSubTilesPerImageTile()
    {
        return ((HPerWconv == 8) && (WPerWconv == 4)) ? 2 : 1;
    }

    static constexpr index_t GetNumImageTilesInVertical()
    {
        // Filter 1x1: always require 1 image tile.
        // Filter 3x3: 8x4 require 2 tiles in vertical, other pixel shapes require 3 tiles.
        return FilterSize == 1 ? 1 : ((HPerWconv == 8) && (WPerWconv == 4)) ? 2 : 3;
    }

    static constexpr index_t GetNumImageSubTilesInVertical()
    {
        return GetNumImageTilesInVertical() * GetNumSubTilesPerImageTile();
    }

    static constexpr index_t GetDataRegSize()
    {
        return GetDataRegSizePerTile() * GetNumImageTilesInVertical();
    }

    static constexpr index_t GetNumDataComponents()
    {
        return GetDataRegSize() * 32 / SizeOfBits<KernelInDataType>();
    }

    // Accum info
    static constexpr index_t GetAccumRegSize()
    {
        return HPerWconv * WPerWconv * GetNumOutputChannels() * SizeOfBits<AccDataType>() /
               WaveSize / 32;
    }

    static constexpr index_t GetNumAccumComponents()
    {
        return GetAccumRegSize() * 32 / SizeOfBits<AccDataType>();
    }

    // Helper functions of origin data index per thread
    static constexpr auto CalculateWeiDataThreadOriginDataIndex()
    {
        auto laneId                      = get_thread_local_1d_id() & (WaveSize - 1);
        constexpr index_t NumCompPerTile = GetNumWeightCompPerTile();
        if constexpr((HPerWconv == 8) && (WPerWconv == 4))
        {
            return make_tuple(laneId / 16,
                              (laneId / 2) & (GetNumOutputChannels() - 1),
                              0,
                              (laneId % 2) * NumCompPerTile);
        }
        else
        {
            return make_tuple(0, (laneId / 2), 0, (laneId % 2) * NumCompPerTile);
        }
    }

    static constexpr auto CalculateInDataThreadOriginDataIndex()
    {
        auto laneId                      = get_thread_local_1d_id() & (WaveSize - 1);
        constexpr index_t NumCompPerTile = GetNumDataCompPerTile();
        const index_t SubC               = (laneId * NumCompPerTile) % GetNumInputChannels();
        const index_t PixelOffset        = laneId * NumCompPerTile / GetNumInputChannels();

        return make_tuple(PixelOffset / WPerWconv, PixelOffset % WPerWconv, SubC);
    }

    static constexpr auto CalculateAccThreadOriginDataIndex()
    {
        auto laneId              = get_thread_local_1d_id() & (WaveSize - 1);
        const index_t accCompIdx = laneId * GetNumAccumComponents() / GetNumSubTilesPerImageTile();
        const index_t subW       = (accCompIdx / GetNumOutputChannels()) % WPerWconv;
        const index_t subH       = (accCompIdx / GetNumOutputChannels()) / WPerWconv;
        const index_t subK       = accCompIdx % GetNumOutputChannels();
        // TODO: modify it if ACO = 1
        constexpr index_t SwizzleComp = 4;
        const index_t subK_8          = (laneId & 1) * SwizzleComp;

        static_assert(Aco == 0, "");
        if constexpr(GetNumAccumComponents() == 4)
        {
            return make_tuple(0, subH, subW, 0, subK);
        }
        else
        {
            return make_tuple(0, subH, subW, 0, subK_8);
        }
    }

    // Pre-defined types
    using KernelWeightDataType = typename std::
        conditional<std::is_same<int4_t, WeiDataType>::value, int8_t, WeiDataType>::type;
    using KernelInDataType = typename std::
        conditional<std::is_same<int4_t, InDataType>::value, int8_t, InDataType>::type;

    using AccDataVec = vector_type<AccDataType, GetNumAccumComponents()>;
    using WeiDataVec = vector_type<KernelWeightDataType, GetNumWeightComponents()>;
    using InDataVec  = vector_type<KernelInDataType, GetNumDataComponents()>;

    using AccDataTileVec =
        vector_type<AccDataType, GetNumAccumComponents() / GetNumSubTilesPerImageTile()>;
    using WeiDataTileVec = vector_type<KernelWeightDataType, GetNumWeightCompPerTile()>;
    using InDataTileVec  = vector_type<KernelInDataType, GetNumDataCompPerTile()>;

    using WeiDataTapeVec = vector_type<KernelWeightDataType, GetNumWeightCompPerTape()>;

    static_assert(GetNumImageSubTilesInVertical() == GetDataRegSize(), "");

    // WCNN intrinsic
    static constexpr auto wconv       = WconvSelector<WeiDataType,
                                                InDataType,
                                                AccDataType,
                                                HPerWconv,
                                                WPerWconv,
                                                FilterSize,
                                                DilationX,
                                                DilationY,
                                                Iters,
                                                Aco>{};
    static constexpr auto wconv_instr = wconv.selected_wconv;
};

} // namespace ck
