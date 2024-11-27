// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_AMD_FMA_FROM_TENSOR_HPP
#define CK_AMD_FMA_FROM_TENSOR_HPP

#include "data_type.hpp"
namespace ck {

#if defined(__gfx1300__) || defined(__gfx1301__) || defined(__gfx1302__)
#define __gfx13__
#endif

// aux_data[ 5: 0] maps to instr[ 63: 58] for VOP2M-VOP6M (MOD0)
// aux_data[17: 6] maps to instr[ 87: 76] for VOP3M-VOP5M (MOD1)
// aux_data[13: 6] maps to instr[ 83: 76] for VOP6M       (MOD1)
// aux_data[25:18] maps to instr[127:120] for VOP5M-VOP6M (MOD2)
// constexpr index_t FmaAuxData_Mod0 = 0;
constexpr index_t FmaAuxData_Mod1 = 6;
// constexpr index_t FmaAuxData_Mod2 = 18;

template <index_t ChanOff>
constexpr index_t GetChanOff()
{
    return (ChanOff / 4) << (FmaAuxData_Mod1 + 4);
};

template <index_t H, index_t W, index_t ChanOff>
struct intrin_fma_from_tensor;

template <index_t ChanOff>
struct intrin_fma_from_tensor<4, 2, ChanOff>
{
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    // i4
    template <class FloatAcc>
    __device__ static void
    Run(const float4_t& inAcc, const int4x8_t& residual1, const float_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = GetChanOff<ChanOff>();
        constexpr bool clamp       = false;
        const int4x8_t residual2   = {};
        outAcc.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_i4_4x2(inAcc,
                                                        bit_cast<uint32_t>(residual1),
                                                        bit_cast<uint32_t>(residual2),
                                                        scale,
                                                        aux_data,
                                                        clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const half4_t& inAcc, const int4x8_t& residual1, const half2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = GetChanOff<ChanOff>();
        constexpr bool clamp       = false;
        const int4x8_t residual2   = {};
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_i4_4x2(inAcc,
                                                        bit_cast<uint32_t>(residual1),
                                                        bit_cast<uint32_t>(residual2),
                                                        scale,
                                                        aux_data,
                                                        clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf4_t& inAcc, const int4x8_t& residual1, const bhalf2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = GetChanOff<ChanOff>();
        constexpr bool clamp       = false;
        const int4x8_t residual2   = {};
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_i4_4x2(inAcc,
                                                         bit_cast<uint32_t>(residual1),
                                                         bit_cast<uint32_t>(residual2),
                                                         scale,
                                                         aux_data,
                                                         clamp);
    }

    // u4
    template <class FloatAcc>
    __device__ static void
    Run(const float4_t& inAcc, const uint4x8_t& residual1, const float_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = GetChanOff<ChanOff>();
        constexpr bool clamp       = false;
        const uint4x8_t residual2  = {};
        outAcc.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_u4_4x2(inAcc,
                                                        bit_cast<uint32_t>(residual1),
                                                        bit_cast<uint32_t>(residual2),
                                                        scale,
                                                        aux_data,
                                                        clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const half4_t& inAcc, const uint4x8_t& residual1, const half2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = GetChanOff<ChanOff>();
        constexpr bool clamp       = false;
        const uint4x8_t residual2  = {};
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_u4_4x2(inAcc,
                                                        bit_cast<uint32_t>(residual1),
                                                        bit_cast<uint32_t>(residual2),
                                                        scale,
                                                        aux_data,
                                                        clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf4_t& inAcc, const uint4x8_t& residual1, const bhalf2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = GetChanOff<ChanOff>();
        constexpr bool clamp       = false;
        const uint4x8_t residual2  = {};
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_u4_4x2(inAcc,
                                                         bit_cast<uint32_t>(residual1),
                                                         bit_cast<uint32_t>(residual2),
                                                         scale,
                                                         aux_data,
                                                         clamp);
    }
#endif
    // i8
    template <class FloatAcc>
    __device__ static void
    Run(const float4_t& inAcc, const int8x4_t& residual1, const float_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const int8x4_t residual2   = {};
        outAcc.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_i8_4x2(inAcc,
                                                        bit_cast<uint32_t>(residual1),
                                                        bit_cast<uint32_t>(residual2),
                                                        scale,
                                                        aux_data,
                                                        clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const half4_t& inAcc, const int8x4_t& residual1, const half2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const int8x4_t residual2   = {};
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_i8_4x2(inAcc,
                                                        bit_cast<uint32_t>(residual1),
                                                        bit_cast<uint32_t>(residual2),
                                                        scale,
                                                        aux_data,
                                                        clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf4_t& inAcc, const int8x4_t& residual1, const bhalf2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const int8x4_t residual2   = {};
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_i8_4x2(inAcc,
                                                         bit_cast<uint32_t>(residual1),
                                                         bit_cast<uint32_t>(residual2),
                                                         scale,
                                                         aux_data,
                                                         clamp);
    }

    // u8
    template <class FloatAcc>
    __device__ static void
    Run(const float4_t& inAcc, const uint8x4_t& residual1, const float_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const uint8x4_t residual2  = {};
        outAcc.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_u8_4x2(inAcc,
                                                        bit_cast<uint32_t>(residual1),
                                                        bit_cast<uint32_t>(residual2),
                                                        scale,
                                                        aux_data,
                                                        clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const half4_t& inAcc, const uint8x4_t& residual1, const half2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const uint8x4_t residual2  = {};
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_u8_4x2(inAcc,
                                                        bit_cast<uint32_t>(residual1),
                                                        bit_cast<uint32_t>(residual2),
                                                        scale,
                                                        aux_data,
                                                        clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf4_t& inAcc, const uint8x4_t& residual1, const bhalf2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const uint8x4_t residual2  = {};
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_u8_4x2(inAcc,
                                                         bit_cast<uint32_t>(residual1),
                                                         bit_cast<uint32_t>(residual2),
                                                         scale,
                                                         aux_data,
                                                         clamp);
    }

    // f8
    template <class FloatAcc>
    __device__ static void
    Run(const float4_t& inAcc, const f8x4_t& residual1, const float_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const f8x4_t residual2     = {};
        outAcc.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_fp8_4x2(inAcc,
                                                         bit_cast<uint32_t>(residual1),
                                                         bit_cast<uint32_t>(residual2),
                                                         scale,
                                                         aux_data,
                                                         clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const half4_t& inAcc, const f8x4_t& residual1, const half2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const f8x4_t residual2     = {};
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_fp8_4x2(inAcc,
                                                         bit_cast<uint32_t>(residual1),
                                                         bit_cast<uint32_t>(residual2),
                                                         scale,
                                                         aux_data,
                                                         clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf4_t& inAcc, const f8x4_t& residual1, const bhalf2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const f8x4_t residual2     = {};
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_fp8_4x2(inAcc,
                                                          bit_cast<uint32_t>(residual1),
                                                          bit_cast<uint32_t>(residual2),
                                                          scale,
                                                          aux_data,
                                                          clamp);
    }

    // bf8
    template <class FloatAcc>
    __device__ static void
    Run(const float4_t& inAcc, const bf8x4_t& residual1, const float_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const bf8x4_t residual2    = {};
        outAcc.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_bf8_4x2(inAcc,
                                                         bit_cast<uint32_t>(residual1),
                                                         bit_cast<uint32_t>(residual2),
                                                         scale,
                                                         aux_data,
                                                         clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const half4_t& inAcc, const bf8x4_t& residual1, const half2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const bf8x4_t residual2    = {};
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_bf8_4x2(inAcc,
                                                         bit_cast<uint32_t>(residual1),
                                                         bit_cast<uint32_t>(residual2),
                                                         scale,
                                                         aux_data,
                                                         clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf4_t& inAcc, const bf8x4_t& residual1, const bhalf2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;

        const bf8x4_t residual2 = {};
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_bf8_4x2(inAcc,
                                                          bit_cast<uint32_t>(residual1),
                                                          bit_cast<uint32_t>(residual2),
                                                          scale,
                                                          aux_data,
                                                          clamp);
    }

    // f16
    template <class FloatAcc>
    __device__ static void Run(const float4_t& inAcc,
                               const half2_t& residual1,
                               const half2_t& residual2,
                               const float_t scale,
                               FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_f16_4x2(
                inAcc, residual1, residual2, scale, aux_data, clamp);
    }

    template <class FloatAcc>
    __device__ static void Run(const half4_t& inAcc,
                               const half2_t& residual1,
                               const half2_t& residual2,
                               const half2_t scale,
                               FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_f16_4x2(
                inAcc, residual1, residual2, scale, aux_data, clamp);
    }

    // bf16
    template <class FloatAcc>
    __device__ static void Run(const float4_t& inAcc,
                               const bhalf2_t& residual1,
                               const bhalf2_t& residual2,
                               const float_t scale,
                               FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f32_bf16_4x2(
                inAcc, residual1, residual2, scale, aux_data, clamp);
    }

    template <class FloatAcc>
    __device__ static void Run(const bhalf4_t& inAcc,
                               const bhalf2_t& residual1,
                               const bhalf2_t& residual2,
                               const bhalf2_t scale,
                               FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_bf16_4x2(
                inAcc, residual1, residual2, scale, aux_data, clamp);
    }
};

template <index_t ChanOff>
struct intrin_fma_from_tensor<4, 4, ChanOff>
{
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    // i4
    template <class FloatAcc>
    __device__ static void
    Run(const half8_t& inAcc, const int4x8_t& residual1, const half2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = GetChanOff<ChanOff>();
        constexpr bool clamp       = false;
        const int4x8_t residual2   = {};
        outAcc.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_i4_4x4(inAcc,
                                                        bit_cast<uint32_t>(residual1),
                                                        bit_cast<uint32_t>(residual2),
                                                        scale,
                                                        aux_data,
                                                        clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf8_t& inAcc, const int4x8_t& residual1, const bhalf2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = GetChanOff<ChanOff>();
        constexpr bool clamp       = false;
        const int4x8_t residual2   = {};
        outAcc.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_i4_4x4(inAcc,
                                                         bit_cast<uint32_t>(residual1),
                                                         bit_cast<uint32_t>(residual2),
                                                         scale,
                                                         aux_data,
                                                         clamp);
    }

    // u4
    template <class FloatAcc>
    __device__ static void
    Run(const half8_t& inAcc, const uint4x8_t& residual1, const half2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = GetChanOff<ChanOff>();
        constexpr bool clamp       = false;
        const uint4x8_t residual2  = {};
        outAcc.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_u4_4x4(inAcc,
                                                        bit_cast<uint32_t>(residual1),
                                                        bit_cast<uint32_t>(residual2),
                                                        scale,
                                                        aux_data,
                                                        clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf8_t& inAcc, const uint4x8_t& residual1, const bhalf2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = GetChanOff<ChanOff>();
        constexpr bool clamp       = false;
        const uint4x8_t residual2  = {};
        outAcc.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_u4_4x4(inAcc,
                                                         bit_cast<uint32_t>(residual1),
                                                         bit_cast<uint32_t>(residual2),
                                                         scale,
                                                         aux_data,
                                                         clamp);
    }
#endif
    // i8
    template <class FloatAcc>
    __device__ static void Run(const half8_t& inAcc,
                               const int8x4_t& residual1,
                               const int8x4_t& residual2,
                               const half2_t scale,
                               FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_i8_4x4(inAcc,
                                                        bit_cast<uint32_t>(residual1),
                                                        bit_cast<uint32_t>(residual2),
                                                        scale,
                                                        aux_data,
                                                        clamp);
    }

    template <class FloatAcc>
    __device__ static void Run(const bhalf8_t& inAcc,
                               const int8x4_t& residual1,
                               const int8x4_t& residual2,
                               const bhalf2_t scale,
                               FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_i8_4x4(inAcc,
                                                         bit_cast<uint32_t>(residual1),
                                                         bit_cast<uint32_t>(residual2),
                                                         scale,
                                                         aux_data,
                                                         clamp);
    }

    // u8
    template <class FloatAcc>
    __device__ static void Run(const half8_t& inAcc,
                               const uint8x4_t& residual1,
                               const uint8x4_t& residual2,
                               const half2_t scale,
                               FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_u8_4x4(inAcc,
                                                        bit_cast<uint32_t>(residual1),
                                                        bit_cast<uint32_t>(residual2),
                                                        scale,
                                                        aux_data,
                                                        clamp);
    }

    template <class FloatAcc>
    __device__ static void Run(const bhalf8_t& inAcc,
                               const uint8x4_t& residual1,
                               const uint8x4_t& residual2,
                               const bhalf2_t scale,
                               FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_u8_4x4(inAcc,
                                                         bit_cast<uint32_t>(residual1),
                                                         bit_cast<uint32_t>(residual2),
                                                         scale,
                                                         aux_data,
                                                         clamp);
    }

    // f8
    template <class FloatAcc>
    __device__ static void Run(const half8_t& inAcc,
                               const f8x4_t& residual1,
                               const f8x4_t& residual2,
                               const half2_t scale,
                               FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_fp8_4x4(inAcc,
                                                         bit_cast<uint32_t>(residual1),
                                                         bit_cast<uint32_t>(residual2),
                                                         scale,
                                                         aux_data,
                                                         clamp);
    }

    template <class FloatAcc>
    __device__ static void Run(const bhalf8_t& inAcc,
                               const f8x4_t& residual1,
                               const f8x4_t& residual2,
                               const bhalf2_t scale,
                               FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_fp8_4x4(inAcc,
                                                          bit_cast<uint32_t>(residual1),
                                                          bit_cast<uint32_t>(residual2),
                                                          scale,
                                                          aux_data,
                                                          clamp);
    }

    // bf8
    template <class FloatAcc>
    __device__ static void Run(const half8_t& inAcc,
                               const bf8x4_t& residual1,
                               const bf8x4_t& residual2,
                               const half2_t scale,
                               FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_bf8_4x4(inAcc,
                                                         bit_cast<uint32_t>(residual1),
                                                         bit_cast<uint32_t>(residual2),
                                                         scale,
                                                         aux_data,
                                                         clamp);
    }

    template <class FloatAcc>
    __device__ static void Run(const bhalf8_t& inAcc,
                               const bf8x4_t& residual1,
                               const bf8x4_t& residual2,
                               const bhalf2_t scale,
                               FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        outAcc.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_bf8_4x4(inAcc,
                                                          bit_cast<uint32_t>(residual1),
                                                          bit_cast<uint32_t>(residual2),
                                                          scale,
                                                          aux_data,
                                                          clamp);
    }

    // f16
    template <class FloatAcc>
    __device__ static void Run(const half8_t& inAcc,
                               const half2_t& residual1,
                               const half2_t& residual2,
                               const half2_t& residual3,
                               const half2_t& residual4,
                               const half2_t scale,
                               FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        half4_t inAcc1;
        half4_t inAcc2;
        inAcc1[0] = inAcc[0];
        inAcc1[1] = inAcc[1];
        inAcc1[2] = inAcc[2];
        inAcc1[3] = inAcc[3];
        inAcc2[0] = inAcc[4];
        inAcc2[1] = inAcc[5];
        inAcc2[2] = inAcc[6];
        inAcc2[3] = inAcc[7];
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_f16_4x4(
                inAcc1, residual1, residual2, scale, aux_data, clamp);
        // TODO: permute scale
        outAcc.template AsType<half4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_f16_4x4(
                inAcc2, residual3, residual4, scale, aux_data, clamp);
    }

    // bf16
    template <class FloatAcc>
    __device__ static void Run(const bhalf8_t& inAcc,
                               const bhalf2_t& residual1,
                               const bhalf2_t& residual2,
                               const bhalf2_t& residual3,
                               const bhalf2_t& residual4,
                               const bhalf2_t scale,
                               FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        bhalf4_t inAcc1;
        bhalf4_t inAcc2;
        inAcc1[0] = inAcc[0];
        inAcc1[1] = inAcc[1];
        inAcc1[2] = inAcc[2];
        inAcc1[3] = inAcc[3];
        inAcc2[0] = inAcc[4];
        inAcc2[1] = inAcc[5];
        inAcc2[2] = inAcc[6];
        inAcc2[3] = inAcc[7];
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_bf16_4x4(
                inAcc1, residual1, residual2, scale, aux_data, clamp);
        // TODO: permute scale
        outAcc.template AsType<bhalf4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_bf16_4x4(
                inAcc2, residual3, residual4, scale, aux_data, clamp);
    }
};

template <index_t ChanOff>
struct intrin_fma_from_tensor<8, 4, ChanOff>
{
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    // i4
    template <class FloatAcc>
    __device__ static void
    Run(const half8_t& inAcc, const int4x16_t& residual, const half2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = GetChanOff<ChanOff>();
        constexpr bool clamp       = false;
        const int32_t residual1    = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual2    = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_i4_8x4(
                inAcc, residual1, residual2, scale, aux_data, clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf8_t& inAcc, const int4x16_t& residual, const bhalf2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = GetChanOff<ChanOff>();
        constexpr bool clamp       = false;
        const int32_t residual1    = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual2    = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_i4_8x4(
                inAcc, residual1, residual2, scale, aux_data, clamp);
    }

    // u4
    template <class FloatAcc>
    __device__ static void
    Run(const half8_t& inAcc, const uint4x16_t& residual, const half2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = GetChanOff<ChanOff>();
        constexpr bool clamp       = false;
        const int32_t residual1    = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual2    = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_u4_8x4(
                inAcc, residual1, residual2, scale, aux_data, clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf8_t& inAcc, const uint4x16_t& residual, const bhalf2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = GetChanOff<ChanOff>();
        constexpr bool clamp       = false;
        const int32_t residual1    = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual2    = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_u4_8x4(
                inAcc, residual1, residual2, scale, aux_data, clamp);
    }
#endif

    // i8
    template <class FloatAcc>
    __device__ static void
    Run(const half8_t& inAcc, const int8x8_t& residual, const half2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const int32_t residual1    = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual2    = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_i8_8x4(
                inAcc, residual1, residual2, scale, aux_data, clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf8_t& inAcc, const int8x8_t& residual, const bhalf2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const int32_t residual1    = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual2    = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_i8_8x4(
                inAcc, residual1, residual2, scale, aux_data, clamp);
    }

    // u8
    template <class FloatAcc>
    __device__ static void
    Run(const half8_t& inAcc, const uint8x8_t& residual, const half2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const int32_t residual1    = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual2    = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_u8_8x4(
                inAcc, residual1, residual2, scale, aux_data, clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf8_t& inAcc, const uint8x8_t& residual, const bhalf2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const int32_t residual1    = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual2    = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_u8_8x4(
                inAcc, residual1, residual2, scale, aux_data, clamp);
    }

    // f8
    template <class FloatAcc>
    __device__ static void
    Run(const half8_t& inAcc, const f8x8_t& residual, const half2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const int32_t residual1    = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual2    = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_fp8_8x4(
                inAcc, residual1, residual2, scale, aux_data, clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf8_t& inAcc, const f8x8_t& residual, const bhalf2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const int32_t residual1    = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual2    = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_fp8_8x4(
                inAcc, residual1, residual2, scale, aux_data, clamp);
    }

    // bf8
    template <class FloatAcc>
    __device__ static void
    Run(const half8_t& inAcc, const bf8x8_t& residual, const half2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const int32_t residual1    = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual2    = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_bf8_8x4(
                inAcc, residual1, residual2, scale, aux_data, clamp);
    }

    template <class FloatAcc>
    __device__ static void
    Run(const bhalf8_t& inAcc, const bf8x8_t& residual, const bhalf2_t scale, FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const int32_t residual1    = bit_cast<int32x2_t>(residual)[0];
        const int32_t residual2    = bit_cast<int32x2_t>(residual)[1];
        outAcc.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_bf8_8x4(
                inAcc, residual1, residual2, scale, aux_data, clamp);
    }

    // f16
    template <class FloatAcc>
    __device__ static void Run(const half8_t& inAcc,
                               const half4_t& residual12,
                               const half4_t& residual34,
                               const half2_t scale,
                               FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        // c0-c7
        const int32_t residual1 = bit_cast<int32x2_t>(residual12)[0];
        const int32_t residual2 = bit_cast<int32x2_t>(residual12)[1];
        // c8-c15
        const int32_t residual3 = bit_cast<int32x2_t>(residual34)[0];
        const int32_t residual4 = bit_cast<int32x2_t>(residual34)[1];
        half4_t inAcc1;
        half4_t inAcc2;
        inAcc1[0] = inAcc[0];
        inAcc1[1] = inAcc[1];
        inAcc1[2] = inAcc[2];
        inAcc1[3] = inAcc[3];
        inAcc2[0] = inAcc[4];
        inAcc2[1] = inAcc[5];
        inAcc2[2] = inAcc[6];
        inAcc2[3] = inAcc[7];
        // first 4x4 tile
        outAcc.template AsType<half4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_f16_8x4(inAcc1,
                                                         bit_cast<half2_t>(residual1),
                                                         bit_cast<half2_t>(residual3),
                                                         scale,
                                                         aux_data,
                                                         clamp);
        // TODO: permute scale
        // second 4x4 tile
        outAcc.template AsType<half4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_f16_f16_8x4(inAcc1,
                                                         bit_cast<half2_t>(residual2),
                                                         bit_cast<half2_t>(residual4),
                                                         scale,
                                                         aux_data,
                                                         clamp);
    }

    // bf16
    template <class FloatAcc>
    __device__ static void Run(const bhalf8_t& inAcc,
                               const bhalf4_t& residual12,
                               const bhalf4_t& residual34,
                               const bhalf2_t scale,
                               FloatAcc& outAcc)
    {
        constexpr index_t aux_data = 0;
        constexpr bool clamp       = false;
        const int32_t residual1    = bit_cast<int32x2_t>(residual12)[0];
        const int32_t residual2    = bit_cast<int32x2_t>(residual12)[1];
        const int32_t residual3    = bit_cast<int32x2_t>(residual34)[0];
        const int32_t residual4    = bit_cast<int32x2_t>(residual34)[1];

        bhalf4_t inAcc1;
        bhalf4_t inAcc2;
        inAcc1[0] = inAcc[0];
        inAcc1[1] = inAcc[1];
        inAcc1[2] = inAcc[2];
        inAcc1[3] = inAcc[3];
        inAcc2[0] = inAcc[4];
        inAcc2[1] = inAcc[5];
        inAcc2[2] = inAcc[6];
        inAcc2[3] = inAcc[7];
        // first 4x4 tile
        outAcc.template AsType<bhalf4_t>()(Number<0>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_bf16_8x4(inAcc1,
                                                           bit_cast<bhalf2_t>(residual1),
                                                           bit_cast<bhalf2_t>(residual2),
                                                           scale,
                                                           aux_data,
                                                           clamp);
        // TODO: permute scale
        // second 4x4 tile
        outAcc.template AsType<bhalf4_t>()(Number<1>{}) =
            __builtin_amdgcn_fma_from_tensor_bf16_bf16_8x4(inAcc1,
                                                           bit_cast<bhalf2_t>(residual3),
                                                           bit_cast<bhalf2_t>(residual4),
                                                           scale,
                                                           aux_data,
                                                           clamp);
    }
};

} // namespace ck
#endif // CK_AMD_FMA_FROM_TENSOR_HPP
