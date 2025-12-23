// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef CK_AMD_WMMA_HPP
#define CK_AMD_WMMA_HPP

#include "ck/utility/amd_inline_asm.hpp"
#include "data_type.hpp"
#include "dtype_fp64.hpp"
// TODO: Add arch limitation
namespace ck {

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx11_generic__)
#define __gfx11__
#endif

#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx12_generic__)
#define __gfx120__
#endif

#if defined(__gfx1250__) || defined(__gfx1251__)
#define __gfx125__
#endif

/********************************WAVE32 MODE***********************************************/

// src: fp16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f16_w32;

template <>
struct intrin_wmma_f32_16x16x16_f16_w32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
        // * Inline assembly need to elimate the duplicated data load, compiler won't help you
        // delete them.
        // amd_assembly_wmma_f32_16x16x16_f16_w32(
        //     reg_a, reg_b, reg_c.template AsType<float8_t>()(Number<0>{}));
#if defined(__gfx11__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf16_w32;

template <>
struct intrin_wmma_f32_16x16x16_bf16_w32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: fp16, dst: fp16
template <index_t MPerWave, index_t NPerWave, index_t Opsel>
struct intrin_wmma_f16_16x16x16_f16_w32;

template <index_t Opsel>
struct intrin_wmma_f16_16x16x16_f16_w32<16, 16, Opsel>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx11__)
        reg_c.template AsType<half16_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
            reg_a, reg_b, reg_c.template AsType<half16_t>()[Number<0>{}], Opsel);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: bf16
template <index_t MPerWave, index_t NPerWave, index_t Opsel>
struct intrin_wmma_bf16_16x16x16_bf16_w32;

template <index_t Opsel>
struct intrin_wmma_bf16_16x16x16_bf16_w32<16, 16, Opsel>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx11__) || defined(__gfx120__)
        reg_c.template AsType<bhalf16_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(
                reg_a, reg_b, reg_c.template AsType<bhalf16_t>()[Number<0>{}], Opsel);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: iu8, dst: i32
template <index_t MPerWave, index_t NPerWave, bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w32;

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w32<16, 16, neg_a, neg_b, clamp>
{
    template <class FloatC>
    __device__ static void Run(const int8x16_t& reg_a, const int8x16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                neg_a,
                bit_cast<int32x4_t>(reg_a),
                neg_b,
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

/********************************WAVE64 MODE***********************************************/

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f16_w64;

template <>
struct intrin_wmma_f32_16x16x16_f16_w64<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x16_f16_w64(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf16_w64;

template <>
struct intrin_wmma_f32_16x16x16_bf16_w64<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w64(
                reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: fp16, dst: fp16
template <index_t MPerWave, index_t NPerWave, index_t Opsel>
struct intrin_wmma_f16_16x16x16_f16_w64;

template <index_t Opsel>
struct intrin_wmma_f16_16x16x16_f16_w64<16, 16, Opsel>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx11__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
            reg_a, reg_b, reg_c.template AsType<half8_t>()[Number<0>{}], Opsel);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: bf16
template <index_t MPerWave, index_t NPerWave, index_t Opsel>
struct intrin_wmma_bf16_16x16x16_bf16_w64;

template <index_t Opsel>
struct intrin_wmma_bf16_16x16x16_bf16_w64<16, 16, Opsel>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx11__)
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64(
                reg_a, reg_b, reg_c.template AsType<bhalf8_t>()[Number<0>{}], Opsel);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: iu8, dst: i32
template <index_t MPerWave, index_t NPerWave, bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w64;

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w64<16, 16, neg_a, neg_b, clamp>
{
    template <class FloatC>
    __device__ static void Run(const int8x16_t& reg_a, const int8x16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<int32x4_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(
                neg_a,
                bit_cast<int32x4_t>(reg_a),
                neg_b,
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<int32x4_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// gfx12
/********************************WAVE32 MODE***********************************************/

// src: fp16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f16_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_f16_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half8_t& reg_a, const half8_t& reg_b, FloatC& reg_c)
    {
        // * Inline assembly need to elimate the duplicated data load, compiler won't help you
        // delete them.
        // amd_assembly_wmma_f32_16x16x16_f16_w32(
        //     reg_a, reg_b, reg_c.template AsType<float8_t>()(Number<0>{}));
#if defined(__gfx120__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf16_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_bf16_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf8_t& reg_a, const bhalf8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx120__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: iu8, dst: i32
template <index_t MPerWave, index_t NPerWave, bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w32_gfx12;

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w32_gfx12<16, 16, neg_a, neg_b, clamp>
{
    template <class FloatC>
    __device__ static void Run(const int8x8_t& reg_a, const int8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx120__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                neg_a,
                bit_cast<int32x2_t>(reg_a),
                neg_b,
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: f8, f8, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f8f8_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_f8f8_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx120__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: f8, bf8, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f8bf8_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_f8bf8_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx120__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf8, f8, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf8f8_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_bf8f8_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx120__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf8, bf8, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf8bf8_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_bf8bf8_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx120__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// gfx125x
/********************************WAVE32 MODE***********************************************/
// src: fp16, dst: fp16
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x32_f16;

template <>
struct intrin_wmma_f16_16x16x32_f16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x32_f16(
            0, reg_a, 0, reg_b, 0, reg_c.template AsType<half8_t>()[Number<0>{}], false, false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: bf16
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_bf16_16x16x32_bf16;

template <>
struct intrin_wmma_bf16_16x16x32_bf16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx125__)
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_bf16_16x16x32_bf16(
            0, reg_a, 0, reg_b, 0, reg_c.template AsType<bhalf8_t>()[Number<0>{}], false, false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: fp16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x32_f16;

template <>
struct intrin_wmma_f32_16x16x32_f16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
        // * Inline assembly need to elimate the duplicated data load, compiler won't help you
        // delete them.
        // amd_assembly_wmma_f32_16x16x16_f16_w32(
        //     reg_a, reg_b, reg_c.template AsType<float8_t>()(Number<0>{}));
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x32_f16(
            0, reg_a, 0, reg_b, 0, reg_c.template AsType<float8_t>()[Number<0>{}], false, false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x32_bf16;

template <>
struct intrin_wmma_f32_16x16x32_bf16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x32_bf16(
            0, reg_a, 0, reg_b, 0, reg_c.template AsType<float8_t>()[Number<0>{}], false, false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: bf16
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_bf16f32_16x16x32_bf16;

template <>
struct intrin_wmma_bf16f32_16x16x32_bf16<16, 16>
{
    template <class FloatC, class FloatD>
    __device__ static void
    Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c, FloatD& reg_d)
    {
#if defined(__gfx125__)
        reg_d
            .template AsType<bhalf8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_bf16f32_16x16x32_bf16(
            0, reg_a, 0, reg_b, 0, reg_c.template AsType<float8_t>()[Number<0>{}], false, false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_d;
#endif
    }
};

// src: iu8, dst: i32
template <index_t MPerWave, index_t NPerWave, bool neg_a, bool neg_b>
struct intrin_wmma_i32_16x16x64_iu8;

template <bool neg_a, bool neg_b>
struct intrin_wmma_i32_16x16x64_iu8<16, 16, neg_a, neg_b>
{
    template <class FloatC>
    __device__ static void Run(const int8x32_t& reg_a, const int8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x64_iu8(neg_a,
                                                   bit_cast<int32x8_t>(reg_a),
                                                   neg_b,
                                                   bit_cast<int32x8_t>(reg_b),
                                                   reg_c.template AsType<int32x8_t>()[Number<0>{}],
                                                   false,
                                                   false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x64_f8f8;
template <>
struct intrin_wmma_f32_16x16x64_f8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<float8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x64_f8bf8;
template <>
struct intrin_wmma_f32_16x16x64_f8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<float8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x64_bf8f8;
template <>
struct intrin_wmma_f32_16x16x64_bf8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<float8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x64_bf8bf8;
template <>
struct intrin_wmma_f32_16x16x64_bf8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<float8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x64_f8f8;
template <>
struct intrin_wmma_f16_16x16x64_f8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x64_f8bf8;
template <>
struct intrin_wmma_f16_16x16x64_f8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x64_bf8f8;
template <>
struct intrin_wmma_f16_16x16x64_bf8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x64_bf8bf8;
template <>
struct intrin_wmma_f16_16x16x64_bf8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8(
            bit_cast<int32x8_t>(reg_a),
            bit_cast<int32x8_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x128_f8f8;
template <>
struct intrin_wmma_f32_16x16x128_f8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a, const f8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                false,
                false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x128_f8bf8;
template <>
struct intrin_wmma_f32_16x16x128_f8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a, const bf8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                false,
                false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x128_bf8f8;
template <>
struct intrin_wmma_f32_16x16x128_bf8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x64_t& reg_a, const f8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                false,
                false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x128_bf8bf8;
template <>
struct intrin_wmma_f32_16x16x128_bf8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x64_t& reg_a, const bf8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8(
                bit_cast<int32x16_t>(reg_a),
                bit_cast<int32x16_t>(reg_b),
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                false,
                false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x128_f8f8;
template <>
struct intrin_wmma_f16_16x16x128_f8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a, const f8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8(
            bit_cast<int32x16_t>(reg_a),
            bit_cast<int32x16_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x128_f8bf8;
template <>
struct intrin_wmma_f16_16x16x128_f8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a, const bf8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8(
            bit_cast<int32x16_t>(reg_a),
            bit_cast<int32x16_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x128_bf8f8;
template <>
struct intrin_wmma_f16_16x16x128_bf8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x64_t& reg_a, const f8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8(
            bit_cast<int32x16_t>(reg_a),
            bit_cast<int32x16_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f16_16x16x128_bf8bf8;
template <>
struct intrin_wmma_f16_16x16x128_bf8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x64_t& reg_a, const bf8x64_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8(
            bit_cast<int32x16_t>(reg_a),
            bit_cast<int32x16_t>(reg_b),
            0,
            reg_c.template AsType<half8_t>()[Number<0>{}],
            false,
            false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x4_f32;

template <>
struct intrin_wmma_f32_16x16x4_f32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const float2_t& reg_a, const float2_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x4_f32(0,
                                                  bit_cast<float2_t>(reg_a),
                                                  0,
                                                  bit_cast<float2_t>(reg_b),
                                                  0,
                                                  reg_c.template AsType<float8_t>()[Number<0>{}],
                                                  false,
                                                  false);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f64_16x16x4_f64;

template <>
struct intrin_wmma_f64_16x16x4_f64<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const double2_t& reg_a, const double2_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx1251__)
        reg_c.template AsType<double8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f64_16x16x4_f64(0,
                                                  bit_cast<double2_t>(reg_a),
                                                  0,
                                                  bit_cast<double2_t>(reg_b),
                                                  0,
                                                  reg_c.template AsType<double8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, index_t ScaleOpselA, index_t ScaleOpselB>
struct intrin_wmma_scale_f32_16x16x128_f8f6f4;

#ifndef CK_CODE_GEN_RTC
template <index_t ScaleOpselA, index_t ScaleOpselB>
struct intrin_wmma_scale_f32_16x16x128_f8f6f4<16, 16, ScaleOpselA, ScaleOpselB>
{
    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a,
                               const e8m0x4_bexp_t& scale_a,
                               const f8x64_t& reg_b,
                               const e8m0x4_bexp_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                0x0, // OPSEL:0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                reg_a,
                0x0, // OPSEL_HI
                reg_b,
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA, // SCALE_OPSEL[0]
                0,           // SCALE_OPSEL_HI[0]
                // M=laneId % 16 [7:0] K=0..31; [15:8] K=32..63; [23:16] K=64..95; [31:24] K=96..127
                bit_cast<int32_t>(scale_a),
                ScaleOpselB, // SCALE_OPSEL[1]
                0,           // SCALE_OPSEL_HI[1]
                // N=laneId % 16 [7:0] K=0..31; [15:8] K=32..63; [23:16] K=64..95; [31:24] K=96..127
                bit_cast<int32_t>(scale_b),
                0,  // NEG: 0-E8M0
                0); // NEG_HI: 0-E8M0
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf8x64_t& reg_a,
                               const e8m0x4_bexp_t& scale_a,
                               const bf8x64_t& reg_b,
                               const e8m0x4_bexp_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                0x1, // OPSEL:0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                reg_a,
                0x1, // OPSEL_HI
                reg_b,
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                // SCALE_OPSEL[0]
                0,                          // SCALE_OPSEL_HI[0]
                bit_cast<int32_t>(scale_a), // M=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                ScaleOpselB,                // SCALE_OPSEL[1]
                0,                          // SCALE_OPSEL_HI[1]
                bit_cast<int32_t>(scale_b), // N=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                0,                          // NEG
                0);                         // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f6x64_t& reg_a,
                               const e8m0x4_bexp_t& scale_a,
                               const f6x64_t& reg_b,
                               const e8m0x4_bexp_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        // f6x64_t is a vector of 2 f6x32_pk_t, so we have to repack and cast them to int32x6_t
        int32x6_t arg_a_0 = bit_cast<int32x6_t>(reg_a.AsType<f6x32_pk_t>()[Number<0>{}]);
        int32x6_t arg_a_1 = bit_cast<int32x6_t>(reg_a.AsType<f6x32_pk_t>()[Number<1>{}]);
        int32x6_t arg_b_0 = bit_cast<int32x6_t>(reg_b.AsType<f6x32_pk_t>()[Number<0>{}]);
        int32x6_t arg_b_1 = bit_cast<int32x6_t>(reg_b.AsType<f6x32_pk_t>()[Number<1>{}]);
        using arg_type    = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                0x2, // OPSEL:0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                arg_type{arg_a_0[0],
                         arg_a_0[1],
                         arg_a_0[2],
                         arg_a_0[3],
                         arg_a_0[4],
                         arg_a_0[5],
                         arg_a_1[0],
                         arg_a_1[1],
                         arg_a_1[2],
                         arg_a_1[3],
                         arg_a_1[4],
                         arg_a_1[5],
                         0,
                         0,
                         0,
                         0},
                0x2, // OPSEL_HI
                arg_type{arg_b_0[0],
                         arg_b_0[1],
                         arg_b_0[2],
                         arg_b_0[3],
                         arg_b_0[4],
                         arg_b_0[5],
                         arg_b_1[0],
                         arg_b_1[1],
                         arg_b_1[2],
                         arg_b_1[3],
                         arg_b_1[4],
                         arg_b_1[5],
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                // SCALE_OPSEL[0]
                0,                          // SCALE_OPSEL_HI[0]
                bit_cast<int32_t>(scale_a), // M=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                ScaleOpselB,                // SCALE_OPSEL[1]
                0,                          // SCALE_OPSEL_HI[1]
                bit_cast<int32_t>(scale_b), // N=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                0,                          // NEG
                0);                         // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf6x64_t& reg_a,
                               const e8m0x4_bexp_t& scale_a,
                               const bf6x64_t& reg_b,
                               const e8m0x4_bexp_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        // bf6x64_t is a vector of 2 bf6x32_pk_t, so we have to repack and cast them to int32x6_t
        int32x6_t arg_a_0 = bit_cast<int32x6_t>(reg_a.AsType<bf6x32_pk_t>()[Number<0>{}]);
        int32x6_t arg_a_1 = bit_cast<int32x6_t>(reg_a.AsType<bf6x32_pk_t>()[Number<1>{}]);
        int32x6_t arg_b_0 = bit_cast<int32x6_t>(reg_b.AsType<bf6x32_pk_t>()[Number<0>{}]);
        int32x6_t arg_b_1 = bit_cast<int32x6_t>(reg_b.AsType<bf6x32_pk_t>()[Number<1>{}]);
        using arg_type    = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                0x3, // OPSEL:0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                arg_type{arg_a_0[0],
                         arg_a_0[1],
                         arg_a_0[2],
                         arg_a_0[3],
                         arg_a_0[4],
                         arg_a_0[5],
                         arg_a_1[0],
                         arg_a_1[1],
                         arg_a_1[2],
                         arg_a_1[3],
                         arg_a_1[4],
                         arg_a_1[5],
                         0,
                         0,
                         0,
                         0},
                0x3, // OPSEL_HI
                arg_type{arg_b_0[0],
                         arg_b_0[1],
                         arg_b_0[2],
                         arg_b_0[3],
                         arg_b_0[4],
                         arg_b_0[5],
                         arg_b_1[0],
                         arg_b_1[1],
                         arg_b_1[2],
                         arg_b_1[3],
                         arg_b_1[4],
                         arg_b_1[5],
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                // SCALE_OPSEL[0]
                0,                          // SCALE_OPSEL_HI[0]
                bit_cast<int32_t>(scale_a), // M=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                ScaleOpselB,                // SCALE_OPSEL[1]
                0,                          // SCALE_OPSEL_HI[1]
                bit_cast<int32_t>(scale_b), // N=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                0,                          // NEG
                0);                         // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f4x64_t& reg_a,
                               const e8m0x4_bexp_t scale_a,
                               const f4x64_t& reg_b,
                               const e8m0x4_bexp_t scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        int32x8_t arg_a = bit_cast<int32x8_t>(reg_a);
        int32x8_t arg_b = bit_cast<int32x8_t>(reg_b);
        using arg_type  = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                0x4, // 0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                arg_type{arg_a[0],
                         arg_a[1],
                         arg_a[2],
                         arg_a[3],
                         arg_a[4],
                         arg_a[5],
                         arg_a[6],
                         arg_a[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0x4,
                arg_type{arg_b[0],
                         arg_b[1],
                         arg_b[2],
                         arg_b[3],
                         arg_b[4],
                         arg_b[5],
                         arg_b[6],
                         arg_b[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,
                0x0, // 0-E8M0, 1-E5M3, 2-E4M3
                bit_cast<int32_t>(scale_a),
                ScaleOpselB,
                0x0, // 0-E8M0, 1-E5M3, 2-E4M3
                bit_cast<int32_t>(scale_b),
                0,
                0);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f4x64_t& reg_a,
                               const e4m3x4_scale_t& scale_a,
                               const f4x64_t& reg_b,
                               const e4m3x4_scale_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        int32x8_t arg_a = bit_cast<int32x8_t>(reg_a);
        int32x8_t arg_b = bit_cast<int32x8_t>(reg_b);
        using arg_type  = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                0x4, // 0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                arg_type{arg_a[0],
                         arg_a[1],
                         arg_a[2],
                         arg_a[3],
                         arg_a[4],
                         arg_a[5],
                         arg_a[6],
                         arg_a[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0x4,
                arg_type{arg_b[0],
                         arg_b[1],
                         arg_b[2],
                         arg_b[3],
                         arg_b[4],
                         arg_b[5],
                         arg_b[6],
                         arg_b[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,
                0x2, // 0-E8M0, 1-E5M3, 2-E4M3
                bit_cast<int32_t>(scale_a),
                ScaleOpselB,
                0x2, // 0-E8M0, 1-E5M3, 2-E4M3
                bit_cast<int32_t>(scale_b),
                0,
                0);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f4x64_t& reg_a,
                               const e5m3x4_scale_t& scale_a,
                               const f4x64_t& reg_b,
                               const e5m3x4_scale_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        int32x8_t arg_a = bit_cast<int32x8_t>(reg_a);
        int32x8_t arg_b = bit_cast<int32x8_t>(reg_b);
        using arg_type  = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                0x4, // 0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                arg_type{arg_a[0],
                         arg_a[1],
                         arg_a[2],
                         arg_a[3],
                         arg_a[4],
                         arg_a[5],
                         arg_a[6],
                         arg_a[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0x4,
                arg_type{arg_b[0],
                         arg_b[1],
                         arg_b[2],
                         arg_b[3],
                         arg_b[4],
                         arg_b[5],
                         arg_b[6],
                         arg_b[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,
                0x1, // 0-E8M0, 1-E5M3, 2-E4M3
                bit_cast<int32_t>(scale_a),
                ScaleOpselB,
                0x1, // 0-E8M0, 1-E5M3, 2-E4M3
                bit_cast<int32_t>(scale_b),
                0,
                0);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f4x64_t& reg_a,
                               const e4m3x4_scale_t& scale_a,
                               const f4x64_t& reg_b,
                               const e5m3x4_scale_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        int32x8_t arg_a = bit_cast<int32x8_t>(reg_a);
        int32x8_t arg_b = bit_cast<int32x8_t>(reg_b);
        using arg_type  = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                0x4, // 0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                arg_type{arg_a[0],
                         arg_a[1],
                         arg_a[2],
                         arg_a[3],
                         arg_a[4],
                         arg_a[5],
                         arg_a[6],
                         arg_a[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0x4,
                arg_type{arg_b[0],
                         arg_b[1],
                         arg_b[2],
                         arg_b[3],
                         arg_b[4],
                         arg_b[5],
                         arg_b[6],
                         arg_b[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,
                0x2, // 0-E8M0, 1-E5M3, 2-E4M3
                bit_cast<int32_t>(scale_a),
                ScaleOpselB,
                0x1, // 0-E8M0, 1-E5M3, 2-E4M3
                bit_cast<int32_t>(scale_b),
                0,
                0);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f4x64_t& reg_a,
                               const e5m3x4_scale_t& scale_a,
                               const f4x64_t& reg_b,
                               const e4m3x4_scale_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        int32x8_t arg_a = bit_cast<int32x8_t>(reg_a);
        int32x8_t arg_b = bit_cast<int32x8_t>(reg_b);
        using arg_type  = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                0x4, // 0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                arg_type{arg_a[0],
                         arg_a[1],
                         arg_a[2],
                         arg_a[3],
                         arg_a[4],
                         arg_a[5],
                         arg_a[6],
                         arg_a[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0x4,
                arg_type{arg_b[0],
                         arg_b[1],
                         arg_b[2],
                         arg_b[3],
                         arg_b[4],
                         arg_b[5],
                         arg_b[6],
                         arg_b[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,
                0x1, // 0-E8M0, 1-E5M3, 2-E4M3
                bit_cast<int32_t>(scale_a),
                ScaleOpselB,
                0x2, // 0-E8M0, 1-E5M3, 2-E4M3
                bit_cast<int32_t>(scale_b),
                0,
                0);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a,
                               const e8m0x4_bexp_t& scale_a,
                               const f4x64_t& reg_b,
                               const e8m0x4_bexp_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        int32x8_t arg_b = bit_cast<int32x8_t>(reg_b);
        using arg_type  = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                0x0, // 0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                reg_a,
                0x4,
                arg_type{arg_b[0],
                         arg_b[1],
                         arg_b[2],
                         arg_b[3],
                         arg_b[4],
                         arg_b[5],
                         arg_b[6],
                         arg_b[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,
                0x0,
                bit_cast<int32_t>(scale_a),
                ScaleOpselB,
                0x0, // 0-E8M0, 1-E5M3, 2-E4M3
                bit_cast<int32_t>(scale_b),
                0,
                0);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }
};
#endif // #ifndef CK_CODE_GEN_RTC

template <index_t MPerWave, index_t NPerWave, index_t ScaleOpselA, index_t ScaleOpselB>
struct intrin_wmma_scale16_f32_16x16x128_f8f6f4;

#ifndef CK_CODE_GEN_RTC
template <index_t ScaleOpselA, index_t ScaleOpselB>
struct intrin_wmma_scale16_f32_16x16x128_f8f6f4<16, 16, ScaleOpselA, ScaleOpselB>
{
    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a,
                               const e8m0x8_bexp_t& scale_a,
                               const f8x64_t& reg_b,
                               const e8m0x8_bexp_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
                0x0, // OPSEL:0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                reg_a,
                0x0, // OPSEL_HI
                reg_b,
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                // SCALE_OPSEL[0]
                0,                          // SCALE_OPSEL_HI[0]
                bit_cast<int64_t>(scale_a), // M=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                ScaleOpselB,                // SCALE_OPSEL[1]
                0,                          // SCALE_OPSEL_HI[1]
                bit_cast<int64_t>(scale_b), // N=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                0,                          // NEG
                0);                         // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf8x64_t& reg_a,
                               const e8m0x8_bexp_t& scale_a,
                               const bf8x64_t& reg_b,
                               const e8m0x8_bexp_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
                0x1, // OPSEL:0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                reg_a,
                0x1, // OPSEL_HI
                reg_b,
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                // SCALE_OPSEL[0]
                0,                          // SCALE_OPSEL_HI[0]
                bit_cast<int64_t>(scale_a), // M=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                ScaleOpselB,                // SCALE_OPSEL[1]
                0,                          // SCALE_OPSEL_HI[1]
                bit_cast<int64_t>(scale_b), // N=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                0,                          // NEG
                0);                         // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f6x64_t& reg_a,
                               const e8m0x8_bexp_t& scale_a,
                               const f6x64_t& reg_b,
                               const e8m0x8_bexp_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        // f6x64_t is a vector of 2 f6x32_pk_t, so we have to repack and cast them to int32x6_t
        int32x6_t arg_a_0 = bit_cast<int32x6_t>(reg_a.AsType<f6x32_pk_t>()[Number<0>{}]);
        int32x6_t arg_a_1 = bit_cast<int32x6_t>(reg_a.AsType<f6x32_pk_t>()[Number<1>{}]);
        int32x6_t arg_b_0 = bit_cast<int32x6_t>(reg_b.AsType<f6x32_pk_t>()[Number<0>{}]);
        int32x6_t arg_b_1 = bit_cast<int32x6_t>(reg_b.AsType<f6x32_pk_t>()[Number<1>{}]);
        using arg_type    = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
                0x2, // OPSEL:0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                arg_type{arg_a_0[0],
                         arg_a_0[1],
                         arg_a_0[2],
                         arg_a_0[3],
                         arg_a_0[4],
                         arg_a_0[5],
                         arg_a_1[0],
                         arg_a_1[1],
                         arg_a_1[2],
                         arg_a_1[3],
                         arg_a_1[4],
                         arg_a_1[5],
                         0,
                         0,
                         0,
                         0},
                0x2, // OPSEL_HI
                arg_type{arg_b_0[0],
                         arg_b_0[1],
                         arg_b_0[2],
                         arg_b_0[3],
                         arg_b_0[4],
                         arg_b_0[5],
                         arg_b_1[0],
                         arg_b_1[1],
                         arg_b_1[2],
                         arg_b_1[3],
                         arg_b_1[4],
                         arg_b_1[5],
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                // SCALE_OPSEL[0]
                0,                          // SCALE_OPSEL_HI[0]
                bit_cast<int64_t>(scale_a), // M=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                ScaleOpselB,                // SCALE_OPSEL[1]
                0,                          // SCALE_OPSEL_HI[1]
                bit_cast<int64_t>(scale_b), // N=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                0,                          // NEG
                0);                         // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf6x64_t& reg_a,
                               const e8m0x8_bexp_t& scale_a,
                               const bf6x64_t& reg_b,
                               const e8m0x8_bexp_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        // bf6x64_t is a vector of 2 bf6x32_pk_t, so we have to repack and cast them to int32x6_t
        int32x6_t arg_a_0 = bit_cast<int32x6_t>(reg_a.AsType<bf6x32_pk_t>()[Number<0>{}]);
        int32x6_t arg_a_1 = bit_cast<int32x6_t>(reg_a.AsType<bf6x32_pk_t>()[Number<1>{}]);
        int32x6_t arg_b_0 = bit_cast<int32x6_t>(reg_b.AsType<bf6x32_pk_t>()[Number<0>{}]);
        int32x6_t arg_b_1 = bit_cast<int32x6_t>(reg_b.AsType<bf6x32_pk_t>()[Number<1>{}]);
        using arg_type    = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
                0x3, // OPSEL:0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                arg_type{arg_a_0[0],
                         arg_a_0[1],
                         arg_a_0[2],
                         arg_a_0[3],
                         arg_a_0[4],
                         arg_a_0[5],
                         arg_a_1[0],
                         arg_a_1[1],
                         arg_a_1[2],
                         arg_a_1[3],
                         arg_a_1[4],
                         arg_a_1[5],
                         0,
                         0,
                         0,
                         0},
                0x3, // OPSEL_HI
                arg_type{arg_b_0[0],
                         arg_b_0[1],
                         arg_b_0[2],
                         arg_b_0[3],
                         arg_b_0[4],
                         arg_b_0[5],
                         arg_b_1[0],
                         arg_b_1[1],
                         arg_b_1[2],
                         arg_b_1[3],
                         arg_b_1[4],
                         arg_b_1[5],
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,                // SCALE_OPSEL[0]
                0,                          // SCALE_OPSEL_HI[0]
                bit_cast<int64_t>(scale_a), // M=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                ScaleOpselB,                // SCALE_OPSEL[1]
                0,                          // SCALE_OPSEL_HI[1]
                bit_cast<int64_t>(scale_b), // N=laneId [7:0] K=0..31; [15:8] K=32..63; [23:16]
                                            // K=64..95; [31:24] K=96..127
                0,                          // NEG
                0);                         // NEG_HI
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f4x64_t& reg_a,
                               const e8m0x8_bexp_t& scale_a,
                               const f4x64_t& reg_b,
                               const e8m0x8_bexp_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        int32x8_t arg_a = bit_cast<int32x8_t>(reg_a);
        int32x8_t arg_b = bit_cast<int32x8_t>(reg_b);
        using arg_type  = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
                0x4, // 0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                arg_type{arg_a[0],
                         arg_a[1],
                         arg_a[2],
                         arg_a[3],
                         arg_a[4],
                         arg_a[5],
                         arg_a[6],
                         arg_a[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0x4,
                arg_type{arg_b[0],
                         arg_b[1],
                         arg_b[2],
                         arg_b[3],
                         arg_b[4],
                         arg_b[5],
                         arg_b[6],
                         arg_b[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,
                0,
                bit_cast<int64_t>(scale_a),
                ScaleOpselB,
                0,
                bit_cast<int64_t>(scale_b),
                0,
                0);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f4x64_t& reg_a,
                               const e4m3x8_scale_t& scale_a,
                               const f4x64_t& reg_b,
                               const e4m3x8_scale_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        int32x8_t arg_a = bit_cast<int32x8_t>(reg_a);
        int32x8_t arg_b = bit_cast<int32x8_t>(reg_b);
        using arg_type  = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
                0x4, // 0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                arg_type{arg_a[0],
                         arg_a[1],
                         arg_a[2],
                         arg_a[3],
                         arg_a[4],
                         arg_a[5],
                         arg_a[6],
                         arg_a[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0x4,
                arg_type{arg_b[0],
                         arg_b[1],
                         arg_b[2],
                         arg_b[3],
                         arg_b[4],
                         arg_b[5],
                         arg_b[6],
                         arg_b[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,
                0x2, // 0-E8M0, 1-E5M3, 2-E4M3
                bit_cast<int64_t>(scale_a),
                ScaleOpselB,
                0x2, // 0-E8M0, 1-E5M3, 2-E4M3
                bit_cast<int64_t>(scale_b),
                0,
                0);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f4x64_t& reg_a,
                               const e5m3x8_scale_t& scale_a,
                               const f4x64_t& reg_b,
                               const e5m3x8_scale_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        int32x8_t arg_a = bit_cast<int32x8_t>(reg_a);
        int32x8_t arg_b = bit_cast<int32x8_t>(reg_b);
        using arg_type  = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
                0x4, // 0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                arg_type{arg_a[0],
                         arg_a[1],
                         arg_a[2],
                         arg_a[3],
                         arg_a[4],
                         arg_a[5],
                         arg_a[6],
                         arg_a[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0x4,
                arg_type{arg_b[0],
                         arg_b[1],
                         arg_b[2],
                         arg_b[3],
                         arg_b[4],
                         arg_b[5],
                         arg_b[6],
                         arg_b[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,
                0x1, // 0-E8M0, 1-E5M3, 2-E4M3
                bit_cast<int64_t>(scale_a),
                ScaleOpselB,
                0x1, // 0-E8M0, 1-E5M3, 2-E4M3
                bit_cast<int64_t>(scale_b),
                0,
                0);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f4x64_t& reg_a,
                               const e4m3x8_scale_t& scale_a,
                               const f4x64_t& reg_b,
                               const e5m3x8_scale_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        int32x8_t arg_a = bit_cast<int32x8_t>(reg_a);
        int32x8_t arg_b = bit_cast<int32x8_t>(reg_b);
        using arg_type  = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
                0x4, // 0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                arg_type{arg_a[0],
                         arg_a[1],
                         arg_a[2],
                         arg_a[3],
                         arg_a[4],
                         arg_a[5],
                         arg_a[6],
                         arg_a[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0x4,
                arg_type{arg_b[0],
                         arg_b[1],
                         arg_b[2],
                         arg_b[3],
                         arg_b[4],
                         arg_b[5],
                         arg_b[6],
                         arg_b[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,
                0x2, // 0-E8M0, 1-E5M3, 2-E4M3
                bit_cast<int64_t>(scale_a),
                ScaleOpselB,
                0x1, // 0-E8M0, 1-E5M3, 2-E4M3
                bit_cast<int64_t>(scale_b),
                0,
                0);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f4x64_t& reg_a,
                               const e5m3x8_scale_t& scale_a,
                               const f4x64_t& reg_b,
                               const e4m3x8_scale_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        int32x8_t arg_a = bit_cast<int32x8_t>(reg_a);
        int32x8_t arg_b = bit_cast<int32x8_t>(reg_b);
        using arg_type  = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
                0x4, // 0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                arg_type{arg_a[0],
                         arg_a[1],
                         arg_a[2],
                         arg_a[3],
                         arg_a[4],
                         arg_a[5],
                         arg_a[6],
                         arg_a[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0x4,
                arg_type{arg_b[0],
                         arg_b[1],
                         arg_b[2],
                         arg_b[3],
                         arg_b[4],
                         arg_b[5],
                         arg_b[6],
                         arg_b[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,
                0x1, // 0-E8M0, 1-E5M3, 2-E4M3
                bit_cast<int64_t>(scale_a),
                ScaleOpselB,
                0x2, // 0-E8M0, 1-E5M3, 2-E4M3
                bit_cast<int64_t>(scale_b),
                0,
                0);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f8x64_t& reg_a,
                               const e8m0x8_bexp_t scale_a,
                               const f4x64_t& reg_b,
                               const e8m0x8_bexp_t scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        int32x8_t arg_b = bit_cast<int32x8_t>(reg_b);
        using arg_type  = int32x16_t;
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
                0x0, // 0-FP8 E4M3; 1-FP8 E5M2; 2-FP6 E2M3; 3-FP6 E3M2; 4-FP4 E2M1
                reg_a,
                0x4,
                arg_type{arg_b[0],
                         arg_b[1],
                         arg_b[2],
                         arg_b[3],
                         arg_b[4],
                         arg_b[5],
                         arg_b[6],
                         arg_b[7],
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0},
                0,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                ScaleOpselA,
                0,
                bit_cast<int64_t>(scale_a),
                ScaleOpselB,
                0,
                bit_cast<int64_t>(scale_b),
                0,
                0);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }
};
#endif // #ifndef CK_CODE_GEN_RTC

template <index_t MPerWave, index_t NPerWave, index_t ScaleOpselB>
struct intrin_wmma_scale_f32_32x16x128_f4;

template <index_t ScaleOpselB>
struct intrin_wmma_scale_f32_32x16x128_f4<32, 16, ScaleOpselB>
{
    template <class FloatC>
    __device__ static void Run(const f4x128_t& reg_a,
                               const int32_t scale_a,
                               const f4x64_t& reg_b,
                               const int32_t scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx125__)
        int32x16_t arg_a = bit_cast<int32x16_t>(reg_a);
        int32x8_t arg_b  = bit_cast<int32x8_t>(reg_b);
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_scale_f32_32x16x128_f4(
                arg_a,
                arg_b,
                0,
                reg_c.template AsType<float16_t>()[Number<0>{}],
                1, // fix ScaleOpselA as 1
                0,
                scale_a,
                ScaleOpselB,
                0,
                scale_b,
                0,
                0);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }
};

} // namespace ck
#endif
