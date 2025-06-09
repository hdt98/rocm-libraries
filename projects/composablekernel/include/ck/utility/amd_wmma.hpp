// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_AMD_WMMA_HPP
#define CK_AMD_WMMA_HPP

#include "ck/utility/amd_inline_asm.hpp"
#include "data_type.hpp"
// TODO: Add arch limitation
namespace ck {

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx11_generic__)
#define __gfx11__
#endif

#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx12_generic__)
#define __gfx12__
#endif

#if defined(__gfx1300__) || defined(__gfx1301__) || defined(__gfx1302__) || \
    defined(__gfx130E__) || defined(__gfx130F__) || defined(__gfx13_generic__)
#define __gfx13__
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

    template <class FloatC>
    __device__ static void Run(const half8_t& reg_a, const half8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx12__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void
    Run(const half8_t& reg_a, const half8_t& reg_b, FloatC& reg_c, const int& k_multiplier)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_clamp(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
        ignore = k_multiplier;
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

    template <class FloatC>
    __device__ static void Run(const bhalf8_t& reg_a, const bhalf8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx12__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void
    Run(const bhalf8_t& reg_a, const bhalf8_t& reg_b, FloatC& reg_c, const int& k_multiplier)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_clamp(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
        ignore = k_multiplier;
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

    template <class FloatC>
    __device__ static void Run(const half8_t& reg_a, const half8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx12__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(
                reg_a, reg_b, reg_c.template AsType<half8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void
    Run(const half8_t& reg_a, const half8_t& reg_b, FloatC& reg_c, const int& k_multiplier)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x16_f16_clamp(
                reg_a, reg_b, reg_c.template AsType<half8_t>()[Number<0>{}], 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
        ignore = k_multiplier;
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
#if defined(__gfx11__) || defined(__gfx12__)
        reg_c.template AsType<bhalf16_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(
                reg_a, reg_b, reg_c.template AsType<bhalf16_t>()[Number<0>{}], Opsel);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bhalf8_t& reg_a, const bhalf8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx12__)
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12(
                reg_a, reg_b, reg_c.template AsType<bhalf8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void
    Run(const bhalf8_t& reg_a, const bhalf8_t& reg_b, FloatC& reg_c, const int& k_multiplier)
    {
#if defined(__gfx13__)
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_bf16_16x16x16_bf16_clamp(
                reg_a, reg_b, reg_c.template AsType<bhalf8_t>()[Number<0>{}], 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
        ignore = k_multiplier;
    }
};

// src: iu8, dst: i32
template <index_t MPerWave, index_t NPerWave, bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w32;

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w32<16, 16, neg_a, neg_b, clamp>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        if constexpr(sizeof(reg_a) == sizeof(int32x4_t))
        {
            reg_c.template AsType<int32x8_t>()(Number<0>{}) =
                __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    neg_a,
                    bit_cast<int32x4_t>(reg_a),
                    neg_b,
                    bit_cast<int32x4_t>(reg_b),
                    reg_c.template AsType<int32x8_t>()[Number<0>{}],
                    clamp);
        }
        else
#elif defined(__gfx12__)
        if constexpr(sizeof(reg_a) == sizeof(int32x2_t))
        {
            reg_c.template AsType<int32x8_t>()(Number<0>{}) =
                __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                    neg_a,
                    bit_cast<int32x2_t>(reg_a),
                    neg_b,
                    bit_cast<int32x2_t>(reg_b),
                    reg_c.template AsType<int32x8_t>()[Number<0>{}],
                    clamp);
        }
        else
#endif
        {
            ignore = reg_a;
            ignore = reg_b;
            ignore = reg_c;
        }
    }

    template <class FloatA, class FloatB, class FloatC>
    __device__ static void
    Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c, const int& k_multiplier)
    {
#if defined(__gfx13__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_clamp(
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
        ignore = k_multiplier;
    }
};

/**********************************GFX13***************************************************/
// for gfx13 usage
template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_f16f16_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_f16f16_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const half8_t& reg_a, const half8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1, "gfx13 this opcode only support kMultiplier = 1.");
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_clamp(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: fp32
template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_bf16bf16_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_bf16bf16_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const bhalf8_t& reg_a, const bhalf8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1, "gfx13 this opcode only support kMultiplier = 1.");
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_clamp(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_f16f16_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_f16f16_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const half8_t& reg_a, const half8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1, "gfx13 this opcode only support kMultiplier = 1.");
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x16_f16_clamp(
                reg_a, reg_b, reg_c.template AsType<half8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_bf16_16x16_bf16bf16_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_bf16_16x16_bf16bf16_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const bhalf8_t& reg_a, const bhalf8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1, "gfx13 this opcode only support kMultiplier = 1.");
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_bf16_16x16x16_bf16_clamp(
                reg_a, reg_b, reg_c.template AsType<bhalf8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave,
          index_t NPerWave,
          bool neg_a,
          bool neg_b,
          bool clamp,
          index_t kMultiplier>
struct intrin_wmma_i32_16x16_iu8iu8_w32;

template <bool neg_a, bool neg_b, bool clamp, index_t kMultiplier>
struct intrin_wmma_i32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, kMultiplier>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1 || kMultiplier == 2,
                      "gfx13 this opcode only support kMultiplier = 1 or 2.");
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_clamp(
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

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, 2>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x32_iu8_clamp(
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

template <index_t MPerWave,
          index_t NPerWave,
          bool neg_a,
          bool neg_b,
          bool clamp,
          index_t kMultiplier>
struct intrin_wmma_f32_16x16_iu8iu8_w32;

template <bool neg_a, bool neg_b, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, kMultiplier>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_iu8_clamp(
                neg_a,
                bit_cast<int32x2_t>(reg_a),
                neg_b,
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

constexpr index_t VOP5M_MOD1_OFFSET = 6;
// this field in mod1
template <typename SrcAType, typename SrcBType>
constexpr int32_t get_f8f6f4_data_format()
{
    int32_t srcAType = static_cast<int32_t>(SrcAType::RawType);
    int32_t srcBType = static_cast<int32_t>(SrcBType::RawType);
    return (srcAType | (srcBType << 3)) << VOP5M_MOD1_OFFSET;
}

// this field in mod1
template <index_t ABlockSel, index_t BBlockSel>
constexpr int32_t scale_select()
{
    return ((ABlockSel) | (BBlockSel << 2)) << (VOP5M_MOD1_OFFSET + 8);
}

// src: f8f6f4, dst: fp32
template <index_t MPerWave,
          index_t NPerWave,
          typename SrcAType,
          typename SrcBType,
          index_t ABlockSel,
          index_t BBlockSel,
          bool clamp>
struct intrin_wmma_f32_16x16_f8f6f4_w32;

template <typename SrcAType, typename SrcBType, index_t ABlockSel, index_t BBlockSel, bool clamp>
struct intrin_wmma_f32_16x16_f8f6f4_w32<16, 16, SrcAType, SrcBType, ABlockSel, BBlockSel, clamp>
{
    template <class FloatC>
    __device__ static void Run(const typename SrcAType::vec_t& reg_a,
                               const typename SrcBType::vec_t& reg_b,
                               const int32_t& a_scale,
                               const int32_t& b_scale,
                               FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x64_f8f6f4_clamp(
                reg_a,
                reg_b,
                reg_c.template AsType<float8_t>()[Number<0>{}],
                a_scale,
                b_scale,
                get_f8f6f4_data_format<SrcAType, SrcBType>() | scale_select<ABlockSel, BBlockSel>(),
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = a_scale;
        ignore = b_scale;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_f32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, 2>
{
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x32_iu8_clamp(
                neg_a,
                bit_cast<int32x4_t>(reg_a),
                neg_b,
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_f8f8_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_f8f8_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1,
                      "gfx13 this opcode for this A, B type only support kMultiplier = 1.");
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_clamp(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool clamp>
struct intrin_wmma_f32_16x16_f8f8_w32<16, 16, clamp, 2>
{
    template <class FloatC>
    __device__ static void Run(const f8x16_t& reg_a, const f8x16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x32_fp8_fp8_clamp(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_f8bf8_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_f8bf8_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1,
                      "gfx13 this opcode for this A, B type only support kMultiplier = 1.");
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_clamp(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool clamp>
struct intrin_wmma_f32_16x16_f8bf8_w32<16, 16, clamp, 2>
{
    template <class FloatC>
    __device__ static void Run(const f8x16_t& reg_a, const bf8x16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x32_fp8_bf8_clamp(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_bf8f8_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_bf8f8_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1,
                      "gfx13 this opcode for this A, B type only support kMultiplier = 1.");
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_clamp(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool clamp>
struct intrin_wmma_f32_16x16_bf8f8_w32<16, 16, clamp, 2>
{
    template <class FloatC>
    __device__ static void Run(const bf8x16_t& reg_a, const f8x16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x32_bf8_fp8_clamp(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_bf8bf8_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_bf8bf8_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1,
                      "gfx13 this opcode for this A, B type only support kMultiplier = 1.");
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_clamp(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool clamp>
struct intrin_wmma_f32_16x16_bf8bf8_w32<16, 16, clamp, 2>
{
    template <class FloatC>
    __device__ static void Run(const bf8x16_t& reg_a, const bf8x16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x32_bf8_bf8_clamp(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_f8bf8_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_f8bf8_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1,
                      "gfx13 this opcode for this A, B type only support kMultiplier = 1.");
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x16_fp8_bf8_clamp(
                reg_a, reg_b, reg_c.template AsType<half8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool clamp>
struct intrin_wmma_f16_16x16_f8bf8_w32<16, 16, clamp, 2>
{
    template <class FloatC>
    __device__ static void Run(const f8x16_t& reg_a, const bf8x16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x32_fp8_bf8_clamp(
                reg_a, reg_b, reg_c.template AsType<half8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_bf8f8_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_bf8f8_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1,
                      "gfx13 this opcode for this A, B type only support kMultiplier = 1.");
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x16_bf8_fp8_clamp(
                reg_a, reg_b, reg_c.template AsType<half8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool clamp>
struct intrin_wmma_f16_16x16_bf8f8_w32<16, 16, clamp, 2>
{
    template <class FloatC>
    __device__ static void Run(const bf8x16_t& reg_a, const f8x16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x32_bf8_fp8_clamp(
                reg_a, reg_b, reg_c.template AsType<half8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_bf8bf8_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_bf8bf8_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1,
                      "gfx13 this opcode for this A, B type only support kMultiplier = 1.");
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x16_bf8_bf8_clamp(
                reg_a, reg_b, reg_c.template AsType<half8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool clamp>
struct intrin_wmma_f16_16x16_bf8bf8_w32<16, 16, clamp, 2>
{
    template <class FloatC>
    __device__ static void Run(const bf8x16_t& reg_a, const bf8x16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x32_bf8_bf8_clamp(
                reg_a, reg_b, reg_c.template AsType<half8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_f8f8_w32;

template <bool clamp, index_t kMultiplier>
struct intrin_wmma_f16_16x16_f8f8_w32<16, 16, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        static_assert(kMultiplier == 1,
                      "gfx13 this opcode for this A, B type only support kMultiplier = 1.");
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x16_fp8_fp8_clamp(
                reg_a, reg_b, reg_c.template AsType<half8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool clamp>
struct intrin_wmma_f16_16x16_f8f8_w32<16, 16, clamp, 2>
{
    template <class FloatC>
    __device__ static void Run(const f8x16_t& reg_a, const f8x16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<half8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f16_16x16x32_fp8_fp8_clamp(
                reg_a, reg_b, reg_c.template AsType<half8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave,
          index_t NPerWave,
          bool neg_a,
          bool neg_b,
          bool clamp,
          index_t kMultiplier>
struct intrin_wmma_i32_16x16_iu4iu4_w32;

template <bool neg_a, bool neg_b, bool clamp, index_t kMultiplier>
struct intrin_wmma_i32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const int32_t& reg_a, const int32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x16_iu4_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<int32x8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, 2>
{
    template <class FloatC>
    __device__ static void Run(const int32x2_t& reg_a, const int32x2_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x32_iu4_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<int32x8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, 4>
{
    template <class FloatC>
    __device__ static void Run(const int32x4_t& reg_a, const int32x4_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x64_iu4_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<int32x8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave,
          index_t NPerWave,
          bool neg_a,
          bool neg_b,
          bool clamp,
          index_t kMultiplier>
struct intrin_wmma_f32_16x16_iu4iu4_w32;

template <bool neg_a, bool neg_b, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, kMultiplier>
{
    template <class FloatC>
    __device__ static void Run(const int32_t& reg_a, const int32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_iu4_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_f32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, 2>
{
    template <class FloatC>
    __device__ static void Run(const int32x2_t& reg_a, const int32x2_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x32_iu4_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_f32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, 4>
{
    template <class FloatC>
    __device__ static void Run(const int32x4_t& reg_a, const int32x4_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx13__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x64_iu4_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave,
          index_t NPerWave,
          bool neg_a,
          bool neg_b,
          bool clamp,
          index_t kMultiplier>
struct intrin_wmma_f32i32_16x16_iu8iu8_w32;

template <bool neg_a, bool neg_b, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32i32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, kMultiplier>
{
    template <class FloatA, class FloatB, class FloatC, class FloatD>
    __device__ static void
    Run(const FloatA& reg_a, const FloatB& reg_b, const FloatC& reg_c, FloatD& reg_d)
    {
#if defined(__gfx13__)
        reg_d.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32i32_16x16x16_iu8_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<int32x8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_d;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_f32i32_16x16_iu8iu8_w32<16, 16, neg_a, neg_b, clamp, 2>
{
    template <class FloatA, class FloatB, class FloatC, class FloatD>
    __device__ static void
    Run(const FloatA& reg_a, const FloatB& reg_b, const FloatC& reg_c, FloatD& reg_d)
    {
#if defined(__gfx13__)
        reg_d.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32i32_16x16x32_iu8_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<int32x8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_d;
#endif
    }
};

template <index_t MPerWave,
          index_t NPerWave,
          bool neg_a,
          bool neg_b,
          bool clamp,
          index_t kMultiplier>
struct intrin_wmma_f32i32_16x16_iu4iu4_w32;

template <bool neg_a, bool neg_b, bool clamp, index_t kMultiplier>
struct intrin_wmma_f32i32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, kMultiplier>
{
    template <class FloatC, class FloatD>
    __device__ static void
    Run(const int32_t& reg_a, const int32_t& reg_b, const FloatC& reg_c, FloatD& reg_d)
    {
#if defined(__gfx13__)
        reg_d.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32i32_16x16x16_iu4_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<int32x8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_d;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_f32i32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, 2>
{
    template <class FloatC, class FloatD>
    __device__ static void
    Run(const int32x2_t& reg_a, const int32x2_t& reg_b, const FloatC& reg_c, FloatD& reg_d)
    {
#if defined(__gfx13__)
        reg_d.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32i32_16x16x32_iu4_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<int32x8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_d;
#endif
    }
};

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_f32i32_16x16_iu4iu4_w32<16, 16, neg_a, neg_b, clamp, 4>
{
    template <class FloatC, class FloatD>
    __device__ static void
    Run(const int32x4_t& reg_a, const int32x4_t& reg_b, const FloatC& reg_c, FloatD& reg_d)
    {
#if defined(__gfx13__)
        reg_d.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32i32_16x16x64_iu4_clamp(
                neg_a, reg_a, neg_b, reg_b, reg_c.template AsType<int32x8_t>()[Number<0>{}], clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
        ignore = reg_d;
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
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
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
#if defined(__gfx12__)
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
#if defined(__gfx12__)
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
    template <class FloatA, class FloatB, class FloatC>
    __device__ static void Run(const FloatA& reg_a, const FloatB& reg_b, FloatC& reg_c)
    {
#if defined(__gfx12__)
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
#if defined(__gfx12__)
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
#if defined(__gfx12__)
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
#if defined(__gfx12__)
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
#if defined(__gfx12__)
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

} // namespace ck
#endif
