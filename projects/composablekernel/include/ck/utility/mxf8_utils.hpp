// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/utility/numeric_limits.hpp"
#include "ck/utility/mxfp_utils.hpp"

#if defined(__gfx950__) && __HIP_DEVICE_COMPILE__
#define CK_MX_ARCH_950 1
#else
#define CK_MX_ARCH_950 0
#endif
#if defined(__gfx1250__) && __HIP_DEVICE_COMPILE__
#define CK_MX_ARCH_1250 1
#else
#define CK_MX_ARCH_1250 0
#endif

#if CK_MX_ARCH_950 || CK_MX_ARCH_1250
#define CK_MX_FP8_CVT_FAST_PATH 1
#else
#define CK_MX_FP8_CVT_FAST_PATH 0
#endif

namespace ck {

namespace fp8_impl {

// FUNCTION: cast_to_f8_from_f32_scaled
#if CK_MX_FP8_CVT_FAST_PATH

// Forward declarations

template <ck_fp8_interpretation_t interpret>
static __device__ float cast_to_f32_from_f8_scaled(float scale, fp8_storage_t v);

template <ck_fp8_interpretation_t interpret>
static __device__ float2_t cast_to_f32_from_f8_scaled(float scale, fp8x2_storage_t v);

template <ck_fp8_interpretation_t interpret, typename Ts = float, int Opsel = 0>
static __device__ float8_t cast_to_f32_from_f8_scaled(Ts scale, fp8x8_storage_t v);

template <ck_fp8_interpretation_t interpret, bool stochastic_rounding = false>
static __device__ fp8_storage_t cast_to_f8_from_f32_scaled(float v,
                                                           unsigned int rng = 0,
                                                           float scale      = 1.0f);

template <ck_fp8_interpretation_t interpret, bool stochastic_rounding = false>
static __device__ fp8x2_storage_t cast_to_f8_from_f32_scaled(float2_t v,
                                                             unsigned int rng = 0,
                                                             float scale      = 1.0f);

template <ck_fp8_interpretation_t interpret, bool stochastic_rounding = false>
static __device__ fp8x8_storage_t cast_to_f8_from_f32_scaled(float8_t v,
                                                             unsigned int rng = 0,
                                                             float scale      = 1.0f);

// Implementations for different architectures
#if CK_MX_ARCH_950

template <ck_fp8_interpretation_t interpret>
static __device__ float cast_to_f32_from_f8_scaled(float scale, fp8_storage_t v)
{
    union
    {
        unsigned int i32val;
        unsigned char i8val[4];
    } val;
    val.i8val[0] = v;

    static_assert(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP ||
                      interpret == ck_fp8_interpretation_t::CK_E5M2_OCP,
                  "Only OCP interpretations are supported");

    if constexpr(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
    {
        return __builtin_amdgcn_cvt_scalef32_f32_fp8(val.i32val, scale, 0);
    }
    else
    {
        return __builtin_amdgcn_cvt_scalef32_f32_bf8(val.i32val, scale, 0);
    }
}

template <ck_fp8_interpretation_t interpret>
static __device__ float2_t cast_to_f32_from_f8_scaled(float scale, fp8x2_storage_t v)
{
    const auto i16val = bit_cast<uint16_t>(v);

    static_assert(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP ||
                      interpret == ck_fp8_interpretation_t::CK_E5M2_OCP,
                  "Only OCP interpretations are supported");

    if constexpr(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
    {
        return __builtin_amdgcn_cvt_scalef32_pk_f32_fp8(i16val, scale, 0);
    }
    else
    {
        return __builtin_amdgcn_cvt_scalef32_pk_f32_bf8(i16val, scale, 0);
    }
}

template <ck_fp8_interpretation_t interpret, typename Ts, int Opsel>
static __device__ float8_t cast_to_f32_from_f8_scaled(Ts scale, fp8x8_storage_t v)
{
    static_assert(std::is_same_v<Ts, float>, "Ts must be float");
    union
    {
        float8_t v8f32x1;
        float2_t v2f32x4[4];
    } out;

    union
    {
        fp8x8_storage_t v8f8x1;
        fp8x2_storage_t v2f8x4[4];
    } in{v};

    ck::static_for<0, 4, 1>{}([&](auto i) {
        out.v2f32x4[i] = cast_to_f32_from_f8_scaled<interpret>(scale, in.v2f8x4[i]);
    });

    return out.v8f32x1;
}

template <ck_fp8_interpretation_t interpret, bool stochastic_rounding>
static __device__ fp8_storage_t cast_to_f8_from_f32_scaled(float v, unsigned int rng, float scale)
{
    fp8_storage_t i8data;
    union
    {
        float fval;
        unsigned int i32val;
    } val;

    union
    {
        uint32_t ival;
        vector_type<int16_t, 2>::type v2i16;
        fp8_storage_t v4i8[4];
    } ret{};

    // unsigned int ival = 0;
    val.fval = v;

    if constexpr(stochastic_rounding)
    {
        ret.ival =
            (interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
                ? __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(ret.ival, val.fval, rng, scale, 0)
                : __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(ret.ival, val.fval, rng, scale, 0);

        i8data = ret.v4i8[0];
    }
    else
    {
        // RNE CVT
        // llvm.amdgcn.cvt.scalef32.pk.fp8.f32
        // v2i16 old_vdst, float srcA, float srcB, float scale, bool dst_lo_hi_sel
        if constexpr(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
        {
            // If fval / scale > max fp8, returns Nan
            ret.v2i16 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(/*old_vdst*/ ret.v2i16,
                                                                 val.fval,
                                                                 val.fval,
                                                                 scale,
                                                                 /*dst_lo_hi_sel*/ false);
        }
        else
        {
            // If fval / scale > max bf8, returns Inf
            ret.v2i16 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f32(/*old_vdst*/ ret.v2i16,
                                                                 val.fval,
                                                                 val.fval,
                                                                 scale,
                                                                 /*dst_lo_hi_sel*/ false);
        }

        i8data = ret.v4i8[0];
    }
    return i8data;
}

template <ck_fp8_interpretation_t interpret, bool stochastic_rounding>
static __device__ fp8x2_storage_t cast_to_f8_from_f32_scaled(float2_t v,
                                                             unsigned int rng,
                                                             float scale)
{
    union
    {
        uint32_t ival;
        vector_type<int16_t, 2>::type v2i16;
        StaticallyIndexedArray<fp8x2_storage_t, 2> v2f8x2;
    } ret{};

    if constexpr(stochastic_rounding)
    {
        fp8x2_storage_t f8x2;
        if constexpr(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
        {
            ret.ival = __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(ret.ival, v[0], rng, scale, 0);
            f8x2[0]  = ret.v2f8x2(Number<0>{})[0];
            ret.ival = __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(ret.ival, v[1], rng, scale, 0);
            f8x2[1]  = ret.v2f8x2(Number<0>{})[0];
        }
        else
        {
            ret.ival = __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(ret.ival, v[0], rng, scale, 0);
            f8x2[0]  = ret.v2f8x2(Number<0>{})[0];
            ret.ival = __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(ret.ival, v[1], rng, scale, 0);
            f8x2[1]  = ret.v2f8x2(Number<0>{})[0];
        }
        return f8x2;
    }
    else
    {
        // RNE CVT
        // llvm.amdgcn.cvt.scalef32.pk.fp8.f32
        // v2i16 old_vdst, float srcA, float srcB, float scale, bool dst_lo_hi_sel
        if constexpr(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
        {
            // If fval / scale > max fp8, returns Nan
            ret.v2i16 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(/*old_vdst*/ ret.v2i16,
                                                                 v[0],
                                                                 v[1],
                                                                 scale,
                                                                 /*dst_lo_hi_sel*/ false);
        }
        else
        {
            // If fval / scale > max bf8, returns Inf
            ret.v2i16 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f32(/*old_vdst*/ ret.v2i16,
                                                                 v[0],
                                                                 v[1],
                                                                 scale,
                                                                 /*dst_lo_hi_sel*/ false);
        }

        return ret.v2f8x2(Number<0>{});
    }
}

template <ck_fp8_interpretation_t interpret, bool stochastic_rounding>
static __device__ fp8x8_storage_t cast_to_f8_from_f32_scaled(float8_t v,
                                                             unsigned int rng,
                                                             float scale)
{
    union
    {
        uint32x2_t ival;
        fp8x8_storage_t v8f8x1;
        fp8x2_storage_t v2f8x4[4];
    } ret{};

    union
    {
        float8_t vfloat_8x1;
        float2_t v2floatx4[4];
    } in{v};

    ck::static_for<0, 4, 1>{}([&](auto i) {
        ret.v2f8x4[i] =
            cast_to_f8_from_f32_scaled<interpret, stochastic_rounding>(in.v2floatx4[i], rng, scale);
    });

    return ret.v8f8x1;
}

#elif CK_MX_ARCH_1250

template <ck_fp8_interpretation_t interpret>
static __device__ float cast_to_f32_from_f8_scaled(float scale, fp8_storage_t v)
{
    // gfx1250 only support packed 8 scaled conversion and pk4I8 scale factor
    union
    {
        uint32x2_t i32x2val;
        unsigned char i8val[8];
    } val;
    val.i8val[0]    = v;
    uint32_t scale4 = bit_cast<uint32_t>(utils::get_exponent_value(e8m0_bexp_t(scale)));

    static_assert(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP ||
                      interpret == ck_fp8_interpretation_t::CK_E5M2_OCP,
                  "Only OCP interpretations are supported");

    if constexpr(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
    {
        float8_t out = __builtin_amdgcn_cvt_scale_pk8_f32_fp8(val.i32x2val, scale4, 0); // Opsel = 0
        return out[0];
    }
    else
    {
        float8_t out = __builtin_amdgcn_cvt_scale_pk8_f32_bf8(val.i32x2val, scale4, 0); // Opsel = 0
        return out[0];
    }
}

template <ck_fp8_interpretation_t interpret>
static __device__ float2_t cast_to_f32_from_f8_scaled(float scale, fp8x2_storage_t v)
{
    // gfx1250 only have packed 8 scale conversion and pk4I8 scale factor
    union
    {
        uint32x2_t i32x2val;
        fp8x2_storage_t v2f8x4[4];
    } val{};
    val.v2f8x4[0] = v;
    union
    {
        float2_t arr[4];
        float8_t val;
    } out{};
    uint32_t scale4 = bit_cast<uint32_t>(utils::get_exponent_value(e8m0_bexp_t(scale)));

    static_assert(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP ||
                      interpret == ck_fp8_interpretation_t::CK_E5M2_OCP,
                  "Only OCP interpretations are supported");

    if constexpr(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
    {
        out.val = __builtin_amdgcn_cvt_scale_pk8_f32_fp8(val.i32x2val, scale4, 0); // Opsel = 0
        return out.arr[0];
    }
    else
    {
        out.val = __builtin_amdgcn_cvt_scale_pk8_f32_bf8(val.i32x2val, scale4, 0); // Opsel = 0
        return out.arr[0];
    }
}

template <ck_fp8_interpretation_t interpret, typename Ts, int Opsel>
static __device__ float8_t cast_to_f32_from_f8_scaled(Ts scale, fp8x8_storage_t v)
{
    static_assert(sizeof(Ts) == 4,
                  "Ts must be float or uint32_t");

    uint32_t scale4 = (ck::is_same_v<Ts, float>)
                          ? bit_cast<uint32_t>(utils::get_exponent_value(e8m0_bexp_t(scale)))
                          : bit_cast<uint32_t>(scale);

    const auto v_uint2 = bit_cast<uint32x2_t>(v);

    static_assert(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP ||
                      interpret == ck_fp8_interpretation_t::CK_E5M2_OCP,
                  "Only OCP interpretations are supported");

    if constexpr(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
    {
        return __builtin_amdgcn_cvt_scale_pk8_f32_fp8(v_uint2, scale4, Opsel);
    }
    else
    {
        return __builtin_amdgcn_cvt_scale_pk8_f32_bf8(v_uint2, scale4, Opsel);
    }
}

template <ck_fp8_interpretation_t interpret, bool stochastic_rounding>
static __device__ fp8_storage_t cast_to_f8_from_f32_scaled(float v, unsigned int rng, float scale)
{
    fp8_storage_t i8data;
    // gfx1250 only support packed 8 scaled conversion
    union
    {
        uint32x2_t ival;
        fp8_storage_t vf8x8[8];
    } ret{};

    if constexpr(stochastic_rounding)
    {
        ret.ival = (interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
                       ? __builtin_amdgcn_cvt_scalef32_sr_pk8_fp8_f32(float8_t(v), rng, scale)
                       : __builtin_amdgcn_cvt_scalef32_sr_pk8_bf8_f32(float8_t(v), rng, scale);
        i8data   = ret.vf8x8[0];
    }
    else
    {
        ret.ival = (interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
                       ? __builtin_amdgcn_cvt_scalef32_pk8_fp8_f32(float8_t(v), scale)
                       : __builtin_amdgcn_cvt_scalef32_pk8_bf8_f32(float8_t(v), scale);
        i8data   = ret.vf8x8[0];
    }
    return i8data;
}

template <ck_fp8_interpretation_t interpret, bool stochastic_rounding>
static __device__ fp8x2_storage_t cast_to_f8_from_f32_scaled(float2_t v,
                                                             unsigned int rng,
                                                             float scale)
{
    // gfx1250 only support packed 8 scaled conversion
    union
    {
        float8_t val;
        float2_t v2f32x4[4];
    } v_in{};
    v_in.v2f32x4[0] = v;
    union
    {
        uint32x2_t ival;
        fp8x2_storage_t v2f8x4[4];
    } ret{};

    if constexpr(stochastic_rounding)
    {
        ret.ival = (interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
                       ? __builtin_amdgcn_cvt_scalef32_sr_pk8_fp8_f32(v_in.val, rng, scale)
                       : __builtin_amdgcn_cvt_scalef32_sr_pk8_bf8_f32(v_in.val, rng, scale);
        return ret.v2f8x4[0];
    }
    else
    {
        ret.ival = (interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
                       ? __builtin_amdgcn_cvt_scalef32_pk8_fp8_f32(v_in.val, scale)
                       : __builtin_amdgcn_cvt_scalef32_pk8_bf8_f32(v_in.val, scale);
        return ret.v2f8x4[0];
    }
}

template <ck_fp8_interpretation_t interpret, bool stochastic_rounding>
static __device__ fp8x8_storage_t cast_to_f8_from_f32_scaled(float8_t v,
                                                             unsigned int rng,
                                                             float scale)
{
    union
    {
        uint32x2_t ival;
        fp8x8_storage_t v8f8x1;
        fp8x2_storage_t v2f8x4[4];
    } ret{};

    if constexpr(stochastic_rounding)
    {
        if constexpr(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
        {
            ret.ival = __builtin_amdgcn_cvt_scalef32_sr_pk8_fp8_f32(v, rng, scale);
        }
        else
        {
            ret.ival = __builtin_amdgcn_cvt_scalef32_sr_pk8_bf8_f32(v, rng, scale);
        }
    }
    else
    {
        if constexpr(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
        {
            ret.ival = __builtin_amdgcn_cvt_scalef32_pk8_fp8_f32(v, scale);
        }
        else
        {
            ret.ival = __builtin_amdgcn_cvt_scalef32_pk8_bf8_f32(v, scale);
        }
    }

    return ret.v8f8x1;
}
#endif // CK_MX_ARCH_1250

#endif // CK_MX_FP8_CVT_FAST_PATH

// FUNCTION: cvt_float_to_fp8_scaled
/**
 * \brief convert float to @p fp8_storage_t with scaling
 *
 * \tparam interp interpretation of fp8
 * \param f float number
 * \param scale scaling factor
 * \return fp8_storage_t
 */
template <ck_fp8_interpretation_t interp, bool stochastic_rounding = false>
__host__ __device__ static inline fp8_storage_t cvt_float_to_fp8_scaled(const float f, float scale)
{
    __is_interpret_supported(interp);
    uint32_t rng = 0;
    if constexpr(stochastic_rounding)
    {
#if CK_MX_FP8_CVT_FAST_PATH // GFX950, GFX1250
        // use HW clock for stochastic input multiply by incremented thread id
        rng = __builtin_amdgcn_prng_b32(__builtin_readcyclecounter() *
                                        (get_thread_global_1d_id() + 1));
#else
        constexpr int seed = 1254739;
        rng                = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&f), f);
#endif
    }

#if CK_MX_FP8_CVT_FAST_PATH
    return cast_to_f8_from_f32_scaled<interp, stochastic_rounding>(f, rng, scale);
#else
    if constexpr(interp == ck_fp8_interpretation_t::CK_E4M3_OCP)
    {
        return cast_to_f8<float, 3, 4, false, true, stochastic_rounding>(f / scale, rng);
    }
    else if constexpr(interp == ck_fp8_interpretation_t::CK_E5M2_OCP)
    {
        return cast_to_f8<float, 2, 5, false, true, stochastic_rounding>(f / scale, rng);
    }
    else
    {
        __hip_assert(false && "FP8 type is not supported by current target device");
        return 0;
    }
#endif
}

/**
 * \brief convert 2xfloat to @p 2xfp8_storage_t with scaling
 *
 * \tparam interp interpretation of fp8
 * \param f 2xfloat
 * \param scale scaling factor
 * \return 2xfp8_storage_t
 */
template <ck_fp8_interpretation_t interp, bool stochastic_rounding = false>
__host__ __device__ static inline fp8x2_storage_t cvt_float_to_fp8_scaled(const float2_t f,
                                                                          float scale)
{
    __is_interpret_supported(interp);
    uint32_t rng = 0;
    if constexpr(stochastic_rounding)
    {
#if CK_MX_FP8_CVT_FAST_PATH // GFX950, GFX1250
        // use HW clock for stochastic input multiply by incremented thread id
        rng = __builtin_amdgcn_prng_b32(__builtin_readcyclecounter() *
                                        (get_thread_global_1d_id() + 1));
#else
        constexpr int seed = 1254739;
        rng                = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&f), f[0]);
#endif
    }

#if CK_MX_FP8_CVT_FAST_PATH
    return cast_to_f8_from_f32_scaled<interp, stochastic_rounding>(f, rng, scale);
#else
    if constexpr(interp == ck_fp8_interpretation_t::CK_E4M3_OCP)
    {
        return {cast_to_f8<float, 3, 4, false, true, stochastic_rounding>(f[0] / scale, rng),
                cast_to_f8<float, 3, 4, false, true, stochastic_rounding>(f[1] / scale, rng)};
    }
    else if constexpr(interp == ck_fp8_interpretation_t::CK_E5M2_OCP)
    {
        return {cast_to_f8<float, 2, 5, false, true, stochastic_rounding>(f[0] / scale, rng),
                cast_to_f8<float, 2, 5, false, true, stochastic_rounding>(f[1] / scale, rng)};
    }
    else
    {
        __hip_assert(false && "FP8 type is not supported by current target device");
        return 0;
    }
#endif
}

/**
 * \brief convert 8xfloat to @p 8xfp8_storage_t with scaling
 *
 * \tparam interp interpretation of fp8
 * \param f 8xfloat
 * \param scale scaling factor
 * \return 8xfp8_storage_t
 */
template <ck_fp8_interpretation_t interp, bool stochastic_rounding = false>
__host__ __device__ static inline fp8x8_storage_t cvt_float_to_fp8_scaled(const float8_t f,
                                                                          float scale)
{
    __is_interpret_supported(interp);

    uint32_t rng = 0;
    if constexpr(stochastic_rounding)
    {
#if CK_MX_FP8_CVT_FAST_PATH
        // use HW clock for stochastic input multiply by incremented thread id
        rng = __builtin_amdgcn_prng_b32(__builtin_readcyclecounter() *
                                        (get_thread_global_1d_id() + 1));
#else
        constexpr int seed = 1254739;
        rng                = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&f), f[0]);
#endif
    }

    union
    {
        float8_t vfloat_8x1;
        float2_t vfloat_2x4[4];
        float_t vfloat_1x8[8];
    } in{f};

    union
    {
        fp8x8_storage_t vfp8_8x1;
        fp8x2_storage_t vfp8_2x4[4];
        fp8_storage_t vfp8_1x8[8];
    } out{};

#if CK_MX_FP8_CVT_FAST_PATH
    out.vfp8_8x1 =
        cast_to_f8_from_f32_scaled<interp, stochastic_rounding>(in.vfloat_8x1, rng, scale);
#else
    if constexpr(interp == ck_fp8_interpretation_t::CK_E4M3_OCP)
    {
        ck::static_for<0, 8, 1>{}([&](auto i) {
            out.vfp8_1x8[i] = cast_to_f8<float, 3, 4, false, true, stochastic_rounding>(
                in.vfloat_1x8[i] / scale, rng);
        });
    }
    else if constexpr(interp == ck_fp8_interpretation_t::CK_E5M2_OCP)
    {
        ck::static_for<0, 8, 1>{}([&](auto i) {
            out.vfp8_1x8[i] = cast_to_f8<float, 2, 5, false, true, stochastic_rounding>(
                in.vfloat_1x8[i] / scale, rng);
        });
    }
#endif // different arch support
    return out.vfp8_8x1;
}

} // namespace fp8_impl

// Declare a template function for fp8 conversion using SR
template <typename Y, typename X>
__host__ __device__ constexpr Y mxf8_convert_sr(X x, float scale);

// Declare a template function for fp8 conversion using RNE
template <typename Y, typename X>
__host__ __device__ constexpr Y mxf8_convert_rne(X x, float scale);

// convert fp32 to fp8 with rounding to nearest even
template <>
inline __host__ __device__ f8_ocp_t mxf8_convert_rne<f8_ocp_t, float>(float x, float scale)
{
    return f8_ocp_t{fp8_impl::cvt_float_to_fp8_scaled<f8_ocp_t::default_interpret>(x, scale)};
}

// convert fp32 to bf8 with rounding to nearest even
template <>
inline __host__ __device__ bf8_ocp_t mxf8_convert_rne<bf8_ocp_t, float>(float x, float scale)
{
    return bf8_ocp_t{fp8_impl::cvt_float_to_fp8_scaled<bf8_ocp_t::default_interpret>(x, scale)};
}

// convert fp32x2 to fp8x2 with rounding to nearest even
template <>
inline __host__ __device__ f8x2_ocp_t mxf8_convert_rne<f8x2_ocp_t, float2_t>(float2_t x,
                                                                             float scale)
{
    return f8x2_ocp_t{fp8_impl::cvt_float_to_fp8_scaled<f8_ocp_t::default_interpret>(x, scale)};
}

// convert fp32x2 to bf8x2 with rounding to nearest even
template <>
inline __host__ __device__ bf8x2_ocp_t mxf8_convert_rne<bf8x2_ocp_t, float2_t>(float2_t x,
                                                                               float scale)
{
    return bf8x2_ocp_t{fp8_impl::cvt_float_to_fp8_scaled<bf8_ocp_t::default_interpret>(x, scale)};
}

// convert fp32x8 to fp8x8 with rounding to nearest even
template <>
inline __host__ __device__ f8x8_ocp_t mxf8_convert_rne<f8x8_ocp_t, float8_t>(float8_t x,
                                                                             float scale)
{
    return f8x8_ocp_t{fp8_impl::cvt_float_to_fp8_scaled<f8_ocp_t::default_interpret>(x, scale)};
}

// convert fp32x8 to bf8x8 with rounding to nearest even
template <>
inline __host__ __device__ bf8x8_ocp_t mxf8_convert_rne<bf8x8_ocp_t, float8_t>(float8_t x,
                                                                               float scale)
{
    return bf8x8_ocp_t{fp8_impl::cvt_float_to_fp8_scaled<bf8_ocp_t::default_interpret>(x, scale)};
}

// convert fp32x16 to fp8x16 with rounding to nearest even
template <>
inline __host__ __device__ f8x16_ocp_t mxf8_convert_rne<f8x16_ocp_t, float16_t>(float16_t x,
                                                                                float scale)
{
    union
    {
        float16_t float_1x16;
        float8_t float_8x2[2];
    } in{x};

    union
    {
        f8x16_ocp_t fp8_1x16;
        f8x8_ocp_t fp8_8x2[2];
    } out{};

    ck::static_for<0, 2, 1>{}(
        [&](auto i) { out.fp8_8x2[i] = mxf8_convert_rne<f8x8_ocp_t>(in.float_8x2[i], scale); });

    return out.fp8_1x16;
}

// convert fp32x16 to bf8x16 with rounding to nearest even
template <>
inline __host__ __device__ bf8x16_ocp_t mxf8_convert_rne<bf8x16_ocp_t, float16_t>(float16_t x,
                                                                                  float scale)
{
    union
    {
        float16_t float_1x16;
        float8_t float_8x2[2];
    } in{x};

    union
    {
        bf8x16_ocp_t bf8_1x16;
        bf8x8_ocp_t bf8_8x2[2];
    } out{};

    ck::static_for<0, 2, 1>{}(
        [&](auto i) { out.bf8_8x2[i] = mxf8_convert_rne<bf8x8_ocp_t>(in.float_8x2[i], scale); });

    return out.bf8_1x16;
}

// convert fp32x32 to fp8x32 with rounding to nearest even
template <>
inline __host__ __device__ f8x32_ocp_t mxf8_convert_rne<f8x32_ocp_t, float32_t>(float32_t x,
                                                                                float scale)
{
    union
    {
        float32_t float_1x32;
        float16_t float_16x2[2];
    } in{x};

    union
    {
        f8x32_ocp_t fp8_1x32;
        f8x16_ocp_t fp8_16x2[2];
    } out{};

    ck::static_for<0, 2, 1>{}(
        [&](auto i) { out.fp8_16x2[i] = mxf8_convert_rne<f8x16_ocp_t>(in.float_16x2[i], scale); });

    return out.fp8_1x32;
}

// convert fp32x32 to bf8x32 with rounding to nearest even
template <>
inline __host__ __device__ bf8x32_ocp_t mxf8_convert_rne<bf8x32_ocp_t, float32_t>(float32_t x,
                                                                                  float scale)
{
    union
    {
        float32_t float_1x32;
        float16_t float_16x2[2];
    } in{x};

    union
    {
        bf8x32_ocp_t bf8_1x32;
        bf8x16_ocp_t bf8_16x2[2];
    } out{};

    ck::static_for<0, 2, 1>{}(
        [&](auto i) { out.bf8_16x2[i] = mxf8_convert_rne<bf8x16_ocp_t>(in.float_16x2[i], scale); });

    return out.bf8_1x32;
}

// convert fp32 to fp8 with stochastic rounding
template <>
inline __host__ __device__ f8_ocp_t mxf8_convert_sr<f8_ocp_t, float>(float x, float scale)
{
    return f8_ocp_t{fp8_impl::cvt_float_to_fp8_scaled<f8_ocp_t::default_interpret, true>(x, scale)};
}

// convert fp32 to bf8 with stochastic rounding
template <>
inline __host__ __device__ bf8_ocp_t mxf8_convert_sr<bf8_ocp_t, float>(float x, float scale)
{
    return bf8_ocp_t{
        fp8_impl::cvt_float_to_fp8_scaled<bf8_ocp_t::default_interpret, true>(x, scale)};
}

// convert fp32x2 to fp8x2 with stochastic rounding
template <>
inline __host__ __device__ f8x2_ocp_t mxf8_convert_sr<f8x2_ocp_t, float2_t>(float2_t x, float scale)
{
    return f8x2_ocp_t{
        fp8_impl::cvt_float_to_fp8_scaled<f8_ocp_t::default_interpret, true>(x, scale)};
}

// convert fp32x2 to bf8x2 with stochastic rounding
template <>
inline __host__ __device__ bf8x2_ocp_t mxf8_convert_sr<bf8x2_ocp_t, float2_t>(float2_t x,
                                                                              float scale)
{
    return bf8x2_ocp_t{
        fp8_impl::cvt_float_to_fp8_scaled<bf8_ocp_t::default_interpret, true>(x, scale)};
}

// convert fp32x8 to fp8x8 with rounding to nearest even
template <>
inline __host__ __device__ f8x8_ocp_t mxf8_convert_sr<f8x8_ocp_t, float8_t>(float8_t x, float scale)
{
    return f8x8_ocp_t{
        fp8_impl::cvt_float_to_fp8_scaled<f8_ocp_t::default_interpret, true>(x, scale)};
}

// convert fp32x8 to bf8x8 with rounding to nearest even
template <>
inline __host__ __device__ bf8x8_ocp_t mxf8_convert_sr<bf8x8_ocp_t, float8_t>(float8_t x,
                                                                              float scale)
{
    return bf8x8_ocp_t{
        fp8_impl::cvt_float_to_fp8_scaled<bf8_ocp_t::default_interpret, true>(x, scale)};
}

// convert fp32x16 to fp8x16 with stochastic rounding
template <>
inline __host__ __device__ f8x16_ocp_t mxf8_convert_sr<f8x16_ocp_t, float16_t>(float16_t x,
                                                                               float scale)
{
    union
    {
        float16_t float_1x16;
        float8_t float_8x2[2];
    } in{x};

    union
    {
        f8x16_ocp_t fp8_1x16;
        f8x8_ocp_t fp8_8x2[2];
    } out{};

    ck::static_for<0, 2, 1>{}(
        [&](auto i) { out.fp8_8x2[i] = mxf8_convert_sr<f8x8_ocp_t>(in.float_8x2[i], scale); });

    return out.fp8_1x16;
}

// convert fp32x16 to bf8x16 with stochastic rounding
template <>
inline __host__ __device__ bf8x16_ocp_t mxf8_convert_sr<bf8x16_ocp_t, float16_t>(float16_t x,
                                                                                 float scale)
{
    union
    {
        float16_t float_1x16;
        float8_t float_8x2[2];
    } in{x};

    union
    {
        bf8x16_ocp_t bf8_1x16;
        bf8x8_ocp_t bf8_8x2[2];
    } out{};

    ck::static_for<0, 2, 1>{}(
        [&](auto i) { out.bf8_8x2[i] = mxf8_convert_sr<bf8x8_ocp_t>(in.float_8x2[i], scale); });

    return out.bf8_1x16;
}

// convert fp32x32 to fp8x32 with stochastic rounding
template <>
inline __host__ __device__ f8x32_ocp_t mxf8_convert_sr<f8x32_ocp_t, float32_t>(float32_t x,
                                                                               float scale)
{
    union
    {
        float32_t float_1x32;
        float16_t float_16x2[2];
    } in{x};

    union
    {
        f8x32_ocp_t fp8_1x32;
        f8x16_ocp_t fp8_16x2[2];
    } out{};

    ck::static_for<0, 2, 1>{}(
        [&](auto i) { out.fp8_16x2[i] = mxf8_convert_sr<f8x16_ocp_t>(in.float_16x2[i], scale); });

    return out.fp8_1x32;
}

// convert fp32x32 to bf8x32 with stochastic rounding
template <>
inline __host__ __device__ bf8x32_ocp_t mxf8_convert_sr<bf8x32_ocp_t, float32_t>(float32_t x,
                                                                                 float scale)
{
    union
    {
        float32_t float_1x32;
        float16_t float_16x2[2];
    } in{x};

    union
    {
        bf8x32_ocp_t bf8_1x32;
        bf8x16_ocp_t bf8_16x2[2];
    } out{};

    ck::static_for<0, 2, 1>{}(
        [&](auto i) { out.bf8_16x2[i] = mxf8_convert_sr<bf8x16_ocp_t>(in.float_16x2[i], scale); });

    return out.bf8_1x32;
}

} // namespace ck
