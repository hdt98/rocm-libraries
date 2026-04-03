// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: host — float/typed conversions, verification tolerances.
//
// Host-side type conversion utilities for rocm_ck examples.
// Compiled with GCC (not hipcc), so hip's half/bfloat16 operator
// overloads are not available. Uses _Float16 (GCC built-in) for fp16
// and raw bit manipulation for bf16/fp8/bf8.

#pragma once

#include <rocm_ck/datatype_utils.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rocm_ck {

// ============================================================================
// BF16 conversions
// ============================================================================

/// Convert float to bf16 using round-to-nearest-even.
inline std::uint16_t floatToBf16Bits(float f)
{
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    // Round to nearest even: add rounding bias based on LSB + trailing bits
    u += 0x7FFF + ((u >> 16) & 1);
    return static_cast<std::uint16_t>(u >> 16);
}

/// Convert bf16 bits back to float.
inline float bf16BitsToFloat(std::uint16_t bits)
{
    std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// ============================================================================
// FP8/BF8 conversions
// ============================================================================
//
// Four formats, two encoding families:
//
//   FNUZ (gfx942 native):  bias = 2^(k-1), no negative zero, NaN = 0x80
//   OCP  (gfx950 native):  bias = 2^(k-1)-1, has negative zero, IEEE-like NaN
//
//   E4M3: 4 exponent bits, 3 mantissa bits (FP8)
//   E5M2: 5 exponent bits, 2 mantissa bits (BF8)
//
// Conversion logic adapted from CK Tile's float8.hpp. Simplified for
// host-only float↔uint8_t with round-to-nearest-even and saturation.

namespace detail {

/// Count leading zeros in a 32-bit unsigned integer.
inline int clz(std::uint32_t x)
{
    if(x == 0)
        return 32;
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_clz(x);
#else
    int n = 0;
    if((x & 0xFFFF0000) == 0)
    {
        n += 16;
        x <<= 16;
    }
    if((x & 0xFF000000) == 0)
    {
        n += 8;
        x <<= 8;
    }
    if((x & 0xF0000000) == 0)
    {
        n += 4;
        x <<= 4;
    }
    if((x & 0xC0000000) == 0)
    {
        n += 2;
        x <<= 2;
    }
    if((x & 0x80000000) == 0)
    {
        n += 1;
    }
    return n;
#endif
}

/// Generic float → fp8 conversion with round-to-nearest-even and saturation.
/// Template parameters: exp/mant widths of the destination fp8 format.
/// IsFnuz selects FNUZ vs OCP encoding for NaN, zero, and bias.
template <int DstExp, int DstMant, int DstBias, bool IsFnuz>
inline std::uint8_t floatToFp8Generic(float src)
{
    // Float format: sign(1) + exp(8) + mant(23), bias=127
    constexpr int SrcExp  = 8;
    constexpr int SrcMant = 23;
    constexpr int SrcBias = 127;

    std::uint32_t src_bits;
    std::memcpy(&src_bits, &src, sizeof(src_bits));

    std::uint32_t sign     = src_bits >> 31;
    int src_exponent       = static_cast<int>((src_bits >> SrcMant) & 0xFF);
    std::uint32_t mantissa = src_bits & 0x7FFFFF;

    // --- Special values for fp8 output ---
    std::uint8_t signed_max; // saturated max (with sign)
    std::uint8_t nan_out;

    if constexpr(IsFnuz)
    {
        signed_max = static_cast<std::uint8_t>((sign << (DstExp + DstMant)) | 0x7F);
        nan_out    = 0x80;
    }
    else
    {
        if constexpr(DstExp == 4) // E4M3 OCP: max = exp=15, mant=6 (0x7E), NaN = 0x7F
        {
            signed_max = static_cast<std::uint8_t>((sign << (DstExp + DstMant)) | 0x7E);
            nan_out    = static_cast<std::uint8_t>((sign << (DstExp + DstMant)) | 0x7F);
        }
        else // E5M2 OCP: max = exp=30, mant=3 (0x7B), Inf = 0x7C, NaN = 0x7F
        {
            signed_max = static_cast<std::uint8_t>((sign << (DstExp + DstMant)) | 0x7B);
            nan_out    = static_cast<std::uint8_t>((sign << (DstExp + DstMant)) | 0x7F);
        }
    }

    // Max representable |float| for this fp8 format
    constexpr std::uint32_t ifmax = []() -> std::uint32_t {
        if constexpr(DstExp == 5)
            return 0x47600000u; // 57344.0f
        else if constexpr(IsFnuz)
            return 0x43700000u; // 240.0f
        else
            return 0x43E00000u; // 448.0f
    }();

    // Handle float NaN and Inf
    if(src_exponent == 0xFF)
        return (mantissa != 0) ? nan_out : signed_max; // NaN→NaN, Inf→saturate

    // Handle overflow
    if((src_bits & 0x7FFFFFFF) > ifmax)
        return signed_max;

    // Handle zero
    if(src_exponent == 0 && mantissa == 0)
    {
        if constexpr(IsFnuz)
            return 0; // FNUZ: no negative zero
        else
            return static_cast<std::uint8_t>(sign << (DstExp + DstMant));
    }

    // --- Normal conversion with round-to-nearest-even ---
    constexpr int f8_denorm_exp = 1 - DstBias; // actual exponent of smallest fp8 denormal

    int act_exponent;
    int exponent_diff;

    if(src_exponent == 0)
    {
        // Float denormal → extremely small, will be fp8 zero
        act_exponent  = 1 - SrcBias;
        exponent_diff = f8_denorm_exp - act_exponent;
    }
    else
    {
        act_exponent = src_exponent - SrcBias;
        if(act_exponent <= f8_denorm_exp)
            exponent_diff = f8_denorm_exp - act_exponent;
        else
            exponent_diff = 0;
        mantissa |= (1u << SrcMant); // add implicit 1
    }

    // Too small → zero
    if(exponent_diff > DstMant + 1)
    {
        if constexpr(IsFnuz)
            return 0;
        else
            return static_cast<std::uint8_t>(sign << (DstExp + DstMant));
    }

    // Detect midpoint (tie) before shifting — shift may destroy residual bits
    bool midpoint = (mantissa & ((1u << (SrcMant - DstMant + exponent_diff)) - 1)) ==
                    (1u << (SrcMant - DstMant + exponent_diff - 1));

    if(exponent_diff > 0)
        mantissa >>= exponent_diff;
    else if(exponent_diff == -1)
        mantissa <<= 1;

    bool implicit_one = (mantissa & (1u << SrcMant)) != 0;
    int f8_exponent   = (act_exponent + exponent_diff) + DstBias - (implicit_one ? 0 : 1);

    // Round-to-nearest-even: add rounding bias to truncated bits
    std::uint32_t drop_mask = (1u << (SrcMant - DstMant)) - 1;
    bool odd                = (mantissa & (1u << (SrcMant - DstMant))) != 0;
    // RNE: on tie, round to even (toward the value with LSB=0)
    mantissa += (midpoint ? (odd ? mantissa : mantissa - 1u) : mantissa) & drop_mask;

    // Handle carry from rounding
    if(f8_exponent == 0)
    {
        if(mantissa & (1u << SrcMant))
            f8_exponent = 1; // denormal overflowed to normal
    }
    else
    {
        if(mantissa & (1u << (SrcMant + 1)))
        {
            mantissa >>= 1;
            f8_exponent++;
        }
    }

    mantissa >>= (SrcMant - DstMant);

    // Overflow after rounding → saturate
    constexpr int max_exp = (1 << DstExp) - 1;
    if(f8_exponent > max_exp)
    {
        mantissa    = (1 << DstMant) - 1;
        f8_exponent = max_exp;
    }

    // Underflow to zero
    if(f8_exponent == 0 && mantissa == 0)
    {
        if constexpr(IsFnuz)
            return 0;
        else
            return static_cast<std::uint8_t>(sign << (DstExp + DstMant));
    }

    mantissa &= (1u << DstMant) - 1;
    return static_cast<std::uint8_t>((sign << (DstExp + DstMant)) | (f8_exponent << DstMant) |
                                     mantissa);
}

/// Generic fp8 → float conversion.
template <int SrcExp, int SrcMant, int SrcBias, bool IsFnuz>
inline float fp8ToFloatGeneric(std::uint8_t x)
{
    constexpr int DstExp          = 8;
    constexpr int DstMant         = 23;
    constexpr std::uint8_t absmsk = static_cast<std::uint8_t>((1 << (SrcExp + SrcMant)) - 1);

    if(x == 0)
        return 0.0f;

    std::uint32_t sign     = static_cast<std::uint32_t>(x >> (SrcExp + SrcMant));
    std::uint32_t mantissa = x & ((1 << SrcMant) - 1);
    int exponent           = (x & absmsk) >> SrcMant;

    // Handle special values
    if constexpr(IsFnuz)
    {
        if(x == 0x80) // FNUZ NaN
        {
            std::uint32_t nan_bits = 0x7FC00000u; // quiet NaN
            float result;
            std::memcpy(&result, &nan_bits, sizeof(result));
            return result;
        }
    }
    else
    {
        if(x == 0x80) // OCP negative zero
            return -0.0f;

        if constexpr(SrcExp == 4)
        {
            // E4M3 OCP: NaN when exp=15 AND mant=7
            if((x & 0x7F) == 0x7F)
            {
                std::uint32_t nan_bits = 0x7FC00000u;
                float result;
                std::memcpy(&result, &nan_bits, sizeof(result));
                return result;
            }
        }
        else
        {
            // E5M2 OCP: Inf when exp=31, mant=0; NaN when exp=31, mant≠0
            if((x & 0x7C) == 0x7C)
            {
                if((x & 0x03) == 0)
                {
                    // Inf
                    std::uint32_t inf_bits = sign ? 0xFF800000u : 0x7F800000u;
                    float result;
                    std::memcpy(&result, &inf_bits, sizeof(result));
                    return result;
                }
                std::uint32_t nan_bits = 0x7FC00000u;
                float result;
                std::memcpy(&result, &nan_bits, sizeof(result));
                return result;
            }
        }
    }

    // Exponent adjustment: map fp8 bias to float bias (127)
    // exp_low_cutoff accounts for the bias difference
    constexpr int exp_low_cutoff = (1 << (DstExp - 1)) - (1 << (SrcExp - 1)) + 1 - (IsFnuz ? 1 : 0);

    // Handle fp8 denormals: normalize by shifting mantissa left
    if(exponent == 0)
    {
        int sh = 1 + clz(mantissa) - (32 - SrcMant);
        mantissa <<= sh;
        exponent += 1 - sh;
        mantissa &= (1u << SrcMant) - 1;
    }

    exponent += exp_low_cutoff - 1;
    mantissa <<= (DstMant - SrcMant);

    // Handle potential float denormal output (rare, only for fp8 denormals)
    if(exponent <= 0)
    {
        mantissa |= (1u << DstMant);
        mantissa >>= (1 - exponent);
        exponent = 0;
    }

    std::uint32_t result_bits =
        (sign << 31) | (static_cast<std::uint32_t>(exponent) << DstMant) | mantissa;
    float result;
    std::memcpy(&result, &result_bits, sizeof(result));
    return result;
}

} // namespace detail

// --- Public FP8/BF8 conversion functions ---

// E4M3 FNUZ (FP8, gfx942 native): bias=8, max=240, NaN=0x80, no -0
inline std::uint8_t floatToFp8e4m3Fnuz(float f)
{
    return detail::floatToFp8Generic<4, 3, 8, true>(f);
}
inline float fp8e4m3FnuzToFloat(std::uint8_t bits)
{
    return detail::fp8ToFloatGeneric<4, 3, 8, true>(bits);
}

// E5M2 FNUZ (BF8, gfx942 native): bias=16, max=57344, NaN=0x80, no -0
inline std::uint8_t floatToFp8e5m2Fnuz(float f)
{
    return detail::floatToFp8Generic<5, 2, 16, true>(f);
}
inline float fp8e5m2FnuzToFloat(std::uint8_t bits)
{
    return detail::fp8ToFloatGeneric<5, 2, 16, true>(bits);
}

// E4M3 OCP (FP8, gfx950 native): bias=7, max=448, NaN=0x7F, has -0
inline std::uint8_t floatToFp8e4m3Ocp(float f)
{
    return detail::floatToFp8Generic<4, 3, 7, false>(f);
}
inline float fp8e4m3OcpToFloat(std::uint8_t bits)
{
    return detail::fp8ToFloatGeneric<4, 3, 7, false>(bits);
}

// E5M2 OCP (BF8, gfx950 native): bias=15, max=57344, NaN=0x7F, has -0, has Inf
inline std::uint8_t floatToFp8e5m2Ocp(float f)
{
    return detail::floatToFp8Generic<5, 2, 15, false>(f);
}
inline float fp8e5m2OcpToFloat(std::uint8_t bits)
{
    return detail::fp8ToFloatGeneric<5, 2, 15, false>(bits);
}

// ============================================================================
// Dispatch: float ↔ typed buffer
// ============================================================================

/// Convert a float to the device type and store into a byte buffer.
inline void floatToTyped(DataType dt, float value, void* dst)
{
    switch(dt)
    {
    case DataType::FP64: *static_cast<double*>(dst) = static_cast<double>(value); break;
    case DataType::FP32: *static_cast<float*>(dst) = value; break;
    case DataType::FP16: *static_cast<_Float16*>(dst) = static_cast<_Float16>(value); break;
    case DataType::BF16: *static_cast<std::uint16_t*>(dst) = floatToBf16Bits(value); break;
    case DataType::FP8_FNUZ: *static_cast<std::uint8_t*>(dst) = floatToFp8e4m3Fnuz(value); break;
    case DataType::BF8_FNUZ: *static_cast<std::uint8_t*>(dst) = floatToFp8e5m2Fnuz(value); break;
    case DataType::FP8_OCP: *static_cast<std::uint8_t*>(dst) = floatToFp8e4m3Ocp(value); break;
    case DataType::BF8_OCP: *static_cast<std::uint8_t*>(dst) = floatToFp8e5m2Ocp(value); break;
    case DataType::I8: {
        int clamped = std::clamp(static_cast<int>(std::round(value)), -128, 127);
        *static_cast<std::int8_t*>(dst) = static_cast<std::int8_t>(clamped);
        break;
    }
    case DataType::I32: {
        auto clamped                     = std::clamp(static_cast<std::int64_t>(std::round(value)),
                                  INT64_C(-2147483648),
                                  INT64_C(2147483647));
        *static_cast<std::int32_t*>(dst) = static_cast<std::int32_t>(clamped);
        break;
    }
    case DataType::I4:
    case DataType::I16:
    case DataType::I64:
    case DataType::U8:
    case DataType::U16:
    case DataType::U32:
    case DataType::U64:
        std::fprintf(stderr, "%s: host conversion not implemented\n", dataTypeName(dt));
        std::abort();
    }
}

/// Read a typed value from a byte buffer and convert to float.
inline float typedToFloat(DataType dt, const void* src)
{
    switch(dt)
    {
    case DataType::FP64: return static_cast<float>(*static_cast<const double*>(src));
    case DataType::FP32: return *static_cast<const float*>(src);
    case DataType::FP16: return static_cast<float>(*static_cast<const _Float16*>(src));
    case DataType::BF16: return bf16BitsToFloat(*static_cast<const std::uint16_t*>(src));
    case DataType::FP8_FNUZ: return fp8e4m3FnuzToFloat(*static_cast<const std::uint8_t*>(src));
    case DataType::BF8_FNUZ: return fp8e5m2FnuzToFloat(*static_cast<const std::uint8_t*>(src));
    case DataType::FP8_OCP: return fp8e4m3OcpToFloat(*static_cast<const std::uint8_t*>(src));
    case DataType::BF8_OCP: return fp8e5m2OcpToFloat(*static_cast<const std::uint8_t*>(src));
    case DataType::I8: return static_cast<float>(*static_cast<const std::int8_t*>(src));
    case DataType::I32: return static_cast<float>(*static_cast<const std::int32_t*>(src));
    case DataType::I4:
    case DataType::I16:
    case DataType::I64:
    case DataType::U8:
    case DataType::U16:
    case DataType::U32:
    case DataType::U64:
        std::fprintf(stderr, "%s: host conversion not implemented\n", dataTypeName(dt));
        std::abort();
    }
    return 0.0f; // unreachable — silences -Wreturn-type
}

/// Tolerance for verification based on output data type.
///
/// Used in the formula: |result - ref| <= tol * |ref| + tol
/// This combines relative error (tol * |ref|) and absolute error (tol).
///
/// Values are calibrated to ~1 ULP at magnitude 1.0 for each format:
///   FP8 E4M3 (3 mant bits): ULP at 1.0 = 2^-3 = 0.125
///   BF8 E5M2 (2 mant bits): ULP at 1.0 = 2^-2 = 0.25
inline float toleranceFor(DataType dt)
{
    switch(dt)
    {
    case DataType::FP64: return 1e-12f;
    case DataType::FP32: return 1e-5f;
    case DataType::FP16: return 1e-2f;
    case DataType::BF16: return 1e-1f;
    case DataType::FP8_FNUZ:
    case DataType::FP8_OCP: return 0.125f; // E4M3: 3 mantissa bits
    case DataType::BF8_FNUZ:
    case DataType::BF8_OCP: return 0.25f; // E5M2: 2 mantissa bits
    case DataType::I4:
    case DataType::I8:
    case DataType::I16:
    case DataType::I32:
    case DataType::I64:
    case DataType::U8:
    case DataType::U16:
    case DataType::U32:
    case DataType::U64: return 0.0f; // exact for integer types
    }
    return 1e-5f;
}

} // namespace rocm_ck
