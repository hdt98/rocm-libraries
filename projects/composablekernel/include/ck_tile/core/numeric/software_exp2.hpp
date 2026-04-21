// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"

namespace ck_tile {

namespace detail {

// Degree-3 minimax polynomial coefficients for 2^f on [0, 1).
// From FlashAttention-4 (Tri Dao, 2025). Max relative error ~0.18%.
inline constexpr float kExp2C0 = 1.0f;
inline constexpr float kExp2C1 = 0.695146143436431884765625f;
inline constexpr float kExp2C2 = 0.227564394474029541015625f;
inline constexpr float kExp2C3 = 0.077119089663028717041015625f;

// Software exp2(x) using FMA polynomial approximation.
// Replaces transcendental v_exp_f32 (1/4 VALU throughput on CDNA3/4)
// with 3x v_fma_f32 (full VALU throughput), freeing the trans unit
// for other work. Useful when MFMA is the bottleneck.
inline __device__ float exp2_fma(float x)
{
    // Clamp to valid IEEE 754 exponent range
    float xc = __builtin_fmaxf(x, -126.0f);
    xc       = __builtin_fminf(xc, 127.0f);

    // Split into integer and fractional parts: x = n + f, f in [0, 1)
    int n    = static_cast<int>(xc);
    float nf = static_cast<float>(n);
    // Correct for truncation toward zero (need floor)
    n -= (xc < nf) ? 1 : 0;
    nf      = static_cast<float>(n);
    float f = xc - nf;

    // Evaluate degree-3 minimax polynomial: p(f) ~= 2^f
    float p = __builtin_fmaf(kExp2C3, f, kExp2C2);
    p       = __builtin_fmaf(p, f, kExp2C1);
    p       = __builtin_fmaf(p, f, kExp2C0);

    // Multiply by 2^n via IEEE 754 exponent addition
    return bit_cast<float>(bit_cast<uint32_t>(p) + static_cast<uint32_t>(n << 23));
}

} // namespace detail

// Inline wrapper for FMHA forward conditional-rescale exp2 calls.
// Dispatches to the FMA polynomial when CK_TILE_FMHA_FWD_FAST_EXP2 is enabled
// (the whole point of the SW exp2 path), or to the hardware v_exp_f32
// otherwise. Replaces the previous CK_TILE_EXP2 preprocessor macro with a
// single call site that the optimizer can still inline.
inline __device__ float fmha_fast_exp2(float x)
{
#if CK_TILE_FMHA_FWD_FAST_EXP2
    return detail::exp2_fma(x);
#else
    return exp2(x);
#endif
}

} // namespace ck_tile
