// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"

namespace ck_tile {

// Degree-3 minimax polynomial coefficients for 2^f on [0, 1).
// From FlashAttention-4 (Tri Dao, 2025). Max relative error ~0.18%.
namespace detail {
    CK_TILE_DEVICE static constexpr float kExp2C0 = 1.0f;
    CK_TILE_DEVICE static constexpr float kExp2C1 = 0.695146143436431884765625f;
    CK_TILE_DEVICE static constexpr float kExp2C2 = 0.227564394474029541015625f;
    CK_TILE_DEVICE static constexpr float kExp2C3 = 0.077119089663028717041015625f;
} // namespace detail

// Software exp2(x) using FMA polynomial approximation.
// Replaces Trans-unit v_exp_f32 (1/4 VALU throughput on CDNA3/4)
// with 3x v_fma_f32 (full VALU throughput), freeing the transcendental
// unit for other work. Useful when MFMA is the bottleneck.
CK_TILE_DEVICE float exp2_fma(float x)
{
    // Clamp to valid IEEE 754 exponent range
    float xc = __builtin_fmaxf(x, -126.0f);
    xc       = __builtin_fminf(xc, 127.0f);

    // Split into integer and fractional parts: x = n + f, f in [0, 1)
    int n    = static_cast<int>(xc);
    float nf = static_cast<float>(n);
    // Correct for truncation toward zero (need floor)
    n -= (xc < nf) ? 1 : 0;
    nf = static_cast<float>(n);
    float f = xc - nf;

    // Evaluate degree-3 minimax polynomial: p(f) ≈ 2^f
    float p = __builtin_fmaf(detail::kExp2C3, f, detail::kExp2C2);
    p       = __builtin_fmaf(p, f, detail::kExp2C1);
    p       = __builtin_fmaf(p, f, detail::kExp2C0);

    // Multiply by 2^n via IEEE 754 exponent addition
    return bit_cast<float>(bit_cast<uint32_t>(p) + static_cast<uint32_t>(n << 23));
}

// ILP-friendly pair version: compute exp2(a) and exp2(b) with
// interleaved FMA operations to keep the VALU pipeline full.
CK_TILE_DEVICE void exp2_fma_pair(float a, float b, float& out_a, float& out_b)
{
    float ac = __builtin_fmaxf(a, -126.0f);
    float bc = __builtin_fmaxf(b, -126.0f);
    ac       = __builtin_fminf(ac, 127.0f);
    bc       = __builtin_fminf(bc, 127.0f);

    int na    = static_cast<int>(ac);
    int nb    = static_cast<int>(bc);
    float nfa = static_cast<float>(na);
    float nfb = static_cast<float>(nb);
    na -= (ac < nfa) ? 1 : 0;
    nb -= (bc < nfb) ? 1 : 0;
    nfa    = static_cast<float>(na);
    nfb    = static_cast<float>(nb);
    float fa = ac - nfa;
    float fb = bc - nfb;

    float pa = __builtin_fmaf(detail::kExp2C3, fa, detail::kExp2C2);
    float pb = __builtin_fmaf(detail::kExp2C3, fb, detail::kExp2C2);
    pa       = __builtin_fmaf(pa, fa, detail::kExp2C1);
    pb       = __builtin_fmaf(pb, fb, detail::kExp2C1);
    pa       = __builtin_fmaf(pa, fa, detail::kExp2C0);
    pb       = __builtin_fmaf(pb, fb, detail::kExp2C0);

    out_a = bit_cast<float>(bit_cast<uint32_t>(pa) + static_cast<uint32_t>(na << 23));
    out_b = bit_cast<float>(bit_cast<uint32_t>(pb) + static_cast<uint32_t>(nb << 23));
}

} // namespace ck_tile
