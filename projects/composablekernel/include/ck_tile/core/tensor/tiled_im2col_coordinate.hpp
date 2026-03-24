// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"

namespace ck_tile {

// ============================================================================
// TiledIm2ColMetadata
// ============================================================================
// Precomputed constants for the tile-aware im2col index calculation for a 2D
// NHWGC grouped convolution. Populated at host time in MakeATileMetadata() on
// TransformConvFwdToGemm_V2, then passed to device kernels.
//
// The A-matrix offset decomposes as (see im2col_tile_analysis_general.md):
//
//   offset(m, k) = M_base(m) + K_offset(k)
//
//   M_base(m)   = n_conv * NStride
//               + ho     * SH_HiStride
//               + wo     * step_w
//               - pad_offset
//
//   K_offset(k) = y * DH_HiStride + x * DW_WiStride + c_conv
//
// Within a tile, M_base is incrementally updated:
//   M_base[i] = M_base[0] + i * step_w + floor((wo_0 + i) / Wo) * wrap_delta

struct TiledIm2ColMetadata
{
    // ---- M-dimension constants ----
    index_t NStride;      // Hi * Wi * G * C
    index_t HiStride;     // Wi * G * C
    index_t WiStride;     // G * C
    index_t SH_HiStride;  // SH * HiStride  (ho contribution to M_base)
    index_t step_w;       // SW * WiStride  (wo contribution to M_base / step per wo)
    index_t wrap_delta;   // SH_HiStride - Wo * step_w  (M_base correction at ho-boundary)
    index_t pad_offset;   // PH * HiStride + PW * WiStride  (subtracted from M_base)

    // ---- K-dimension constants ----
    index_t DH_HiStride;  // DH * HiStride  (y contribution to K_offset)
    index_t DW_WiStride;  // DW * WiStride  (x contribution to K_offset)

    // ---- Dimension sizes (for decode and validity) ----
    index_t C;            // input channels per group
    index_t XC;           // X * C  (x-period in K dimension)
    index_t Wo;           // output width  (wo-period in M dimension)
    index_t HoWo;         // Ho * Wo  (n_conv-period in M dimension)
    index_t Hi;           // input height  (for validity: 0 <= ih < Hi)
    index_t Wi;           // input width   (for validity: 0 <= iw < Wi)

    // ---- Raw conv params for validity check ----
    index_t SH;           // ConvStrideH
    index_t SW;           // ConvStrideW
    index_t DH;           // ConvDilationH
    index_t DW;           // ConvDilationW
    index_t PH;           // InLeftPadH
    index_t PW;           // InLeftPadW

    // ---- GEMM bounds (replaces pad_tensor_view OOB handling) ----
    index_t M_gemm;       // N * Ho * Wo  — elements beyond this in M are OOB
    index_t K_gemm;       // Y * X * C    — elements beyond this in K are OOB
};

// ============================================================================
// TiledIm2ColCoordinate
// ============================================================================
// Lightweight coordinate for the A (input/im2col) tile window.
// Replaces the generic BottomTensorCoord + transform-chain evaluation.
//
// Usage pattern:
//   init(m_gemm, k_gemm, meta)       — once per thread coordinate (5 divmod)
//   move_step(dm, dk, meta)          — on coordinate move (0–5 divmod depending on dims)
//   get_offset()                     — every load (1 add)
//   is_valid()                       — every load (1 bool read)
//
// Key property: M_base is invariant across K-loop steps (when dm=0).
// K-only moves cost 2 divmod (K decode only, M_base unchanged).

struct TiledIm2ColCoordinate
{
    index_t M_base;    // offset contribution from m_gemm  (constant across K-loop)
    index_t K_offset;  // offset contribution from k_gemm  (updated per K-step)
    index_t m_gemm;    // current absolute M index (for move_step)
    index_t k_gemm;    // current absolute K index (for move_step)
    bool    valid;     // true iff (m_gemm, k_gemm) maps to non-padded input

    // -------------------------------------------------------------------------
    // Full initialization from (m_gemm, k_gemm).
    // Cost: 5 integer divisions.
    // Called once per thread coordinate at block start.
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void init(index_t m,
                                   index_t k,
                                   const TiledIm2ColMetadata& meta)
    {
        m_gemm = m;
        k_gemm = k;

        // Decode m → (n_conv, ho, wo)  [3 divmod]
        const index_t n_conv = m / meta.HoWo;
        const index_t rem_m  = m % meta.HoWo;
        const index_t ho     = rem_m / meta.Wo;
        const index_t wo     = rem_m % meta.Wo;

        // M_base: depends only on m_gemm
        M_base = n_conv * meta.NStride
               + ho     * meta.SH_HiStride
               + wo     * meta.step_w
               - meta.pad_offset;

        // Decode k → (y, x, c_conv)  [2 divmod]
        const index_t y      = k / meta.XC;
        const index_t rem_k  = k % meta.XC;
        const index_t x      = rem_k / meta.C;
        const index_t c_conv = rem_k % meta.C;

        // K_offset: depends only on k_gemm
        K_offset = y * meta.DH_HiStride + x * meta.DW_WiStride + c_conv;

        // Validity: GEMM bounds + spatial padding check
        // ih = ho*SH + y*DH - PH, iw = wo*SW + x*DW - PW
        const index_t ih = ho * meta.SH + y * meta.DH - meta.PH;
        const index_t iw = wo * meta.SW + x * meta.DW - meta.PW;
        valid = (m < meta.M_gemm) && (k < meta.K_gemm) &&
                (ih >= 0 && ih < meta.Hi) && (iw >= 0 && iw < meta.Wi);
    }

    // -------------------------------------------------------------------------
    // Move by (dm, dk) in the global (M, K) space.
    // Called by move_tensor_coordinate for both K-loop steps and within-tile moves.
    //
    // When dm == 0 (pure K step): 2 divmod, M_base unchanged.
    // When dm != 0: full reinit, 5 divmod.
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void move_step(index_t dm,
                                        index_t dk,
                                        const TiledIm2ColMetadata& meta)
    {
        if(dm == 0)
        {
            // K-only step: update K_offset, M_base unchanged  [2 divmod]
            k_gemm += dk;
            const index_t y      = k_gemm / meta.XC;
            const index_t rem_k  = k_gemm % meta.XC;
            const index_t x      = rem_k / meta.C;
            const index_t c_conv = rem_k % meta.C;
            K_offset = y * meta.DH_HiStride + x * meta.DW_WiStride + c_conv;

            // Validity: K bounds + spatial padding (M_gemm bound unchanged from init)
            const index_t rem_m = m_gemm % meta.HoWo;
            const index_t ho    = rem_m / meta.Wo;
            const index_t wo    = rem_m % meta.Wo;
            const index_t ih    = ho * meta.SH + y * meta.DH - meta.PH;
            const index_t iw    = wo * meta.SW + x * meta.DW - meta.PW;
            valid = (k_gemm < meta.K_gemm) &&
                    (ih >= 0 && ih < meta.Hi) && (iw >= 0 && iw < meta.Wi);
        }
        else
        {
            // General step: full reinit  [5 divmod]
            init(m_gemm + dm, k_gemm + dk, meta);
        }
    }

    CK_TILE_HOST_DEVICE index_t get_offset() const { return M_base + K_offset; }
    CK_TILE_HOST_DEVICE bool    is_valid()   const { return valid; }
};

// ============================================================================
// Trait helper: detect if a descriptor has tiled im2col metadata
// ============================================================================

template <typename T, typename = void>
struct has_im2col_meta : std::false_type
{
};

template <typename T>
struct has_im2col_meta<T, std::void_t<decltype(std::declval<T>().get_im2col_meta())>>
    : std::true_type
{
};

template <typename T>
inline constexpr bool has_im2col_meta_v = has_im2col_meta<T>::value;

} // namespace ck_tile
