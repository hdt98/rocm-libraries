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
    index_t step_w;       // SW * WiStride  (wo contribution to M_base / M_base step per row)
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
};

// ============================================================================
// TiledIm2ColCoordinate
// ============================================================================
// Lightweight coordinate for the A (input/im2col) tile window.
// Replaces the generic BottomTensorCoord + transform-chain evaluation.
//
// Usage pattern:
//   init(m_gemm, k_gemm, meta)  — once per thread coordinate (5 divmod)
//   move_k(k_new, meta)         — once per K-step (2 divmod, no M cost)
//   get_offset()                — every load (1 add)
//   is_valid()                  — every load (1 bool read)

struct TiledIm2ColCoordinate
{
    index_t M_base;   // offset contribution from m_gemm
    index_t K_offset; // offset contribution from k_gemm
    bool    valid;    // true iff (m_gemm, k_gemm) maps to non-padded input

    // -------------------------------------------------------------------------
    // Full initialization from (m_gemm, k_gemm).
    // Cost: 5 integer divisions.
    // Called once per thread coordinate at block start.
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void init(index_t m_gemm,
                                   index_t k_gemm,
                                   const TiledIm2ColMetadata& meta)
    {
        // Decode m_gemm → (n_conv, ho, wo)  [3 divmod]
        const index_t n_conv = m_gemm / meta.HoWo;
        const index_t rem_m  = m_gemm % meta.HoWo;
        const index_t ho     = rem_m / meta.Wo;
        const index_t wo     = rem_m % meta.Wo;

        // M_base: depends only on m_gemm
        M_base = n_conv * meta.NStride
               + ho     * meta.SH_HiStride
               + wo     * meta.step_w
               - meta.pad_offset;

        // Decode k_gemm → (y, x, c_conv)  [2 divmod]
        const index_t y      = k_gemm / meta.XC;
        const index_t rem_k  = k_gemm % meta.XC;
        const index_t x      = rem_k / meta.C;
        const index_t c_conv = rem_k % meta.C;

        // K_offset: depends only on k_gemm
        K_offset = y * meta.DH_HiStride + x * meta.DW_WiStride + c_conv;

        // Validity: ih = ho*SH + y*DH - PH, iw = wo*SW + x*DW - PW
        const index_t ih = ho * meta.SH + y * meta.DH - meta.PH;
        const index_t iw = wo * meta.SW + x * meta.DW - meta.PW;
        valid = (ih >= 0 && ih < meta.Hi) && (iw >= 0 && iw < meta.Wi);
    }

    // -------------------------------------------------------------------------
    // Incremental K update. Called once per K-step in the pipeline loop.
    // Cost: 2 integer divisions (covers K_offset recomputation from k_new).
    // M_base is unaffected — no divisions needed for it.
    //
    // Future optimisation: replace with fully-incremental update using deltas
    //   ΔK(x unchanged): +1
    //   ΔK(x wraps):     DW_WiStride - (C-1)
    //   ΔK(y wraps):     DH_HiStride - X*DW_WiStride - (C-1)
    // For now we recompute from k_new to keep the code simple and correct.
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void move_k(index_t k_new,
                                     index_t m_gemm,
                                     const TiledIm2ColMetadata& meta)
    {
        const index_t y      = k_new / meta.XC;
        const index_t rem_k  = k_new % meta.XC;
        const index_t x      = rem_k / meta.C;
        const index_t c_conv = rem_k % meta.C;
        K_offset = y * meta.DH_HiStride + x * meta.DW_WiStride + c_conv;

        // Update validity with new K values
        const index_t rem_m  = m_gemm % meta.HoWo;
        const index_t ho     = rem_m / meta.Wo;
        const index_t wo     = rem_m % meta.Wo;
        const index_t ih     = ho * meta.SH + y * meta.DH - meta.PH;
        const index_t iw     = wo * meta.SW + x * meta.DW - meta.PW;
        valid = (ih >= 0 && ih < meta.Hi) && (iw >= 0 && iw < meta.Wi);
    }

    CK_TILE_HOST_DEVICE index_t get_offset() const { return M_base + K_offset; }
    CK_TILE_HOST_DEVICE bool    is_valid()   const { return valid; }
};

} // namespace ck_tile
