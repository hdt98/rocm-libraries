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
    index_t X;            // filter width  (number of x steps per y row)
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
//   move_step(dm, dk, meta)          — on coordinate move (0 or 5 divmod)
//   get_offset()                     — every load (1 add)
//   is_valid()                       — every load (1 bool read)
//
// Key property: M_base is invariant across K-loop steps (when dm=0).
//
// Branching note (Issue 2): the `if(dm == 0)` check in move_step is a runtime
// branch, but in the pipeline's K-loop the step is always {0, KPerBlock} where
// 0 is a compile-time constant — so the compiler eliminates the dm!=0 branch
// body entirely from the unrolled K-loop.  No performance impact observed.
//
// Precomputation (Issue 3): c_conv_, x_, y_ are stored in the coordinate and
// updated INCREMENTALLY in move_step(dm=0) via additions and comparisons only
// — 0 divmods for the K-only step (vs 2 before).  Trade-off: +2 VGPRs per
// coordinate vs the gain from eliminated divmods.  For the COMPUTE_V3 pipeline
// with NumCoord=2 the tiled kernel sits exactly at the 64-VGPR occupancy
// boundary (2 concurrent workgroups per SIMD); adding 2 more VGPRs may move
// it to 68 VGPRs (1 workgroup).  Benchmark carefully to confirm net benefit.

struct TiledIm2ColCoordinate
{
    index_t M_base;    // offset contribution from m_gemm  (constant across K-loop)
    index_t K_offset;  // offset contribution from k_gemm  (updated per K-step)
    index_t m_gemm;    // current absolute M index (needed for dm!=0 reinit)
    index_t k_gemm;    // current absolute K index (needed for K_gemm bound check)
    index_t c_conv_;   // current c_conv = k_gemm % C  (incremental K tracking)
    index_t x_;        // current x filter index       (incremental K tracking)
    index_t y_;        // current y filter index       (incremental K tracking)
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
        y_      = k / meta.XC;
        const index_t rem_k = k % meta.XC;
        x_      = rem_k / meta.C;
        c_conv_ = rem_k % meta.C;

        // K_offset: depends only on k_gemm
        K_offset = y_ * meta.DH_HiStride + x_ * meta.DW_WiStride + c_conv_;

        // Validity: GEMM bounds + spatial padding check
        // ih = ho*SH + y*DH - PH, iw = wo*SW + x*DW - PW
        const index_t ih = ho * meta.SH + y_ * meta.DH - meta.PH;
        const index_t iw = wo * meta.SW + x_ * meta.DW - meta.PW;
        valid = (m < meta.M_gemm) && (k < meta.K_gemm) &&
                (ih >= 0 && ih < meta.Hi) && (iw >= 0 && iw < meta.Wi);
    }

    // -------------------------------------------------------------------------
    // Move by (dm, dk) in the global (M, K) space.
    //
    // When dm == 0 (pure K step): 0 divmod.
    //   c_conv_, x_, y_ are updated incrementally with additions and comparisons.
    //   K_offset is recomputed from the updated (y_, x_, c_conv_) in O(1).
    //   Validity is updated at x/y boundaries only.
    //
    // When dm != 0: full reinit, 5 divmod.  Rare — only during tile setup.
    //
    // The branch is well-predicted: in the K-loop dm is always compile-time 0.
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void move_step(index_t dm,
                                        index_t dk,
                                        const TiledIm2ColMetadata& meta)
    {
        if(dm == 0)
        {
            // K-only step: update (c_conv_, x_, y_) incrementally.  [0 divmod]
            k_gemm  += dk;
            c_conv_ += dk;
            // c_conv has stride 1 in K_offset — always apply the base delta first.
            K_offset += dk;

            // Handle c_conv_ overflow → x_ increment
            if(c_conv_ >= meta.C)
            {
                c_conv_ -= meta.C;
                x_++;
                // x moved by +1 (contributes +DW_WiStride) and c_conv wrapped by −C.
                // The base delta (dk) was already applied above; the net correction is:
                //   +DW_WiStride − C    (x step minus the c_conv rollback)
                K_offset += meta.DW_WiStride - meta.C;

                // Handle x_ overflow → y_ increment
                if(x_ >= meta.X)
                {
                    x_ = 0;
                    y_++;
                    // y moved by +1 (contributes +DH_HiStride) and x wrapped (−X*DW_WiStride).
                    K_offset += meta.DH_HiStride - meta.X * meta.DW_WiStride;
                }

                // Spatial validity changes when x_ or y_ changes.
                // Recompute using the compiler-hoisted ho/wo from m_gemm.
                const index_t rem_m = m_gemm % meta.HoWo;
                const index_t ho    = rem_m / meta.Wo;
                const index_t wo    = rem_m % meta.Wo;
                const index_t ih    = ho * meta.SH + y_ * meta.DH - meta.PH;
                const index_t iw    = wo * meta.SW + x_ * meta.DW - meta.PW;
                valid = (k_gemm < meta.K_gemm) &&
                        (ih >= 0 && ih < meta.Hi) && (iw >= 0 && iw < meta.Wi);
            }
            else
            {
                // c_conv moved within same x/y → base delta already applied.
                // Spatial validity unchanged; only K_gemm bound may change.
                valid = (k_gemm < meta.K_gemm) && valid;
            }
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
