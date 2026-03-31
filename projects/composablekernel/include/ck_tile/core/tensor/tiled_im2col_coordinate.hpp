// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/utility/magic_div.hpp"

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
//   offset(m, k) = M_base_(m) + K_offset_(k)
//
//   M_base_(m)   = n_conv * NStride
//               + ho     * SH_HiStride
//               + wo     * step_w
//               - pad_offset
//
//   K_offset_(k) = y * DH_HiStride + x * DW_WiStride + c_conv
//
// Within a tile, M_base_ is incrementally updated:
//   M_base_[i] = M_base_[0] + i * step_w + floor((wo_0 + i) / Wo) * wrap_delta

struct TiledIm2ColMetadata
{
    // ---- M-dimension constants ----
    index_t NStride;      // Hi * Wi * G * C
    index_t HiStride;     // Wi * G * C
    index_t WiStride;     // G * C
    index_t SH_HiStride;  // SH * HiStride  (ho contribution to M_base_)
    index_t step_w;       // SW * WiStride  (wo contribution to M_base_ / step per wo)
    index_t wrap_delta;   // SH_HiStride - Wo * step_w  (M_base_ correction at ho-boundary)
    index_t pad_offset;   // PH * HiStride + PW * WiStride  (subtracted from M_base_)

    // ---- K-dimension constants ----
    index_t DH_HiStride;  // DH * HiStride  (y contribution to K_offset_)
    index_t DW_WiStride;  // DW * WiStride  (x contribution to K_offset_)

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

    // ---- Magic-division precomputed multipliers (host-computed, device-used) ----
    // Replaces runtime integer division in init_m / init_k with multiply+shift.
    // Computed once in MakeATileMetadata(); valid for non-negative dividends < 2^31.
    mdiv2 mdiv_HoWo;      // magic divisor for HoWo  (used in init_m: m / HoWo)
    mdiv2 mdiv_Wo;        // magic divisor for Wo    (used in init_m: rem / Wo)
    mdiv2 mdiv_XC;        // magic divisor for XC    (used in init_k: k / XC)
    mdiv2 mdiv_C;         // magic divisor for C     (used in init_k: rem / C)
};

// ============================================================================
// TiledIm2ColCoordinate
// ============================================================================
// Lightweight coordinate for the A (input/im2col) tile window.
// Replaces the generic BottomTensorCoord + transform-chain evaluation.
//
// Usage pattern:
//   init_m(m_gemm, meta)             — M part: 3 divmods, stores ho_/wo_
//   init_k(k_gemm, meta)             — K part: 2 divmods + validity
//   init(m_gemm, k_gemm, meta)       — convenience wrapper: calls init_m + init_k
//   move_step(dm, dk, meta)          — on coordinate move (0 divmod when dm==0)
//   get_offset()                     — every load (1 add)
//   is_valid()                       — every load (1 bool read)
//
// Key property: M_base_ is invariant across K-loop steps (when dm=0).
//
// init_m / init_k split (wave-uniform M optimisation):
//   When KPerBlock >= warp_size * VecSizeA all 64 lanes in a warp share the
//   same m_gemm.  prepare_coords applies amd_wave_read_first_lane() on m_gemm
//   and calls init_m() with the resulting scalar value — the 3 M divmods then
//   execute on the scalar unit (SALU) rather than the vector unit (VALU).
//   init_k() is called per-lane with the lane-specific k_gemm.
//
// ho_ / wo_ caching:
//   ho_ and wo_ are precomputed in init_m() and stored in the coordinate.
//   move_step(dm=0) reads them directly, eliminating the 2 divmods that were
//   previously needed to recompute ho/wo from m_gemm at every x_/y_ boundary.
//
// Branching note: the `if(dm == 0)` check in move_step is a runtime branch,
// but in the pipeline's K-loop the step is always {0, KPerBlock} where 0 is a
// compile-time constant — so the compiler eliminates the dm!=0 branch body
// entirely from the unrolled K-loop.  No performance impact observed.

struct TiledIm2ColCoordinate
{
    index_t M_base_;    // offset contribution from m_gemm  (constant across K-loop)
    index_t K_offset_;  // offset contribution from k_gemm  (updated per K-step)
    index_t m_gemm;    // current absolute M index (needed for dm!=0 reinit)
    index_t k_gemm;    // current absolute K index (needed for K_gemm bound check)
    index_t c_conv_;   // current c_conv = k_gemm % C  (incremental K tracking)
    index_t ho_;       // output row   from m_gemm (cached to avoid divmod in move_step)
    index_t wo_;       // output col   from m_gemm (cached to avoid divmod in move_step)
    bool    valid;     // true iff (m_gemm, k_gemm) maps to non-padded input

    // TODO: Make a template parameter.
    static constexpr index_t KPerBlock = 32;

    // -------------------------------------------------------------------------
    // M part: decode m_gemm → (n_conv, ho_, wo_) and compute M_base_.
    // Cost: 3 integer divisions.
    //
    // When m_gemm is wave-uniform, the caller should apply
    // amd_wave_read_first_lane() before calling this method so that the 3
    // divmods execute on the scalar unit (SALU) instead of the vector unit.
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void init_m(index_t m, const TiledIm2ColMetadata& meta)
    {
        m_gemm = m;

        // Decode m → (n_conv, ho_, wo_) using magic division  [3 multiply-shift]
        uint32_t n_conv, rem_m, ho, wo;
        meta.mdiv_HoWo.divmod(static_cast<uint32_t>(m),     meta.HoWo, n_conv, rem_m);
        meta.mdiv_Wo  .divmod(static_cast<uint32_t>(rem_m), meta.Wo,   ho,     wo);
        ho_ = static_cast<index_t>(ho);
        wo_ = static_cast<index_t>(wo);

        // M_base_: depends only on m_gemm
        M_base_ = static_cast<index_t>(n_conv) * meta.NStride
               + ho_    * meta.SH_HiStride
               + wo_    * meta.step_w
               - meta.pad_offset;
    }

    // -------------------------------------------------------------------------
    // K part: decode k_gemm → (y_, x_, c_conv_), compute K_offset_ and validity.
    // Cost: 2 integer divisions.
    // Must be called after init_m() (uses ho_, wo_ for the validity check).
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void init_k(index_t k, const TiledIm2ColMetadata& meta)
    {
        k_gemm = k;

        // Decode k → (y_, x_, c_conv_) using magic division  [2 multiply-shift]
        uint32_t y, rem_k, x, c;
        meta.mdiv_XC.divmod(static_cast<uint32_t>(k),     meta.XC, y, rem_k);
        meta.mdiv_C .divmod(static_cast<uint32_t>(rem_k), meta.C,  x, c);
        y_      = static_cast<index_t>(y);
        x_      = static_cast<index_t>(x);
        c_conv_ = static_cast<index_t>(c);

        // K_offset_: depends only on k_gemm
        K_offset_ = y_ * meta.DH_HiStride + x_ * meta.DW_WiStride + c_conv_;
    }

    // Fast path: valid when C % KPerBlock == 0 (no C-boundary crossing within tile).
    // k_start = k_tile_idx * KPerBlock (wave-uniform scalar)
    // k_loc   = per-lane offset within tile ∈ [0, KPerBlock)
    // y, x are wave-uniform: 2 scalar magic-divmods (SALU when caller uses read_first_lane)
    CK_TILE_HOST_DEVICE void init_k_aligned(index_t k_start,
                                            index_t k_loc,
                                            const TiledIm2ColMetadata& meta)
    {
        // Decode k_start → (y_, x_, c_base) using magic division  [2 multiply-shift]
        // k_start is wave-uniform → these execute on SALU when promoted by caller.
        uint32_t y, rem_ks, x, c_base;
        meta.mdiv_XC.divmod(static_cast<uint32_t>(k_start),     meta.XC, y, rem_ks);
        meta.mdiv_C .divmod(static_cast<uint32_t>(rem_ks),      meta.C,  x, c_base);
        y_      = static_cast<index_t>(y);
        x_      = static_cast<index_t>(x);
        c_conv_ = static_cast<index_t>(c_base) + k_loc;  // no modulo — no C-boundary crossing

        k_gemm    = k_start + k_loc;
        K_offset_ = y_ * meta.DH_HiStride + x_ * meta.DW_WiStride + c_conv_;
    }


    CK_TILE_HOST_DEVICE void init_valid(const TiledIm2ColMetadata& meta)
    {
        // Validity: GEMM bounds + spatial padding check (uses cached ho_, wo_)

        /*
        const index_t ih = ho_ * meta.SH + y_ * meta.DH - meta.PH;
        const index_t iw = wo_ * meta.SW + x_ * meta.DW - meta.PW;
        valid = (m_gemm < meta.M_gemm) && (k_gemm < meta.K_gemm) &&
                (ih >= 0 && ih < meta.Hi) && (iw >= 0 && iw < meta.Wi);
                */
        // Speacialized case: unit stride, dilation, and padding — only GEMM bounds check is needed.
        const index_t ih = ho_ + y_ - meta.PH;
        const index_t iw = wo_ + x_ - meta.PW;
        valid = (m_gemm < meta.M_gemm) && (k_gemm < meta.K_gemm) &&
                (ih >= 0 && ih < meta.Hi) && (iw >= 0 && iw < meta.Wi);
    }


    // -------------------------------------------------------------------------
    // Convenience wrapper: full init from (m_gemm, k_gemm).
    // Cost: 5 integer divisions (3 M + 2 K).
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void init(index_t m,
                                   index_t k,
                                   const TiledIm2ColMetadata& meta)
    {
        init_m(m, meta);
        init_k(k, meta);
        init_valid(meta);
    }

    // -------------------------------------------------------------------------
    // Move by (dm, dk) in the global (M, K) space.
    //
    // When dm == 0 (pure K step): 0 divmod.
    //   c_conv_, x_, y_ are updated incrementally with additions and comparisons.
    //   K_offset_ is recomputed from the updated (y_, x_, c_conv_) in O(1).
    //   Validity at x_/y_ boundaries uses cached ho_, wo_ — 0 divmods.
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
            // c_conv has stride 1 in K_offset_ — always apply the base delta first.
            K_offset_ += dk;

            // Handle c_conv_ overflow → x_ increment
            if(c_conv_ >= meta.C)
            {
                c_conv_ -= meta.C;
                x_++;
                // x moved by +1 (contributes +DW_WiStride) and c_conv wrapped by −C.
                // The base delta (dk) was already applied above; the net correction is:
                //   +DW_WiStride − C    (x step minus the c_conv rollback)
                K_offset_ += meta.DW_WiStride - meta.C;

                // Handle x_ overflow → y_ increment
                if(x_ >= meta.X)
                {
                    x_ = 0;
                    y_++;
                    // y moved by +1 (contributes +DH_HiStride) and x wrapped (−X*DW_WiStride).
                    K_offset_ += meta.DH_HiStride - meta.X * meta.DW_WiStride;
                }

                // Spatial validity changes when x_ or y_ changes.
                // Use cached ho_, wo_ — no divmods needed.
                const index_t ih = ho_ * meta.SH + y_ * meta.DH - meta.PH;
                const index_t iw = wo_ * meta.SW + x_ * meta.DW - meta.PW;
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

    CK_TILE_HOST_DEVICE index_t get_offset() const { return M_base_ + K_offset_; }
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
