// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/utility/magic_div.hpp"

namespace ck_tile {

// ============================================================================
// Im2ColTensor — distinguishes which tensor an Im2ColCoordinate addresses
// ============================================================================
enum class Im2ColTensor
{
    FwdInput,  // GEMM-A: input tensor (NHWGC), requires full (y,x,c) K-decode + padding check
    FwdOutput, // GEMM-C: output tensor (NHWGK), K-dim is trivially k_conv = n_gemm
};

// ============================================================================
// Im2ColMetadata
// ============================================================================
// Precomputed constants for the tile-aware im2col index calculation for a 2D
// NHWGC/NHWGK grouped convolution. Populated at host time in MakeATileMetadata()
// (FwdInput) or MakeCTileMetadata() (FwdOutput) on TransformConvFwdToGemm_V2,
// then passed to device kernels.
//
// FwdInput:
//   offset(m, k) = M_base(m) + K_offset(k)
//   M_base(m)    = n_conv * NStride + ho * SH_HiStride + wo * step_w - pad_offset
//   K_offset(k)  = y * DH_HiStride + x * DW_WiStride + c_conv
//
// FwdOutput:
//   offset(m, n) = M_base(m) + n * KStride
//   M_base(m)    = n_conv * NStride_out + ho * HoStride + wo * WoStride
//   K_offset(n)  = n  (trivial: KStride == 1 for contiguous K, or scaled by KStride)
//
// Within a tile, M_base is incrementally updated:
//   M_base[i] = M_base[0] + i * step_w + floor((wo_0 + i) / Wo) * wrap_delta

struct Im2ColMetadata
{
    // ---- M-dimension constants (shared by FwdInput and FwdOutput) ----
    index_t NStride;      // outermost stride (NStride for input; NStride_out for output)
    index_t HiStride;     // second spatial stride (HiStride = Wi*G*C for input; HoStride = Wo*G*K for output)
    index_t WiStride;     // innermost spatial stride (WiStride = G*C for input; WoStride = G*K for output)
    index_t SH_HiStride;  // SH * HiStride  (ho contribution to M_base; SH=1 for output)
    index_t step_w;       // SW * WiStride  (wo contribution to M_base / step per wo; SW=1 for output)
    index_t wrap_delta;   // SH_HiStride - Wo * step_w  (M_base correction at ho-boundary)
    index_t pad_offset;   // PH * HiStride + PW * WiStride  (subtracted from M_base; 0 for output)

    // ---- K-dimension constants (FwdInput only; unused for FwdOutput) ----
    index_t DH_HiStride;  // DH * HiStride  (y contribution to K_offset)
    index_t DW_WiStride;  // DW * WiStride  (x contribution to K_offset)

    // ---- N-dimension constants (FwdOutput only; unused for FwdInput) ----
    index_t KStride;      // stride for k_conv in output tensor (= 1 for contiguous K)

    // ---- Dimension sizes (for decode and validity) ----
    index_t C;            // input channels per group   (FwdInput: K-decode period)
    index_t X;            // filter width               (FwdInput: x steps per y row)
    index_t XC;           // X * C                      (FwdInput: x-period in K dimension)
    index_t Wo;           // output width               (shared: wo-period in M dimension)
    index_t HoWo;         // Ho * Wo                    (shared: n_conv-period in M dimension)
    index_t Hi;           // input height               (FwdInput: validity: 0 <= ih < Hi)
    index_t Wi;           // input width                (FwdInput: validity: 0 <= iw < Wi)

    // ---- Raw conv params for validity check (FwdInput only) ----
    index_t SH;           // ConvStrideH
    index_t SW;           // ConvStrideW
    index_t DH;           // ConvDilationH
    index_t DW;           // ConvDilationW
    index_t PH;           // InLeftPadH
    index_t PW;           // InLeftPadW

    // ---- GEMM bounds ----
    index_t M_gemm;       // N * Ho * Wo  — elements beyond this in M are OOB
    index_t K_gemm;       // Y * X * C    (FwdInput) or K (FwdOutput) — OOB bound in K/N dimension

    // ---- KPerBlock (for aligned fast path, FwdInput only) ----
    index_t KPerBlock;

    // ---- Magic-division precomputed multipliers (host-computed, device-used) ----
    // Replaces runtime integer division in init_m / init_k with multiply+shift.
    mdiv2 mdiv_HoWo;      // magic divisor for HoWo  (used in init_m: m / HoWo)
    mdiv2 mdiv_Wo;        // magic divisor for Wo    (used in init_m: rem / Wo)
    mdiv2 mdiv_XC;        // magic divisor for XC    (used in init_k: k / XC, FwdInput only)
    mdiv2 mdiv_C;         // magic divisor for C     (used in init_k: rem / C, FwdInput only)
};

// ============================================================================
// Im2ColCoordinate<Im2ColTensor>
// ============================================================================
// Primary template — instantiated for FwdInput and FwdOutput.
// Each specialization provides:
//   init_m(m, meta)        — decode m_gemm → (n_conv, ho, wo), compute M_base
//   init_k(k, meta)        — decode k/n index, compute K_offset (or N_offset)
//   init(m, k, meta)       — convenience: init_m + init_k + init_valid
//   move_step(dm, dk, meta)— incremental move in (M, K/N) space
//   get_offset()           — M_base + K_offset
//   is_valid()             — validity flag

template <Im2ColTensor Tensor>
struct Im2ColCoordinate;

// ============================================================================
// Im2ColCoordinate<FwdInput>
// ============================================================================
// Coordinate for the A (input/im2col) tile window.
// Replaces the generic BottomTensorCoord + transform-chain evaluation.
//
// Key property: M_base is invariant across K-loop steps (when dm=0).
//
// init_m / init_k split (wave-uniform M optimisation):
//   When KPerBlock >= warp_size * VecSizeA all 64 lanes in a warp share the
//   same m_gemm.  prepare_coords applies amd_wave_read_first_lane() on m_gemm
//   and calls init_m() with the resulting scalar value — the 3 M divmods then
//   execute on the scalar unit (SALU) rather than the vector unit (VALU).
//   init_k() is called per-lane with the lane-specific k_gemm.

template <>
struct Im2ColCoordinate<Im2ColTensor::FwdInput>
{
    index_t M_base;     // offset contribution from m_gemm  (constant across K-loop)
    index_t K_offset_;  // offset contribution from k_gemm  (updated per K-step)
    index_t m_gemm;     // current absolute M index (needed for dm!=0 reinit)
    index_t k_gemm;     // current absolute K index (needed for K_gemm bound check)
    index_t c_conv_;    // current c_conv = k_gemm % C  (incremental K tracking)
    index_t x_;         // current x filter index         (incremental K tracking)
    index_t y_;         // current y filter index         (incremental K tracking)
    index_t ho_;        // output row from m_gemm (cached to avoid divmod in move_step)
    index_t wo_;        // output col from m_gemm (cached to avoid divmod in move_step)
    bool    valid;      // true iff (m_gemm, k_gemm) maps to non-padded input

    // -------------------------------------------------------------------------
    // M part: decode m_gemm → (n_conv, ho_, wo_) and compute M_base.
    // Cost: 3 integer divisions (via magic multiply-shift).
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void init_m(index_t m, const Im2ColMetadata& meta)
    {
        m_gemm = m;

        uint32_t n_conv, rem_m, ho, wo;
        meta.mdiv_HoWo.divmod(static_cast<uint32_t>(m),     meta.HoWo, n_conv, rem_m);
        meta.mdiv_Wo  .divmod(static_cast<uint32_t>(rem_m), meta.Wo,   ho,     wo);
        ho_ = static_cast<index_t>(ho);
        wo_ = static_cast<index_t>(wo);

        M_base = static_cast<index_t>(n_conv) * meta.NStride
               + ho_    * meta.SH_HiStride
               + wo_    * meta.step_w
               - meta.pad_offset;
    }

    // -------------------------------------------------------------------------
    // K part: decode k_gemm → (y_, x_, c_conv_), compute K_offset_ and validity.
    // Cost: 2 integer divisions.
    // Must be called after init_m() (uses ho_, wo_ for the validity check).
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void init_k(index_t k, const Im2ColMetadata& meta)
    {
        k_gemm = k;

        uint32_t y, rem_k, x, c;
        meta.mdiv_XC.divmod(static_cast<uint32_t>(k),     meta.XC, y, rem_k);
        meta.mdiv_C .divmod(static_cast<uint32_t>(rem_k), meta.C,  x, c);
        y_      = static_cast<index_t>(y);
        x_      = static_cast<index_t>(x);
        c_conv_ = static_cast<index_t>(c);

        K_offset_ = y_ * meta.DH_HiStride + x_ * meta.DW_WiStride + c_conv_;
    }

    // Fast path: valid when C % KPerBlock == 0 (no C-boundary crossing within tile).
    // k_start = k_tile_idx * KPerBlock (wave-uniform scalar)
    // k_loc   = per-lane offset within tile ∈ [0, KPerBlock)
    // y, x are wave-uniform: 2 scalar magic-divmods (SALU when caller uses read_first_lane)
    CK_TILE_HOST_DEVICE void init_k_aligned(index_t k_start,
                                            index_t k_loc,
                                            const Im2ColMetadata& meta)
    {
        uint32_t y, rem_ks, x, c_base;
        meta.mdiv_XC.divmod(static_cast<uint32_t>(k_start),     meta.XC, y, rem_ks);
        meta.mdiv_C .divmod(static_cast<uint32_t>(rem_ks),      meta.C,  x, c_base);
        y_      = static_cast<index_t>(y);
        x_      = static_cast<index_t>(x);
        c_conv_ = static_cast<index_t>(c_base) + k_loc;  // no modulo — no C-boundary crossing

        k_gemm    = k_start + k_loc;
        K_offset_ = y_ * meta.DH_HiStride + x_ * meta.DW_WiStride + c_conv_;
    }

    CK_TILE_HOST_DEVICE void init_valid(const Im2ColMetadata& meta)
    {
        const index_t ih = ho_ + y_ - meta.PH;
        const index_t iw = wo_ + x_ - meta.PW;
        valid = (m_gemm < meta.M_gemm) && (k_gemm < meta.K_gemm) &&
                (ih >= 0 && ih < meta.Hi) && (iw >= 0 && iw < meta.Wi);
    }

    CK_TILE_HOST_DEVICE void init(index_t m, index_t k, const Im2ColMetadata& meta)
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
    //
    // When dm != 0: full reinit, 5 divmod.  Rare — only during tile setup.
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void move_step(index_t dm, index_t dk, const Im2ColMetadata& meta)
    {
        if(dm == 0)
        {
            k_gemm  += dk;
            c_conv_ += dk;
            K_offset_ += dk;

            if(c_conv_ >= meta.C)
            {
                c_conv_ -= meta.C;
                x_++;
                K_offset_ += meta.DW_WiStride - meta.C;

                if(x_ >= meta.X)
                {
                    x_ = 0;
                    y_++;
                    K_offset_ += meta.DH_HiStride - meta.X * meta.DW_WiStride;
                }

                const index_t ih = ho_ * meta.SH + y_ * meta.DH - meta.PH;
                const index_t iw = wo_ * meta.SW + x_ * meta.DW - meta.PW;
                valid = (k_gemm < meta.K_gemm) &&
                        (ih >= 0 && ih < meta.Hi) && (iw >= 0 && iw < meta.Wi);
            }
            else
            {
                valid = (k_gemm < meta.K_gemm) && valid;
            }
        }
        else
        {
            init(m_gemm + dm, k_gemm + dk, meta);
        }
    }

    CK_TILE_HOST_DEVICE index_t get_offset() const { return M_base + K_offset_; }
    CK_TILE_HOST_DEVICE bool    is_valid()   const { return valid; }
};

// ============================================================================
// Im2ColCoordinate<FwdOutput>
// ============================================================================
// Coordinate for the C (output) tile window in fwd convolution.
// Output tensor layout: NHWGK  (n_conv, ho, wo, g, k_conv)
//
// Address decomposition:
//   offset(m, n) = M_base(m) + n * KStride
//
//   M_base(m) = n_conv * NStride_out
//             + ho * HoStride        (= ho * step_w_out, since SH=1 for output)
//             + wo * WoStride        (= wo * step_w,     since SW=1 for output)
//
//   N_offset_(n) = n * KStride       (trivially linear; no y/x/c decode needed)
//
// Validity: pure GEMM bounds — no padding check on the output tensor.
//   valid = (m_gemm < M_gemm) && (n_gemm < K_gemm)   [K_gemm stores N_gemm = K for output]

template <>
struct Im2ColCoordinate<Im2ColTensor::FwdOutput>
{
    index_t M_base;     // offset contribution from m_gemm
    index_t N_offset_;  // offset contribution from n_gemm  (n * KStride)
    index_t m_gemm;     // current absolute M index
    index_t n_gemm;     // current absolute N index (= k_conv output channel index)
    index_t ho_;        // output row cached from init_m (for move_step wrap detection)
    index_t wo_;        // output col cached from init_m
    bool    valid;      // true iff (m_gemm, n_gemm) is within GEMM bounds

    // -------------------------------------------------------------------------
    // M part: identical structure to FwdInput — decode m_gemm → (n_conv, ho, wo).
    // Cost: 3 integer divisions (via magic multiply-shift).
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void init_m(index_t m, const Im2ColMetadata& meta)
    {
        m_gemm = m;

        uint32_t n_conv, rem_m, ho, wo;
        meta.mdiv_HoWo.divmod(static_cast<uint32_t>(m),     meta.HoWo, n_conv, rem_m);
        meta.mdiv_Wo  .divmod(static_cast<uint32_t>(rem_m), meta.Wo,   ho,     wo);
        ho_ = static_cast<index_t>(ho);
        wo_ = static_cast<index_t>(wo);

        M_base = static_cast<index_t>(n_conv) * meta.NStride
               + ho_ * meta.SH_HiStride
               + wo_ * meta.step_w
               - meta.pad_offset;   // pad_offset == 0 for FwdOutput
    }

    // -------------------------------------------------------------------------
    // N part: trivially linear — n_gemm is the output channel index k_conv.
    // No filter decode needed; KStride is the stride of k_conv in output memory.
    // Cost: 0 integer divisions.
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void init_k(index_t n, const Im2ColMetadata& meta)
    {
        n_gemm    = n;
        N_offset_ = n * meta.KStride;
    }

    CK_TILE_HOST_DEVICE void init_valid(const Im2ColMetadata& meta)
    {
        valid = (m_gemm < meta.M_gemm) && (n_gemm < meta.K_gemm);
    }

    CK_TILE_HOST_DEVICE void init(index_t m, index_t n, const Im2ColMetadata& meta)
    {
        init_m(m, meta);
        init_k(n, meta);
        init_valid(meta);
    }

    // -------------------------------------------------------------------------
    // Move by (dm, dn) in the global (M, N) space.
    //
    // When dm == 0 (pure N step): 0 divmod.  N_offset_ updated with 1 multiply.
    // When dm != 0: full reinit, 3 divmod (M only; N is trivial).
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void move_step(index_t dm, index_t dn, const Im2ColMetadata& meta)
    {
        if(dm == 0)
        {
            n_gemm    += dn;
            N_offset_ += dn * meta.KStride;
            valid = (n_gemm < meta.K_gemm) && valid;
        }
        else
        {
            init(m_gemm + dm, n_gemm + dn, meta);
        }
    }

    CK_TILE_HOST_DEVICE index_t get_offset() const { return M_base + N_offset_; }
    CK_TILE_HOST_DEVICE bool    is_valid()   const { return valid; }
};

// ============================================================================
// Trait helper: detect if a descriptor has im2col metadata
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

// ============================================================================
// Trait helper: extract Im2ColTensor kind from a descriptor's coordinate type
// ============================================================================
// Defaults to FwdInput for backward compatibility with existing descriptors
// that expose get_im2col_meta() but do not declare im2col_tensor_kind.

template <typename T, typename = void>
struct im2col_tensor_kind_of
{
    static constexpr Im2ColTensor value = Im2ColTensor::FwdInput;
};

template <typename T>
struct im2col_tensor_kind_of<T, std::void_t<decltype(T::im2col_tensor_kind)>>
{
    static constexpr Im2ColTensor value = T::im2col_tensor_kind;
};

template <typename T>
inline constexpr Im2ColTensor im2col_tensor_kind_of_v = im2col_tensor_kind_of<T>::value;

} // namespace ck_tile
