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
    FwdInput,     // GEMM-A: input tensor (NHWGC), requires full (y,x,c) K-decode + padding check
    FwdOutput,    // GEMM-C: output tensor (NHWGK), K-dim is trivially k_conv = n_gemm
    BwdDataInput, // GEMM-A: output-gradient tensor (NHWGK); K decomposes as (ydot,xdot,k_out)
    BwdDataOutput,// GEMM-C: input-gradient tensor (NHWGC); N-dim is c_conv; validity = no-pad
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

    // ---- N-dimension constants (FwdOutput only; unused for FwdInput/BwdDataInput) ----
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

    // ---- BwdDataInput fields (BwdDataInput only) ----
    // dY tensor layout: NHWGK  (n, ho, wo, g, k)
    // GemmM = (N, HTildeSlice, WTildeSlice),  GemmK = (YDotSlice, XDotSlice, K)
    //
    // M decode:  m_gemm → (n, htilde_local, wtilde_local)
    //   htilde_abs = htilde_local + IHTildeSliceBegin
    //   wtilde_abs = wtilde_local + IWTildeSliceBegin
    //   M_base  = n * NStride_dy + htilde_abs * HoStride_dy + wtilde_abs * WoStride_dy
    //           = n * NStride_dy + (htilde_local + IHTildeSliceBegin) * HoStride_dy
    //                            + (wtilde_local + IWTildeSliceBegin) * WoStride_dy
    //
    // K decode:  k_gemm → (ydot, xdot, k_out)  with periods (YDotSlice, XDotSlice, K)
    //   K_offset = -ydot * (DH/gcd * HoStride_dy) - xdot * (DW/gcd * WoStride_dy) + k_out
    //
    // Validity: ho = htilde_abs - ydot*(DH/gcd) ∈ [0, Ho)
    //           wo = wtilde_abs - xdot*(DW/gcd) ∈ [0, Wo)

    // M-side strides for dY (NHWGK)
    index_t NStride_dy;       // Ho * Wo * G * K
    index_t HoStride_dy;      // Wo * G * K
    index_t WoStride_dy;      // G * K  (= 1 when G=K=1)

    // K-offset coefficients (negative per dot step)
    index_t DH_gcd_HoStride;  // (DH / GcdSH_DH) * HoStride_dy   (step per ydot)
    index_t DW_gcd_WoStride;  // (DW / GcdSW_DW) * WoStride_dy   (step per xdot)
    index_t DH_gcd;           // DH / GcdSH_DH  — spatial step per ydot (for validity)
    index_t DW_gcd;           // DW / GcdSW_DW  — spatial step per xdot (for validity)

    // Tilde-slice origin (added to decoded htilde_local / wtilde_local to get htilde_abs)
    index_t IHTildeSliceBegin; // first htilde that touches the non-pad input region
    index_t IWTildeSliceBegin; // first wtilde that touches the non-pad input region

    // Tilde-slice widths (sizes of the slice in M dimension)
    index_t HTildeSlice;
    index_t WTildeSlice;

    // K decomposition periods
    index_t K;            // output channels per group  (innermost K-period in GemmK)
    index_t XDotSlice;    // ceil((X - IdxXTilde) / XTilde)  (mid K-period)

    // Validity bounds for output spatial dimensions
    index_t Ho;           // output height (htilde_abs - ydot*(DH/gcd) must be in [0, Ho))
    index_t Wo_bwd;       // output width  (wtilde_abs - xdot*(DW/gcd) must be in [0, Wo))

    // Magic divisors for BwdDataInput M and K decode
    mdiv2 mdiv_HTildeSlice_WTildeSlice; // for n: m / (HTildeSlice * WTildeSlice)
    mdiv2 mdiv_WTildeSlice;             // for htilde_local: rem / WTildeSlice
    mdiv2 mdiv_XDotSlice_K;             // for ydot: k / (XDotSlice * K)
    mdiv2 mdiv_K;                       // for xdot: rem / K

    // ---- BwdDataOutput fields (BwdDataOutput only) ----
    // dX tensor layout: NHWGC  (n, hi, wi, g, c)
    // GemmM = (N, HTildeSlice, WTildeSlice),  GemmN = C
    //
    // M decode:  m_gemm → (n, htilde_local, wtilde_local)
    //   htilde_abs = htilde_local + IHTildeSliceBegin
    //   hi = IdxYTilde * DH + htilde_abs * SH - PH
    //   wi = IdxXTilde * DW + wtilde_abs * SW - PW
    //   M_base = n * NStride_dx + hi * HiStride_dx + wi * WiStride_dx
    //
    // N is trivially c_conv (contiguous C-dimension).
    //
    // Validity: hi ∈ [0, Hi)  and  wi ∈ [0, Wi)
    //   (elements in the padded region are excluded)

    // dX (NHWGC) strides
    index_t NStride_dx;     // Hi * Wi * G * C
    index_t HiStride_dx;    // Wi * G * C
    index_t WiStride_dx;    // G * C

    // Constants folded from tilde + conv params (per sub-GEMM)
    index_t IdxYTilde_DH;   // IdxYTilde * DH  (constant hi contribution from current YTilde)
    index_t IdxXTilde_DW;   // IdxXTilde * DW  (constant wi contribution from current XTilde)
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
// Im2ColCoordinate<BwdDataInput>
// ============================================================================
// Coordinate for the A (dY / output-gradient) tile window in bwd-data convolution.
// dY tensor layout: NHWGK  →  GemmM = (N, HTildeSlice, WTildeSlice),
//                             GemmK = (YDotSlice, XDotSlice, K)
//
// Address formula:
//   offset(m, k) = M_base(m) + K_offset(k)
//
//   M_base(m)    = n * NStride_dy
//                + (htilde_local + IHTildeSliceBegin) * HoStride_dy
//                + (wtilde_local + IWTildeSliceBegin) * WoStride_dy
//
//   K_offset(k)  = -ydot * DH_gcd_HoStride
//                  -xdot * DW_gcd_WoStride
//                  + k_out
//   Note: K_offset is negative for non-zero ydot/xdot; combined offset ≥ 0 for valid elements.
//
// Validity: ho = (htilde_local + IHTildeSliceBegin) - ydot * (DH/gcd) ∈ [0, Ho)
//           wo = (wtilde_local + IWTildeSliceBegin) - xdot * (DW/gcd) ∈ [0, Wo)
//           m_gemm < M_gemm,  k_gemm < K_gemm

template <>
struct Im2ColCoordinate<Im2ColTensor::BwdDataInput>
{
    index_t M_base;         // offset from m_gemm (constant across K-loop when dm=0)
    index_t K_offset_;      // offset from k_gemm (may be negative; total offset ≥ 0 when valid)
    index_t m_gemm;         // current absolute M index
    index_t k_gemm;         // current absolute K index
    index_t htilde_local_;  // htilde within slice (htilde_abs = htilde_local_ + IHTildeSliceBegin)
    index_t wtilde_local_;  // wtilde within slice
    index_t htilde_abs_;    // cached htilde_abs = htilde_local_ + IHTildeSliceBegin
    index_t wtilde_abs_;    // cached wtilde_abs = wtilde_local_ + IWTildeSliceBegin
    index_t ydot_;          // current YDot index
    index_t xdot_;          // current XDot index
    index_t k_out_;         // current K (output channel) index within slice
    bool    valid;          // true iff the (m, k) combination addresses a real dY element

    // -------------------------------------------------------------------------
    // M part: decode m_gemm → (n, htilde_local, wtilde_local), compute M_base.
    // Cost: 3 integer divisions (via magic multiply-shift).
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void init_m(index_t m, const Im2ColMetadata& meta)
    {
        m_gemm = m;

        uint32_t n, rem_m, htloc, wtloc;
        meta.mdiv_HTildeSlice_WTildeSlice.divmod(
            static_cast<uint32_t>(m), meta.HTildeSlice * meta.WTildeSlice, n, rem_m);
        meta.mdiv_WTildeSlice.divmod(
            static_cast<uint32_t>(rem_m), meta.WTildeSlice, htloc, wtloc);

        htilde_local_ = static_cast<index_t>(htloc);
        wtilde_local_ = static_cast<index_t>(wtloc);
        htilde_abs_   = htilde_local_ + meta.IHTildeSliceBegin;
        wtilde_abs_   = wtilde_local_ + meta.IWTildeSliceBegin;

        M_base = static_cast<index_t>(n) * meta.NStride_dy
               + htilde_abs_ * meta.HoStride_dy
               + wtilde_abs_ * meta.WoStride_dy;
    }

    // -------------------------------------------------------------------------
    // K part: decode k_gemm → (ydot, xdot, k_out), compute K_offset_.
    // Cost: 2 integer divisions.
    // Must be called after init_m() (uses htilde_abs_, wtilde_abs_ for validity).
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void init_k(index_t k, const Im2ColMetadata& meta)
    {
        k_gemm = k;

        uint32_t ydot, rem_k, xdot, kout;
        meta.mdiv_XDotSlice_K.divmod(
            static_cast<uint32_t>(k), meta.XDotSlice * meta.K, ydot, rem_k);
        meta.mdiv_K.divmod(static_cast<uint32_t>(rem_k), meta.K, xdot, kout);

        ydot_   = static_cast<index_t>(ydot);
        xdot_   = static_cast<index_t>(xdot);
        k_out_  = static_cast<index_t>(kout);

        // K_offset_ is negative for non-zero dot indices.
        // The combined offset (M_base + K_offset_) is always non-negative for valid elements.
        K_offset_ = -ydot_ * meta.DH_gcd_HoStride
                    - xdot_ * meta.DW_gcd_WoStride
                    + k_out_;
    }

    CK_TILE_HOST_DEVICE void init_valid(const Im2ColMetadata& meta)
    {
        // ho and wo are the output spatial coordinates in dY that this (m,k) addresses.
        // For the element to exist in dY, both must be in [0, Ho) and [0, Wo) respectively.
        const index_t ho = htilde_abs_ - ydot_ * meta.DH_gcd;
        const index_t wo = wtilde_abs_ - xdot_ * meta.DW_gcd;
        valid = (m_gemm < meta.M_gemm) && (k_gemm < meta.K_gemm) &&
                (ho >= 0 && ho < meta.Ho) && (wo >= 0 && wo < meta.Wo_bwd);
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
    //   k_out_, xdot_, ydot_ updated incrementally with additions and comparisons.
    //
    // When dm != 0: full reinit, 5 divmod.  Rare — only during tile setup.
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void move_step(index_t dm, index_t dk, const Im2ColMetadata& meta)
    {
        if(dm == 0)
        {
            k_gemm += dk;
            k_out_ += dk;

            if(k_out_ >= meta.K)
            {
                // k_out_ crossed a K boundary — recompute xdot_ and ydot_ from scratch.
                // This is a less-frequent event (period K), so full recomputation is acceptable.
                k_out_ -= meta.K;
                xdot_++;

                if(xdot_ >= meta.XDotSlice)
                {
                    xdot_ = 0;
                    ydot_++;
                }

                // Recompute K_offset_ from the updated (ydot_, xdot_, k_out_).
                K_offset_ = -ydot_ * meta.DH_gcd_HoStride
                            - xdot_ * meta.DW_gcd_WoStride
                            + k_out_;

                // Recompute validity.
                const index_t ho = htilde_abs_ - ydot_ * meta.DH_gcd;
                const index_t wo = wtilde_abs_ - xdot_ * meta.DW_gcd;
                valid = (k_gemm < meta.K_gemm) &&
                        (ho >= 0 && ho < meta.Ho) && (wo >= 0 && wo < meta.Wo_bwd);
            }
            else
            {
                K_offset_ += dk;
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
// Im2ColCoordinate<BwdDataOutput>
// ============================================================================
// Coordinate for the C (dX / input-gradient) tile window in bwd-data convolution.
// dX tensor layout: NHWGC  →  GemmM = (N, HTildeSlice, WTildeSlice),
//                             GemmN = C (trivially linear)
//
// Address formula:
//   offset(m, n) = M_base(m) + n
//
//   M_base(m) = n_batch * NStride_dx
//             + hi * HiStride_dx
//             + wi * WiStride_dx
//
//   where hi = IdxYTilde_DH + (htilde_local + IHTildeSliceBegin) * SH - PH
//         wi = IdxXTilde_DW + (wtilde_local + IWTildeSliceBegin) * SW - PW
//
// Validity: hi ∈ [0, Hi)  and  wi ∈ [0, Wi)
//   Elements in the padded region must not be written to the input tensor.
//   m_gemm < M_gemm  and  n < C (= K_gemm for BwdDataOutput).

template <>
struct Im2ColCoordinate<Im2ColTensor::BwdDataOutput>
{
    index_t M_base;         // offset from m_gemm (constant across N-loop when dm=0)
    index_t N_offset_;      // offset from n_gemm (= n, since C-stride = 1)
    index_t m_gemm;         // current absolute M index
    index_t n_gemm;         // current absolute N index (= c_conv)
    index_t htilde_local_;  // htilde within slice
    index_t wtilde_local_;  // wtilde within slice
    bool    valid;          // true iff (m, n) maps to a non-padded dX element

    // -------------------------------------------------------------------------
    // M part: decode m_gemm → (n_batch, htilde_local, wtilde_local), compute M_base.
    // Cost: 3 integer divisions (via magic multiply-shift).
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void init_m(index_t m, const Im2ColMetadata& meta)
    {
        m_gemm = m;

        uint32_t n_batch, rem_m, htloc, wtloc;
        meta.mdiv_HTildeSlice_WTildeSlice.divmod(
            static_cast<uint32_t>(m), meta.HTildeSlice * meta.WTildeSlice, n_batch, rem_m);
        meta.mdiv_WTildeSlice.divmod(
            static_cast<uint32_t>(rem_m), meta.WTildeSlice, htloc, wtloc);

        htilde_local_ = static_cast<index_t>(htloc);
        wtilde_local_ = static_cast<index_t>(wtloc);

        const index_t htilde_abs = htilde_local_ + meta.IHTildeSliceBegin;
        const index_t wtilde_abs = wtilde_local_ + meta.IWTildeSliceBegin;

        // Map tilde coordinates to input spatial coordinates
        const index_t hi = meta.IdxYTilde_DH + htilde_abs * meta.SH - meta.PH;
        const index_t wi = meta.IdxXTilde_DW + wtilde_abs * meta.SW - meta.PW;

        M_base = static_cast<index_t>(n_batch) * meta.NStride_dx
               + hi * meta.HiStride_dx
               + wi * meta.WiStride_dx;
    }

    // -------------------------------------------------------------------------
    // N part: trivially linear — n_gemm is the input channel index c_conv.
    // Cost: 0 integer divisions.
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void init_k(index_t n, const Im2ColMetadata& meta)
    {
        n_gemm    = n;
        N_offset_ = n;   // C-stride = 1 (contiguous channels)
        (void)meta;
    }

    CK_TILE_HOST_DEVICE void init_valid(const Im2ColMetadata& meta)
    {
        const index_t htilde_abs = htilde_local_ + meta.IHTildeSliceBegin;
        const index_t wtilde_abs = wtilde_local_ + meta.IWTildeSliceBegin;
        const index_t hi = meta.IdxYTilde_DH + htilde_abs * meta.SH - meta.PH;
        const index_t wi = meta.IdxXTilde_DW + wtilde_abs * meta.SW - meta.PW;
        valid = (m_gemm < meta.M_gemm) && (n_gemm < meta.K_gemm) &&
                (hi >= 0 && hi < meta.Hi) && (wi >= 0 && wi < meta.Wi);
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
    // When dm == 0 (pure N step): 0 divmod.  N_offset_ updated trivially.
    // When dm != 0: full reinit, 3 divmod.  Rare — only during tile setup.
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE void move_step(index_t dm, index_t dn, const Im2ColMetadata& meta)
    {
        if(dm == 0)
        {
            n_gemm    += dn;
            N_offset_ += dn;
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
