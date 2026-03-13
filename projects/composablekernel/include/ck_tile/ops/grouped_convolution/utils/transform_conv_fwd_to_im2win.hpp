// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/grouped_convolution/utils/convolution_specialization.hpp"

namespace ck_tile {

// ═══════════════════════════════════════════════════════════════════════
// TransformConvFwdToIm2win
// ═══════════════════════════════════════════════════════════════════════
//
// Implements the *true* im2win transformation for 2D forward grouped
// convolution (arXiv:2306.14316), mapping convolution to GEMM:
//
//   A[M=K, K_gemm=C×Y×X]    ← weight W[K,C,Y,X] (GKCYX, channels-first)
//   B[N=N×Ho×Wo, K_gemm]    ← input  I[N,C,Hi,Wi] (GNCHW, channels-first)
//   C[M=K, N=N×Ho×Wo]       ← output O[N,K,Ho,Wo] (GNKHW, channels-first)
//
// Unlike im2col (M=N×Ho×Wo, N=K), true im2win has M=K (small for our
// target problems with K=4,8,16) so MFMA tiles can match M exactly.
//
// The B (input) descriptor is provided in two forms:
//
//   MakeBDescriptor_N_K_Composite<GNCHW>()  — Approach 1 (composite):
//     Chains elementary transforms (pad, embed, merge) following the
//     im2col style in transform_conv_fwd_to_gemm.hpp.
//
//   MakeBDescriptor_N_K_Direct<GNCHW>()     — Approach 2 (direct):
//     Uses the explicit index formulas from im2win_transform.md directly
//     as a flat stride descriptor, following the style of
//     TransformConvBwdDataToGemm_v1 in old CK.
//     Padding is expressed via negative strides; out-of-bounds accesses
//     are suppressed by the kernel's kPadK=true tile window.
//
// The default alias MakeBDescriptor_N_K selects Approach 1.
//
// Scope: 2D convolution, NumGroupsToMerge=1.
// ═══════════════════════════════════════════════════════════════════════

template <index_t NDimSpatial,
          ConvolutionSpecialization ConvSpecialization,
          index_t VectorSizeA,
          index_t VectorSizeB,
          index_t VectorSizeC,
          index_t NumGroupsToMerge    = 1,
          bool SplitN                 = false,
          bool UseDirectTransform     = false, // select B descriptor approach
          typename ADataType          = float,
          typename CDataType          = float,
          typename IndexType          = index_t>
struct TransformConvFwdToIm2win
{
    static_assert(NDimSpatial == 2,
                  "TransformConvFwdToIm2win currently supports 2D convolution only.");
    static_assert(NumGroupsToMerge == 1,
                  "Channels-first GNCHW/GKCYX only supports NumGroupsToMerge == 1. "
                  "Use NHWGC/GKYXC for group merging.");

    private:
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};
    static constexpr auto I3 = number<3>{};
    static constexpr auto I4 = number<4>{};

    static constexpr long_index_t TwoGB = (long_index_t{1} << 31);

    public:
    CK_TILE_HOST constexpr IndexType GetN() const { return N_; }
    CK_TILE_HOST constexpr IndexType GetOriginalN() const { return original_N_; }
    CK_TILE_HOST constexpr TransformConvFwdToIm2win() {}

    // ── Constructor ───────────────────────────────────────────────────
    // Argument conventions (indices into the length arrays):
    //   [G=0, N=1, C=2, Hi=3, Wi=4]   for input
    //   [G=0, K=1, C=2,  Y=3,  X=4]   for weight
    //   [G=0, N=1, K=2, Ho=3, Wo=4]   for output
    template <typename ConvDimsType,
              typename ConvSpatialDimsType,
              index_t NDim                                   = NDimSpatial,
              typename std::enable_if<NDim == 2, bool>::type = false>
    CK_TILE_HOST TransformConvFwdToIm2win(const ConvDimsType& a_g_n_c_wis_lengths,
                                          const ConvDimsType& b_g_k_c_xs_lengths,
                                          const ConvDimsType& c_g_n_k_wos_lengths,
                                          const ConvSpatialDimsType& conv_filter_strides,
                                          const ConvSpatialDimsType& conv_filter_dilations,
                                          const ConvSpatialDimsType& input_left_pads,
                                          const ConvSpatialDimsType& input_right_pads)
        : G_{a_g_n_c_wis_lengths[I0]},
          N_{c_g_n_k_wos_lengths[I1]},
          original_N_{c_g_n_k_wos_lengths[I1]},
          C_{b_g_k_c_xs_lengths[I2]},
          Hi_{a_g_n_c_wis_lengths[I3]},
          Wi_{a_g_n_c_wis_lengths[I4]},
          K_{c_g_n_k_wos_lengths[I2]},
          Ho_{c_g_n_k_wos_lengths[I3]},
          Wo_{c_g_n_k_wos_lengths[I4]},
          Y_{b_g_k_c_xs_lengths[I3]},
          X_{b_g_k_c_xs_lengths[I4]},
          ConvStrideH_{conv_filter_strides[I0]},
          ConvStrideW_{conv_filter_strides[I1]},
          ConvDilationH_{conv_filter_dilations[I0]},
          ConvDilationW_{conv_filter_dilations[I1]},
          InLeftPadH_{input_left_pads[I0]},
          InLeftPadW_{input_left_pads[I1]},
          InRightPadH_{input_right_pads[I0]},
          InRightPadW_{input_right_pads[I1]},
          // Physical strides for I[N,C,Hi,Wi] (NCHW, packed, per group)
          NStride_{C_ * Hi_ * Wi_},
          CStride_{Hi_ * Wi_},
          HiStride_{Wi_},
          WiStride_{1}
    {
        static_assert(std::is_same_v<ConvSpatialDimsType, std::array<IndexType, NDimSpatial>> ||
                      std::is_same_v<ConvSpatialDimsType, ck_tile::array<IndexType, NDimSpatial>>);
        static_assert(std::is_same_v<ConvDimsType, std::array<IndexType, NDimSpatial + I3>> ||
                      std::is_same_v<ConvDimsType, ck_tile::array<IndexType, NDimSpatial + I3>>);

        // 2 GB size check (using 2 bytes = fp16, the smallest supported type).
        // ADataType/CDataType default to float but actual element size may be smaller.
        // We check against the most conservative assumption (2 bytes = fp16/bf16).
        constexpr long_index_t bytes_per_elem = 2; // fp16 / bf16 minimum
        const long_index_t in_size  = static_cast<long_index_t>(G_) * N_ * C_ * Hi_ * Wi_;
        const long_index_t out_size = static_cast<long_index_t>(G_) * N_ * K_ * Ho_ * Wo_;
        if(in_size * bytes_per_elem > TwoGB)
            throw std::runtime_error("Im2win: input tensor exceeds 2 GB. Use split-N.");
        if(out_size * bytes_per_elem > TwoGB)
            throw std::runtime_error("Im2win: output tensor exceeds 2 GB. Use split-N.");
    }

    // ═══════════════════════════════════════════════════════════════════
    // A descriptor: Weight W → A[M=K, K_gemm=C×Y×X]
    //
    // Simple 2D naive descriptor — no sliding window, no padding.
    // For K=4, C=4, Y=X=3: only 144 elements, fits in registers.
    // ═══════════════════════════════════════════════════════════════════
    template <typename ALayout,
              typename std::enable_if<std::is_same_v<ALayout, tensor_layout::convolution::GKCYX>,
                                      bool>::type = false>
    CK_TILE_HOST auto MakeADescriptor_M_K() const
    {
        static_assert(NumGroupsToMerge == 1,
                      "GKCYX weight only supports NumGroupsToMerge==1.");

        // Physical strides for W[K,C,Y,X] (KCYX order, per group).
        const IndexType KStride = C_ * Y_ * X_;
        const IndexType XStride = 1; // innermost

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
        {
            // K_gemm = C * 9 (compile-time Y*X=9)
            return make_naive_tensor_descriptor(
                make_tuple(K_, C_ * number<9>{}),
                make_tuple(KStride, XStride),
                number<VectorSizeA>{}, I1);
        }
        else
        {
            // General: A[K, C*Y*X] with strides [C*Y*X, 1].
            return make_naive_tensor_descriptor(
                make_tuple(K_, C_ * Y_ * X_),
                make_tuple(KStride, XStride),
                number<VectorSizeA>{}, I1);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // B descriptor — Approach 1 (composite transforms)
    //
    // Maps I[N,C,Hi,Wi] (GNCHW, per group) to B[N=N×Ho×Wo, K_gemm=C×Y×X]
    // via the chain: physical → pad Hi/Wi → embed (Y,Ho)×(X,Wo) → merge.
    //
    // This is the same approach as transform_conv_fwd_to_gemm.hpp:
    //   Step 1: I[N, C, Hi, Wi]                (physical, NCHW strides)
    //   Step 2: I[N, C, Hi+LPH+RPH, Wi+LPW+RPW] (pad to allow filter overlap)
    //   Step 3: I[N, C, Y, Ho, X, Wo]           (embed: hi=y*Dy+ho*Sy, wi=x*Dx+wo*Sx)
    //   Step 4: B[N×Ho×Wo, C×Y×X]               (merge to GEMM shape)
    // ═══════════════════════════════════════════════════════════════════
    template <typename BLayout,
              typename std::enable_if<
                  NDimSpatial == 2 &&
                      std::is_same_v<BLayout, tensor_layout::convolution::GNCHW>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeBDescriptor_N_K_Composite() const
    {
        static_assert(NumGroupsToMerge == 1,
                      "GNCHW input only supports NumGroupsToMerge==1.");

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            // 1×1, stride=1, no padding: I[N,C,Hi,Wi] directly becomes B[N×Ho×Wo, C].
            const auto in_n_c_ho_wo_desc = make_naive_tensor_descriptor(
                make_tuple(N_, C_, Ho_, Wo_),
                make_tuple(NStride_, CStride_, HiStride_, WiStride_),
                number<VectorSizeB>{}, I1);

            return transform_tensor_descriptor(
                in_n_c_ho_wo_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0, 2, 3>{}, sequence<1>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
        {
            // Step 1: I[N, C, Hi, Wi]
            const auto in_n_c_hi_wi_desc = make_naive_tensor_descriptor(
                make_tuple(N_, C_, Hi_, Wi_),
                make_tuple(NStride_, CStride_, HiStride_, WiStride_),
                number<VectorSizeB>{}, I1);

            // Step 2: pad Hi and Wi
            const auto in_n_c_hip_wip_desc = transform_tensor_descriptor(
                in_n_c_hi_wi_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_pass_through_transform(C_),
                           make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                           make_pad_transform(Wi_, InLeftPadW_, InRightPadW_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

            // Step 3: embed (Y=3,Ho) onto padded Hi and (X=3,Wo) onto padded Wi
            const auto in_n_c_y_ho_x_wo_desc = transform_tensor_descriptor(
                in_n_c_hip_wip_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_pass_through_transform(C_),
                           make_embed_transform(make_tuple(number<3>{}, Ho_),
                                               make_tuple(ConvDilationH_, ConvStrideH_)),
                           make_embed_transform(make_tuple(number<3>{}, Wo_),
                                               make_tuple(ConvDilationW_, ConvStrideW_))),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4, 5>{}));

            // Step 4: merge → B[N=N×Ho×Wo, K_gemm=C×Y×X]
            return transform_tensor_descriptor(
                in_n_c_y_ho_x_wo_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                           make_merge_transform(make_tuple(C_, number<3>{}, number<3>{}))),
                make_tuple(sequence<0, 3, 5>{}, sequence<1, 2, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Pad0)
        {
            // 1×1 filter, no padding: embed stride into spatial.
            const auto in_n_c_hi_wi_desc = make_naive_tensor_descriptor(
                make_tuple(N_, C_, Hi_, Wi_),
                make_tuple(NStride_, CStride_, HiStride_, WiStride_),
                number<VectorSizeB>{}, I1);

            const auto in_n_c_ho_wo_desc = transform_tensor_descriptor(
                in_n_c_hi_wi_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_pass_through_transform(C_),
                           make_embed_transform(make_tuple(Ho_), make_tuple(ConvStrideH_)),
                           make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_))),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

            return transform_tensor_descriptor(
                in_n_c_ho_wo_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0, 2, 3>{}, sequence<1>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else
        {
            // General: arbitrary Y, X, stride, dilation, padding.
            const auto in_n_c_hi_wi_desc = make_naive_tensor_descriptor(
                make_tuple(N_, C_, Hi_, Wi_),
                make_tuple(NStride_, CStride_, HiStride_, WiStride_),
                number<VectorSizeB>{}, I1);

            const auto in_n_c_hip_wip_desc = transform_tensor_descriptor(
                in_n_c_hi_wi_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_pass_through_transform(C_),
                           make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                           make_pad_transform(Wi_, InLeftPadW_, InRightPadW_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

            const auto in_n_c_y_ho_x_wo_desc = transform_tensor_descriptor(
                in_n_c_hip_wip_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_pass_through_transform(C_),
                           make_embed_transform(make_tuple(Y_, Ho_),
                                               make_tuple(ConvDilationH_, ConvStrideH_)),
                           make_embed_transform(make_tuple(X_, Wo_),
                                               make_tuple(ConvDilationW_, ConvStrideW_))),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4, 5>{}));

            return transform_tensor_descriptor(
                in_n_c_y_ho_x_wo_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                           make_merge_transform(make_tuple(C_, Y_, X_))),
                make_tuple(sequence<0, 3, 5>{}, sequence<1, 2, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // B descriptor — Approach 2 (direct formula)
    //
    // Uses the explicit index formulas from im2win_transform.md directly
    // as a flat stride descriptor (style of TransformConvBwdDataToGemm_v1).
    //
    // From the formulas (group index g dropped, per-group view):
    //
    //   B[n_merged, k_gemm] = I[i_n, i_c, i_h, i_w]   where
    //
    //   n_merged = n*(Ho*Wo) + ho*Wo + wo              (n,ho,wo → B row)
    //   k_gemm   = c*(Y*X)  + fy*X  + fx              (c,fy,fx → B col)
    //
    //   i_n = n                                         (same as n_merged // (Ho*Wo))
    //   i_c = c                                         (same as k_gemm // (Y*X))
    //   i_h = ho*Sy + fy*Dy - LPH                      (may be negative → padding)
    //   i_w = wo*Sx + fx*Dx - LPW
    //
    // Physical address from I[N,C,Hi,Wi] (NCHW, packed, per group):
    //   addr = n * NStride + c * CStride
    //        + (ho*Sy + fy*Dy - LPH) * HiStride
    //        + (wo*Sx + fx*Dx - LPW) * WiStride
    //
    // This is linear in (n, ho, wo, c, fy, fx) with coefficients:
    //   stride(n)  = NStride                 [n contributes via i_n]
    //   stride(ho) = Sy * HiStride           [ho contributes via i_h]
    //   stride(wo) = Sx * WiStride           [wo contributes via i_w]
    //   stride(c)  = CStride
    //   stride(fy) = Dy * HiStride           [fy contributes via i_h]
    //   stride(fx) = Dx * WiStride           [fx contributes via i_w]
    //   offset     = -LPH * HiStride - LPW * WiStride  (padding offset)
    //
    // The descriptor is built as a 6D naive tensor with these strides,
    // then merged to [N=N×Ho×Wo, K_gemm=C×Y×X].  Padding (i_h < 0 or
    // i_h >= Hi etc.) produces out-of-bounds addresses; the kernel's
    // kPadK=true tile window zeroes these automatically.
    //
    // Note: this approach stores all physical strides in the transformer
    // (NStride_, CStride_, HiStride_, WiStride_) and uses them directly,
    // avoiding the multi-step composite chain of Approach 1.
    // ═══════════════════════════════════════════════════════════════════
    template <typename BLayout,
              typename std::enable_if<
                  NDimSpatial == 2 &&
                      std::is_same_v<BLayout, tensor_layout::convolution::GNCHW>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeBDescriptor_N_K_Direct() const
    {
        static_assert(NumGroupsToMerge == 1,
                      "GNCHW input only supports NumGroupsToMerge==1.");

        // Physical address = n*NStride + c*CStride
        //                  + (ho*Sy + fy*Dy - LPH)*HiStride
        //                  + (wo*Sx + fx*Dx - LPW)*WiStride
        //
        // Effective strides for each dimension:
        const IndexType n_eff  = NStride_;                        // stride of n
        const IndexType ho_eff = ConvStrideH_ * HiStride_;        // stride of ho
        const IndexType wo_eff = ConvStrideW_ * WiStride_;        // stride of wo
        const IndexType c_eff  = CStride_;                        // stride of c
        const IndexType fy_eff = ConvDilationH_ * HiStride_;      // stride of filter y
        const IndexType fx_eff = ConvDilationW_ * WiStride_;      // stride of filter x

        // Effective lengths for the 6D descriptor:
        //   N, Ho, Wo  → contribute to n_merged (GEMM N dimension)
        //   C, Y, X    → contribute to k_gemm   (GEMM K dimension)
        //
        // The "address origin" for padding is absorbed into the pointer offset
        // that the kernel applies: b_ptr = in_ptr - LPH*HiStride - LPW*WiStride
        // so the descriptor itself starts at index 0 for (ho=0, wo=0, fy=0, fx=0).
        //
        // Build the 6D descriptor with strides [n_eff, ho_eff, wo_eff, c_eff, fy_eff, fx_eff]:
        const auto in_6d_desc = make_naive_tensor_descriptor(
            make_tuple(N_, Ho_, Wo_, C_, Y_, X_),
            make_tuple(n_eff, ho_eff, wo_eff, c_eff, fy_eff, fx_eff),
            number<VectorSizeB>{}, I1);

        // Merge [N, Ho, Wo] → N_gemm and [C, Y, X] → K_gemm:
        return transform_tensor_descriptor(
            in_6d_desc,
            make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                       make_merge_transform(make_tuple(C_, Y_, X_))),
            make_tuple(sequence<0, 1, 2>{}, sequence<3, 4, 5>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    // Default B descriptor for GNCHW — dispatches on the UseDirectTransform
    // template parameter:
    //   false (default) → Approach 1: composite transforms
    //   true            → Approach 2: direct formula
    //
    // When UseDirectTransform=true the kernel must adjust b_ptr by
    // GetBPtrAdjust() so that the descriptor's zero-based addressing
    // correctly maps to the padded physical input space.
    template <typename BLayout,
              typename std::enable_if<
                  std::is_same_v<BLayout, tensor_layout::convolution::GNCHW>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeBDescriptor_N_K() const
    {
        if constexpr(UseDirectTransform)
            return MakeBDescriptor_N_K_Direct<BLayout>();
        else
            return MakeBDescriptor_N_K_Composite<BLayout>();
    }

    // Returns the byte-offset to add to b_ptr when UseDirectTransform=true.
    // The direct descriptor's index-0 corresponds to (ho=0, wo=0, fy=0, fx=0)
    // which maps to physical input position (i_h = -LPH, i_w = -LPW) —
    // in the left-padding zone.  Subtracting this offset from b_ptr brings
    // the origin of the descriptor to the physical start of the input tensor,
    // so that index arithmetic produces the correct physical addresses.
    //
    // For UseDirectTransform=false (composite) this is always 0.
    CK_TILE_HOST long_index_t GetBPtrAdjust() const
    {
        if constexpr(UseDirectTransform)
            return -(static_cast<long_index_t>(InLeftPadH_) * HiStride_ +
                     static_cast<long_index_t>(InLeftPadW_) * WiStride_);
        else
            return 0;
    }

    // ── GKYXC A descriptor (channels-last weight, for NHWGC path) ─────
    // K_gemm = merge([Y, X, C]): C innermost, consistent with NHWGC B.
    template <typename ALayout,
              typename std::enable_if<std::is_same_v<ALayout, tensor_layout::convolution::GKYXC>,
                                      bool>::type = false>
    CK_TILE_HOST auto MakeADescriptor_M_K() const
    {
        const IndexType KStride = Y_ * X_ * C_;
        const IndexType CStride = 1;
        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
            return make_naive_tensor_descriptor(
                make_tuple(K_ * NumGroupsToMerge, C_ * number<9>{}),
                make_tuple(C_ * number<9>{}, CStride),
                number<VectorSizeA>{}, I1);
        else
            return make_naive_tensor_descriptor(
                make_tuple(K_ * NumGroupsToMerge, Y_ * X_ * C_),
                make_tuple(KStride, CStride),
                number<VectorSizeA>{}, I1);
    }

    // ── NHWGC B descriptor (channels-last input, for group-merging) ───
    // Physical I[N,Hi,Wi,G,C]; G adjacent to C → group stride = C.
    // K_gemm = merge([Y, X, C]): C innermost, consistent with GKYXC A.
    template <typename BLayout,
              typename std::enable_if<
                  NDimSpatial == 2 &&
                      std::is_same_v<BLayout, tensor_layout::convolution::NHWGC>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeBDescriptor_N_K() const
    {
        const IndexType NStride  = Hi_ * Wi_ * G_ * C_;
        const IndexType HiStride = Wi_ * G_ * C_;
        const IndexType WiStride = G_ * C_;
        const IndexType CStride  = 1;
        const IndexType GStride  = C_;

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
        {
            const auto base = (NumGroupsToMerge == 1)
                ? make_naive_tensor_descriptor(make_tuple(N_, Hi_, Wi_, C_),
                      make_tuple(NStride, HiStride, WiStride, CStride), number<VectorSizeB>{}, I1)
                : make_naive_tensor_descriptor(make_tuple(N_, Hi_, Wi_, NumGroupsToMerge, C_),
                      make_tuple(NStride, HiStride, WiStride, GStride, CStride),
                      number<VectorSizeB>{}, I1);

            if constexpr(NumGroupsToMerge == 1)
            {
                const auto padded = transform_tensor_descriptor(
                    base,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));
                const auto embedded = transform_tensor_descriptor(
                    padded,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(number<3>{}, Ho_),
                                                   make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(number<3>{}, Wo_),
                                                   make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1,2>{}, sequence<3,4>{}, sequence<5>{}));
                return transform_tensor_descriptor(
                    embedded,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                               make_merge_transform(make_tuple(number<3>{}, number<3>{}, C_))),
                    make_tuple(sequence<0,2,4>{}, sequence<1,3,5>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else
            {
                const auto padded = transform_tensor_descriptor(
                    base,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{},sequence<1>{},sequence<2>{},sequence<3>{},sequence<4>{}),
                    make_tuple(sequence<0>{},sequence<1>{},sequence<2>{},sequence<3>{},sequence<4>{}));
                const auto embedded = transform_tensor_descriptor(
                    padded,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(number<3>{}, Ho_),
                                                   make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(number<3>{}, Wo_),
                                                   make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{},sequence<1>{},sequence<2>{},sequence<3>{},sequence<4>{}),
                    make_tuple(sequence<0>{},sequence<1,2>{},sequence<3,4>{},sequence<5>{},sequence<6>{}));
                return transform_tensor_descriptor(
                    embedded,
                    make_tuple(make_merge_transform(make_tuple(NumGroupsToMerge, N_, Ho_, Wo_)),
                               make_merge_transform(make_tuple(number<3>{}, number<3>{}, C_))),
                    make_tuple(sequence<5,0,2,4>{}, sequence<1,3,6>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
        else
        {
            // General case (Gm=1 only for simplicity here)
            const auto base = make_naive_tensor_descriptor(
                make_tuple(N_, Hi_, Wi_, C_),
                make_tuple(NStride, HiStride, WiStride, CStride),
                number<VectorSizeB>{}, I1);
            const auto padded = transform_tensor_descriptor(
                base,
                make_tuple(make_pass_through_transform(N_),
                           make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                           make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0>{},sequence<1>{},sequence<2>{},sequence<3>{}),
                make_tuple(sequence<0>{},sequence<1>{},sequence<2>{},sequence<3>{}));
            const auto embedded = transform_tensor_descriptor(
                padded,
                make_tuple(make_pass_through_transform(N_),
                           make_embed_transform(make_tuple(Y_, Ho_),
                                               make_tuple(ConvDilationH_, ConvStrideH_)),
                           make_embed_transform(make_tuple(X_, Wo_),
                                               make_tuple(ConvDilationW_, ConvStrideW_)),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0>{},sequence<1>{},sequence<2>{},sequence<3>{}),
                make_tuple(sequence<0>{},sequence<1,2>{},sequence<3,4>{},sequence<5>{}));
            return transform_tensor_descriptor(
                embedded,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                           make_merge_transform(make_tuple(Y_, X_, C_))),
                make_tuple(sequence<0,2,4>{}, sequence<1,3,5>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // C descriptor: Output O → C[M=K, N=N×Ho×Wo]
    //
    // Two output layouts supported:
    //
    //   GNKHW (channels-first):
    //     O[N, K, Ho, Wo] per group; K has stride Ho×Wo (non-unit).
    //     Vectorised stores along N (Wo innermost, stride 1).
    //
    //   NHWGK (channels-last, K unit-stride):
    //     O[N, Ho, Wo, G, K]; K innermost (stride 1).
    //     Vectorised stores along M=K work efficiently.
    // ═══════════════════════════════════════════════════════════════════

    // ── GNKHW output (channels-first, primary) ────────────────────────
    template <typename CLayout,
              typename std::enable_if<
                  NDimSpatial == 2 &&
                      std::is_same_v<CLayout, tensor_layout::convolution::GNKHW>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeCDescriptor_M_N() const
    {
        static_assert(NumGroupsToMerge == 1,
                      "GNKHW output only supports NumGroupsToMerge==1.");

        // Physical O[N,K,Ho,Wo] per group (NKHW order, G stripped by pointer offset).
        const IndexType NStride  = K_ * Ho_ * Wo_;
        const IndexType KStride  = Ho_ * Wo_; // stride(M=K) — non-unit
        const IndexType HoStride = Wo_;
        const IndexType WoStride = 1; // unit stride (innermost of N_gemm merge)

        const auto out_n_k_ho_wo_desc = make_naive_tensor_descriptor(
            make_tuple(N_, K_, Ho_, Wo_),
            make_tuple(NStride, KStride, HoStride, WoStride),
            number<VectorSizeC>{}, I1);

        // C[M=K, N=N×Ho×Wo]:
        //   M = K → dimension 1 (stride = Ho×Wo)
        //   N = merge(N=dim0, Ho=dim2, Wo=dim3) with Wo innermost (stride 1)
        return transform_tensor_descriptor(
            out_n_k_ho_wo_desc,
            make_tuple(make_pass_through_transform(K_),
                       make_merge_transform(make_tuple(N_, Ho_, Wo_))),
            make_tuple(sequence<1>{}, sequence<0, 2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    // ── NHWGK output (channels-last, K unit-stride) ───────────────────
    template <typename CLayout,
              typename std::enable_if<
                  NDimSpatial == 2 &&
                      std::is_same_v<CLayout, tensor_layout::convolution::NHWGK>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeCDescriptor_M_N() const
    {
        static_assert(NumGroupsToMerge == 1,
                      "NHWGK output with NumGroupsToMerge==1 only in this path.");

        // Physical O[N,Ho,Wo,G,K]; group G stripped by pointer offset.
        // c_ptr = out_ptr + g_base * K  (stride of G dimension in NHWGK = K).
        const IndexType NStride  = Ho_ * Wo_ * G_ * K_;
        const IndexType HoStride = Wo_ * G_ * K_;
        const IndexType WoStride = G_ * K_;
        const IndexType KStride  = 1; // K innermost — vectorised stores along M=K ✓

        const auto out_n_ho_wo_k_desc = make_naive_tensor_descriptor(
            make_tuple(N_, Ho_, Wo_, K_),
            make_tuple(NStride, HoStride, WoStride, KStride),
            number<VectorSizeC>{}, I1);

        // C[M=K, N=N×Ho×Wo]:
        //   M = K → dimension 3 (stride = 1)
        //   N = merge(N=dim0, Ho=dim1, Wo=dim2)
        return transform_tensor_descriptor(
            out_n_ho_wo_k_desc,
            make_tuple(make_pass_through_transform(K_),
                       make_merge_transform(make_tuple(N_, Ho_, Wo_))),
            make_tuple(sequence<3>{}, sequence<0, 1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    // ═══════════════════════════════════════════════════════════════════
    // GEMM dimension queries
    // ═══════════════════════════════════════════════════════════════════

    /// M = K  (output channels — small for target use case K=4,8,16)
    CK_TILE_HOST IndexType GetGemmM() const { return K_; }

    /// N_gemm = N × Ho × Wo  (batch × output spatial)
    CK_TILE_HOST IndexType GetGemmN() const { return N_ * Ho_ * Wo_; }

    /// K_gemm = C × Y × X  (input channels × filter)
    CK_TILE_HOST IndexType GetGemmK() const { return C_ * Y_ * X_; }

    /// GemmBatch = G  (one GEMM per conv group)
    CK_TILE_HOST IndexType GetGemmBatch() const { return G_; }

    // ═══════════════════════════════════════════════════════════════════
    // Per-group strides (used by the kernel to advance pointers)
    // ═══════════════════════════════════════════════════════════════════

    /// Weight stride per group in GKCYX: K×C×Y×X elements.
    CK_TILE_HOST long_index_t GetGroupStrideA() const
    {
        return static_cast<long_index_t>(K_) * C_ * Y_ * X_;
    }

    /// Input stride per group in GNCHW: N×C×Hi×Wi elements.
    CK_TILE_HOST long_index_t GetGroupStrideB() const
    {
        return static_cast<long_index_t>(original_N_) * C_ * Hi_ * Wi_;
    }

    /// Output stride per group in NHWGK: K elements (stride of G dim).
    CK_TILE_HOST long_index_t GetGroupStrideC() const
    {
        return static_cast<long_index_t>(K_);
    }

    /// Output stride per group in GNKHW: N×K×Ho×Wo elements.
    CK_TILE_HOST long_index_t GetGroupStrideCGnkhw() const
    {
        return static_cast<long_index_t>(original_N_) * K_ * Ho_ * Wo_;
    }

    // ═══════════════════════════════════════════════════════════════════
    // Stored problem parameters (all per-group, G applied externally)
    // ═══════════════════════════════════════════════════════════════════
    IndexType G_{};
    IndexType N_{};           ///< Current N (== original_N_ for no split)
    IndexType original_N_{};
    IndexType C_{};
    IndexType Hi_{}, Wi_{};
    IndexType K_{};
    IndexType Ho_{}, Wo_{};
    IndexType Y_{},  X_{};
    IndexType ConvStrideH_,  ConvStrideW_{};
    IndexType ConvDilationH_, ConvDilationW_{};
    IndexType InLeftPadH_,   InLeftPadW_{};
    IndexType InRightPadH_,  InRightPadW_{};

    // Physical strides for I[N,C,Hi,Wi] per group (NCHW, packed).
    // Stored to support Approach 2 (direct formula) without recomputing.
    IndexType NStride_{};   ///< = C * Hi * Wi
    IndexType CStride_{};   ///< = Hi * Wi
    IndexType HiStride_{};  ///< = Wi
    IndexType WiStride_{};  ///< = 1
};

} // namespace ck_tile
