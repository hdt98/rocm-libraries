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
// Implements the *true* im2win GEMM mapping (paper: arXiv:2306.14316):
//
//   A[M=K, K_gemm=C×Y×X]   ← weight W[K,C,Y,X]          (GKCYX, channels-first)
//   B[N=N×Ho×Wo, K_gemm]   ← input  I[N,C,Hi,Wi]         (GNCHW, channels-first)
//   C[M=K, N=N×Ho×Wo]      ← output O[G,N,K,Ho,Wo]       (GNKHW, channels-first)
//
// Unlike im2col (M=N×Ho×Wo, N=K), true im2win has M=K (output channels) as
// the row dimension.  For small K the 4×64×16 MFMA instruction matches M=K=4
// exactly, eliminating the M-tile waste that im2col suffers in the N direction.
//
// Additional layouts supported:
//   MakeADescriptor_M_K<GKYXC>()  — channels-last weight (K_gemm=Y×X×C)
//   MakeBDescriptor_N_K<NHWGC>()  — channels-last input  (enables group merging)
//   MakeCDescriptor_M_N<NHWGK>()  — channels-last output (K unit-stride)
//
// Channels-first (GNCHW/GKCYX/GNKHW) only supports NumGroupsToMerge == 1.
// Group merging requires channels-last (NHWGC/GKYXC) and Gm power-of-2.
// ═══════════════════════════════════════════════════════════════════════

template <index_t NDimSpatial,
          ConvolutionSpecialization ConvSpecialization,
          index_t VectorSizeA,
          index_t VectorSizeB,
          index_t VectorSizeC,
          index_t NumGroupsToMerge = 1,
          bool SplitN              = false,
          typename ADataType       = float,
          typename CDataType       = float,
          typename IndexType       = index_t>
struct TransformConvFwdToIm2win
{
    static_assert(NDimSpatial == 2,
                  "TransformConvFwdToIm2win currently supports 2D convolution only.");

    private:
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};
    static constexpr auto I3 = number<3>{};
    static constexpr auto I4 = number<4>{};

    static constexpr long_index_t TwoGB = (long_index_t{1} << 31);

    template <typename ConvDimsType>
    static IndexType GetSplitedNSize(const ConvDimsType& a_g_n_c_wis_lengths,
                                     const ConvDimsType& c_g_n_k_wos_lengths)
    {
        const index_t num_dims = a_g_n_c_wis_lengths.size();
        ConvDimsType a_strides, c_strides;
        a_strides[num_dims - 1] = 1;
        c_strides[num_dims - 1] = 1;
        for(index_t i = num_dims - 2; i >= 0; i--)
        {
            a_strides[i] = a_strides[i + 1] * a_g_n_c_wis_lengths[i + 1];
            c_strides[i] = c_strides[i + 1] * c_g_n_k_wos_lengths[i + 1];
        }
        long_index_t a_size = 1, c_size = 1;
        for(index_t i = 1; i < num_dims; i++)
        {
            a_size += static_cast<long_index_t>(a_g_n_c_wis_lengths[i] - 1) *
                      static_cast<long_index_t>(a_strides[i]);
            c_size += static_cast<long_index_t>(c_g_n_k_wos_lengths[i] - 1) *
                      static_cast<long_index_t>(c_strides[i]);
        }
        const long_index_t element_space =
            ck_tile::max(a_size * sizeof(ADataType), c_size * sizeof(CDataType));
        const IndexType N = a_g_n_c_wis_lengths[I1];
        if(element_space > TwoGB)
        {
            const auto divisor = ck_tile::integer_divide_ceil(element_space, TwoGB);
            if(divisor <= static_cast<double>(N))
            {
                for(IndexType d = divisor; d * d <= N; d++)
                    if(N % d == 0) return N / d;
                return 1;
            }
            return 1;
        }
        return N;
    }

    public:
    CK_TILE_HOST constexpr IndexType GetN() const { return N_; }
    CK_TILE_HOST constexpr IndexType GetOriginalN() const { return original_N_; }
    CK_TILE_HOST constexpr TransformConvFwdToIm2win() {}

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
          InRightPadW_{input_right_pads[I1]}
    {
        static_assert(std::is_same_v<ConvSpatialDimsType, std::array<IndexType, NDimSpatial>> ||
                      std::is_same_v<ConvSpatialDimsType, ck_tile::array<IndexType, NDimSpatial>>);
        static_assert(std::is_same_v<ConvDimsType, std::array<IndexType, NDimSpatial + I3>> ||
                      std::is_same_v<ConvDimsType, ck_tile::array<IndexType, NDimSpatial + I3>>);
        if constexpr(SplitN)
            N_ = GetSplitedNSize(a_g_n_c_wis_lengths, c_g_n_k_wos_lengths);
    }

    // ─────────────────────────────────────────────────────────────────────
    // A descriptors: Weight W → A[M=K*Gm, K_gemm]
    // ─────────────────────────────────────────────────────────────────────

    // ── GKCYX (channels-first weight, NumGroupsToMerge=1 only) ────────
    template <typename ALayout,
              typename std::enable_if<std::is_same_v<ALayout, tensor_layout::convolution::GKCYX>,
                                      bool>::type = false>
    CK_TILE_HOST auto MakeADescriptor_M_K() const
    {
        static_assert(NumGroupsToMerge == 1,
                      "GKCYX weight only supports NumGroupsToMerge==1. "
                      "Use GKYXC for group merging.");
        const IndexType KStride = C_ * Y_ * X_;
        const IndexType CStride = Y_ * X_;
        const IndexType YStride = X_;
        const IndexType XStride = 1;

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
        {
            return make_naive_tensor_descriptor(
                make_tuple(K_, C_ * number<9>{}),
                make_tuple(KStride, XStride),
                number<VectorSizeA>{}, I1);
        }
        else
        {
            const auto wei_k_c_y_x_desc = make_naive_tensor_descriptor(
                make_tuple(K_, C_, Y_, X_),
                make_tuple(KStride, CStride, YStride, XStride),
                number<VectorSizeA>{}, I1);
            return transform_tensor_descriptor(
                wei_k_c_y_x_desc,
                make_tuple(make_pass_through_transform(K_),
                           make_merge_transform(make_tuple(C_, Y_, X_))),
                make_tuple(sequence<0>{}, sequence<1, 2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    // ── GKYXC (channels-last weight, NumGroupsToMerge>=1) ─────────────
    // K_gemm = merge([Y, X, C]): C innermost → vectorised A loads.
    // For Gm>1: A[M=Gm*K, K_gemm=Y*X*C] — Gm groups stacked contiguously.
    template <typename ALayout,
              typename std::enable_if<std::is_same_v<ALayout, tensor_layout::convolution::GKYXC>,
                                      bool>::type = false>
    CK_TILE_HOST auto MakeADescriptor_M_K() const
    {
        // Physical W[K, Y, X, C] per group (or Gm*K rows for Gm>1).
        // KStride = Y*X*C (distance between consecutive K rows in GKYXC).
        const IndexType KStride = Y_ * X_ * C_;
        const IndexType CStride = 1; // C innermost

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
        {
            // K_gemm = C * 9 (compile-time), M = Gm * K (runtime)
            return make_naive_tensor_descriptor(
                make_tuple(K_ * NumGroupsToMerge, C_ * number<9>{}),
                make_tuple(C_ * number<9>{}, CStride),
                number<VectorSizeA>{}, I1);
        }
        else
        {
            // General: A[Gm*K, Y*X*C] — rows are contiguous in GKYXC memory.
            return make_naive_tensor_descriptor(
                make_tuple(K_ * NumGroupsToMerge, Y_ * X_ * C_),
                make_tuple(KStride, CStride),
                number<VectorSizeA>{}, I1);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // B descriptors: Input I → B[N=Gm×N×Ho×Wo, K_gemm]
    // ─────────────────────────────────────────────────────────────────────

    // ── GNCHW (channels-first input, NumGroupsToMerge=1 only) ─────────
    template <typename BLayout,
              typename std::enable_if<
                  NDimSpatial == 2 &&
                      std::is_same_v<BLayout, tensor_layout::convolution::GNCHW>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeBDescriptor_N_K() const
    {
        static_assert(NumGroupsToMerge == 1,
                      "GNCHW input only supports NumGroupsToMerge==1. "
                      "Use NHWGC for group merging.");
        const IndexType NStride  = C_ * Hi_ * Wi_;
        const IndexType CStride  = Hi_ * Wi_;
        const IndexType HiStride = Wi_;
        const IndexType WiStride = 1;

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            const auto in_n_c_ho_wo_desc = make_naive_tensor_descriptor(
                make_tuple(N_, C_, Ho_, Wo_),
                make_tuple(NStride, CStride, HiStride, WiStride),
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
            const auto in_n_c_hi_wi_desc = make_naive_tensor_descriptor(
                make_tuple(N_, C_, Hi_, Wi_),
                make_tuple(NStride, CStride, HiStride, WiStride),
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
                           make_embed_transform(make_tuple(number<3>{}, Ho_),
                                               make_tuple(ConvDilationH_, ConvStrideH_)),
                           make_embed_transform(make_tuple(number<3>{}, Wo_),
                                               make_tuple(ConvDilationW_, ConvStrideW_))),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4, 5>{}));
            return transform_tensor_descriptor(
                in_n_c_y_ho_x_wo_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                           make_merge_transform(make_tuple(C_, number<3>{}, number<3>{}))),
                make_tuple(sequence<0, 3, 5>{}, sequence<1, 2, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Pad0)
        {
            const auto in_n_c_hi_wi_desc = make_naive_tensor_descriptor(
                make_tuple(N_, C_, Hi_, Wi_),
                make_tuple(NStride, CStride, HiStride, WiStride),
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
            const auto in_n_c_hi_wi_desc = make_naive_tensor_descriptor(
                make_tuple(N_, C_, Hi_, Wi_),
                make_tuple(NStride, CStride, HiStride, WiStride),
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

    // ── NHWGC (channels-last input, NumGroupsToMerge>=1) ──────────────
    // K_gemm = merge([Y, X, C]): C innermost — consistent with GKYXC A.
    //
    // For Gm=1: B[N=N×Ho×Wo, K_gemm=Y×X×C]
    //   Physical I[N, Hi, Wi, C] with full NHWGC strides (G not stripped).
    //
    // For Gm>1: B[N=Gm×N×Ho×Wo, K_gemm=Y×X×C]
    //   Physical I[N, Hi, Wi, Gm, C] with G_local stride = C (NHWGC adjacent).
    //   merge([N, Ho, Wo, Gm]) → N = Gm*N*Ho*Wo.
    template <typename BLayout,
              typename std::enable_if<
                  NDimSpatial == 2 &&
                      std::is_same_v<BLayout, tensor_layout::convolution::NHWGC>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeBDescriptor_N_K() const
    {
        // Full NHWGC strides for I[N, Hi, Wi, G, C].
        const IndexType NStride  = Hi_ * Wi_ * G_ * C_;
        const IndexType HiStride = Wi_ * G_ * C_;
        const IndexType WiStride = G_ * C_;
        const IndexType CStride  = 1;  // C innermost
        const IndexType GStride  = C_; // stride of G in NHWGC = C

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Ho_, Wo_, C_),
                    make_tuple(NStride, HiStride, WiStride, CStride),
                    number<VectorSizeB>{}, I1);
                return transform_tensor_descriptor(
                    in_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0, 1, 2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else
            {
                const auto in_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Ho_, Wo_, NumGroupsToMerge, C_),
                    make_tuple(NStride, HiStride, WiStride, GStride, CStride),
                    number<VectorSizeB>{}, I1);
                // N_gemm = merge([Gm, N, Ho, Wo]): Gm OUTERMOST, consistent with C descriptor.
                return transform_tensor_descriptor(
                    in_desc,
                    make_tuple(make_merge_transform(make_tuple(NumGroupsToMerge, N_, Ho_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<3, 0, 1, 2>{}, sequence<4>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_hi_wi_c = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, C_),
                    make_tuple(NStride, HiStride, WiStride, CStride),
                    number<VectorSizeB>{}, I1);
                const auto in_padded = transform_tensor_descriptor(
                    in_n_hi_wi_c,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));
                const auto in_embedded = transform_tensor_descriptor(
                    in_padded,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(number<3>{}, Ho_),
                                                   make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(number<3>{}, Wo_),
                                                   make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{}, sequence<5>{}));
                return transform_tensor_descriptor(
                    in_embedded,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                               make_merge_transform(make_tuple(number<3>{}, number<3>{}, C_))),
                    make_tuple(sequence<0, 2, 4>{}, sequence<1, 3, 5>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else
            {
                // Gm>1: I[N, Hi, Wi, Gm, C] — Gm local groups, stride(Gm)=C.
                const auto in_n_hi_wi_gm_c = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(NStride, HiStride, WiStride, GStride, CStride),
                    number<VectorSizeB>{}, I1);
                const auto in_padded = transform_tensor_descriptor(
                    in_n_hi_wi_gm_c,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{},
                               sequence<3>{}, sequence<4>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{},
                               sequence<3>{}, sequence<4>{}));
                const auto in_embedded = transform_tensor_descriptor(
                    in_padded,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(number<3>{}, Ho_),
                                                   make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(number<3>{}, Wo_),
                                                   make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{},
                               sequence<3>{}, sequence<4>{}),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{},
                               sequence<5>{}, sequence<6>{}));
                // N_gemm = merge([Gm, N, Ho, Wo]): Gm OUTERMOST, consistent with C descriptor.
                // K_gemm = merge([Y=3, X=3, C]): C innermost.
                // After embed: [N=0, Y=1, Ho=2, X=3, Wo=4, Gm=5, C=6]
                return transform_tensor_descriptor(
                    in_embedded,
                    make_tuple(make_merge_transform(make_tuple(NumGroupsToMerge, N_, Ho_, Wo_)),
                               make_merge_transform(make_tuple(number<3>{}, number<3>{}, C_))),
                    make_tuple(sequence<5, 0, 2, 4>{}, sequence<1, 3, 6>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, C_),
                    make_tuple(NStride, HiStride, WiStride, CStride),
                    number<VectorSizeB>{}, I1);
                const auto in_strided = transform_tensor_descriptor(
                    in_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Ho_), make_tuple(ConvStrideH_)),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));
                return transform_tensor_descriptor(
                    in_strided,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0, 1, 2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else
            {
                const auto in_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(NStride, HiStride, WiStride, GStride, CStride),
                    number<VectorSizeB>{}, I1);
                const auto in_strided = transform_tensor_descriptor(
                    in_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Ho_), make_tuple(ConvStrideH_)),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{},
                               sequence<3>{}, sequence<4>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{},
                               sequence<3>{}, sequence<4>{}));
                // N_gemm = merge([Gm, N, Ho, Wo]): Gm OUTERMOST.
                return transform_tensor_descriptor(
                    in_strided,
                    make_tuple(make_merge_transform(make_tuple(NumGroupsToMerge, N_, Ho_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<3, 0, 1, 2>{}, sequence<4>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
        else
        {
            // General case.
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_hi_wi_c = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, C_),
                    make_tuple(NStride, HiStride, WiStride, CStride),
                    number<VectorSizeB>{}, I1);
                const auto in_padded = transform_tensor_descriptor(
                    in_n_hi_wi_c,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));
                const auto in_embedded = transform_tensor_descriptor(
                    in_padded,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Y_, Ho_),
                                                   make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                   make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{}, sequence<5>{}));
                return transform_tensor_descriptor(
                    in_embedded,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                               make_merge_transform(make_tuple(Y_, X_, C_))),
                    make_tuple(sequence<0, 2, 4>{}, sequence<1, 3, 5>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else
            {
                const auto in_n_hi_wi_gm_c = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(NStride, HiStride, WiStride, GStride, CStride),
                    number<VectorSizeB>{}, I1);
                const auto in_padded = transform_tensor_descriptor(
                    in_n_hi_wi_gm_c,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{},
                               sequence<3>{}, sequence<4>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{},
                               sequence<3>{}, sequence<4>{}));
                const auto in_embedded = transform_tensor_descriptor(
                    in_padded,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Y_, Ho_),
                                                   make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                   make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{},
                               sequence<3>{}, sequence<4>{}),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{},
                               sequence<5>{}, sequence<6>{}));
                // N_gemm = merge([Gm, N, Ho, Wo]): Gm OUTERMOST.
                // K_gemm = merge([Y, X, C]): C innermost.
                // After embed: [N=0, Y=1, Ho=2, X=3, Wo=4, Gm=5, C=6]
                return transform_tensor_descriptor(
                    in_embedded,
                    make_tuple(make_merge_transform(make_tuple(NumGroupsToMerge, N_, Ho_, Wo_)),
                               make_merge_transform(make_tuple(Y_, X_, C_))),
                    make_tuple(sequence<5, 0, 2, 4>{}, sequence<1, 3, 6>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // C descriptor: Output O → C[M=K*Gm, N=Gm×N×Ho×Wo] (NHWGK)
    // ─────────────────────────────────────────────────────────────────────
    //
    // For NumGroupsToMerge=1:  simple C[K, N×Ho×Wo] with M=K unit-stride.
    //
    // For NumGroupsToMerge=Gm: XOR-diagonal trick (same physical layout,
    //   M and N swap roles vs im2col because true im2win has M=K, not M=spatial).
    //
    //   Steps (adapted from transform_conv_fwd_to_gemm.hpp):
    //   1. 6D physical [N, Ho, Wo, Gm, K, 1] strides [N_s, Ho_s, Wo_s, K, 1, K]
    //   2. Merge spatial [N,Ho,Wo]→NDoHoWo; pad dummy [1→Gm]
    //      → [NDoHoWo, Gm, K, Gm_padded]
    //   3. XOR on (Gm, Gm_padded): ensures g_M==g_N entries are the valid ones
    //      → [NDoHoWo, Gm_xor1, K, Gm_xor2]
    //   4. (im2win) M = merge([Gm_xor2, K]) from dims {3,2}
    //              N = merge([Gm_xor1, NDoHoWo]) from dims {1,0}
    //      (im2col would have been: M = merge([NDoHoWo, Gm_xor1]),
    //                               N = merge([K, Gm_xor2]))
    //
    // Physical offset for valid (g_M==g_N=g):
    //   spatial*(G*K) + g*K + k  ✓
    template <typename CLayout,
              typename std::enable_if<
                  NDimSpatial == 2 &&
                      std::is_same_v<CLayout, tensor_layout::convolution::NHWGK>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeCDescriptor_M_N() const
    {
        const IndexType NStride  = Ho_ * Wo_ * G_ * K_;
        const IndexType HoStride = Wo_ * G_ * K_;
        const IndexType WoStride = G_ * K_;
        const IndexType KStride  = 1;   // K innermost
        const IndexType GStride  = K_;  // stride(G) in NHWGK = K

        if constexpr(NumGroupsToMerge == 1)
        {
            // Simple: C[M=K, N=N×Ho×Wo], K unit-stride.
            const auto out_n_ho_wo_k_desc = make_naive_tensor_descriptor(
                make_tuple(N_, Ho_, Wo_, K_),
                make_tuple(NStride, HoStride, WoStride, KStride),
                number<VectorSizeC>{}, I1);
            return transform_tensor_descriptor(
                out_n_ho_wo_k_desc,
                make_tuple(make_pass_through_transform(K_),
                           make_merge_transform(make_tuple(N_, Ho_, Wo_))),
                make_tuple(sequence<3>{}, sequence<0, 1, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else
        {
            // XOR-diagonal for merged groups (Gm must be a power of 2).
            static_assert(NumGroupsToMerge == 2  || NumGroupsToMerge == 4  ||
                          NumGroupsToMerge == 8  || NumGroupsToMerge == 16 ||
                          NumGroupsToMerge == 32 || NumGroupsToMerge == 64,
                          "NumGroupsToMerge must be a power of 2 for group merging.");

            const IndexType NDoHoWo = N_ * Ho_ * Wo_;

            // Step 1: 6D [N, Ho, Wo, Gm, K, 1] — the dummy dim (size 1) has stride K.
            const auto n_ho_wo_gm_k_1_desc = make_naive_tensor_descriptor(
                make_tuple(N_, Ho_, Wo_, NumGroupsToMerge, K_, 1),
                make_tuple(NStride, HoStride, WoStride, GStride, KStride, GStride),
                number<VectorSizeC>{}, I1);

            // Step 2: merge spatial → NDoHoWo; pad dummy from 1 → Gm.
            const auto nhw_gm_k_gmpad_desc = transform_tensor_descriptor(
                n_ho_wo_gm_k_1_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                           make_pass_through_transform(NumGroupsToMerge),
                           make_pass_through_transform(K_),
                           make_pad_transform(1, 0, NumGroupsToMerge - 1)),
                make_tuple(sequence<0, 1, 2>{}, sequence<3>{}, sequence<4>{}, sequence<5>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));
            // → [NDoHoWo=0, Gm=1, K=2, Gm_padded=3]

            // Step 3: XOR on (Gm=dim1, Gm_padded=dim3).
            const auto nhw_gxor1_k_gxor2_desc = transform_tensor_descriptor(
                nhw_gm_k_gmpad_desc,
                make_tuple(make_pass_through_transform(NDoHoWo),
                           make_xor_transform(make_tuple(NumGroupsToMerge, NumGroupsToMerge)),
                           make_pass_through_transform(K_)),
                make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}));
            // → [NDoHoWo=0, Gm_xor1=1, K=2, Gm_xor2=3]

            // Step 4 (im2win): M = merge([Gm_xor2, K]), N = merge([Gm_xor1, NDoHoWo]).
            //   m = g_M * K + k   (Gm_xor2 outer, K inner)
            //   n = g_N * NDoHoWo + spatial  (Gm_xor1 outer, NDoHoWo inner)
            return transform_tensor_descriptor(
                nhw_gxor1_k_gxor2_desc,
                make_tuple(make_merge_transform(make_tuple(NumGroupsToMerge, K_)),
                           make_merge_transform(make_tuple(NumGroupsToMerge, NDoHoWo))),
                make_tuple(sequence<3, 2>{}, sequence<1, 0>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    // ── C descriptor: Output O → C[M=K, N=N×Ho×Wo] (GNKHW) ──────────────
    //
    // Channels-first output layout: O[G, N, K, Ho, Wo] (group stripped).
    // In GNKHW, K has stride Ho×Wo (not unit), so vectorised epilogue stores
    // are along the N_gemm dimension (Wo innermost, stride 1).
    //
    // Descriptor mapping (c_ptr = out_ptr + g * N*K*Ho*Wo):
    //   C[m=k, n=n*Ho*Wo+ho*Wo+wo] → physical = n*(K*Ho*Wo) + k*(Ho*Wo) + ho*Wo + wo
    //   stride(M=k)     = Ho*Wo      ← non-unit (K not innermost in GNKHW)
    //   stride(N=spatial) = 1 for Wo (innermost of the merged spatial dims)
    //
    // Note: For NumGroupsToMerge=1 only.
    template <typename CLayout,
              typename std::enable_if<
                  NDimSpatial == 2 &&
                      std::is_same_v<CLayout, tensor_layout::convolution::GNKHW>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeCDescriptor_M_N() const
    {
        static_assert(NumGroupsToMerge == 1,
                      "GNKHW output only supports NumGroupsToMerge==1.");

        // Physical strides for O[N, K, Ho, Wo] (NKHW order, group stripped).
        const IndexType NStride  = K_ * Ho_ * Wo_;
        const IndexType KStride  = Ho_ * Wo_; // stride(M=K) — non-unit
        const IndexType HoStride = Wo_;
        const IndexType WoStride = 1; // unit stride (innermost of N merge)

        const auto out_n_k_ho_wo_desc = make_naive_tensor_descriptor(
            make_tuple(N_, K_, Ho_, Wo_),
            make_tuple(NStride, KStride, HoStride, WoStride),
            number<VectorSizeC>{},
            I1);

        // C[M=K, N=N×Ho×Wo]:
        //   M = K → dimension 1 (KStride = Ho*Wo)
        //   N = merge(N=dim0, Ho=dim2, Wo=dim3) with Wo innermost (stride 1)
        return transform_tensor_descriptor(
            out_n_k_ho_wo_desc,
            make_tuple(make_pass_through_transform(K_),
                       make_merge_transform(make_tuple(N_, Ho_, Wo_))),
            make_tuple(sequence<1>{}, sequence<0, 2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    // ─────────────────────────────────────────────────────────────────────
    // GEMM dimension queries
    // ─────────────────────────────────────────────────────────────────────

    /// M = K * NumGroupsToMerge  (output channels × merged groups)
    CK_TILE_HOST IndexType GetGemmM() const { return K_ * NumGroupsToMerge; }

    /// N = N × Ho × Wo × NumGroupsToMerge  (spatial × merged groups)
    CK_TILE_HOST IndexType GetGemmN() const { return N_ * Ho_ * Wo_ * NumGroupsToMerge; }

    /// K_gemm = C × Y × X  (unchanged by group merging)
    CK_TILE_HOST IndexType GetGemmK() const { return C_ * Y_ * X_; }

    /// GemmBatch = G / NumGroupsToMerge
    CK_TILE_HOST IndexType GetGemmBatch() const
    {
        return integer_divide_ceil(G_, NumGroupsToMerge);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Per-single-group strides (kernel args multiply by NumGroupsToMerge)
    // ─────────────────────────────────────────────────────────────────────

    /// Weight stride per group in GKCYX/GKYXC: K*C*Y*X elements.
    CK_TILE_HOST long_index_t GetGroupStrideA() const
    {
        return static_cast<long_index_t>(K_) * C_ * Y_ * X_;
    }

    /// Input stride per group in GNCHW: N*C*Hi*Wi elements.
    CK_TILE_HOST long_index_t GetGroupStrideB() const
    {
        return static_cast<long_index_t>(original_N_) * C_ * Hi_ * Wi_;
    }

    /// Input stride per group in NHWGC: C elements (stride of G dimension).
    CK_TILE_HOST long_index_t GetGroupStrideBNhwgc() const
    {
        return static_cast<long_index_t>(C_);
    }

    /// Output stride per group in NHWGK: K elements (stride of G dimension).
    CK_TILE_HOST long_index_t GetGroupStrideC() const
    {
        return static_cast<long_index_t>(K_);
    }

    /// Output stride per group in GNKHW: N*K*Ho*Wo elements (G is outermost).
    CK_TILE_HOST long_index_t GetGroupStrideCGnkhw() const
    {
        return static_cast<long_index_t>(original_N_) * K_ * Ho_ * Wo_;
    }

    // ─────────────────────────────────────────────────────────────────────
    IndexType G_{};
    IndexType N_{};
    IndexType original_N_{};
    IndexType C_{};
    IndexType Hi_{};
    IndexType Wi_{};
    IndexType K_{};
    IndexType Ho_{};
    IndexType Wo_{};
    IndexType Y_{};
    IndexType X_{};
    IndexType ConvStrideH_{};
    IndexType ConvStrideW_{};
    IndexType ConvDilationH_{};
    IndexType ConvDilationW_{};
    IndexType InLeftPadH_{};
    IndexType InLeftPadW_{};
    IndexType InRightPadH_{};
    IndexType InRightPadW_{};
};

} // namespace ck_tile
