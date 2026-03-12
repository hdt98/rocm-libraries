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
// Implements the *true* im2win transformation for forward grouped
// convolution as described in:
//   "Im2Win: Memory-Efficient Convolution on GPUs" (arXiv:2306.14316)
//
// Problem layout (channels-first, GNCHW):
//   Input  I[G, N, C, Hi, Wi]  — tensor I
//   Weight W[G, K, C,  Y,  X]  — tensor W
//   Output O[G, N, K, Ho, Wo]  — tensor O (stored as NHWGK for efficiency)
//
// True im2win GEMM mapping  (paper, §3):
//   M      = K             (A rows:    output channels — small for our use case)
//   N_gemm = N × Ho × Wo  (B cols:    batch × output spatial — large)
//   K_gemm = C × Y × X    (reduction: input channels × filter)
//
//   C[M=K, N=N×Ho×Wo] = A[M=K, K_gemm] × B[N=N×Ho×Wo, K_gemm]ᵀ
//
// where:
//   A[k, kg] = W[k, c(kg), y(kg), x(kg)]            — weight (simple 2D)
//   B[m, kg] = I[n(m), c(kg), ho(m)*Sy+y(kg)*Dy-LPH,  — input (sliding
//                             wo(m)*Sx+x(kg)*Dx-LPW]      window, no buffer)
//
// Advantage over im2col for small K:
//   im2col has N_gemm = K → tiles waste (1 - K/N_Tile) of their N capacity.
//   im2win has M = K → the 4×64×16 MFMA instruction matches M=K=4 exactly,
//   eliminating M-dimension waste entirely.
//
// Tensor descriptor outputs:
//   MakeADescriptor_M_K<GKCYX>() → A[M=K, K_gemm=C×Y×X]  from W[K, C, Y, X]
//   MakeBDescriptor_N_K<GNCHW>() → B[N=N×Ho×Wo, K_gemm]  from I[N, C, Hi, Wi]
//   MakeCDescriptor_M_N<NHWGK>() → C[M=K, N=N×Ho×Wo]
//
// Scope: 2D convolution, GKCYX weight, GNCHW input, NHWGK output.
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
    static_assert(NumGroupsToMerge == 1,
                  "TransformConvFwdToIm2win currently supports NumGroupsToMerge == 1 only.");

    private:
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};
    static constexpr auto I3 = number<3>{};
    static constexpr auto I4 = number<4>{};

    // 2 GB memory threshold for split-N decisions
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
                {
                    if(N % d == 0)
                        return N / d;
                }
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
        {
            N_ = GetSplitedNSize(a_g_n_c_wis_lengths, c_g_n_k_wos_lengths);
        }
    }

    // ── A descriptor: Weight W → A[M=K, K_gemm] ─────────────────────
    //
    // Two weight layouts are supported, differing in K_gemm element ordering:
    //
    //   GKCYX (channels-first):  W[K, C, Y, X], K_gemm = merge([C, Y, X])
    //                            kg = c*(Y*X) + y*X + x  (C outermost)
    //                            Pair with GNCHW input.
    //
    //   GKYXC (channels-last):   W[K, Y, X, C], K_gemm = merge([Y, X, C])
    //                            kg = y*(X*C) + x*C + c  (C innermost, vectorisable)
    //                            Pair with NHWGC input.
    //
    // Both descriptors are simple 2D naive descriptors (no sliding window).
    // ─────────────────────────────────────────────────────────────────

    // ── A descriptor for GKCYX (channels-first weight) ───────────────
    template <typename ALayout,
              typename std::enable_if<std::is_same_v<ALayout, tensor_layout::convolution::GKCYX>,
                                      bool>::type = false>
    CK_TILE_HOST auto MakeADescriptor_M_K() const
    {
        // Physical strides for W[K, C, Y, X] (KCYX order, group stripped).
        const IndexType KStride = C_ * Y_ * X_;
        const IndexType CStride = Y_ * X_;
        const IndexType YStride = X_;
        const IndexType XStride = 1;

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
        {
            return make_naive_tensor_descriptor(
                make_tuple(K_, C_ * number<9>{}),
                make_tuple(KStride, XStride),
                number<VectorSizeA>{},
                I1);
        }
        else
        {
            const auto wei_k_c_y_x_desc = make_naive_tensor_descriptor(
                make_tuple(K_, C_, Y_, X_),
                make_tuple(KStride, CStride, YStride, XStride),
                number<VectorSizeA>{},
                I1);

            return transform_tensor_descriptor(
                wei_k_c_y_x_desc,
                make_tuple(make_pass_through_transform(K_),
                           make_merge_transform(make_tuple(C_, Y_, X_))),
                make_tuple(sequence<0>{}, sequence<1, 2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    // ── A descriptor for GKYXC (channels-last weight) ─────────────────
    // K_gemm = merge([Y, X, C]): C innermost → vectorised A loads along C.
    // Use with NHWGC input (MakeBDescriptor_N_K<NHWGC>).
    template <typename ALayout,
              typename std::enable_if<std::is_same_v<ALayout, tensor_layout::convolution::GKYXC>,
                                      bool>::type = false>
    CK_TILE_HOST auto MakeADescriptor_M_K() const
    {
        // Physical strides for W[K, Y, X, C] (KYXC order, group stripped).
        const IndexType KStride = Y_ * X_ * C_;
        const IndexType YStride = X_ * C_;
        const IndexType XStride = C_;
        const IndexType CStride = 1; // C innermost

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
        {
            // K_gemm = C * 9  (Y=X=3 at compile time, C innermost)
            return make_naive_tensor_descriptor(
                make_tuple(K_, C_ * number<9>{}),
                make_tuple(KStride, CStride),
                number<VectorSizeA>{},
                I1);
        }
        else
        {
            const auto wei_k_y_x_c_desc = make_naive_tensor_descriptor(
                make_tuple(K_, Y_, X_, C_),
                make_tuple(KStride, YStride, XStride, CStride),
                number<VectorSizeA>{},
                I1);

            return transform_tensor_descriptor(
                wei_k_y_x_c_desc,
                make_tuple(make_pass_through_transform(K_),
                           make_merge_transform(make_tuple(Y_, X_, C_))),
                make_tuple(sequence<0>{}, sequence<1, 2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    // ── B descriptor: Input I → B[N=N×Ho×Wo, K_gemm=C×Y×X] ──────────
    //
    // In true im2win, the im2win-transformed input is the B matrix.
    // Physical layout: I[N, C, Hi, Wi] (GNCHW, group stripped).
    //
    // The sliding-window descriptor chain:
    //   Step 1 (physical):  I[N, C, Hi, Wi]
    //   Step 2 (pad):       I[N, C, Hi+LPH+RPH, Wi+LPW+RPW]
    //   Step 3 (embed):     I[N, C, Y, Ho, X, Wo]
    //                           hi = y*Dy + ho*Sy,  wi = x*Dx + wo*Sx
    //   Step 4 (merge):     B[N=N×Ho×Wo, K_gemm=C×Y×X]
    //
    // This is the same descriptor chain used in im2col for the A matrix,
    // just with M renamed to N (the B role in true im2win).
    //
    // Supported layout: GNCHW
    template <typename BLayout,
              typename std::enable_if<
                  NDimSpatial == 2 &&
                      std::is_same_v<BLayout, tensor_layout::convolution::GNCHW>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeBDescriptor_N_K() const
    {
        // Physical strides for I[N, C, Hi, Wi] (NCHW order, group stripped).
        const IndexType NStride  = C_ * Hi_ * Wi_;
        const IndexType CStride  = Hi_ * Wi_;
        const IndexType HiStride = Wi_;
        const IndexType WiStride = 1;

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            const auto in_n_c_ho_wo_desc = make_naive_tensor_descriptor(
                make_tuple(N_, C_, Ho_, Wo_),
                make_tuple(NStride, CStride, HiStride, WiStride),
                number<VectorSizeB>{},
                I1);

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
                number<VectorSizeB>{},
                I1);

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
                           make_merge_transform(
                               make_tuple(C_, number<3>{}, number<3>{}))),
                make_tuple(sequence<0, 3, 5>{}, sequence<1, 2, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Pad0)
        {
            const auto in_n_c_hi_wi_desc = make_naive_tensor_descriptor(
                make_tuple(N_, C_, Hi_, Wi_),
                make_tuple(NStride, CStride, HiStride, WiStride),
                number<VectorSizeB>{},
                I1);

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
            // General case.
            const auto in_n_c_hi_wi_desc = make_naive_tensor_descriptor(
                make_tuple(N_, C_, Hi_, Wi_),
                make_tuple(NStride, CStride, HiStride, WiStride),
                number<VectorSizeB>{},
                I1);

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

    // ── B descriptor: Input I → B[N=N×Ho×Wo, K_gemm] (NHWGC) ─────────
    //
    // Channels-last input layout.  In NHWGC, the G dimension is adjacent
    // to C (innermost), so the group pointer offset is just g_base * C.
    // This enables group merging analogous to im2col's NHWGC support.
    //
    // K_gemm = merge([Y, X, C]):  C innermost → vectorised B loads.
    // Use with MakeADescriptor_M_K<GKYXC>().
    //
    // Physical layout after group strip: I[N, Hi, Wi, C] with strides
    //   [Hi*Wi*G*C,  Wi*G*C,  G*C,  1]
    // (G is NOT stripped from the strides — b_ptr = in_ptr + g_base*C
    //  handles the group offset, but N still strides over the full G*C.)
    //
    // Supported layout: NHWGC
    template <typename BLayout,
              typename std::enable_if<
                  NDimSpatial == 2 &&
                      std::is_same_v<BLayout, tensor_layout::convolution::NHWGC>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeBDescriptor_N_K() const
    {
        // Full NHWGC strides for I[N, Hi, Wi, G, C].
        // b_ptr = in_ptr + g_base * C_ handles the group offset.
        const IndexType NStride  = Hi_ * Wi_ * G_ * C_;
        const IndexType HiStride = Wi_ * G_ * C_;
        const IndexType WiStride = G_ * C_;
        const IndexType CStride  = 1; // C innermost

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            const auto in_n_ho_wo_c_desc = make_naive_tensor_descriptor(
                make_tuple(N_, Ho_, Wo_, C_),
                make_tuple(NStride, HiStride, WiStride, CStride),
                number<VectorSizeB>{},
                I1);

            return transform_tensor_descriptor(
                in_n_ho_wo_c_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0, 1, 2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
        {
            // Step 1: I[N, Hi, Wi, C] with full NHWGC strides.
            const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N_, Hi_, Wi_, C_),
                make_tuple(NStride, HiStride, WiStride, CStride),
                number<VectorSizeB>{},
                I1);

            // Step 2: pad Hi, Wi.
            const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                           make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

            // Step 3: embed filter dims.
            const auto in_n_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                in_n_hip_wip_c_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_embed_transform(make_tuple(number<3>{}, Ho_),
                                               make_tuple(ConvDilationH_, ConvStrideH_)),
                           make_embed_transform(make_tuple(number<3>{}, Wo_),
                                               make_tuple(ConvDilationW_, ConvStrideW_)),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{}, sequence<5>{}));

            // Step 4: merge → B[N=N*Ho*Wo, K_gemm=Y*X*C] (C innermost).
            return transform_tensor_descriptor(
                in_n_y_ho_x_wo_c_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                           make_merge_transform(make_tuple(number<3>{}, number<3>{}, C_))),
                make_tuple(sequence<0, 2, 4>{}, sequence<1, 3, 5>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Pad0)
        {
            const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N_, Hi_, Wi_, C_),
                make_tuple(NStride, HiStride, WiStride, CStride),
                number<VectorSizeB>{},
                I1);

            const auto in_n_ho_wo_c_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_embed_transform(make_tuple(Ho_), make_tuple(ConvStrideH_)),
                           make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

            return transform_tensor_descriptor(
                in_n_ho_wo_c_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0, 1, 2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else
        {
            // General case.
            const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N_, Hi_, Wi_, C_),
                make_tuple(NStride, HiStride, WiStride, CStride),
                number<VectorSizeB>{},
                I1);

            const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                           make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

            const auto in_n_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                in_n_hip_wip_c_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_embed_transform(make_tuple(Y_, Ho_),
                                               make_tuple(ConvDilationH_, ConvStrideH_)),
                           make_embed_transform(make_tuple(X_, Wo_),
                                               make_tuple(ConvDilationW_, ConvStrideW_)),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{}, sequence<5>{}));

            return transform_tensor_descriptor(
                in_n_y_ho_x_wo_c_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                           make_merge_transform(make_tuple(Y_, X_, C_))),
                make_tuple(sequence<0, 2, 4>{}, sequence<1, 3, 5>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    // ── C descriptor: Output O → C[M=K, N=N×Ho×Wo] ───────────────────
    //
    // In true im2win, the output C matrix has M=K (output channels) as
    // rows and N=N×Ho×Wo (spatial positions) as columns.
    //
    // Physical layout: O[N, Ho, Wo, G, K] (NHWGK, group stripped).
    // In NHWGK, K is innermost (stride 1) — enables vectorised epilogue
    // stores along the M=K dimension.
    //
    // Descriptor mapping (from c_ptr = output_ptr + g_base*K):
    //   C[m=k, n=n*Ho*Wo+ho*Wo+wo] → physical = n*(Ho*Wo*G*K) + ho*(Wo*G*K)
    //                                           + wo*(G*K) + k
    //   stride(M=k)      = 1      ← unit stride, vectorisable ✓
    //   stride(N=spatial) = G*K
    //
    // Supported layout: NHWGK
    template <typename CLayout,
              typename std::enable_if<
                  NDimSpatial == 2 &&
                      std::is_same_v<CLayout, tensor_layout::convolution::NHWGK>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeCDescriptor_M_N() const
    {
        // Full NHWGK strides for O[N, Ho, Wo, G, K].
        // c_ptr already points to g_base*K, so the G offset is handled externally.
        const IndexType NStride  = Ho_ * Wo_ * G_ * K_;
        const IndexType HoStride = Wo_ * G_ * K_;
        const IndexType WoStride = G_ * K_;
        const IndexType KStride  = 1; // K innermost — unit stride

        // Physical 4D descriptor for the (N_, Ho_, Wo_, K_) slice.
        const auto out_n_ho_wo_k_desc = make_naive_tensor_descriptor(
            make_tuple(N_, Ho_, Wo_, K_),
            make_tuple(NStride, HoStride, WoStride, KStride),
            number<VectorSizeC>{},
            I1);

        // Map to GEMM shape [M=K, N=N×Ho×Wo]:
        //   M = K   → dimension 3 (innermost in NHWGK)
        //   N = merge(N, Ho, Wo) → dimensions 0, 1, 2
        return transform_tensor_descriptor(
            out_n_ho_wo_k_desc,
            make_tuple(make_pass_through_transform(K_),
                       make_merge_transform(make_tuple(N_, Ho_, Wo_))),
            make_tuple(sequence<3>{}, sequence<0, 1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    // ── GEMM dimension queries ────────────────────────────────────────

    /// M dimension = K (output channels per group — small for target use case)
    CK_TILE_HOST IndexType GetGemmM() const { return K_; }

    /// N_gemm dimension = N × Ho × Wo (batch × output spatial — large)
    CK_TILE_HOST IndexType GetGemmN() const { return N_ * Ho_ * Wo_; }

    /// K_gemm dimension = C × Y × X (reduction dimension — unchanged)
    CK_TILE_HOST IndexType GetGemmK() const { return C_ * Y_ * X_; }

    // ── Group (batch) queries ─────────────────────────────────────────

    /// Number of GEMM batch iterations = number of convolution groups G.
    CK_TILE_HOST IndexType GetGemmBatch() const { return G_; }

    /// Stride to advance the A (weight) pointer by one group.
    /// GKCYX layout: one group's weight occupies K × C × Y × X elements.
    CK_TILE_HOST long_index_t GetGroupStrideA() const
    {
        return static_cast<long_index_t>(K_) * C_ * Y_ * X_;
    }

    /// Stride to advance the B (input) pointer by one group.
    /// GNCHW layout: one group's input occupies original_N × C × Hi × Wi elements.
    CK_TILE_HOST long_index_t GetGroupStrideB() const
    {
        return static_cast<long_index_t>(original_N_) * C_ * Hi_ * Wi_;
    }

    /// Stride to advance the C (output) pointer by one group.
    /// NHWGK layout: stride(G) = K (G is adjacent to K in memory).
    CK_TILE_HOST long_index_t GetGroupStrideC() const
    {
        return static_cast<long_index_t>(K_);
    }

    // ── Stored problem parameters ─────────────────────────────────────
    IndexType G_{};
    IndexType N_{};          ///< Current N (may be reduced by split-N)
    IndexType original_N_{}; ///< Original batch size before any split
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
