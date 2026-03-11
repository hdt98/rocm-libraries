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
// Implements the im2win (image-to-window) transformation for forward
// grouped convolution as described in:
//   "Im2Win: Memory-Efficient Convolution on GPUs" (arXiv:2306.14316)
//
// Problem layout (channels-first):
//   Input  I[G, N, C, Hi, Wi]  — tensor I
//   Weight W[G, K, C,  Y,  X]  — tensor W
//   Output O[G, N, K, Ho, Wo]  — tensor O
//
// im2win tensor I' definition (from the paper, equation 1):
//   I'(g, i, r, m, k*Y + u) = I(g, i, r, m + u*Sy, k + v*Sx)
//
// where:
//   g  ∈ [0, G-1]     — group index
//   i  ∈ [0, N-1]     — batch index
//   r  ∈ [0, C-1]     — input channel index
//   m  ∈ [0, Ho-1]    — output row (y) index
//   k  ∈ [0, Wo-1]    — output column (x) index  (NOTE: called 'n' in paper)
//   u  ∈ [0, Y-1]     — filter row index
//   v  ∈ [0, X-1]     — filter column index  (implicit in k*Y+u encoding)
//
// The key difference from im2col is which dimension is "outer":
//   im2col: A[M=N×Ho×Wo, K_gemm=C×Y×X]   — outer dim is output spatial
//   im2win: A[M=N×Ho×Wo, K_gemm=C×Y×X]   — same M/K shape, but the
//           index mapping groups rows sharing the same input column k
//           (Wo position), rather than grouping by filter window.
//
// GEMM problem dimensions:
//   M      = N × Ho × Wo   (A rows:    batch × output spatial)
//   N_gemm = K             (B cols:    output channels)
//   K_gemm = C × Y × X    (reduction: input channels × filter)
//
// Tensor descriptor outputs:
//   MakeADescriptor_M_K()  → A[M, K_gemm]  mapped from I[N, C, Hi, Wi]
//   MakeBDescriptor_N_K()  → B[N_gemm, K_gemm] = W[K, C×Y×X]
//   MakeCDescriptor_M_N()  → C[M, N_gemm] = O[N×Ho×Wo, K]
//
// Current scope: 2D convolution only (NDimSpatial == 2), targeting the
// GNCHW input layout (G, N, C, Hi, Wi) used in the problem statement.
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

    // ── Split-N helper ────────────────────────────────────────────────
    // Mirrors the same logic used in TransformConvFwdToGemm so that the
    // im2win kernel can also handle tensors larger than 2 GB.
    template <typename ConvDimsType>
    static IndexType GetSplitedNSize(const ConvDimsType& a_g_n_c_wis_lengths,
                                     const ConvDimsType& c_g_n_k_wos_lengths)
    {
        const index_t num_dims = a_g_n_c_wis_lengths.size();

        // Build contiguous strides for input and output tensors.
        ConvDimsType a_strides, c_strides;
        a_strides[num_dims - 1] = 1;
        c_strides[num_dims - 1] = 1;
        for(index_t i = num_dims - 2; i >= 0; i--)
        {
            a_strides[i] = a_strides[i + 1] * a_g_n_c_wis_lengths[i + 1];
            c_strides[i] = c_strides[i + 1] * c_g_n_k_wos_lengths[i + 1];
        }

        // Compute element-space size starting from dim 1 (skip group dim).
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
    // ── Public accessors for split-N ──────────────────────────────────
    CK_TILE_HOST constexpr IndexType GetN() const { return N_; }
    CK_TILE_HOST constexpr IndexType GetOriginalN() const { return original_N_; }

    // ── Default constructor ───────────────────────────────────────────
    CK_TILE_HOST constexpr TransformConvFwdToIm2win() {}

    // ── 2D constructor ────────────────────────────────────────────────
    // Argument layout follows the CK convention used by GroupedConvHostArgs:
    //   a_g_n_c_wis_lengths: [G, N, C, Hi, Wi]
    //   b_g_k_c_xs_lengths:  [G, K, C,  Y,  X]
    //   c_g_n_k_wos_lengths: [G, N, K, Ho, Wo]
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

    // ── A descriptor: Input I → A[M, K_gemm] ─────────────────────────
    //
    // Input tensor physical layout: I[N, C, Hi, Wi]
    // (Group dimension is stripped before this call — each group invocation
    //  receives a pointer offset by g * (N*C*Hi*Wi).)
    //
    // im2win mapping (unit stride, unit dilation, zero padding — the primary target):
    //
    //   I'(n, c, ho, wo, y, x) = I(n, c, ho*Sy + y*Dy, wo*Sx + x*Dx)
    //
    // Descriptor chain:
    //   Step 1 (physical):  I[N, C, Hi, Wi]
    //   Step 2 (pad):       I[N, C, Hi+LPadH+RPadH, Wi+LPadW+RPadW]
    //   Step 3 (embed):     I[N, C, Y, Ho, X, Wo]
    //                           hi = y*Dy + ho*Sy,  wi = x*Dx + wo*Sx
    //   Step 4 (merge):     A[M=N×Ho×Wo, K_gemm=C×Y×X]
    //
    // The embed transform re-indexes two output dims (filter, output-spatial)
    // back onto one padded input dim, which is exactly what the paper calls
    // "sliding window without materialising the buffer".
    //
    // Supported layout: GNCHW  (input = [N, C, Hi, Wi] after group strip)
    template <typename ALayout,
              typename std::enable_if<
                  NDimSpatial == 2 &&
                      std::is_same_v<ALayout, tensor_layout::convolution::GNCHW>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeADescriptor_M_K() const
    {
        // Physical strides for I[N, C, Hi, Wi] (NCHW order, no group dim here).
        const IndexType NStride  = C_ * Hi_ * Wi_;
        const IndexType CStride  = Hi_ * Wi_;
        const IndexType HiStride = Wi_;
        const IndexType WiStride = 1;

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            // ── 1×1, stride=1, no padding ────────────────────────────
            // For 1×1 filters with unit stride and no padding,
            // hi = ho, wi = wo.  The input collapses to I[N, C, Ho, Wo],
            // which is already in GEMM form.
            //
            //   A[M=N×Ho×Wo, K_gemm=C]
            const auto in_n_c_ho_wo_desc = make_naive_tensor_descriptor(
                make_tuple(N_, C_, Ho_, Wo_),
                make_tuple(NStride, CStride, HiStride, WiStride),
                number<VectorSizeA>{},
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
            // ── 3×3 filter (Y=X=3 known at compile time) ─────────────
            // Step 1: physical layout I[N, C, Hi, Wi].
            const auto in_n_c_hi_wi_desc = make_naive_tensor_descriptor(
                make_tuple(N_, C_, Hi_, Wi_),
                make_tuple(NStride, CStride, HiStride, WiStride),
                number<VectorSizeA>{},
                I1);

            // Step 2: pad Hi and Wi for filter footprint.
            const auto in_n_c_hip_wip_desc = transform_tensor_descriptor(
                in_n_c_hi_wi_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_pass_through_transform(C_),
                           make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                           make_pad_transform(Wi_, InLeftPadW_, InRightPadW_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

            // Step 3: embed Y,Ho onto padded Hi and X,Wo onto padded Wi.
            //   hi_padded = y * ConvDilationH + ho * ConvStrideH
            //   wi_padded = x * ConvDilationW + wo * ConvStrideW
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

            // Step 4: merge to GEMM shape A[M=N×Ho×Wo, K_gemm=C×Y×X].
            //   M     = N × Ho × Wo  (dims 0, 3, 5)
            //   K_gemm = C × Y  × X  (dims 1, 2, 4)
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
            // ── 1×1 filter, no padding ───────────────────────────────
            // hi = ho * ConvStrideH,  wi = wo * ConvStrideW.
            // No padding step needed.
            const auto in_n_c_hi_wi_desc = make_naive_tensor_descriptor(
                make_tuple(N_, C_, Hi_, Wi_),
                make_tuple(NStride, CStride, HiStride, WiStride),
                number<VectorSizeA>{},
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
            // ── General case (arbitrary Y, X, stride, dilation, padding) ──
            // Step 1: physical layout I[N, C, Hi, Wi].
            const auto in_n_c_hi_wi_desc = make_naive_tensor_descriptor(
                make_tuple(N_, C_, Hi_, Wi_),
                make_tuple(NStride, CStride, HiStride, WiStride),
                number<VectorSizeA>{},
                I1);

            // Step 2: pad.
            const auto in_n_c_hip_wip_desc = transform_tensor_descriptor(
                in_n_c_hi_wi_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_pass_through_transform(C_),
                           make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                           make_pad_transform(Wi_, InLeftPadW_, InRightPadW_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

            // Step 3: embed filter dims onto padded spatial dims.
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

            // Step 4: merge.
            //   M      = N × Ho × Wo  (dims 0, 3, 5)
            //   K_gemm = C × Y  × X   (dims 1, 2, 4)
            return transform_tensor_descriptor(
                in_n_c_y_ho_x_wo_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                           make_merge_transform(make_tuple(C_, Y_, X_))),
                make_tuple(sequence<0, 3, 5>{}, sequence<1, 2, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    // ── B descriptor: Weight W → B[N_gemm=K, K_gemm=C×Y×X] ──────────
    //
    // Physical layout: W[K, C, Y, X] (group dimension stripped by pointer
    // offset, as for the A descriptor).
    //
    // The B matrix is the weight tensor viewed as a 2-D matrix:
    //   B[N_gemm = K,  K_gemm = C × Y × X]
    //
    // For Filter3x3 specialisation the filter size is a compile-time
    // constant (9), which lets the compiler fold it into the descriptor.
    //
    // Supported layout: GKCYX  (weight = [K, C, Y, X] after group strip)
    template <
        typename BLayout,
        typename std::enable_if<std::is_same_v<BLayout, tensor_layout::convolution::GKCYX>,
                                bool>::type = false>
    CK_TILE_HOST auto MakeBDescriptor_N_K() const
    {
        // Physical strides for W[K, C, Y, X] (KCYX order).
        const IndexType KStride = C_ * Y_ * X_;
        const IndexType CStride = Y_ * X_;
        const IndexType YStride = X_;
        const IndexType XStride = 1;

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
        {
            // Filter size is known at compile time: Y=X=3, so Y×X=9.
            return make_naive_tensor_descriptor(
                make_tuple(K_, C_ * number<9>{}),
                make_tuple(KStride, XStride),
                number<VectorSizeB>{},
                I1);
        }
        else
        {
            // General: flatten C×Y×X at runtime.
            const auto wei_k_c_y_x_desc = make_naive_tensor_descriptor(
                make_tuple(K_, C_, Y_, X_),
                make_tuple(KStride, CStride, YStride, XStride),
                number<VectorSizeB>{},
                I1);

            return transform_tensor_descriptor(
                wei_k_c_y_x_desc,
                make_tuple(make_pass_through_transform(K_),
                           make_merge_transform(make_tuple(C_, Y_, X_))),
                make_tuple(sequence<0>{}, sequence<1, 2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    // ── C descriptor: Output O → C[M=N×Ho×Wo, N_gemm=K] ─────────────
    //
    // Physical layout: O[N, K, Ho, Wo] (GNKHW, group dimension stripped).
    //
    // GNKHW has K between N and the spatial dims.  We cannot directly merge
    // dims {0, 2, 3} of [N, K, Ho, Wo] because K (stride Ho×Wo) is
    // interleaved — the merge transform requires contiguous logical dimensions.
    //
    // Solution: two-step reindex.
    //   Step 1: view [N, K, Ho, Wo] as [N, K, Ho×Wo] by merging the two
    //           inner spatial dims.
    //   Step 2: transpose to [N, Ho×Wo, K] by reorder, then merge
    //           [N, Ho×Wo] → M and pass K → N_gemm.
    //
    //   physical[n, k, ho, wo] = n*(K*Ho*Wo) + k*(Ho*Wo) + ho*Wo + wo
    //   logical (m, k) where m = n*Ho*Wo + ho*Wo + wo
    //   → physical = (m / (Ho*Wo)) * K*Ho*Wo
    //              + k * Ho*Wo
    //              + (m % (Ho*Wo))
    //
    // Supported layout: GNKHW  (output = [N, K, Ho, Wo] after group strip)
    template <typename CLayout,
              typename std::enable_if<
                  NDimSpatial == 2 &&
                      std::is_same_v<CLayout, tensor_layout::convolution::GNKHW>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeCDescriptor_M_N() const
    {
        // Physical strides for O[N, K, Ho, Wo] (NKHW order).
        const IndexType NStride  = K_ * Ho_ * Wo_;
        const IndexType KStride  = Ho_ * Wo_;
        const IndexType HoStride = Wo_;
        const IndexType WoStride = 1;

        // Step 1: [N, K, Ho, Wo] — physical base descriptor.
        const auto out_n_k_ho_wo_desc = make_naive_tensor_descriptor(
            make_tuple(N_, K_, Ho_, Wo_),
            make_tuple(NStride, KStride, HoStride, WoStride),
            number<VectorSizeC>{},
            I1);

        // Step 2: merge the two inner spatial dims → [N, K, HoWo].
        const auto out_n_k_howo_desc = transform_tensor_descriptor(
            out_n_k_ho_wo_desc,
            make_tuple(make_pass_through_transform(N_),
                       make_pass_through_transform(K_),
                       make_merge_transform(make_tuple(Ho_, Wo_))),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

        // Step 3: merge [N, HoWo] → M, keep K as N_gemm.
        // Now dims are [N=0, K=1, HoWo=2].  Merge dims {0, 2} for M, dim {1} for N_gemm.
        // Physical offset = n*(K*HoWo) + k*(HoWo) + howo
        //                 = n*(K*Ho*Wo) + k*(Ho*Wo) + (ho*Wo + wo)  ✓
        return transform_tensor_descriptor(
            out_n_k_howo_desc,
            make_tuple(make_merge_transform(make_tuple(N_, Ho_ * Wo_)),
                       make_pass_through_transform(K_)),
            make_tuple(sequence<0, 2>{}, sequence<1>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    // ── C descriptor: Output O → C[M=N×Ho×Wo, N_gemm=K] (NHWGK) ─────
    //
    // Alternative channels-last output layout where K is innermost (stride 1).
    // This matches what the CShuffleEpilogue expects: the N_gemm (K) dimension
    // must be contiguous in memory for vectorised stores.
    //
    // Physical layout after group strip: O[N, Ho, Wo, K]  (NHWK order).
    // Strides: [Ho×Wo×K, Wo×K, K, 1] — K innermost.
    //
    // Supported layout: NHWGK  (output = [N, Ho, Wo, G, K], G merged into outer loop)
    template <typename CLayout,
              typename std::enable_if<
                  NDimSpatial == 2 &&
                      std::is_same_v<CLayout, tensor_layout::convolution::NHWGK>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeCDescriptor_M_N() const
    {
        // Strides for NHWGK = [N, Ho, Wo, G, K].
        // The group pointer offset handles the G dimension, but N must still
        // step over the full Ho×Wo×G×K stride in the global buffer.
        const IndexType NStride  = Ho_ * Wo_ * G_ * K_; // full N stride over G groups
        const IndexType HoStride = Wo_ * G_ * K_;
        const IndexType WoStride = G_ * K_;
        const IndexType KStride  = 1; // K innermost — vectorisable

        const auto out_n_ho_wo_k_desc = make_naive_tensor_descriptor(
            make_tuple(N_, Ho_, Wo_, K_),
            make_tuple(NStride, HoStride, WoStride, KStride),
            number<VectorSizeC>{},
            I1);

        // Merge (N, Ho, Wo) → M; K → N_gemm.  Dims {0,1,2} are contiguous here.
        return transform_tensor_descriptor(
            out_n_ho_wo_k_desc,
            make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                       make_pass_through_transform(K_)),
            make_tuple(sequence<0, 1, 2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    // ── GEMM dimension queries ────────────────────────────────────────

    /// M dimension = N × Ho × Wo
    CK_TILE_HOST IndexType GetGemmM() const { return N_ * Ho_ * Wo_; }

    /// N_gemm dimension = K (output channels per group)
    CK_TILE_HOST IndexType GetGemmN() const { return K_; }

    /// K_gemm dimension = C × Y × X (reduction dimension)
    CK_TILE_HOST IndexType GetGemmK() const { return C_ * Y_ * X_; }

    // ── Group (batch) queries ─────────────────────────────────────────
    // For grouped convolution the GEMM is run independently for each of
    // the G groups.  The kernel advances the input/weight/output pointers
    // by the strides below for every group index.

    /// Number of GEMM batch iterations = number of convolution groups G.
    CK_TILE_HOST IndexType GetGemmBatch() const { return G_; }

    /// Number of input elements per group in the GNCHW layout.
    /// Stride to add to the base input pointer to reach group g:
    ///   ptr_g = ptr_base + g * GetGroupStrideA()
    CK_TILE_HOST long_index_t GetGroupStrideA() const
    {
        return static_cast<long_index_t>(original_N_) * C_ * Hi_ * Wi_;
    }

    /// Number of weight elements per group in the GKCYX layout.
    CK_TILE_HOST long_index_t GetGroupStrideB() const
    {
        return static_cast<long_index_t>(K_) * C_ * Y_ * X_;
    }

    /// Number of output elements per group in the GNKHW layout.
    CK_TILE_HOST long_index_t GetGroupStrideC() const
    {
        return static_cast<long_index_t>(original_N_) * K_ * Ho_ * Wo_;
    }

    // ── Stored problem parameters ─────────────────────────────────────
    // All values are per-group (group offset applied externally).
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
