// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/core/utility/env.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp"
#include "ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_im2win.hpp"

namespace ck_tile {

// ═══════════════════════════════════════════════════════════════════════
// GroupedConvFwdIm2winKernelArgs
// ═══════════════════════════════════════════════════════════════════════
//
// Device-side kernel arguments for im2win forward convolution.
// Mirrors GroupedConvFwdKernelArgs but uses TransformConvFwdToIm2win
// to build the A, B, and C tensor descriptors.
//
// Layout contract (channels-first):
//   Input  : GNCHW  (G, N, C, Hi, Wi)
//   Weight : GKCYX  (G, K, C,  Y,  X)
//   Output : GNKHW  (G, N, K, Ho, Wo)
// ═══════════════════════════════════════════════════════════════════════
template <typename GroupedConvTraitsType_, typename CDElementwise_>
struct GroupedConvFwdIm2winKernelArgs
{
    using ConvToIm2winTransformer =
        TransformConvFwdToIm2win<GroupedConvTraitsType_::NDimSpatial,
                                 GroupedConvTraitsType_::ConvSpecialization,
                                 GroupedConvTraitsType_::VectorSizeA,
                                 GroupedConvTraitsType_::VectorSizeB,
                                 GroupedConvTraitsType_::VectorSizeC,
                                 /*NumGroupsToMerge=*/1,
                                 /*SplitN=*/true>;
    using CDElementwise                 = CDElementwise_;
    static constexpr index_t NumDTensor = GroupedConvTraitsType_::NumDTensor;

    static constexpr index_t NonSpatialDims = 3;

    // ── Constructor for GNCHW / GKCYX / {GNKHW or NHWGK} layout ─────
    // NHWGK is preferred for output (K innermost → vectorised stores work).
    // GNKHW is kept for future use but currently produces strided K stores.
    template <
        typename InLay  = typename GroupedConvTraitsType_::InLayout,
        typename WeiLay = typename GroupedConvTraitsType_::WeiLayout,
        typename OutLay = typename GroupedConvTraitsType_::OutLayout,
        typename std::enable_if<
            std::is_same_v<InLay, tensor_layout::convolution::GNCHW> &&
                std::is_same_v<WeiLay, tensor_layout::convolution::GKCYX> &&
                (std::is_same_v<OutLay, tensor_layout::convolution::GNKHW> ||
                 std::is_same_v<OutLay, tensor_layout::convolution::NHWGK>),
            bool>::type = false>
    CK_TILE_HOST GroupedConvFwdIm2winKernelArgs(const GroupedConvFwdHostArgs<CDElementwise>& args)
        : elfunc(args.elfunc)
    {
        in_g_n_c_wis_lengths = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.N_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.input_spatial_lengths_[0]),
                                 static_cast<index_t>(args.input_spatial_lengths_[1])};
        wei_g_k_c_xs_lengths = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.K_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.filter_spatial_lengths_[0]),
                                 static_cast<index_t>(args.filter_spatial_lengths_[1])};
        out_g_n_k_wos_lengths = {static_cast<index_t>(args.G_),
                                  static_cast<index_t>(args.N_),
                                  static_cast<index_t>(args.K_),
                                  static_cast<index_t>(args.output_spatial_lengths_[0]),
                                  static_cast<index_t>(args.output_spatial_lengths_[1])};

        conv_filter_strides   = {static_cast<index_t>(args.conv_filter_strides_[0]),
                               static_cast<index_t>(args.conv_filter_strides_[1])};
        conv_filter_dilations = {static_cast<index_t>(args.conv_filter_dilations_[0]),
                                 static_cast<index_t>(args.conv_filter_dilations_[1])};
        input_left_pads       = {static_cast<index_t>(args.input_left_pads_[0]),
                           static_cast<index_t>(args.input_left_pads_[1])};
        input_right_pads      = {static_cast<index_t>(args.input_right_pads_[0]),
                            static_cast<index_t>(args.input_right_pads_[1])};

        k_batch = args.k_batch;

        in_ptr  = args.in_ptr;
        wei_ptr = args.wei_ptr;
        for(index_t d = 0; d < NumDTensor; d++)
        {
            ds_ptr[d] = args.ds_ptr[d];
        }
        out_ptr = args.out_ptr;

        // Build the im2win transformer and derive descriptors.
        transformer_ = ConvToIm2winTransformer{in_g_n_c_wis_lengths,
                                               wei_g_k_c_xs_lengths,
                                               out_g_n_k_wos_lengths,
                                               conv_filter_strides,
                                               conv_filter_dilations,
                                               input_left_pads,
                                               input_right_pads};

        a_grid_desc_m_k =
            transformer_.template MakeADescriptor_M_K<typename GroupedConvTraitsType_::InLayout>();
        b_grid_desc_n_k =
            transformer_.template MakeBDescriptor_N_K<typename GroupedConvTraitsType_::WeiLayout>();
        c_grid_desc_m_n =
            transformer_.template MakeCDescriptor_M_N<typename GroupedConvTraitsType_::OutLayout>();

        GemmM     = a_grid_desc_m_k.get_length(number<0>{});
        GemmN     = b_grid_desc_n_k.get_length(number<0>{});
        GemmK     = a_grid_desc_m_k.get_length(number<1>{});
        GemmBatch = transformer_.GetGemmBatch(); // one GEMM batch per group

        // Split-N support: each group's input/output strides across N.
        n_per_split = transformer_.GetN();
        original_n  = transformer_.GetOriginalN();
        n_splits    = integer_divide_ceil(original_n, n_per_split);

        // Per-group element strides (input / weight always GNCHW / GKCYX).
        group_stride_a = transformer_.GetGroupStrideA();
        group_stride_b = transformer_.GetGroupStrideB();

        // Output layout determines how group and N-batch strides are computed.
        if constexpr(std::is_same_v<OutLay, tensor_layout::convolution::NHWGK>)
        {
            // NHWGK = [N, Ho, Wo, G, K]:  stride(G) = K,  stride(N) = Ho*Wo*G*K.
            group_stride_c      = static_cast<long_index_t>(transformer_.K_);
            output_batch_stride = static_cast<index_t>(transformer_.Ho_) *
                                  static_cast<index_t>(transformer_.Wo_) *
                                  static_cast<index_t>(GemmBatch) *   // G
                                  static_cast<index_t>(transformer_.K_);
        }
        else
        {
            // GNKHW = [G, N, K, Ho, Wo]: stride(G) = N*K*Ho*Wo, stride(N) = K*Ho*Wo.
            group_stride_c      = transformer_.GetGroupStrideC();
            output_batch_stride = static_cast<index_t>(transformer_.K_) *
                                  static_cast<index_t>(transformer_.Ho_) *
                                  static_cast<index_t>(transformer_.Wo_);
        }

        // Input N-batch stride: GNCHW stride(N within group) = C * Hi * Wi.
        input_batch_stride = static_cast<index_t>(transformer_.C_) *
                             static_cast<index_t>(transformer_.Hi_) *
                             static_cast<index_t>(transformer_.Wi_);

        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
        {
            std::cout << "[im2win] GemmM=" << GemmM << " GemmN=" << GemmN << " GemmK=" << GemmK
                      << " GemmBatch(G)=" << GemmBatch
                      << " group_stride_a=" << group_stride_a
                      << " group_stride_b=" << group_stride_b
                      << " group_stride_c=" << group_stride_c
                      << " n_per_split=" << n_per_split << " n_splits=" << n_splits
                      << std::endl;
        }
    }

    // ── Descriptor type aliases (needed for kernel template deduction) ─
    using AGridDescMK = remove_cvref_t<
        decltype(ConvToIm2winTransformer{}.template MakeADescriptor_M_K<
                 typename GroupedConvTraitsType_::InLayout>())>;
    using BGridDescNK = remove_cvref_t<
        decltype(ConvToIm2winTransformer{}.template MakeBDescriptor_N_K<
                 typename GroupedConvTraitsType_::WeiLayout>())>;
    using CGridDescMN = remove_cvref_t<
        decltype(ConvToIm2winTransformer{}.template MakeCDescriptor_M_N<
                 typename GroupedConvTraitsType_::OutLayout>())>;

    // ── Data members ──────────────────────────────────────────────────
    array<index_t, NonSpatialDims + GroupedConvTraitsType_::NDimSpatial> in_g_n_c_wis_lengths;
    array<index_t, NonSpatialDims + GroupedConvTraitsType_::NDimSpatial> wei_g_k_c_xs_lengths;
    array<index_t, NonSpatialDims + GroupedConvTraitsType_::NDimSpatial> out_g_n_k_wos_lengths;

    array<index_t, GroupedConvTraitsType_::NDimSpatial> conv_filter_strides;
    array<index_t, GroupedConvTraitsType_::NDimSpatial> conv_filter_dilations;
    array<index_t, GroupedConvTraitsType_::NDimSpatial> input_left_pads;
    array<index_t, GroupedConvTraitsType_::NDimSpatial> input_right_pads;

    index_t k_batch;
    index_t GemmM;
    index_t GemmN;
    index_t GemmK;
    index_t GemmBatch;

    const void* in_ptr;
    const void* wei_ptr;
    std::array<const void*, NumDTensor> ds_ptr;
    void* out_ptr;
    const CDElementwise elfunc;

    AGridDescMK a_grid_desc_m_k;
    BGridDescNK b_grid_desc_n_k;
    CGridDescMN c_grid_desc_m_n;

    long_index_t group_stride_a; ///< Elements per group in input tensor
    long_index_t group_stride_b; ///< Elements per group in weight tensor
    long_index_t group_stride_c; ///< Elements per group in output tensor

    // Split-N support
    index_t n_splits        = 1; ///< Number of batch splits
    index_t n_per_split     = 1; ///< Batch size per split
    index_t original_n      = 1; ///< Original batch size

    // Strides for advancing one N-slice (within a group, after group offset).
    index_t input_batch_stride  = 0;
    index_t output_batch_stride = 0;

    ConvToIm2winTransformer transformer_;
};

// ═══════════════════════════════════════════════════════════════════════
// GroupedConvolutionForwardIm2winKernel
// ═══════════════════════════════════════════════════════════════════════
//
// GPU kernel template for im2win-based 2D grouped convolution.
//
// The im2win transform re-indexes the input so that each output spatial
// position (m = n*Ho*Wo + ho*Wo + wo) has a corresponding row in A that
// contains all the input values needed for all filter positions (k_gemm =
// C×Y×X). This is expressed entirely through tensor descriptors — no
// temporary buffer is materialised.
//
// Grid / block mapping:
//   blockIdx.z → group index  g ∈ [0, G-1]
//   blockIdx.y → N-split index  (split-N support)
//   blockIdx.x → GEMM tile index, flattened from (M-tile, N-tile)
//                via TilePartitioner
//
// The operator() function:
//   1. Computes per-group pointer offsets.
//   2. Applies the N-split offset if batch splitting is active.
//   3. Calls MakeABlockWindow / MakeBBlockWindow to create tile windows
//      backed by the im2win descriptor.
//   4. Delegates to GemmPipeline (unchanged from the im2col kernel).
//   5. Stores the accumulated C tile via EpiloguePipeline.
//
// Template parameters mirror GroupedConvolutionForwardKernel exactly so
// that the same GemmPipeline and EpiloguePipeline can be reused.
// ═══════════════════════════════════════════════════════════════════════
template <typename GroupedConvTraitsType_,
          typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_>
struct GroupedConvolutionForwardIm2winKernel
{
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using GemmALayout      = remove_cvref_t<typename GemmPipeline::ALayout>;
    using GemmBLayout      = remove_cvref_t<typename GemmPipeline::BLayout>;
    using GemmCLayout      = remove_cvref_t<typename GemmPipeline::CLayout>;

    using InLayout  = remove_cvref_t<typename GroupedConvTraitsType_::InLayout>;
    using WeiLayout = remove_cvref_t<typename GroupedConvTraitsType_::WeiLayout>;
    using OutLayout = remove_cvref_t<typename GroupedConvTraitsType_::OutLayout>;
    using DsLayout  = remove_cvref_t<typename GroupedConvTraitsType_::DsLayout>;

    using GemmDsLayout                  = remove_cvref_t<typename EpiloguePipeline::DsLayout>;
    static constexpr index_t NumDTensor = GroupedConvTraitsType_::NumDTensor;

    static constexpr index_t kBlockSize = GemmPipeline::BlockSize;
    static constexpr index_t NDimSpatial = GroupedConvTraitsType_::NDimSpatial;
    static constexpr ConvolutionSpecialization ConvSpecialization =
        GroupedConvTraitsType_::ConvSpecialization;

    using InDataType  = remove_cvref_t<typename GemmPipeline::ADataType>;
    using WeiDataType = remove_cvref_t<typename GemmPipeline::BDataType>;
    using DsDataType  = remove_cvref_t<typename EpiloguePipeline::DsDataType>;
    using OutDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;
    using CDElementwise = typename EpiloguePipeline::CDElementwise;

    using KernelArgs = GroupedConvFwdIm2winKernelArgs<GroupedConvTraitsType_, CDElementwise>;

    static_assert(GemmPipeline::kPadM && GemmPipeline::kPadN && GemmPipeline::kPadK,
                  "im2win kernel requires padded GEMM pipeline (kPadM/N/K must be true).");
    static_assert(std::is_same_v<GemmALayout, tensor_layout::gemm::RowMajor>,
                  "im2win: A matrix must be RowMajor.");
    static_assert(std::is_same_v<GemmBLayout, tensor_layout::gemm::ColumnMajor>,
                  "im2win: B matrix must be ColumnMajor.");
    static_assert(std::is_same_v<GemmCLayout, tensor_layout::gemm::RowMajor>,
                  "im2win: C matrix must be RowMajor.");

    // ── Static factory helpers ────────────────────────────────────────

    [[nodiscard]] CK_TILE_HOST static std::string GetName()
    {
        return concat('_',
                      "grouped_convolution_forward_im2win",
                      InLayout::name,
                      WeiLayout::name,
                      OutLayout::name,
                      "gemm",
                      GemmPipeline::GetName(),
                      "epilogue",
                      EpiloguePipeline::GetName(),
                      getConvSpecializationString(ConvSpecialization));
    }

    [[nodiscard]] CK_TILE_HOST static std::string GetTypeString() { return GetName(); }

    CK_TILE_HOST static auto GridSize(const KernelArgs& kargs)
    {
        // x: GEMM tile grid (M × N tiles flattened by TilePartitioner)
        // y: N-split index (1 if no split-N)
        // z: group index
        return dim3(TilePartitioner::GridSize(kargs.GemmM, kargs.GemmN),
                    kargs.n_splits,
                    kargs.GemmBatch);
    }

    CK_TILE_HOST static auto BlockSize()
    {
        return is_wave32() ? dim3(kBlockSize / 2) : dim3(kBlockSize);
    }

    CK_TILE_HOST static KernelArgs MakeKernelArgs(const GroupedConvFwdHostArgs<CDElementwise>& h)
    {
        return KernelArgs{h};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_HOST static bool IsSupportedArgument(const KernelArgs& kargs)
    {
        // Only 2D, unit-strided layouts supported for now.
        if constexpr(NDimSpatial != 2)
            return false;

        if constexpr(!std::is_same_v<InLayout, tensor_layout::convolution::GNCHW> ||
                     !std::is_same_v<WeiLayout, tensor_layout::convolution::GKCYX> ||
                     (!std::is_same_v<OutLayout, tensor_layout::convolution::GNKHW> &&
                      !std::is_same_v<OutLayout, tensor_layout::convolution::NHWGK>))
        {
            return false;
        }

        const index_t ConvC = kargs.wei_g_k_c_xs_lengths[number<2>{}];
        const index_t ConvK = kargs.wei_g_k_c_xs_lengths[number<1>{}];

        // Vectorised access checks.
        if(ConvC % GroupedConvTraitsType_::VectorSizeA != 0)
            return false;
        if(ConvK % GroupedConvTraitsType_::VectorSizeC != 0)
            return false;

        // Filter specialisation checks.
        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
        {
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                if(kargs.wei_g_k_c_xs_lengths[i + 3] != 3)
                    return false;
            }
        }
        else if constexpr(ConvSpecialization ==
                          ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                if(kargs.wei_g_k_c_xs_lengths[i + 3] != 1 ||
                   kargs.conv_filter_strides[i] != 1 || kargs.input_left_pads[i] != 0 ||
                   kargs.input_right_pads[i] != 0)
                    return false;
            }
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Pad0)
        {
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                if(kargs.wei_g_k_c_xs_lengths[i + 3] != 1 ||
                   kargs.input_left_pads[i] != 0 || kargs.input_right_pads[i] != 0)
                    return false;
            }
        }

        return true;
    }

    // ── Tile-window factory helpers (device-side) ─────────────────────
    //
    // NOTE: The `block_idx_m` / `block_idx_n` arguments are *tile indices*
    // (0, 1, 2, ...) returned by TilePartitioner::GetOutputTileIndex.
    // make_tile_window expects *element* indices, so we multiply by the
    // per-block extent before passing as the window origin.

    template <typename ADescType>
    CK_TILE_DEVICE static auto
    MakeABlockWindow(const InDataType* a_ptr, const ADescType& a_desc, index_t block_idx_m)
    {
        const auto a_view = make_tensor_view<address_space_enum::global>(a_ptr, a_desc);
        const auto a_pad  = pad_tensor_view(
            a_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::KPerBlock>{}),
            sequence<true, true>{});
        return make_tile_window(
            a_pad,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::KPerBlock>{}),
            {block_idx_m * TilePartitioner::MPerBlock, 0});
    }

    template <typename BDescType>
    CK_TILE_DEVICE static auto
    MakeBBlockWindow(const WeiDataType* b_ptr, const BDescType& b_desc, index_t block_idx_n)
    {
        const auto b_view = make_tensor_view<address_space_enum::global>(b_ptr, b_desc);
        const auto b_pad  = pad_tensor_view(
            b_view,
            make_tuple(number<TilePartitioner::NPerBlock>{}, number<TilePartitioner::KPerBlock>{}),
            sequence<true, true>{});
        return make_tile_window(
            b_pad,
            make_tuple(number<TilePartitioner::NPerBlock>{}, number<TilePartitioner::KPerBlock>{}),
            {block_idx_n * TilePartitioner::NPerBlock, 0});
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set, typename CDescType>
    CK_TILE_DEVICE static auto
    MakeCBlockWindow(OutDataType* c_ptr, const CDescType& c_desc, index_t block_idx_m, index_t block_idx_n)
    {
        const auto c_view = make_tensor_view<address_space_enum::global, DstInMemOp>(c_ptr, c_desc);
        const auto c_pad  = pad_tensor_view(
            c_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            sequence<true, true>{});
        return make_tile_window(
            c_pad,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {block_idx_m * TilePartitioner::MPerBlock,
             block_idx_n * TilePartitioner::NPerBlock});
    }

    // ── GPU kernel entry point ────────────────────────────────────────

    CK_TILE_DEVICE void operator()(const KernelArgs& kargs) const
    {
        // Shared memory is used by GemmPipeline and EpiloguePipeline.
        __shared__ char smem[GetSmemSize()];

        // ── Decode block indices ──────────────────────────────────────
        const index_t group_idx   = amd_wave_read_first_lane(blockIdx.z); // g ∈ [0, G)
        const index_t n_split_idx = amd_wave_read_first_lane(blockIdx.y); // split-N batch chunk
        const index_t block_2d    = amd_wave_read_first_lane(blockIdx.x); // flat GEMM tile index

        // Decompose flat tile index into (M-tile, N-tile).
        // GemmSpatiallyLocalTilePartitioner::GetOutputTileIndex is an instance method.
        const auto [block_idx_m, block_idx_n] =
            TilePartitioner{kargs.GemmM, kargs.GemmN}.GetOutputTileIndex(block_2d);

        // ── Per-group pointer offset ──────────────────────────────────
        // Group dimension is outermost in GNCHW / GKCYX / GNKHW.
        const auto* a_g_ptr = reinterpret_cast<const InDataType*>(kargs.in_ptr) +
                              static_cast<long_index_t>(group_idx) * kargs.group_stride_a;
        const auto* b_g_ptr = reinterpret_cast<const WeiDataType*>(kargs.wei_ptr) +
                              static_cast<long_index_t>(group_idx) * kargs.group_stride_b;
        auto* c_g_ptr = reinterpret_cast<OutDataType*>(kargs.out_ptr) +
                        static_cast<long_index_t>(group_idx) * kargs.group_stride_c;

        // ── Split-N offset ────────────────────────────────────────────
        // When split-N is active each y-block processes n_per_split images.
        const auto* a_ptr =
            a_g_ptr +
            static_cast<long_index_t>(n_split_idx) * kargs.n_per_split * kargs.input_batch_stride;
        auto* c_ptr =
            c_g_ptr +
            static_cast<long_index_t>(n_split_idx) * kargs.n_per_split * kargs.output_batch_stride;

        // ── Tile windows ──────────────────────────────────────────────
        auto a_block_window = MakeABlockWindow(a_ptr, kargs.a_grid_desc_m_k, block_idx_m);
        auto b_block_window = MakeBBlockWindow(b_g_ptr, kargs.b_grid_desc_n_k, block_idx_n);

        // ── GEMM pipeline ─────────────────────────────────────────────
        // num_loop = ceil(K_gemm / KPerBlock) — number of K iterations.
        const index_t num_loop =
            amd_wave_read_first_lane(TilePartitioner::GetLoopNum(kargs.GemmK));
        auto c_block_tile =
            GemmPipeline{}.template operator()(a_block_window, b_block_window, num_loop, smem);

        // ── Epilogue (store C + optional elementwise fusion) ──────────
        // No D tensors for im2win forward; empty tuple passed to epilogue.
        auto c_block_window =
            MakeCBlockWindow(c_ptr, kargs.c_grid_desc_m_n, block_idx_m, block_idx_n);
        auto ds_block_windows = generate_tuple([&](auto) { return c_block_window; },
                                               number<NumDTensor>{});
        EpiloguePipeline{kargs.elfunc}.template operator()(
            c_block_window, c_block_tile, ds_block_windows, smem);
    }
};

} // namespace ck_tile
