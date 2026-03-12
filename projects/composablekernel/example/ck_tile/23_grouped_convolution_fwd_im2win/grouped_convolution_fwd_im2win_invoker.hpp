// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// ═══════════════════════════════════════════════════════════════════════
// grouped_convolution_fwd_im2win_invoker.hpp
//
// Thin wrapper that instantiates GroupedConvolutionForwardIm2winKernel
// from a given Im2winConvConfig and launches it.
//
// The layout contract for the im2win kernel:
//   Input:  GNCHW  (G, N, C, Hi, Wi)  — channels-first
//   Weight: GKCYX  (G, K, C,  Y,  X)  — channels-first
//   Output: NHWGK  (N, Ho, Wo, G, K)  — channels-last, K innermost
//
// NHWGK output is required because the CShuffleEpilogue writes vectorised
// stores along the N_gemm=K dimension, which must be stride-1 in memory.
//
// This invoker is intentionally kept separate from the config header and
// the main driver so each piece can be understood and modified independently.
// ═══════════════════════════════════════════════════════════════════════

#include "ck_tile/host.hpp"
#include "ck_tile/ops/grouped_convolution/utils/convolution_specialization.hpp"
#include "ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp"
#include "ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_im2win_kernel.hpp"
#include "im2win_conv_configs.hpp"

struct GroupedConvolutionFwdIm2winInvoker
{
    template <ck_tile::index_t NDimSpatial,
              typename ConvConfig,
              typename InDataType,
              typename WeiDataType,
              typename AccDataType,
              typename OutDataType,
              typename InLayout,
              typename WeiLayout,
              typename OutLayout,
              typename CDElementWise = ck_tile::element_wise::PassThrough>
    static float grouped_conv_fwd(const ck_tile::GroupedConvFwdHostArgs<CDElementWise>& args,
                                  const ck_tile::stream_config& s)
    {
        // ── GEMM tile shape from config ───────────────────────────────────
        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<ConvConfig::M_Tile, ConvConfig::N_Tile, ConvConfig::K_Tile>,
            ck_tile::sequence<ConvConfig::M_Warp, ConvConfig::N_Warp, ConvConfig::K_Warp>,
            ck_tile::sequence<ConvConfig::M_Warp_Tile,
                              ConvConfig::N_Warp_Tile,
                              ConvConfig::K_Warp_Tile>>;

        // ── Convolution traits ────────────────────────────────────────────
        // Default specialisation — handles arbitrary filter sizes with padding.
        constexpr auto ConvSpec     = ck_tile::ConvolutionSpecialization::Default;
        using GroupedConvTraitsType = ck_tile::GroupedConvTraits<
            NDimSpatial,
            ConvSpec,
            InLayout,
            WeiLayout,
            ck_tile::tuple<>, // no D (bias) tensors
            OutLayout,
            ConvConfig::VectorSizeA,
            ConvConfig::VectorSizeB,
            ConvConfig::VectorSizeC,
            ConvConfig::NumGroupsToMerge>;

        // ── Tile partitioner ──────────────────────────────────────────────
        using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
            GemmShape,
            GroupedConvTraitsType::FixedGemmParams::TilePartitionerGroupNum,
            GroupedConvTraitsType::FixedGemmParams::TilePartitionerM01>;

        // ── GEMM pipeline traits and problem ──────────────────────────────
        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<
            GroupedConvTraitsType::FixedGemmParams::kPadM,
            GroupedConvTraitsType::FixedGemmParams::kPadN,
            GroupedConvTraitsType::FixedGemmParams::kPadK,
            ConvConfig::DoubleSmemBuffer,
            typename GroupedConvTraitsType::AsLayoutFwd,
            typename GroupedConvTraitsType::BsLayoutFwd,
            typename GroupedConvTraitsType::CLayoutFwd,
            GroupedConvTraitsType::FixedGemmParams::TransposeC,
            GroupedConvTraitsType::FixedGemmParams::UseStructuredSparsity,
            GroupedConvTraitsType::FixedGemmParams::Persistent,
            ConvConfig::NumWaveGroups>;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
            InDataType,
            WeiDataType,
            AccDataType,
            GemmShape,
            GemmUniversalTraits,
            ConvConfig::Scheduler,
            ck_tile::element_wise::PassThrough,
            ck_tile::element_wise::PassThrough,
            OutDataType,
            GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
            GroupedConvTraitsType::VectorSizeA,
            GroupedConvTraitsType::VectorSizeB>;

        using GemmPipeline = typename Im2winPipelineTypeTraits<
            ConvConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

        // ── Epilogue ──────────────────────────────────────────────────────
        using ConvEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
            InDataType,
            WeiDataType,
            ck_tile::tuple<>,
            AccDataType,
            OutDataType,
            typename GroupedConvTraitsType::ImplicitGemmDsLayout,
            typename GroupedConvTraitsType::FixedGemmParams::ELayout,
            CDElementWise,
            TilePartitioner::MPerBlock,
            TilePartitioner::NPerBlock,
            ConvConfig::M_Warp,
            ConvConfig::N_Warp,
            ConvConfig::M_Warp_Tile,
            ConvConfig::N_Warp_Tile,
            ConvConfig::K_Warp_Tile,
            GroupedConvTraitsType::FixedGemmParams::TransposeC,
            ConvConfig::NumWaveGroups,
            GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
            GroupedConvTraitsType::VectorSizeC>>;

        // ── Kernel type ───────────────────────────────────────────────────
        using Kernel = ck_tile::GroupedConvolutionForwardIm2winKernel<GroupedConvTraitsType,
                                                                       TilePartitioner,
                                                                       GemmPipeline,
                                                                       ConvEpilogue>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Kernel::GridSize(kargs);
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error(
                "im2win: kernel arguments not supported for this configuration.\n"
                "Check that C % VectorSizeA == 0 and K % VectorSizeC == 0.\n");
        }

        if(s.log_level_ > 0)
        {
            // Use kargs and kernel helpers to avoid shadowing the ck_tile::GemmPipeline enum.
            std::cout << "im2win kernel: " << Kernel::GetName() << '\n'
                      << "  shape:    " << GemmShape::GetName() << '\n'
                      << "  grid:  {" << grids.x << ", " << grids.y << ", " << grids.z << "}\n"
                      << "  block: {" << blocks.x << ", " << blocks.y << ", " << blocks.z
                      << "}\n";
        }

        return ck_tile::launch_kernel(
            s, ck_tile::make_kernel<ConvConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
    }
};
