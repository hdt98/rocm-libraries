// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// ═══════════════════════════════════════════════════════════════════════
// grouped_convolution_fwd_nhwgc_im2col_invoker.hpp
//
// Invoker for the im2col-shape forward convolution in NHWGC channels-last
// layout, using the standard GroupedConvolutionForwardKernel.
//
// GEMM shape:
//   A = input  → A[M=N×Ho×Wo, K_gemm=C×Y×X]  (im2col-style)
//   B = weight → B[N_gemm=K,  K_gemm=C×Y×X]
//   C = output → C[M=N×Ho×Wo, N_gemm=K]
//
// This uses the standard im2col transformer (TransformConvFwdToGemm with
// NHWGC layout), which internally applies the same chain of transforms as
// the im2win I' transformation but directly (without naming the intermediate
// I' tensor explicitly).
//
// Purpose: compare against the im2win group-merge kernel to quantify any
// algorithmic difference in the channels-last layout.
// ═══════════════════════════════════════════════════════════════════════

#include "ck_tile/host.hpp"
#include "ck_tile/ops/grouped_convolution/utils/convolution_specialization.hpp"
#include "ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp"
#include "ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp"
#include "../20_grouped_convolution/grouped_convolution_utils.hpp"
#include "nhwgc_im2col_configs.hpp"

struct GroupedConvolutionFwdNhwgcIm2colInvoker
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
        constexpr auto ConvSpec     = ck_tile::ConvolutionSpecialization::Default;
        using GroupedConvTraitsType = ck_tile::GroupedConvTraits<
            NDimSpatial,
            ConvSpec,
            InLayout,
            WeiLayout,
            ck_tile::tuple<>,
            OutLayout,
            ConvConfig::VectorSizeA,
            ConvConfig::VectorSizeB,
            ConvConfig::VectorSizeC,
            ConvConfig::NumGroupsToMerge,
            false,                     // EnableSplitImage
            false,                     // ExplicitGemm
            ConvConfig::UseIm2Win>;    // apply im2win I' transform in A descriptor

        // ── Tile partitioner ──────────────────────────────────────────────
        using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
            GemmShape,
            GroupedConvTraitsType::FixedGemmParams::TilePartitionerGroupNum,
            GroupedConvTraitsType::FixedGemmParams::TilePartitionerM01>;

        // ── GEMM pipeline traits ──────────────────────────────────────────
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

        // Im2col: A = input (InDataType), B = weight (WeiDataType)
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

        using GemmPipeline = typename PipelineTypeTraits<
            ConvConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

        // CShuffleEpilogue — same as standard example 20 im2col kernel
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

        // Standard im2col kernel (GroupedConvolutionForwardKernel, not im2win)
        using Kernel = ck_tile::GroupedConvolutionForwardKernel<GroupedConvTraitsType,
                                                                TilePartitioner,
                                                                GemmPipeline,
                                                                ConvEpilogue>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Kernel::GridSize(kargs);
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping conv!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "nhwgc_im2col kernel: " << Kernel::GetName() << '\n'
                      << "  shape:    " << GemmShape::GetName() << '\n'
                      << "  grid:  {" << grids.x << ", " << grids.y << ", " << grids.z << "}\n"
                      << "  block: {" << blocks.x << ", " << blocks.y << ", " << blocks.z
                      << "}\n";
        }

        return ck_tile::launch_kernel(
            s, ck_tile::make_kernel<ConvConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
    }
};
