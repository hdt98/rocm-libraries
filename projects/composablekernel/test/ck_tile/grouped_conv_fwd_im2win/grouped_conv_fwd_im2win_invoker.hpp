// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// ── Invoker for GroupedConvolutionForwardIm2winKernel ────────────────────────
//
// Mirrors the structure of GroupedConvolutionForwardInvoker (example/ck_tile/
// 20_grouped_convolution/grouped_convolution_forward_invoker.hpp) but wires up
// GroupedConvolutionForwardIm2winKernel with the GNCHW / GKCYX / GNKHW layout.
//
// The tile / warp shape is fixed here for gfx950 fp16, targeting the primary
// use case (small C/K per group, 3×3 filter):
//   M_Tile = 64  (N × Ho × Wo per block)
//   N_Tile = 32  (K per block)
//   K_Tile = 64  (C × Y × X reduction per step)
// ─────────────────────────────────────────────────────────────────────────────

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/grouped_convolution/utils/convolution_specialization.hpp"
#include "ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp"
#include "ck_tile/ops/grouped_convolution/pipeline/grouped_conv_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_im2win_kernel.hpp"

// ── Tile shape for the integration test ──────────────────────────────────────
// These are deliberately modest to keep compile time short.
// M=64, N=32, K=64 with a single M-warp and 2 N-warps.
// Tile shape matching ConvConfigComputeV3 from the grouped conv example:
//   M=16 x N=64 x K=64, warp tile 16x16x32, 1 M-warp / 4 N-warps.
// This combination has defined WarpGemmDispatcher specialisations for all
// supported data types (fp16, bf16).
template <typename PrecType>
struct Im2winConvTestConfig
{
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr ck_tile::index_t VectorSizeA = 4;
    static constexpr ck_tile::index_t VectorSizeB = 4;
    static constexpr ck_tile::index_t VectorSizeC = 4;

    static constexpr bool DoubleSmemBuffer = false;
    static constexpr int  kBlockPerCu      = 1;

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::COMPUTE_V3;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
};

// ── Pipeline trait selector (reuse from example conv_configs.hpp pattern) ─────
template <ck_tile::GemmPipeline PipelineId>
struct Im2winPipelineTypeTraits;

template <>
struct Im2winPipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V3>
{
    template <typename PipelineProblem>
    using GemmPipeline =
        ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem,
                                          ck_tile::GroupedConvUniversalPipelineAgBgCrPolicy>;
};

// ── Invoker struct ────────────────────────────────────────────────────────────
struct GroupedConvFwdIm2winInvoker
{
    template <ck_tile::index_t NDimSpatial,
              typename ConvConfig,
              typename InDataType,
              typename WeiDataType,
              typename AccDataType,
              typename OutDataType,
              typename InLayout,
              typename WeiLayout,
              typename OutLayout>
    static float grouped_conv_fwd(const ck_tile::GroupedConvFwdHostArgs<>& args,
                                  const ck_tile::stream_config& s)
    {
        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<ConvConfig::M_Tile, ConvConfig::N_Tile, ConvConfig::K_Tile>,
            ck_tile::sequence<ConvConfig::M_Warp, ConvConfig::N_Warp, ConvConfig::K_Warp>,
            ck_tile::sequence<ConvConfig::M_Warp_Tile,
                              ConvConfig::N_Warp_Tile,
                              ConvConfig::K_Warp_Tile>>;

        constexpr auto ConvSpec = ck_tile::ConvolutionSpecialization::Default;

        // Traits object — reuses existing GroupedConvTraits but the im2win kernel
        // overrides the descriptor factory.
        using GroupedConvTraitsType =
            ck_tile::GroupedConvTraits<NDimSpatial,
                                       ConvSpec,
                                       InLayout,
                                       WeiLayout,
                                       ck_tile::tuple<>, // no D tensors
                                       OutLayout,
                                       ConvConfig::VectorSizeA,
                                       ConvConfig::VectorSizeB,
                                       ConvConfig::VectorSizeC,
                                       /*NumGroupsToMerge=*/1>;

        using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
            GemmShape,
            GroupedConvTraitsType::FixedGemmParams::TilePartitionerGroupNum,
            GroupedConvTraitsType::FixedGemmParams::TilePartitionerM01>;

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

        using ConvEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
            InDataType,
            WeiDataType,
            ck_tile::tuple<>,
            AccDataType,
            OutDataType,
            typename GroupedConvTraitsType::ImplicitGemmDsLayout,
            typename GroupedConvTraitsType::FixedGemmParams::ELayout,
            ck_tile::element_wise::PassThrough,
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

        using Kernel = ck_tile::GroupedConvolutionForwardIm2winKernel<GroupedConvTraitsType,
                                                                       TilePartitioner,
                                                                       GemmPipeline,
                                                                       ConvEpilogue>;

        auto kargs = Kernel::MakeKernelArgs(args);

        if(!Kernel::IsSupportedArgument(kargs))
            throw std::runtime_error("im2win: arguments not supported by this kernel instance.");

        const dim3 grids  = Kernel::GridSize(kargs);
        const dim3 blocks = Kernel::BlockSize();

        if(s.log_level_ > 0)
        {
            std::cout << "[im2win invoker] kernel=" << Kernel::GetName()
                      << " grid=(" << grids.x << "," << grids.y << "," << grids.z << ")"
                      << " block=(" << blocks.x << "," << blocks.y << "," << blocks.z << ")\n";
        }

        return ck_tile::launch_kernel(
            s, ck_tile::make_kernel<ConvConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
    }
};
