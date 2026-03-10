// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "custom_lds_policy.hpp"

// ---------------------------------------------------------------------------
// PipelineTypeTraits — maps (GemmPipeline enum, Policy) to concrete types.
//
// When LdsPolicy is DefaultLdsPolicyTag, the pipeline's built-in default
// policy is used.  Otherwise, the user-supplied policy is forwarded as the
// second template argument.
// ---------------------------------------------------------------------------
template <ck_tile::GemmPipeline PipelineId, typename LdsPolicy>
struct BankProfilePipelineTraits;

// ---- helpers: default-policy specialisations (LdsPolicy = DefaultLdsPolicyTag) ----
template <>
struct BankProfilePipelineTraits<ck_tile::GemmPipeline::MEMORY, ck_tile::DefaultLdsPolicyTag>
{
    template <typename P>
    using Pipeline = ck_tile::GemmPipelineAgBgCrMem<P>;
};

template <>
struct BankProfilePipelineTraits<ck_tile::GemmPipeline::COMPUTE_V3, ck_tile::DefaultLdsPolicyTag>
{
    template <typename P>
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompV3<P>;
};

template <>
struct BankProfilePipelineTraits<ck_tile::GemmPipeline::COMPUTE_V4, ck_tile::DefaultLdsPolicyTag>
{
    template <typename P>
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompV4<P>;
};

template <>
struct BankProfilePipelineTraits<ck_tile::GemmPipeline::COMPUTE_V5, ck_tile::DefaultLdsPolicyTag>
{
    template <typename P>
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompV5<P>;
};

template <>
struct BankProfilePipelineTraits<ck_tile::GemmPipeline::COMPUTE_V6, ck_tile::DefaultLdsPolicyTag>
{
    template <typename P>
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompV6<P>;
};

template <>
struct BankProfilePipelineTraits<ck_tile::GemmPipeline::COMPUTE_ASYNC, ck_tile::DefaultLdsPolicyTag>
{
    template <typename P>
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompAsync<P>;
};

// ---- custom-policy specialisations (LdsPolicy = any concrete policy type) ----
template <typename LdsPolicy>
struct BankProfilePipelineTraits<ck_tile::GemmPipeline::COMPUTE_V3, LdsPolicy>
{
    template <typename P>
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompV3<P, LdsPolicy>;
};

template <typename LdsPolicy>
struct BankProfilePipelineTraits<ck_tile::GemmPipeline::COMPUTE_V4, LdsPolicy>
{
    template <typename P>
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompV4<P, LdsPolicy>;
};

template <typename LdsPolicy>
struct BankProfilePipelineTraits<ck_tile::GemmPipeline::COMPUTE_V5, LdsPolicy>
{
    template <typename P>
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompV5<P, LdsPolicy>;
};

template <typename LdsPolicy>
struct BankProfilePipelineTraits<ck_tile::GemmPipeline::COMPUTE_V6, LdsPolicy>
{
    template <typename P>
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompV6<P, LdsPolicy>;
};

template <typename LdsPolicy>
struct BankProfilePipelineTraits<ck_tile::GemmPipeline::COMPUTE_ASYNC, LdsPolicy>
{
    template <typename P>
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompAsync<P, LdsPolicy>;
};

// ---------------------------------------------------------------------------
// BankProfileInvoker — assembles the full CK GEMM type chain from config.
//
// Uses UniversalGemmPipelineProblem + PipelineTypeTraits so that the
// profiled kernel matches the real GEMM exactly (same LDS layout, same
// pipeline, same policy).
//
// Config may optionally define:
//   using LdsPolicy = ck_tile::PaddedLdsPolicy;   // custom LDS layout
// If omitted, the pipeline's built-in default policy is used.
// ---------------------------------------------------------------------------

// SFINAE helper: detect whether GemmConfig::LdsPolicy exists
template <typename T, typename = void>
struct config_lds_policy
{
    using type = ck_tile::DefaultLdsPolicyTag;
};
template <typename T>
struct config_lds_policy<T, std::void_t<typename T::LdsPolicy>>
{
    using type = typename T::LdsPolicy;
};

struct BankProfileInvoker
{
    template <typename GemmConfig,
              typename ADataType,
              typename BDataType,
              typename AccDataType,
              typename ALayout,
              typename BLayout,
              typename CLayout>
    static float run(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
    {
        using CDataType = ADataType;

        // ----- Resolve policy type from config (or use default) -----
        using LdsPolicy = typename config_lds_policy<GemmConfig>::type;

        // ----- Gemm shape -----
        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<GemmConfig::M_Tile,
                                                      GemmConfig::N_Tile,
                                                      GemmConfig::K_Tile>,
                                    ck_tile::sequence<GemmConfig::M_Warp,
                                                      GemmConfig::N_Warp,
                                                      GemmConfig::K_Warp>,
                                    ck_tile::sequence<GemmConfig::M_Warp_Tile,
                                                      GemmConfig::N_Warp_Tile,
                                                      GemmConfig::K_Warp_Tile>>;

        // ----- Tile partitioner -----
        using TilePartitioner =
            ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                        GemmConfig::TilePartitionerGroupNum,
                                                        GemmConfig::TilePartitionerM01>;

        // ----- Traits -----
        using GemmTraits =
            ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                              GemmConfig::kPadN,
                                              GemmConfig::kPadK,
                                              GemmConfig::DoubleSmemBuffer,
                                              ALayout,
                                              BLayout,
                                              CLayout,
                                              GemmConfig::TransposeC,
                                              false, /* UseStructuredSparsity */
                                              false, /* Persistent */
                                              GemmConfig::NumWaveGroups,
                                              GemmConfig::Preshuffle>;

        // ----- Pipeline problem -----
        constexpr auto scheduler = GemmConfig::Scheduler;

        using PipelineProblem =
            ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                   BDataType,
                                                   AccDataType,
                                                   GemmShape,
                                                   GemmTraits,
                                                   scheduler>;

        // ----- Pipeline (selected by enum + policy) -----
        using GemmPipeline = typename BankProfilePipelineTraits<
            GemmConfig::Pipeline, LdsPolicy>::template Pipeline<PipelineProblem>;

        // ----- Epilogue -----
        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                              BDataType,
                                              ck_tile::tuple<>,
                                              AccDataType,
                                              CDataType,
                                              ck_tile::tuple<>,
                                              CLayout,
                                              ck_tile::element_wise::PassThrough,
                                              TilePartitioner::MPerBlock,
                                              TilePartitioner::NPerBlock,
                                              GemmConfig::M_Warp,
                                              GemmConfig::N_Warp,
                                              GemmConfig::M_Warp_Tile,
                                              GemmConfig::N_Warp_Tile,
                                              GemmConfig::K_Warp_Tile,
                                              PipelineProblem::TransposeC,
                                              GemmConfig::NumWaveGroups,
                                              false, /* FixedVectorSize */
                                              1,     /* VectorSizeC */
                                              false, /* TiledMMAPermuteN */
                                              1,     /* BlockedXDLN_PerWarp */
                                              GemmConfig::DoubleSmemBuffer>>;

        // ----- Kernel -----
        using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            std::cerr << "Unsupported arguments for tile config "
                      << GemmConfig::M_Tile << "x" << GemmConfig::N_Tile << "x"
                      << GemmConfig::K_Tile << std::endl;
            return -1;
        }

        std::cout << "Tile: " << GemmConfig::M_Tile << "x" << GemmConfig::N_Tile << "x"
                  << GemmConfig::K_Tile
                  << " grid={" << grids.x << "," << grids.y << "," << grids.z << "}"
                  << " block=" << blocks.x
                  << " pipeline=" << GemmPipeline::GetName() << std::endl;

        return ck_tile::launch_kernel(
            s,
            ck_tile::make_kernel<GemmConfig::kBlockPerCu>(
                Kernel{}, grids, blocks, 0, kargs));
    }
};

// ---------------------------------------------------------------------------
// Config base — carries all the policy/pipeline fields.
//
// To use the pipeline's built-in default LDS policy, simply omit LdsPolicy.
// To inject a custom policy:
//   using LdsPolicy = ck_tile::PaddedLdsPolicy;
// ---------------------------------------------------------------------------
struct BankProfileConfigBase
{
    // Padding
    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;

    // Pipeline selection — override in derived configs
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V4;
    static constexpr auto Scheduler                  = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr bool DoubleSmemBuffer           = true;
    static constexpr bool TransposeC                 = false;
    static constexpr bool Preshuffle                 = false;
    static constexpr ck_tile::index_t NumWaveGroups  = 1;

    // Tile partitioner
    static constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
    static constexpr ck_tile::index_t TilePartitionerM01      = 4;

    // Occupancy
    static constexpr int kBlockPerCu = 1;

    // LDS policy: omit (uses pipeline default) or set to a custom type.
    // Example: using LdsPolicy = ck_tile::PaddedLdsPolicy;
};

// ---------------------------------------------------------------------------
// Concrete tile configs.
// ---------------------------------------------------------------------------
struct Config_128x128x32 : BankProfileConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128, N_Tile = 128, K_Tile = 32;
    static constexpr ck_tile::index_t M_Warp = 2, N_Warp = 2, K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 32, N_Warp_Tile = 32, K_Warp_Tile = 16;
};

struct Config_256x128x32 : BankProfileConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256, N_Tile = 128, K_Tile = 32;
    static constexpr ck_tile::index_t M_Warp = 4, N_Warp = 2, K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 32, N_Warp_Tile = 32, K_Warp_Tile = 16;
};

struct Config_128x256x32 : BankProfileConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128, N_Tile = 256, K_Tile = 32;
    static constexpr ck_tile::index_t M_Warp = 2, N_Warp = 4, K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 32, N_Warp_Tile = 32, K_Warp_Tile = 16;
};

struct Config_256x256x32 : BankProfileConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256, N_Tile = 256, K_Tile = 32;
    static constexpr ck_tile::index_t M_Warp = 2, N_Warp = 2, K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 32, N_Warp_Tile = 32, K_Warp_Tile = 16;
};

struct Config_64x64x32 : BankProfileConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 64, N_Tile = 64, K_Tile = 32;
    static constexpr ck_tile::index_t M_Warp = 2, N_Warp = 2, K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 16, N_Warp_Tile = 16, K_Warp_Tile = 16;
};

struct Config_128x128x64 : BankProfileConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128, N_Tile = 128, K_Tile = 64;
    static constexpr ck_tile::index_t M_Warp = 2, N_Warp = 2, K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 32, N_Warp_Tile = 32, K_Warp_Tile = 16;
};
