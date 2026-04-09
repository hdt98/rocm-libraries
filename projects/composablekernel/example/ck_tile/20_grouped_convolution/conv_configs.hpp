// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <variant>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/utility/json_dump.hpp"

struct ConvConfigBase
{
    static constexpr ck_tile::index_t VectorSizeA = 4;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 8;

    static constexpr int kBlockPerCu                = 1;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
    static constexpr ck_tile::index_t NumWaveGroups = 1;

    static constexpr ck_tile::index_t NumGroupsToMerge = 1;

    // Wave-uniform M: false by default; set true in configs where KPerBlock/VecSizeA >= 64.
    static constexpr bool WaveUniformM = false;

    // Enable tile-aware im2col fast coordinate computation (2D Default only).
    // When true, MakeADescriptor_M_K() returns a TiledIm2ColDescriptor whose
    // coordinate functions use M_base + K_offset instead of the full transform chain.
    static constexpr bool UseTiledIm2Col = true;
};

template <typename PrecType>
struct ConvConfigMemoryInterwave : public ConvConfigBase
{
    // Memory friendly for Interwave scheduler
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::MEMORY;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Interwave;
};

template <typename PrecType>
struct ConvConfigMemoryIntrawave : public ConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::MEMORY;
};

template <typename PrecType>
struct ConvConfigComputeV3 : public ConvConfigBase
{
    // Compute V3 only support Intrawave scheduler
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
};

template <typename PrecType>
struct ConvConfigComputeV3_1 : public ConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
};

template <typename PrecType>
struct ConvConfigComputeV3_2 : public ConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;

    static constexpr int kBlockPerCu = 2;
};

template <typename PrecType>
struct ConvConfigComputeV3_WMMA : public ConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;

    static constexpr int kBlockPerCu = 2;
};

template <typename PrecType>
struct ConvConfigComputeV4 : public ConvConfigBase
{
    // Compute V4 only support Intrawave scheduler
    // Using the ping pong reader in the lds level
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 64 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V4;
};

template <typename PrecType>
struct ConvConfigComputeV4_1 : public ConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V4;
};

template <typename PrecType>
struct ConvConfigComputeV5 : public ConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 2;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V5;
    static constexpr ck_tile::index_t NumWaveGroups = 2;
};

template <typename PrecType>
struct ConvConfigComputeV6 : public ConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 32;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V6;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
};

template <typename PrecType>
struct ConvConfigComputeV3_merged_groups : public ConvConfigBase
{
    static constexpr ck_tile::index_t VectorSizeA = 4;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 8;

    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 32;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;

    static constexpr ck_tile::index_t NumGroupsToMerge = 2;
};

// Wave-uniform m_gemm config.
//
// KPerBlock = 256 ensures X0 = min(warp_size, KPerBlock/VecSize) = min(64, 64) = 64,
// meaning all 64 lanes in a warp map exclusively to K.  Consequently m_gemm is identical
// for every thread in a warp (wave-uniform), so Im2ColCoordinate<FwdInput>::init()'s 3 M
// divmods are computed with a wavefront-uniform value — a necessary condition for the
// scalar-unit M_base optimisation.
//
// LDS (fp16, single buffer): (64 + 64) × 256 × 2 = 65536 B = 64 KB (at limit).
// K_gemm = Y×X×C = 9×256 = 2304, KPerBlock = 256 → 9 K-loop iterations.
// N_gemm = K = 256, NPerBlock = 64 → 4 N-tiles.
template <typename PrecType>
struct ConvConfigWaveUniformM : public ConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    // M_Tile / M_Warp = 32, N_Tile / N_Warp = 32 → m32n32k16 MFMA.
    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
    // KPerBlock / VectorSizeA = 256 / 4 = 64 == warp_size → all lanes map to K → wave-uniform M.
    static constexpr bool WaveUniformM = true;
};

template <typename InDataType, typename WeiDataType = InDataType, typename OutDataType = InDataType>
struct ConvTypeConfig;

template <>
struct ConvTypeConfig<ck_tile::half_t>
{
    using InDataType  = ck_tile::half_t;
    using WeiDataType = ck_tile::half_t;
    using AccDataType = float;
    using OutDataType = ck_tile::half_t;
    // ToDo: Add more bias config to support different categories of GEMM.
};

template <>
struct ConvTypeConfig<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t>
{
    using InDataType  = ck_tile::bf16_t;
    using WeiDataType = ck_tile::bf16_t;
    using AccDataType = float;
    using OutDataType = ck_tile::bf16_t;
};

template <ck_tile::GemmPipeline PipelineId>
struct PipelineTypeTraits;

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::BASIC_V1>
{
    template <typename PipelineProblem>
    using GemmPipeline =
        ck_tile::GemmPipelineAGmemBGmemCRegV1<PipelineProblem,
                                              ck_tile::GroupedConvUniversalPipelineAgBgCrPolicy>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAGmemBGmemCRegV1<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::BASIC_V2>
{
    template <typename PipelineProblem>
    using GemmPipeline =
        ck_tile::GemmPipelineAGmemBGmemCRegV2<PipelineProblem,
                                              ck_tile::GroupedConvUniversalPipelineAgBgCrPolicy>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAGmemBGmemCRegV2<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::MEMORY>
{
    template <typename PipelineProblem>
    using GemmPipeline =
        ck_tile::GemmPipelineAgBgCrMem<PipelineProblem,
                                       ck_tile::GroupedConvUniversalPipelineAgBgCrPolicy>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrMem<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V3>
{
    template <typename PipelineProblem>
    using GemmPipeline =
        ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem,
                                          ck_tile::GroupedConvUniversalPipelineAgBgCrPolicy>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V4>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV4<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV4<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V5>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV5<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV5<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V6>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV6<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV6<PipelineProblem>;
};
