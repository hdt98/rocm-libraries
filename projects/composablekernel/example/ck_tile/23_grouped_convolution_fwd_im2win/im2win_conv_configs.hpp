// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// ═══════════════════════════════════════════════════════════════════════
// im2win_conv_configs.hpp — Kernel configurations for im2win forward conv
// ═══════════════════════════════════════════════════════════════════════
//
// Each Im2winConvConfig<PrecType> struct is a self-contained compile-time
// description of one kernel instance.  To try a different configuration,
// change the alias `ActiveConfig` at the bottom of this file (or pass
// a different config template to run_grouped_conv_fwd_im2win_example()).
//
// Every config must expose the following constants that the invoker reads:
//
//   Tile dimensions
//     M_Tile, N_Tile, K_Tile          — block-level tile sizes
//     M_Warp, N_Warp, K_Warp          — warp counts per block
//     M_Warp_Tile, N_Warp_Tile, K_Warp_Tile — per-warp MFMA tile sizes
//
//   Vector widths
//     VectorSizeA  (input  channel dim — must divide C)
//     VectorSizeB  (weight channel dim — must divide C)
//     VectorSizeC  (output channel dim — must divide K)
//
//   Pipeline control
//     Pipeline         (GemmPipeline enum)
//     Scheduler        (GemmPipelineScheduler enum)
//     DoubleSmemBuffer (bool)
//     NumWaveGroups    (index_t)
//     kBlockPerCu      (int — occupancy hint for the kernel launch)
//
// ─────────────────────────────────────────────────────────────────────
// Design note: keeping configs in a single visible header (rather than
// buried inside an invoker template) makes it straightforward to:
//   • compare configs side-by-side,
//   • add new configs without touching any other file,
//   • drive an auto-tuning loop over all configs from a script.
// ═══════════════════════════════════════════════════════════════════════

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/grouped_convolution/pipeline/grouped_conv_universal_pipeline_ag_bg_cr_policy.hpp"

// ── PipelineTypeTraits — same helper as in the existing conv example ──────────
// Maps a GemmPipeline enum value to the concrete pipeline template.
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

template <>
struct Im2winPipelineTypeTraits<ck_tile::GemmPipeline::MEMORY>
{
    template <typename PipelineProblem>
    using GemmPipeline =
        ck_tile::GemmPipelineAgBgCrMem<PipelineProblem,
                                       ck_tile::GroupedConvUniversalPipelineAgBgCrPolicy>;
};

template <>
struct Im2winPipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V4>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV4<PipelineProblem>;
};

// ── Base defaults shared by all im2win configs ────────────────────────────────
struct Im2winConvConfigBase
{
    // Default vectorisation widths.  The invoker's IsSupportedArgument check
    // verifies that C % VectorSizeA == 0 and K % VectorSizeC == 0 at runtime.
    static constexpr ck_tile::index_t VectorSizeA = 4;
    static constexpr ck_tile::index_t VectorSizeB = 4;
    static constexpr ck_tile::index_t VectorSizeC = 4;

    static constexpr int  kBlockPerCu      = 1;
    static constexpr bool DoubleSmemBuffer = false;

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::COMPUTE_V3;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
};

// ══════════════════════════════════════════════════════════════════════
// Available kernel configurations
//
// Naming convention: Im2winConfig_<pipeline>_M<m>N<n>K<k>_W<mw>x<nw>_T<mt>x<nt>x<kt>
//   where M/N/K are block tiles, W are warp counts, T are per-warp MFMA tiles.
//
// Target problem: G=32, C=K∈{4,8,16}, Y=X=3, unit stride/pad
//   GemmM = N×Ho×Wo,  GemmN = K (small!),  GemmK = C×Y×X
// ══════════════════════════════════════════════════════════════════════

// ── Config 1: ComputeV3, matches ConvConfigComputeV3 from the existing example
// Tile 16×64×64, 1×4 warps, warp tile 16×16×32.
// Good balance for small K (K_tile=64 >> K in practice, kPadN handles remainder).
template <typename PrecType>
struct Im2winConfig_CV3_M16N64K64 : public Im2winConvConfigBase
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
};

// ── Config 2: ComputeV3, larger M tile — more spatial parallelism
// Tile 64×64×64, 2×2 warps, warp tile 16×16×32.
template <typename PrecType>
struct Im2winConfig_CV3_M64N64K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;
};

// ── Config 3: ComputeV3, narrow N — optimised for very small K (K=4/8)
// Tile 64×32×64, 4×1 warps, warp tile 16×32×16.
// N_tile=32 wastes less padding when K=4 or K=8.
template <typename PrecType>
struct Im2winConfig_CV3_M64N32K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;
};

// ── Config 4: Memory pipeline, intrawave — memory-bandwidth-optimised
// Useful when GemmK (= C×Y×X) is small enough that the compute pipeline
// cannot fully hide memory latency.
template <typename PrecType>
struct Im2winConfig_Mem_M128N32K64 : public Im2winConvConfigBase
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

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::MEMORY;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
};

// ── Config 5: Memory pipeline, interwave
template <typename PrecType>
struct Im2winConfig_Mem_M128N32K64_Interwave : public Im2winConvConfigBase
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

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::MEMORY;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Interwave;
};

// ── Config 6: ComputeV3, 2 blocks per CU — higher occupancy
// kBlockPerCu=2 can improve hide memory latency on larger problems.
template <typename PrecType>
struct Im2winConfig_CV3_M128N128K64_Occ2 : public Im2winConvConfigBase
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

    static constexpr int kBlockPerCu = 2;
};

// ══════════════════════════════════════════════════════════════════════
// ActiveConfig — the config used when building the example executable.
//
// To benchmark a different config, change this alias and recompile.
// For systematic auto-tuning, drive config selection from a CMake
// variable or a compile-time macro; see the comments in CMakeLists.txt.
// ══════════════════════════════════════════════════════════════════════
template <typename PrecType>
using ActiveIm2winConfig = Im2winConfig_CV3_M16N64K64<PrecType>;
