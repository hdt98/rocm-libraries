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
    static constexpr ck_tile::index_t NumWaveGroups    = 1;
    // NumGroupsToMerge: how many conv groups one block processes sequentially.
    // 1 = no merging (default). Gm > 1 reduces GemmBatch from G to G/Gm,
    // cutting block-dispatch overhead at the cost of Gm sequential GEMM calls
    // per block. Useful when K is tiny (N_Tile >> K) and G is large.
    static constexpr ck_tile::index_t NumGroupsToMerge = 1;

    // B-descriptor approach for the sliding-window input.
    //   false (default) → Approach 1: composite elementary transforms
    //                     (pad → embed → merge)
    //   true            → Approach 2: direct stride formula
    //                     (flat 6D descriptor using index formulas from
    //                      im2win_transform.md, no explicit pad transform)
    // Change a single config to 'true' to benchmark Approach 2.
    static constexpr bool UseDirectTransform = false;
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

// ══════════════════════════════════════════════════════════════════════
// True im2win configs  (M = K, N = N×Ho×Wo)
//
// In true im2win the GEMM shape is:
//   A[M=K, K_gemm=C×Y×X]   ← weight   (small, ~9 KB for G=32,K=4,C=4,3×3)
//   B[N=N×Ho×Wo, K_gemm]   ← input    (large, sliding window)
//   C[M=K, N=N×Ho×Wo]      ← output   (K unit-stride in NHWGK)
//
// With K=4 the WarpGemmMfmaF16F16F32M4N64K16 instruction matches M=K
// exactly — zero M-dimension waste.
// ══════════════════════════════════════════════════════════════════════

// ── Config T0: True im2win, M=4 (K=4), N=256, Memory pipeline
// WarpGemmMfmaF16F16F32M4N64K16: M_WT=4, N_WT=64.
// M_Warp=1, N_Warp=4 → 256 threads, N_Tile=256.
// Optimal for K=4 (zero M waste, large N tiles).
// ComputeV3 pipeline cannot handle M_Tile=4 (internal div-by-zero constraint),
// so we use the Memory pipeline which supports arbitrary small M tiles.
template <typename PrecType>
struct Im2winConfig_Ti_M4N256K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 4;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 4;   // 4×64×16 MFMA
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::MEMORY;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
};

// ── Config T1: True im2win, M=4 (K=4), N=128 — smaller N tile
template <typename PrecType>
struct Im2winConfig_Ti_M4N128K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 4;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 4;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::MEMORY;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
};

// ── Config T2: True im2win, M=8 (K=8), N=128 — for K=8 problems
template <typename PrecType>
struct Im2winConfig_Ti_M8N128K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 8;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 4;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::MEMORY;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
};

// ── Config T3: True im2win, M=16 (K=16), N=64 — for K=16 problems
// M_Tile=16 is large enough for CV3 pipeline.
template <typename PrecType>
struct Im2winConfig_Ti_M16N64K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 4;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::MEMORY;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
};

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

    static constexpr ck_tile::index_t VectorSizeA = 4;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 4;
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
// Tile 128×32×64, 2×1 warps, warp tile 32×32×16.
// N_tile=32 wastes less padding when K=4 or K=8.
// Uses valid MFMA combo (32×32×16 fp16).
template <typename PrecType>
struct Im2winConfig_CV3_M64N32K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
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
// Configs targeting N_gemm = K = 4 (very small output channels per group)
//
// For small K the N_Tile must be as small as possible to avoid wasted
// MFMA computation.  The MFMA minimum N warp tile is 16, so N_Tile=16
// (N_Warp=1) wastes 75 % of the N dimension but is the best we can do
// without group merging.
//
// K_gemm = C × Y × X = 4 × 9 = 36 is also small — one K_Tile=64
// iteration covers it fully (kPadK handles the 28 empty elements).
// ══════════════════════════════════════════════════════════════════════

// ── Config 7: CV3, M=64, N=16 — smallest viable tile for tiny K
// M_Warp=2×N_Warp=1 → 128 threads per block, 2 blocks/CU for occupancy.
template <typename PrecType>
struct Im2winConfig_CV3_M64N16K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 16;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr int kBlockPerCu = 2;
};

// ── Config 8: CV3, M=128, N=16 — larger M tile for better streaming
// M_Warp=4×N_Warp=1 → 256 threads. Each block covers 128 spatial outputs.
template <typename PrecType>
struct Im2winConfig_CV3_M128N16K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 16;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;
};

// ── Config 9: Memory pipeline, M=128, N=16 — BW-optimised for tiny K
// MEMORY pipeline with intrawave scheduler.
// N_WT=16, M_WT=16, K_WT=32 → valid WarpGemmMfmaF16F16F32M16N16K32.
template <typename PrecType>
struct Im2winConfig_Mem_M128N16K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 16;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;  // M_Tile=128=4×16×2... actually
    static constexpr ck_tile::index_t N_Warp_Tile = 16;  // N_Tile=16=1×16
    static constexpr ck_tile::index_t K_Warp_Tile = 32;  // valid: (16,16,32)

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::MEMORY;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
};

// ── Config 10: Memory pipeline, M=128, N=16, interwave scheduler
template <typename PrecType>
struct Im2winConfig_Mem_M128N16K64_Interwave : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 16;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::MEMORY;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Interwave;
};

// ── Config 11: CV3, M=64, N=16, 2 blocks/CU — high-occupancy variant
template <typename PrecType>
struct Im2winConfig_CV3_M64N16K64_Occ2 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 16;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr int kBlockPerCu = 4;
};

// ── Config 12: Memory pipeline, MFMA M64×N4×K16 — exactly matches N_gemm=4!
// WarpGemmMfmaF16F16F32M64N4K16 processes exactly 4 output channels per MFMA.
// This eliminates all N-dimension padding waste for K=4 problems.
// Uses MEMORY pipeline (ComputeV3 has internal constraints that prevent N_Tile=4).
// M_Tile=128, N_Tile=4, K_Tile=64, M_Warp=2, N_Warp=1.
// BlockSize = 2×1×1×64 = 128 threads.
template <typename PrecType>
struct Im2winConfig_Mem_M128N4K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 4;   // exactly K=4!
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 64; // WarpGemmMfmaF16F16F32M64N4K16
    static constexpr ck_tile::index_t N_Warp_Tile = 4;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::MEMORY;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
};

// ── Config 13: Memory pipeline, M64×N4×K16, larger M tile, 2 blocks/CU
template <typename PrecType>
struct Im2winConfig_Mem_M256N4K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 4;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 64;
    static constexpr ck_tile::index_t N_Warp_Tile = 4;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::MEMORY;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr int kBlockPerCu = 2;
};

// ══════════════════════════════════════════════════════════════════════
// Configs with group merging (NumGroupsToMerge > 1)
//
// These are most useful when K (= N_gemm) is small relative to N_Tile,
// because merging reduces block dispatch overhead.  Each block runs Gm
// sequential GEMM sub-problems (one per group) sharing the same M-tile
// spatial computation and the same LDS allocation.
//
// Constraint: G must be divisible by NumGroupsToMerge.
// ══════════════════════════════════════════════════════════════════════

// ── Config 14: CV3, M128×N16, Gm=2 — baseline group-merging
template <typename PrecType>
struct Im2winConfig_CV3_M128N16K64_Gm2 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 16;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr ck_tile::index_t NumGroupsToMerge = 2;
};

// ── Config 15: CV3, M128×N16, Gm=4
template <typename PrecType>
struct Im2winConfig_CV3_M128N16K64_Gm4 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 16;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr ck_tile::index_t NumGroupsToMerge = 4;
};

// ── Config 16: CV3, M128×N16, Gm=8
template <typename PrecType>
struct Im2winConfig_CV3_M128N16K64_Gm8 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 16;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr ck_tile::index_t NumGroupsToMerge = 8;
};

// ── Config 17: CV3, M128×N16, Gm=32 — merge all groups (G=32 target problem)
template <typename PrecType>
struct Im2winConfig_CV3_M128N16K64_Gm32 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 16;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr ck_tile::index_t NumGroupsToMerge = 32;
};

// ── Config 18: CV3, M64×N16, Gm=32 — smaller M tile with full group merge
template <typename PrecType>
struct Im2winConfig_CV3_M64N16K64_Gm32 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 16;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr ck_tile::index_t NumGroupsToMerge = 32;
};

// ══════════════════════════════════════════════════════════════════════
// Group-merging configs  (NHWGC/GKYXC layout, NumGroupsToMerge > 1)
//
// With Gm groups merged:
//   M = Gm * K   (fills the M tile for small K — zero M-tile waste!)
//   N = Gm * N * Ho * Wo  (spatial × Gm, XOR diagonal ensures correctness)
//   GemmBatch = G / Gm     (fewer kernel dispatches)
//
// Target: G=32, K=4 → Gm=32 gives M=128 (exactly one M_Tile=128 block).
// All MEMORY pipeline (ComputeV3 cannot handle small M_Tile < 16).
// ══════════════════════════════════════════════════════════════════════

// ── Config Mg2: Gm=2, M=8 (K=4), MEMORY pipeline
template <typename PrecType>
struct Im2winConfig_Merge_Gm2_M8N128K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 8;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 4;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::MEMORY;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t NumGroupsToMerge = 2;
};

// ── Config Mg4: Gm=4, M=16 (K=4), MEMORY pipeline
// Uses 16×16×32 MFMA (valid), 1 M-warp, 4 N-warps → N_Tile=64, 256 threads.
template <typename PrecType>
struct Im2winConfig_Merge_Gm4_M16N64K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16; // 16×16×32 MFMA (valid)
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::MEMORY;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t NumGroupsToMerge = 4;
};

// ── Config Mg8: Gm=8, M=32 (K=4), MEMORY pipeline
template <typename PrecType>
struct Im2winConfig_Merge_Gm8_M32N128K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 32;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::MEMORY;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t NumGroupsToMerge = 8;
};

// ── Config Mg32: Gm=32, M=128 (K=4), MEMORY pipeline — PRIMARY TARGET
// With K=4 and G=32: M=128 fills M_Tile=128 perfectly (zero M waste).
// GemmBatch = 1 — entire convolution in a single pass.
template <typename PrecType>
struct Im2winConfig_Merge_Gm32_M128N32K64 : public Im2winConvConfigBase
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
    static constexpr ck_tile::index_t NumGroupsToMerge = 32;
};

// Variant: larger N tile for Gm=32
template <typename PrecType>
struct Im2winConfig_Merge_Gm32_M128N64K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::MEMORY;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t NumGroupsToMerge = 32;
};

// ── Config Mg8_M128N32K32: tile (128,32,32), 32×32 MFMA, Gm=8 ─────────────
// Requested: M_per_XDL=32, N_per_XDL=32, tile=(128,32,32), Gm=8.
// With K=4, Gm=8 → M_actual=32. M_Tile=128 covers 4 logical groups × 32
// (kPadM fills unused M). K_Tile=32 < K_gemm=36 → 2 K-loop iterations.
// 32×32×16 MFMA; Block = M_Warp×N_Warp×64 = 4×1×64 = 256 threads.
template <typename PrecType>
struct Im2winConfig_Merge_Gm8_M128N32K32 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 32;

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32; // 32×32×16 MFMA (fp16 valid)
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::MEMORY;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t NumGroupsToMerge = 8;
};

// ══════════════════════════════════════════════════════════════════════
// Config registry and compile-time selection
//
// The CMakeLists builds one binary per config by passing
//   -DIM2WIN_CONFIG_ID=<n>
// to the compiler.  The tuning script iterates over all IDs.
//
// To add a new config:
//   1. Define it above.
//   2. Add a new `elif` branch below.
//   3. Add the corresponding target in CMakeLists.txt.
// ══════════════════════════════════════════════════════════════════════

#ifndef IM2WIN_CONFIG_ID
#define IM2WIN_CONFIG_ID 0   // default: Config 0 (CV3_M16N64K64)
#endif

// True im2win configs (prefix Ti_)
#if   IM2WIN_CONFIG_ID == -4
template <typename P> using ActiveIm2winConfig = Im2winConfig_Ti_M16N64K64<P>;
#elif IM2WIN_CONFIG_ID == -3
template <typename P> using ActiveIm2winConfig = Im2winConfig_Ti_M8N128K64<P>;
#elif IM2WIN_CONFIG_ID == -2
template <typename P> using ActiveIm2winConfig = Im2winConfig_Ti_M4N128K64<P>;
#elif IM2WIN_CONFIG_ID == -1
template <typename P> using ActiveIm2winConfig = Im2winConfig_Ti_M4N256K64<P>;
// Old (wrong-shape) im2win configs kept for comparison
#elif IM2WIN_CONFIG_ID == 0
template <typename P> using ActiveIm2winConfig = Im2winConfig_CV3_M16N64K64<P>;
#elif IM2WIN_CONFIG_ID == 1
template <typename P> using ActiveIm2winConfig = Im2winConfig_CV3_M64N64K64<P>;
#elif IM2WIN_CONFIG_ID == 2
template <typename P> using ActiveIm2winConfig = Im2winConfig_CV3_M64N32K64<P>;
#elif IM2WIN_CONFIG_ID == 3
template <typename P> using ActiveIm2winConfig = Im2winConfig_Mem_M128N32K64<P>;
#elif IM2WIN_CONFIG_ID == 4
template <typename P> using ActiveIm2winConfig = Im2winConfig_Mem_M128N32K64_Interwave<P>;
#elif IM2WIN_CONFIG_ID == 5
template <typename P> using ActiveIm2winConfig = Im2winConfig_CV3_M128N128K64_Occ2<P>;
#elif IM2WIN_CONFIG_ID == 6
template <typename P> using ActiveIm2winConfig = Im2winConfig_CV3_M64N16K64<P>;
#elif IM2WIN_CONFIG_ID == 7
template <typename P> using ActiveIm2winConfig = Im2winConfig_CV3_M128N16K64<P>;
#elif IM2WIN_CONFIG_ID == 8
template <typename P> using ActiveIm2winConfig = Im2winConfig_Mem_M128N16K64<P>;
#elif IM2WIN_CONFIG_ID == 9
template <typename P> using ActiveIm2winConfig = Im2winConfig_Mem_M128N16K64_Interwave<P>;
#elif IM2WIN_CONFIG_ID == 10
template <typename P> using ActiveIm2winConfig = Im2winConfig_CV3_M64N16K64_Occ2<P>;
#elif IM2WIN_CONFIG_ID == 11
template <typename P> using ActiveIm2winConfig = Im2winConfig_Mem_M128N4K64<P>;
#elif IM2WIN_CONFIG_ID == 12
template <typename P> using ActiveIm2winConfig = Im2winConfig_Mem_M256N4K64<P>;
#elif IM2WIN_CONFIG_ID == 13
template <typename P> using ActiveIm2winConfig = Im2winConfig_CV3_M128N16K64_Gm2<P>;
#elif IM2WIN_CONFIG_ID == 14
template <typename P> using ActiveIm2winConfig = Im2winConfig_CV3_M128N16K64_Gm4<P>;
#elif IM2WIN_CONFIG_ID == 15
template <typename P> using ActiveIm2winConfig = Im2winConfig_CV3_M128N16K64_Gm8<P>;
#elif IM2WIN_CONFIG_ID == 16
template <typename P> using ActiveIm2winConfig = Im2winConfig_CV3_M128N16K64_Gm32<P>;
#elif IM2WIN_CONFIG_ID == 17
template <typename P> using ActiveIm2winConfig = Im2winConfig_CV3_M64N16K64_Gm32<P>;
// True group merging configs (NHWGC/GKYXC + XOR diagonal C descriptor)
#elif IM2WIN_CONFIG_ID == 18
template <typename P> using ActiveIm2winConfig = Im2winConfig_Merge_Gm2_M8N128K64<P>;
#elif IM2WIN_CONFIG_ID == 19
template <typename P> using ActiveIm2winConfig = Im2winConfig_Merge_Gm4_M16N64K64<P>;
#elif IM2WIN_CONFIG_ID == 20
template <typename P> using ActiveIm2winConfig = Im2winConfig_Merge_Gm8_M32N128K64<P>;
#elif IM2WIN_CONFIG_ID == 21
template <typename P> using ActiveIm2winConfig = Im2winConfig_Merge_Gm32_M128N32K64<P>;
#elif IM2WIN_CONFIG_ID == 22
template <typename P> using ActiveIm2winConfig = Im2winConfig_Merge_Gm32_M128N64K64<P>;
#elif IM2WIN_CONFIG_ID == 23
template <typename P> using ActiveIm2winConfig = Im2winConfig_Merge_Gm8_M128N32K32<P>;
#else
#error "Unknown IM2WIN_CONFIG_ID"
#endif
