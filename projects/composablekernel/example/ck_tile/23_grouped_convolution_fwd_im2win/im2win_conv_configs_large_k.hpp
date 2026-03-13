// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// ═══════════════════════════════════════════════════════════════════════
// im2win_conv_configs_large_k.hpp
// ═══════════════════════════════════════════════════════════════════════
//
// Kernel configurations targeting the LARGE-K, LARGE-C regime:
//   G=1, N=32, K=2376, C=256, Y=X=3, Hi=Wi=100 (same-padding)
//
// Contrast with the small-K configs in im2win_conv_configs.hpp:
//
//   Small-K: K=4, C=4,   GEMM: M=4, N=1.28M, K_gemm=36  → memory-bound
//            Use 4×64×16 MFMA to match M=K=4 exactly (no M-tile waste)
//
//   Large-K: K=2376, C=256, GEMM: M=2376, N=320K, K_gemm=2304 → compute-bound
//            Arithmetic intensity ≈ 2080 FLOP/byte >> ridge point (≈176)
//            Standard compute-optimised tiles (32×32×16 or 16×16×32 MFMA)
//            Roofline target: 2.7 ms at 1300 TFLOPs peak
//
// Config naming: LK_<pipeline>_M<m>N<n>K<k>
//   VS8 suffix = VectorSize A/B/C = 8  (C=256, K=2376 both divisible by 8)
//
// All configs use GNCHW input / GKCYX weight / NHWGK output.
// All use UseDirectTransform=false (composite B descriptor, default).
// ═══════════════════════════════════════════════════════════════════════

// ── LK Config 0: ComputeV3, 32×32×16 MFMA, M_Tile=128, N_Tile=32 ─────
// 4 M-warps × 1 N-warp → 256 threads.  K_Tile=64 → 36 K-loop iters.
// Grid: ceil(2376/128)=19 × ceil(320K/32)=10K = 190K blocks.
template <typename PrecType>
struct Im2winConfig_LK_CV3_M128N32K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32; // 32×32×16 MFMA
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr ck_tile::index_t VectorSizeA = 8;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 8;
};

// ── LK Config 1: ComputeV3, 32×32×16 MFMA, M_Tile=64, N_Tile=64 ──────
// 2 M-warps × 2 N-warps → 256 threads.  Balanced M/N tile.
// Grid: 38 × 5000 = 190K blocks.
template <typename PrecType>
struct Im2winConfig_LK_CV3_M64N64K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr ck_tile::index_t VectorSizeA = 8;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 8;
};

// ── LK Config 2: ComputeV3, 32×32×16 MFMA, M_Tile=32, N_Tile=128 ─────
// 1 M-warp × 4 N-warps → 256 threads.  Wide N tile for spatial streaming.
// Grid: 75 × 2500 = 187.5K blocks.
template <typename PrecType>
struct Im2winConfig_LK_CV3_M32N128K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 32;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr ck_tile::index_t VectorSizeA = 8;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 8;
};

// ── LK Config 3: Memory pipeline, 16×16×32 MFMA, M_Tile=16, N_Tile=64 ─
// ComputeV3 cannot handle this tile shape (N_Tile=64 with 1×4 warp layout
// triggers a scheduling constraint). Memory pipeline works fine.
// 1 M-warp × 4 N-warps → 256 threads.
// Grid: 149 × 5000 = 745K blocks  (many small tiles, useful baseline).
template <typename PrecType>
struct Im2winConfig_LK_CV3_M16N64K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16; // 16×16×32 MFMA
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr ck_tile::index_t VectorSizeA = 8;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 8;

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::MEMORY;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
};

// ── LK Config 4: Memory pipeline, 16×16×32 MFMA, M_Tile=64, N_Tile=16 ─
// ComputeV3 cannot handle N_Tile=16 with this warp layout either.
// 4 M-warps × 1 N-warp → 256 threads.  Grid: 38 × 20K = 760K blocks.
template <typename PrecType>
struct Im2winConfig_LK_CV3_M64N16K64 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 16;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr ck_tile::index_t VectorSizeA = 8;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 8;

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::MEMORY;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
};

// ── LK Config 5: ComputeV3, 32×32×16 MFMA, M=128, N=32, 2 blocks/CU ─
// Higher occupancy variant of LK Config 0.
template <typename PrecType>
struct Im2winConfig_LK_CV3_M128N32K64_Occ2 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr ck_tile::index_t VectorSizeA = 8;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 8;

    static constexpr int kBlockPerCu = 2;
};

// ── LK Config 6: ComputeV3, 32×32×16 MFMA, M=64, N=64, 2 blocks/CU ──
// Higher occupancy variant of LK Config 1.
template <typename PrecType>
struct Im2winConfig_LK_CV3_M64N64K64_Occ2 : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr ck_tile::index_t VectorSizeA = 8;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 8;

    static constexpr int kBlockPerCu = 2;
};

// ── LK Config 7: Memory pipeline, M=128, N=32 — comparison baseline ───
// Memory pipeline with intrawave scheduler; useful as a compute-vs-memory
// comparison for this compute-bound problem.
template <typename PrecType>
struct Im2winConfig_LK_Mem_M128N32K64 : public Im2winConvConfigBase
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

    static constexpr ck_tile::index_t VectorSizeA = 8;
    static constexpr ck_tile::index_t VectorSizeB = 8;
    static constexpr ck_tile::index_t VectorSizeC = 8;

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::MEMORY;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
};
