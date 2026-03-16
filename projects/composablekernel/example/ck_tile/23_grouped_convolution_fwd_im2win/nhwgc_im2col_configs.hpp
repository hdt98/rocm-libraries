// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// ═══════════════════════════════════════════════════════════════════════
// nhwgc_im2col_configs.hpp — im2col-shape kernel configs for NHWGC layout
// ═══════════════════════════════════════════════════════════════════════
//
// These configs use the standard im2col GEMM shape:
//   M = N×Ho×Wo  (spatial positions)
//   N = K         (output channels)
//   K = C×Y×X    (input channels × filter size)
//
// The input is in NHWGC channels-last layout and the transformation to
// GEMM matrix A[N×Ho×Wo, C×Y×X] follows the standard im2col path
// (TransformConvFwdToGemm with NHWGC layout).
//
// These configs use GroupedConvolutionForwardKernel and CShuffleEpilogue
// (standard CK im2col kernel from example 20), not the im2win kernel.
//
// Important: The standard GroupedConvolutionForwardKernel with
// NumGroupsToMerge > 1 and NHWGC layout requires C==1 (it vectorizes
// over the G dimension). Since our target problem has C=4, all configs
// here use NumGroupsToMerge=1 (single GEMM per conv group).
//
// Purpose: Compare single-group im2col performance against im2win
// group-merge approach to quantify the benefit of group merging.
//
// Layout:
//   Input:  NHWGC  (N, Hi, Wi, G, C)
//   Weight: GKYXC  (G, K, Y, X, C)  — C innermost for vectorised loads
//   Output: NHWGK  (N, Ho, Wo, G, K) — K innermost for vectorised stores
// ═══════════════════════════════════════════════════════════════════════

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

// ── Base defaults shared by all NHWGC im2col configs ─────────────────────────
struct NhwgcIm2colConfigBase
{
    static constexpr ck_tile::index_t VectorSizeA = 4;
    static constexpr ck_tile::index_t VectorSizeB = 4;
    static constexpr ck_tile::index_t VectorSizeC = 4;

    static constexpr int  kBlockPerCu      = 1;
    static constexpr bool DoubleSmemBuffer = false;

    static constexpr ck_tile::GemmPipeline         Pipeline  = ck_tile::GemmPipeline::COMPUTE_V3;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t NumWaveGroups    = 1;
    // NumGroupsToMerge=1: standard im2col (one GEMM per conv group, G separate GEMMs).
    // NumGroupsToMerge>1 with NHWGC requires C==1 (standard kernel limitation),
    // so all configs below use Gm=1.
    static constexpr ck_tile::index_t NumGroupsToMerge = 1;

    // UseIm2Win: when true, the A descriptor applies the im2win I' height-windowing
    // transformation explicitly as a first step before the standard W-direction embed.
    // The final GEMM shape is still im2col-style (M=N×Ho×Wo, N=K).
    static constexpr bool UseIm2Win = false;
};

// ── Config IC0: 16×64×64, 1×4 warps, 16×16×32 MFMA — baseline im2col
// Matches example 20 ConvConfigComputeV3. Good starting point.
template <typename PrecType>
struct NhwgcIm2colConfig_CV3_M16N64K64 : public NhwgcIm2colConfigBase
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

// ── Config IC1: 128×32×64, 4×1 warps, 32×32×16 MFMA — memory pipeline
// Matches Mem_M128N32K64 im2win config for direct comparison (no group merge).
template <typename PrecType>
struct NhwgcIm2colConfig_Mem_M128N32K64 : public NhwgcIm2colConfigBase
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

    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::MEMORY;
};

// ── Config IC2: 128×128×64, 2×2 warps, 32×32×16 MFMA — larger N tile
// N_Tile=128 > K=4 wastes compute but may amortize overhead for large spatial dims.
template <typename PrecType>
struct NhwgcIm2colConfig_CV3_M128N128K64 : public NhwgcIm2colConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;
};

// ── Config IC3: 64×64×64, 2×2 warps, 16×16×32 MFMA — medium tile
template <typename PrecType>
struct NhwgcIm2colConfig_CV3_M64N64K64 : public NhwgcIm2colConfigBase
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

// ── Config IC4: 128×16×64, 8×1 warps, 16×16×32 MFMA — narrow N tile
// N_Tile=16 matches K=4 more closely. 16×16×32 MFMA valid for fp16/bf16.
template <typename PrecType>
struct NhwgcIm2colConfig_Mem_M128N16K64 : public NhwgcIm2colConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 16;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 8;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::MEMORY;
};

// ── Config IC5: 128×32×64, 4×1 warps, 32×32×16 MFMA — Occ2 hint
// Same tile as IC1 but with kBlockPerCu=2 (higher occupancy target).
template <typename PrecType>
struct NhwgcIm2colConfig_Mem_M128N32K64_Occ2 : public NhwgcIm2colConfigBase
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

    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::MEMORY;
    static constexpr int kBlockPerCu = 2;
};

// ── Config IC6: 64×32×64, 2×1 warps, 32×32×16 MFMA — smaller M tile
template <typename PrecType>
struct NhwgcIm2colConfig_CV3_M64N32K64 : public NhwgcIm2colConfigBase
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

// ── UseIm2Win=true variants ───────────────────────────────────────────────────
// Same tile shapes as IC0, IC1, IC2 but with UseIm2Win=true.
// The A descriptor applies the im2win I' height-windowing transformation
// explicitly (embed H first, then embed W) before the standard im2col merge.
// This makes the I' intermediate step explicit in the descriptor chain.

// ── Config IC7: 16×64×64, 1×4 warps — im2win A descriptor variant of IC0
template <typename PrecType>
struct NhwgcIm2colConfig_IW_CV3_M16N64K64 : public NhwgcIm2colConfig_CV3_M16N64K64<PrecType>
{
    static constexpr bool UseIm2Win = true;
};

// ── Config IC8: 128×32×64, 4×1 warps — im2win A descriptor variant of IC1
template <typename PrecType>
struct NhwgcIm2colConfig_IW_Mem_M128N32K64 : public NhwgcIm2colConfig_Mem_M128N32K64<PrecType>
{
    static constexpr bool UseIm2Win = true;
};

// ── Config IC9: 128×128×64, 2×2 warps — im2win A descriptor variant of IC2
template <typename PrecType>
struct NhwgcIm2colConfig_IW_CV3_M128N128K64 : public NhwgcIm2colConfig_CV3_M128N128K64<PrecType>
{
    static constexpr bool UseIm2Win = true;
};

// ── Config IC10: 64×64×64, 2×2 warps — im2win A descriptor variant of IC3
template <typename PrecType>
struct NhwgcIm2colConfig_IW_CV3_M64N64K64 : public NhwgcIm2colConfig_CV3_M64N64K64<PrecType>
{
    static constexpr bool UseIm2Win = true;
};

// Note: ActiveNhwgcIm2colConfig is defined in grouped_convolution_fwd_nhwgc_im2col.cpp
// via the NHWGC_IM2COL_CONFIG_ID compile-time macro.
