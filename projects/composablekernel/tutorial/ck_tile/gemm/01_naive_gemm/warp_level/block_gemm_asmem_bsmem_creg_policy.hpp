// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

// Controls whether to use the A/B-swapped MFMA variant with transposed C register layout.
//
//   0 = WarpGemmMfmaF16F16F32M32N32K8
//       Standard form: calls __builtin_amdgcn_mfma_f32_32x32x8f16(a_vec, b_vec, c_vec).
//       No operand reordering. C accumulator lands in C[M][N] register layout:
//       lane i holds all columns of row i.
//
//   1 = WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution  (default)
//       Swapped form: calls __builtin_amdgcn_mfma_f32_32x32x8f16(b_vec, a_vec, c_vec) --
//       A and B swapped. Computes B^T * A^T = (AB)^T. C accumulator lands in C[N][M]
//       register layout: lane j holds all rows of column j. The result is mathematically
//       identical to the standard form; only the register layout differs. store_tile
//       handles either layout correctly.
//
// Rationale for defaulting to 1: the transposed C layout is more cache-friendly for
// certain store patterns, and allows the store_tile path to issue ds_read_b128 aligned
// to the N dimension rather than the M dimension.
#ifndef CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION
#define CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION 1
#endif

namespace ck_tile {

// Default policy for BlockGemmASmemBSmemCReg.
// Selects the MFMA instruction wrapper and the warp layout (MWarp x NWarp).
struct BlockGemmASmemBSmemCRegPolicy
{
    // Returns a tuple of (WarpGemm, kMWarp, kNWarp).
    //
    // kMWarp = 4, kNWarp = 1: the 4 warps in the block are arranged as a 4x1 grid,
    // covering the M dimension. All 4 warps share the same N range but each covers a
    // different M stripe of the block tile.
    //
    // The WarpGemm type is selected at compile time based on {ADataType, BDataType, CDataType}.
    // This dispatch has no runtime cost -- the if constexpr branches are resolved entirely
    // during template instantiation.
    //
    // Two variants are available for fp16/bf16:
    //
    //   WarpGemmMfmaF16F16F32M32N32K8 (standard, CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION=0):
    //     Calls __builtin_amdgcn_mfma_f32_32x32x8f16(a_vec, b_vec, c_vec).
    //     No operand reordering. C accumulator lands in C[M][N] register layout:
    //     lane i holds all columns of row i.
    //
    //   WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution (swapped, default):
    //     Calls __builtin_amdgcn_mfma_f32_32x32x8f16(b_vec, a_vec, c_vec) -- A and B swapped.
    //     Computes B^T * A^T = (AB)^T. C accumulator lands in C[N][M] register layout:
    //     lane j holds all rows of column j. The result is mathematically identical; only
    //     the register layout differs. store_tile handles either layout correctly.
    //
    // Note on A/B warp distribution encodings:
    //   WarpGemmAttributeMfmaIterateK uses the same tile_distribution_encoding for both
    //   AWarpDstrEncoding and BWarpDstrEncoding. This is because the pipeline always presents
    //   B to the MFMA in [K][N] orientation (column-major from the MFMA's perspective)
    //   regardless of whether the user passed B in row-major or column-major.
    //   On gfx942 the transposition happens when writing B to LDS;
    //   on gfx950 it happens when reading from LDS. There is therefore never a scenario
    //   where the encoding needs to differ based on B's input storage format: the encoding
    //   reflects the MFMA hardware contract, not the storage format.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        // 4 warps arranged in a 4x1 grid (4 warps cover M, 1 covers N).
        constexpr index_t kMWarp = 4;
        constexpr index_t kNWarp = 1;

        // mfma m32 n32 k8: each MFMA call computes a 32x32 output tile, consuming 8 K elements.
        // WarpGemm::kM=32, WarpGemm::kN=32, WarpGemm::kK=8.
        if constexpr(std::is_same_v<typename Problem::ADataType, half_t> &&
                     std::is_same_v<typename Problem::BDataType, half_t> &&
                     std::is_same_v<typename Problem::CDataType, float>)
        {
#if CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION
            return make_tuple(
                WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution{}, kMWarp, kNWarp);
#else
            return make_tuple(WarpGemmMfmaF16F16F32M32N32K8{}, kMWarp, kNWarp);
#endif
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, bf16_t> &&
                          std::is_same_v<typename Problem::BDataType, bf16_t> &&
                          std::is_same_v<typename Problem::CDataType, float>)
        {
#if CK_TILE_ENABLE_TRANSPOSED_C_DISTRIBUTION
            return make_tuple(
                WarpGemmMfmaBf16Bf16F32M32N32K8TransposedCDistribution{}, kMWarp, kNWarp);
#else
            return make_tuple(WarpGemmMfmaBf16Bf16F32M32N32K8{}, kMWarp, kNWarp);
#endif
        }
        else
        {
            static_assert(false, "Unsupported data type configuration for GEMM warp execution.");
        }
    }
};

} // namespace ck_tile
