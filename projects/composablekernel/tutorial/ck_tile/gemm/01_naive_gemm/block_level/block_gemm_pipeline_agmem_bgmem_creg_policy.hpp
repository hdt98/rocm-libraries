// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "../warp_level/block_gemm_asmem_bsmem_creg.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {

// Default policy for BlockGemmPipelineAGmemBGmemCReg.
// A policy class holds compile-time decisions that can be swapped out without
// changing the pipeline logic. This default policy uses:
//   - Simple row-major LDS layouts with no padding or bank-conflict avoidance
//   - DRAM tile distributions designed for coalesced + vectorized global loads
//   - BlockGemmASmemBSmemCReg as the block GEMM implementation
//
// Default policy classes are not templated themselves -- template parameters go
// on the individual member functions so callers pass the Problem type at the call site.
struct BlockGemmPipelineAGmemBGmemCRegPolicy
{
    // ---------------------------------------------------------------------------------
    // LDS layout for matrix A
    //
    // Creates a row-major LDS descriptor for the A tile (kMPerBlock x kKPerBlock).
    // The layout is plain row-major with no padding. The kKPack dimension encodes
    // the 128-bit LDS store/load width: kKPack=8 means 8 fp16 elements per ds_write_b128
    // instruction, which is the widest LDS instruction on AMD CDNA.
    //
    // The descriptor is constructed in two steps:
    //   Step 1: 3D intermediate shape (kMPerBlock, kKPerBlock/kKPack, kKPack)
    //           with strides                (kKPerBlock,            kKPack,      1)
    //           The innermost dimension (kKPack=8) is always contiguous (stride=1),
    //           ensuring the compiler emits a single 128-bit store per thread.
    //   Step 2: Merge the last two dimensions (kKPerBlock/kKPack, kKPack) back into one
    //           K dimension, yielding the final 2D shape (kMPerBlock, kKPerBlock).
    //
    // Why the 3D detour? The merge encodes the alignment contract. By making kKPack
    // the innermost dimension with stride 1, the descriptor guarantees that any
    // (m, k) access is always aligned to a kKPack-element boundary in the K direction,
    // which is the precondition for ds_write_b128.
    //
    // This same descriptor backs BOTH the store path (DRAM->LDS) and the load path
    // (LDS->MFMA). Both sides must use identical descriptors so that thread T writes
    // element (m, k) to the same physical LDS address that thread T' reads it from.
    // Any mismatch would silently corrupt results.
    // 3d + no padding (NAIVE_IMPLEMENTATION)
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t kKPack     = 8; // 8 fp16 = 128 bits = ds_write_b128 width

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock / kKPack>{}, number<kKPack>{}),
            make_tuple(number<kKPerBlock>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return a_lds_block_desc;
    }

    // ---------------------------------------------------------------------------------
    // LDS layout for matrix B
    //
    // Structurally identical to MakeALdsBlockDescriptor but for B's dimensions (kNPerBlock x
    // kKPerBlock). B is stored in [N, K] layout (N as leading dimension), so its LDS tile also has
    // N as the leading dimension, matching the DRAM layout. This avoids any transposition
    // at the LDS store step for B (on this pipeline; other GPU targets may differ).
    // 3d + no padding (NAIVE_IMPLEMENTATION)
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t kKPack     = 8;

        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock / kKPack>{}, number<kKPack>{}),
            make_tuple(number<kKPerBlock>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kNPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return b_lds_block_desc;
    }

    // ---------------------------------------------------------------------------------
    // DRAM tile distribution for matrix A
    //
    // This distribution answers: which thread is responsible for loading which elements
    // of the kMPerBlock x kKPerBlock A tile from global memory?
    //
    // The variables are derived to simultaneously achieve:
    //   (a) Vectorized loads: each thread issues one 128-bit global_load_dwordx4.
    //   (b) Coalesced loads:  within each wavefront instruction, all 64 thread addresses
    //       form a contiguous block with no gaps, minimizing DRAM transactions.
    //
    // For fp16 (sizeof = 2 bytes):
    //   K1 = 16 / 2 = 8   -- elements per 128-bit load (one global_load_dwordx4 per thread)
    //   K0 = 32 / 8 = 4   -- groups of K1 elements covering all kKPerBlock=32 K slots
    //   M2 = 64 / 4 = 16  -- threads within one warp that cover different M rows (64 / K0)
    //   M1 = 256 / 64 = 4 -- number of warps per block (kBlockSize / warp_size)
    //   M0 = 256 / (16*4) = 4  -- times each warp repeats over M to cover all kMPerBlock rows
    //
    // Sanity check (with the default parameters kMPerBlock=256, kKPerBlock=32, kBlockSize=256):
    //   M: M0 * M1 * M2 = 4 * 4 * 16 = 256 = kMPerBlock
    //   K: K0 * K1      = 4 * 8      = 32  = kKPerBlock
    //   Threads: M1 * M2 * K0 = 4 * 16 * 4 = 256 = kBlockSize
    //   Per thread: M0 * K1 = 4 * 8 = 32 elements = 4 x 128-bit loads
    //
    // Why coalescing works: within each wavefront instruction cycle, the 4 threads
    // covering the K dimension load 4 * 8 = 32 consecutive fp16 values, fitting on a cache line.
    // The 16 threads covering M load from different cache lines but collectively span a
    // perfectly contiguous range -- zero wasted bytes per wavefront instruction.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        using ADataType = remove_cvref_t<typename Problem::ADataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t K1 = 16 / sizeof(ADataType); // elements per 128-bit load
        constexpr index_t K0 = kKPerBlock / K1;        // K groups per warp
        constexpr index_t M2 = get_warp_size() / K0;   // M rows per warp per pass
        // coalesce reading for each blocks
        constexpr index_t M1 = kBlockSize / get_warp_size(); // warps per block
        constexpr index_t M0 = kMPerBlock / (M2 * M1);       // warp repetitions over M

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    // ---------------------------------------------------------------------------------
    // DRAM tile distribution for matrix B
    //
    // Structurally identical to MakeADramTileDistribution with N substituted for M.
    // B is stored as [N, K], so the N dimension is the leading dimension in memory.
    // The distribution assigns threads to N rows and K columns in the same way that
    // A's distribution assigns threads to M rows and K columns.
    //
    // For fp16 with kNPerBlock=128, kKPerBlock=32, kBlockSize=256:
    //   K1=8, K0=4, N2=16, N1=4, N0=2
    //   Each thread issues 2 x 128-bit loads, covering N0*K1 = 2*8 = 16 fp16 elements.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        using BDataType = remove_cvref_t<typename Problem::BDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t K1 = 16 / sizeof(BDataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
        // coalesce reading for each blocks
        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = kNPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    // Returns the block GEMM implementation used by this pipeline.
    // BlockGemmASmemBSmemCReg reads A and B from LDS and accumulates C in registers
    // using MFMA instructions.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        return BlockGemmASmemBSmemCReg<Problem>{};
    }
};

} // namespace ck_tile
