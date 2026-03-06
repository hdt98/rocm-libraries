// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_problem.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_default_policy.hpp"
#include "ck_tile/ops/mhc/block/block_gemm_mhc_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/mhc/kernel/block_sinkhorn_reduce.hpp"

// Include the original kernel header for reduction and Sinkhorn kernels
#include "ck_tile/ops/mhc/kernel/mhc_kernel.hpp"

// Optimized MHC Kernel V5:
// =====================================================================
// Key optimizations:
// 1. Reduced split-K factor (grid_k) from 256 to 4-16 range
// 2. Adaptive K-tiles per block based on target grid_k
// 3. Maintains load/compute overlap pattern
// 4. Ready for norm fusion without double-reading A matrix

namespace ck_tile {

template <typename Problem_,
          typename Policy_     = MHCDefaultPolicy,
          typename Activation_ = element_wise::Sigmoid>
struct MHCKernelV5Optimized
{
    using Activation = ck_tile::remove_cvref_t<Activation_>;
    using Problem    = ck_tile::remove_cvref_t<Problem_>;
    using Policy     = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using PhiDataType     = ck_tile::remove_cvref_t<typename Problem::PhiDataType>;

    // Tile sizes from BlockGemmShape
    static constexpr index_t kMTile = Problem::BlockGemmShape::kM; // Batch tile (16)
    static constexpr index_t kNTile = Problem::BlockGemmShape::kN; // Output tile (32)
    static constexpr index_t kKTile = Problem::BlockGemmShape::kK; // K tile for C dimension (64)

    // Check if problem has logical N (for N=24 optimization)
    template <typename T, typename = void>
    struct has_kNTileLogical : std::false_type
    {
    };

    template <typename T>
    struct has_kNTileLogical<T, std::void_t<decltype(T::kNTileLogical)>> : std::true_type
    {
    };

    static constexpr index_t kNTileLogical = []() {
        if constexpr(has_kNTileLogical<Problem>::value)
            return Problem::kNTileLogical;
        else
            return kNTile;
    }();

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    // OPTIMIZED: Target grid_k in the 4-16 range (matching optimal GEMM split_k)
    // This dramatically reduces reduction overhead and improves cache behavior
    CK_TILE_HOST_DEVICE static constexpr index_t GetTargetGridK(index_t nC)
    {
        // MORE AGGRESSIVE: Prioritize large inputs with minimal split-K
        // The GEMM benchmark showed split_k=4 is optimal
        if(nC >= 32768)
            return 4; // Large C: Only 4 splits (was 8, now matching optimal GEMM)
        else if(nC >= 8192)
            return 4; // Medium-large C: 4 splits (optimal for GEMM benchmark)
        else if(nC >= 4096)
            return 2; // Medium C: 2 splits
        else if(nC >= 2048)
            return 2; // Small-medium C: 2 splits
        else
            return 1; // Small C: no splitting
    }

    // OPTIMIZED: Calculate K-tiles per block to achieve target grid_k
    CK_TILE_HOST_DEVICE static constexpr index_t GetKTilesPerBlock(index_t nC)
    {
        const index_t target_grid_k = GetTargetGridK(nC);
        const index_t total_k_tiles = (nC + kKTile - 1) / kKTile;

        // Calculate tiles per block to achieve target grid_k
        // Round up to ensure we don't exceed target_grid_k
        index_t k_tiles_per_block = (total_k_tiles + target_grid_k - 1) / target_grid_k;

        // Ensure at least 1 tile per block
        k_tiles_per_block = ck_tile::max(k_tiles_per_block, index_t(1));

        // Cap at a reasonable maximum to avoid excessive register pressure
        // 64 tiles * 64 elements = 4096 elements per block is a good upper bound
        k_tiles_per_block = ck_tile::min(k_tiles_per_block, index_t(64));

        return k_tiles_per_block;
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return kBlockSize; }

    // Padding to avoid LDS bank conflicts
    static constexpr index_t kKTilePadded = kKTile + 8; // Add 8 elements padding

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        using GemmPolicy = GemmPipelineAGmemBGmemCRegV1DefaultPolicy;
        return GemmPolicy::GetSmemSizeA<Problem>() + GemmPolicy::GetSmemSizeB<Problem>();
    }

    // Grid configuration: 3D grid (M, N, K) for split-K with output tiling
    CK_TILE_HOST static constexpr auto GetGridSize(index_t batch, index_t output_dim, index_t nC)
    {
        const index_t k_tiles_per_block = GetKTilesPerBlock(nC);
        const index_t k_per_block       = kKTile * k_tiles_per_block;
        const index_t grid_m            = (batch + kMTile - 1) / kMTile;
        const index_t grid_n            = (output_dim + kNTile - 1) / kNTile;
        const index_t grid_k            = (nC + k_per_block - 1) / k_per_block;
        return make_tuple(grid_m, grid_n, grid_k);
    }

    CK_TILE_DEVICE void
    operator()(const XDataType* p_x,
               const PhiDataType* p_phi,
               ComputeDataType* p_workspace,                      // [grid_k, batch, output_dim]
               [[maybe_unused]] ComputeDataType* p_partial_norms, // [grid_k, batch]
               index_t batch,
               index_t nC,
               index_t output_dim,
               [[maybe_unused]] index_t n,
               [[maybe_unused]] float r          = 1.0f,
               [[maybe_unused]] float alpha_pre  = 1.0f,
               [[maybe_unused]] float alpha_post = 1.0f,
               [[maybe_unused]] float alpha_res  = 1.0f,
               [[maybe_unused]] float bias       = 0.0f) const
    {
        // Determine adaptive K-tiles per block based on C dimension
        const index_t k_tiles_per_block = GetKTilesPerBlock(nC);
        const index_t k_per_block       = kKTile * k_tiles_per_block;

        // 3D block indexing: (M, N, K)
        const index_t grid_m  = (batch + kMTile - 1) / kMTile;
        const index_t grid_n  = (output_dim + kNTile - 1) / kNTile;
        const index_t grid_mn = grid_m * grid_n;

        const index_t block_id_mn = get_block_id() % grid_mn;
        const index_t block_m     = block_id_mn % grid_m;
        const index_t block_n     = block_id_mn / grid_m;
        const index_t block_k     = get_block_id() / grid_mn;

        const index_t batch_start = block_m * kMTile;
        const index_t out_start   = block_n * kNTile;
        const index_t k_start     = block_k * k_per_block; // Start of this block's K-range
        const index_t k_end       = ck_tile::min(k_start + k_per_block, nC);

        if(batch_start >= batch || out_start >= output_dim || k_start >= nC)
            return;

        // Use GEMM pipeline's swizzled LDS layout (CRITICAL for multi-warp!)
        using GemmPolicy = GemmPipelineAGmemBGmemCRegV1DefaultPolicy;

        constexpr auto a_lds_block_desc = GemmPolicy::MakeALdsBlockDescriptor<Problem>();
        constexpr auto b_lds_block_desc = GemmPolicy::MakeBLdsBlockDescriptor<Problem>();
        constexpr index_t smem_size_a   = GemmPolicy::GetSmemSizeA<Problem>();
        constexpr index_t smem_size_b   = GemmPolicy::GetSmemSizeB<Problem>();

        __shared__ char smem_ptr[smem_size_a + smem_size_b];
        XDataType* x_lds     = reinterpret_cast<XDataType*>(smem_ptr);
        PhiDataType* phi_lds = reinterpret_cast<PhiDataType*>(smem_ptr + smem_size_a);

        constexpr index_t NumWarps = Problem::BlockGemmShape::NumWarps;
        using BlockGemm            = std::conditional_t<NumWarps == 1,
                                                        BlockGemmASmemBSmemCRegV1<Problem, Policy>,
                                                        BlockGemmARegBSmemCRegV2<Problem, Policy>>;

        auto result_tile = BlockGemm::MakeCBlockTile();
        set_tile(result_tile, 0.0f);

        auto x_tensor_full = make_naive_tensor_view<address_space_enum::global>(
            p_x, make_tuple(batch, nC), make_tuple(nC, 1), number<1>{}, number<1>{});
        auto x_tensor_padded = pad_tensor_view(x_tensor_full,
                                               make_tuple(number<kMTile>{}, number<kKTile>{}),
                                               sequence<false, Problem::kPadK>{});

        constexpr auto x_load_tile_dist = []() {
            if constexpr(NumWarps == 1)
                return Problem::MakeXLoadTileDistribution();
            else
                return BlockGemm::template MakeABlockTileDistribution<kMTile, kKTile>();
        }();
        auto x_dram_window = make_tile_window(x_tensor_padded,
                                              make_tuple(number<kMTile>{}, number<kKTile>{}),
                                              {batch_start, k_start},
                                              x_load_tile_dist);

        auto x_lds_tensor = make_tensor_view<address_space_enum::lds>(x_lds, a_lds_block_desc);
        auto x_lds_window =
            make_tile_window(x_lds_tensor, make_tuple(number<kMTile>{}, number<kKTile>{}), {0, 0});

        // For N=24 optimization: load only kNTileLogical columns from phi
        auto phi_tensor_full = make_naive_tensor_view<address_space_enum::global>(
            p_phi, make_tuple(output_dim, nC), make_tuple(1, output_dim), number<1>{}, number<1>{});
        auto phi_tensor_padded =
            pad_tensor_view(phi_tensor_full,
                            make_tuple(number<kNTileLogical>{}, number<kKTile>{}),
                            sequence<false, Problem::kPadK>{});

        constexpr auto phi_load_tile_dist = Problem::MakePhiLoadTileDistribution();
        auto phi_dram_window =
            make_tile_window(phi_tensor_padded,
                             make_tuple(number<kNTileLogical>{}, number<kKTile>{}),
                             {out_start, k_start},
                             phi_load_tile_dist);

        auto phi_lds_tensor = make_tensor_view<address_space_enum::lds>(phi_lds, b_lds_block_desc);
        auto phi_lds_window = make_tile_window(
            phi_lds_tensor, make_tuple(number<kNTileLogical>{}, number<kKTile>{}), {0, 0});

        // OPTIMIZED: Norm computation with single pass over x
        // This allows fusion without re-reading A matrix from DRAM
        // TEMPORARILY COMMENTED OUT to match baseline behavior
        // __shared__ ComputeDataType norm_sums[kMTile];
        const index_t thread_id = get_thread_id();

        // Initialize norm accumulator
        // for(index_t i = thread_id; i < kMTile; i += kBlockSize)
        //     norm_sums[i] = 0.0f;
        // block_sync_lds();

        // K-loop with integrated norm accumulation and GEMM pipeline scheduling
        auto x_tile   = make_static_distributed_tensor<XDataType>(x_load_tile_dist);
        auto phi_tile = make_static_distributed_tensor<PhiDataType>(phi_load_tile_dist);

        // Prologue: Load tile 0 into registers and store to LDS
        if(k_tiles_per_block > 0 && k_start < k_end)
        {
            // Load tile 0
            load_tile(x_tile, x_dram_window);
            load_tile(phi_tile, phi_dram_window);

            // Accumulate norm from tile 0 (before storing to LDS)
            // TEMPORARILY COMMENTED OUT to match baseline behavior
            // constexpr auto x_spans = decltype(x_tile)::get_distributed_spans();
            // ComputeDataType thread_norm_accum[kMTile] = {0.0f};
            //
            // sweep_tile_span(x_spans[number<0>{}], [&](auto idx0) {
            //     sweep_tile_span(x_spans[number<1>{}], [&](auto idx1) {
            //         const auto tile_idx = get_x_indices_from_distributed_indices(
            //             x_tile.get_tile_distribution(), make_tuple(idx0, idx1));
            //         const index_t local_m = tile_idx.at(number<0>{});
            //         const index_t global_m = batch_start + local_m;
            //
            //         if(global_m < batch)
            //         {
            //             ComputeDataType val =
            //             type_convert<ComputeDataType>(x_tile[make_tuple(idx0, idx1)]);
            //             thread_norm_accum[local_m] += val * val;
            //         }
            //     });
            // });

            // Store tile 0 to LDS
            if constexpr(NumWarps == 1)
                store_tile(x_lds_window, x_tile);
            store_tile(phi_lds_window, phi_tile);

            // Move to tile 1
            move_tile_window(x_dram_window, {0, kKTile});
            move_tile_window(phi_dram_window, {0, kKTile});

            // Accumulate thread's norm contributions
            // TEMPORARILY COMMENTED OUT to match baseline behavior
            // for(index_t m = 0; m < kMTile; ++m)
            // {
            //     if(thread_norm_accum[m] != 0.0f)
            //         atomicAdd(&norm_sums[m], thread_norm_accum[m]);
            // }
        }

        // Main loop: Process tiles with load/compute overlap
        for(index_t k_tile_idx = 0; k_tile_idx < k_tiles_per_block; ++k_tile_idx)
        {
            const index_t k_current = k_start + k_tile_idx * kKTile;
            if(k_current >= k_end)
                break;

            const bool has_next_tile = (k_tile_idx + 1 < k_tiles_per_block) &&
                                       (k_start + (k_tile_idx + 1) * kKTile < k_end);

            // Single-warp path
            if constexpr(NumWarps == 1)
            {
                if(has_next_tile)
                {
                    // Load next tile (tile i+1) into registers
                    load_tile(x_tile, x_dram_window);
                    load_tile(phi_tile, phi_dram_window);

                    // Accumulate norm from next tile
                    // TEMPORARILY COMMENTED OUT to match baseline behavior
                    // constexpr auto x_spans = decltype(x_tile)::get_distributed_spans();
                    // ComputeDataType thread_norm_accum[kMTile] = {0.0f};
                    //
                    // sweep_tile_span(x_spans[number<0>{}], [&](auto idx0) {
                    //     sweep_tile_span(x_spans[number<1>{}], [&](auto idx1) {
                    //         const auto tile_idx = get_x_indices_from_distributed_indices(
                    //             x_tile.get_tile_distribution(), make_tuple(idx0, idx1));
                    //         const index_t local_m = tile_idx.at(number<0>{});
                    //         const index_t global_m = batch_start + local_m;
                    //
                    //         if(global_m < batch)
                    //         {
                    //             ComputeDataType val =
                    //             type_convert<ComputeDataType>(x_tile[make_tuple(idx0, idx1)]);
                    //             thread_norm_accum[local_m] += val * val;
                    //         }
                    //     });
                    // });
                    //
                    // // Accumulate thread's norm contributions
                    // for(index_t m = 0; m < kMTile; ++m)
                    // {
                    //     if(thread_norm_accum[m] != 0.0f)
                    //         atomicAdd(&norm_sums[m], thread_norm_accum[m]);
                    // }
                }

                // Sync before GEMM
                block_sync_lds();

                // GEMM on current tile from LDS
                BlockGemm{}(result_tile, x_lds_window, phi_lds_window);

                if(has_next_tile)
                {
                    // Sync before storing next tile
                    block_sync_lds();

                    // Store next tile to LDS
                    store_tile(x_lds_window, x_tile);
                    store_tile(phi_lds_window, phi_tile);

                    // Move to tile i+2
                    move_tile_window(x_dram_window, {0, kKTile});
                    move_tile_window(phi_dram_window, {0, kKTile});
                }
            }
            else
            {
                // Multi-warp path: x_tile holds current data
                auto x_tile_current = x_tile;

                if(has_next_tile)
                {
                    // Load next tile
                    load_tile(x_tile, x_dram_window);
                    load_tile(phi_tile, phi_dram_window);

                    // Accumulate norm from next tile
                    // TEMPORARILY COMMENTED OUT to match baseline behavior
                    // constexpr auto x_spans = decltype(x_tile)::get_distributed_spans();
                    // ComputeDataType thread_norm_accum[kMTile] = {0.0f};
                    //
                    // sweep_tile_span(x_spans[number<0>{}], [&](auto idx0) {
                    //     sweep_tile_span(x_spans[number<1>{}], [&](auto idx1) {
                    //         const auto tile_idx = get_x_indices_from_distributed_indices(
                    //             x_tile.get_tile_distribution(), make_tuple(idx0, idx1));
                    //         const index_t local_m = tile_idx.at(number<0>{});
                    //         const index_t global_m = batch_start + local_m;
                    //
                    //         if(global_m < batch)
                    //         {
                    //             ComputeDataType val =
                    //             type_convert<ComputeDataType>(x_tile[make_tuple(idx0, idx1)]);
                    //             thread_norm_accum[local_m] += val * val;
                    //         }
                    //     });
                    // });
                    //
                    // // Accumulate thread's norm contributions
                    // for(index_t m = 0; m < kMTile; ++m)
                    // {
                    //     if(thread_norm_accum[m] != 0.0f)
                    //         atomicAdd(&norm_sums[m], thread_norm_accum[m]);
                    // }
                }

                // Sync before GEMM
                block_sync_lds();

                // GEMM uses current tile from registers
                BlockGemm{}(result_tile, x_tile_current, phi_lds_window);

                if(has_next_tile)
                {
                    // Sync before storing next tile
                    block_sync_lds();

                    // Store next tile to LDS
                    store_tile(phi_lds_window, phi_tile);

                    // Move to tile i+2
                    move_tile_window(x_dram_window, {0, kKTile});
                    move_tile_window(phi_dram_window, {0, kKTile});
                }
            }
        }

        // Finalize and store norm partials
        // TEMPORARILY COMMENTED OUT to match baseline behavior
        // block_sync_lds();
        for(index_t i = thread_id; i < kMTile; i += kBlockSize)
        {
            const index_t global_m = batch_start + i;
            if(global_m < batch)
                p_partial_norms[block_k * batch + global_m] = 0.0f; // was: norm_sums[i]
        }

        // Store partial GEMM results to workspace buffer
        const index_t grid_k_total = (nC + k_per_block - 1) / k_per_block;

        constexpr auto result_spans = decltype(result_tile)::get_distributed_spans();
        sweep_tile_span(result_spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(result_spans[number<1>{}], [&](auto idx1) {
                const auto tile_idx = get_x_indices_from_distributed_indices(
                    result_tile.get_tile_distribution(), make_tuple(idx0, idx1));

                const index_t local_m  = tile_idx.at(number<0>{});
                const index_t local_n  = tile_idx.at(number<1>{});
                const index_t global_m = batch_start + local_m;
                const index_t global_n = out_start + local_n;

                // For N=24 optimization: only write columns within kNTileLogical
                if(global_m < batch && global_n < output_dim && local_n < kNTileLogical)
                {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    ComputeDataType value  = result_tile[i_j_idx];
                    const index_t workspace_idx =
                        global_m * (grid_k_total * output_dim) + block_k * output_dim + global_n;
                    p_workspace[workspace_idx] = value;
                }
            });
        });
    }
};

// Export the reduction and Sinkhorn kernels from the original implementation
// These don't need optimization as they're already efficient

} // namespace ck_tile
