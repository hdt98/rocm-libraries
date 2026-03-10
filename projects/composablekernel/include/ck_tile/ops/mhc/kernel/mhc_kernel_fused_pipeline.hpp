// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_problem.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_default_policy.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_gemm_pipeline_ag_bg_cr_comp_v3_fused.hpp"

// MHC Kernel with Fused GEMM Pipeline V3
// =====================================================================
// This kernel uses the MhcGemmPipelineAgBgCrCompV3Fused pipeline which
// supports optional fusion functions. This enables single-pass fusion of
// operations like norm computation with the GEMM operation.
//
// Key features:
// 1. Full GEMM V3 pipeline optimizations (60 TFLOPS baseline)
// 2. Optional fusion function for norm computation (single memory pass)
// 3. Clean lambda-based interface for custom fusion operations
// 4. Maintains all V3 scheduling and prefetch optimizations

namespace ck_tile {

template <typename Problem_,
          typename Policy_     = UniversalGemmPipelineAgBgCrPolicy,
          typename Activation_ = element_wise::Sigmoid>
struct MHCKernelFusedPipeline
{
    using Activation = ck_tile::remove_cvref_t<Activation_>;
    using Problem    = ck_tile::remove_cvref_t<Problem_>;
    using Policy     = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using PhiDataType     = ck_tile::remove_cvref_t<typename Problem::PhiDataType>;

    static constexpr index_t kMTile = Problem::BlockGemmShape::kM;
    static constexpr index_t kNTile = Problem::BlockGemmShape::kN;
    static constexpr index_t kKTile = Problem::BlockGemmShape::kK;

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

    // Use the fused pipeline
    using Pipeline = MhcGemmPipelineAgBgCrCompV3Fused<Problem, Policy>;

    CK_TILE_HOST_DEVICE static constexpr index_t GetTargetGridK(index_t nC)
    {
        if(nC >= 32768)
            return 16;
        else if(nC >= 16384)
            return 16;
        else if(nC >= 8192)
            return 16;
        else if(nC >= 4096)
            return 16;
        else if(nC >= 2048)
            return 32;
        else
            return 64;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetKTilesPerBlock(index_t nC)
    {
        const index_t target_grid_k = GetTargetGridK(nC);
        const index_t total_k_tiles = (nC + kKTile - 1) / kKTile;
        index_t k_tiles_per_block   = (total_k_tiles + target_grid_k - 1) / target_grid_k;
        k_tiles_per_block           = ck_tile::max(k_tiles_per_block, index_t(1));

        index_t max_tiles = 64;
        if(nC >= 131072)
            max_tiles = 256;
        else if(nC >= 65536)
            max_tiles = 128;
        else if(nC >= 32768)
            max_tiles = 64;

        k_tiles_per_block = ck_tile::min(k_tiles_per_block, max_tiles);
        return k_tiles_per_block;
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return kBlockSize; }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Pipeline::GetSmemSize(); }

    CK_TILE_HOST static constexpr auto GetGridSize(index_t batch, index_t output_dim, index_t nC)
    {
        const index_t k_tiles_per_block = GetKTilesPerBlock(nC);
        const index_t k_per_block       = kKTile * k_tiles_per_block;
        const index_t grid_m            = (batch + kMTile - 1) / kMTile;
        const index_t grid_n            = (output_dim + kNTile - 1) / kNTile;
        const index_t grid_k            = (nC + k_per_block - 1) / k_per_block;
        return make_tuple(grid_m, grid_n, grid_k);
    }

    CK_TILE_DEVICE void operator()(const XDataType* p_x,
                                   const PhiDataType* p_phi,
                                   ComputeDataType* p_workspace,
                                   [[maybe_unused]] ComputeDataType* p_partial_norms,
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
        const index_t k_tiles_per_block = GetKTilesPerBlock(nC);
        const index_t k_per_block       = kKTile * k_tiles_per_block;

        const index_t grid_m  = (batch + kMTile - 1) / kMTile;
        const index_t grid_n  = (output_dim + kNTile - 1) / kNTile;
        const index_t grid_mn = grid_m * grid_n;

        const index_t block_id_mn = get_block_id() % grid_mn;
        const index_t block_m     = block_id_mn % grid_m;
        const index_t block_n     = block_id_mn / grid_m;
        const index_t block_k     = get_block_id() / grid_mn;

        const index_t batch_start = block_m * kMTile;
        const index_t out_start   = block_n * kNTile;
        const index_t k_start     = block_k * k_per_block;
        const index_t k_end       = ck_tile::min(k_start + k_per_block, nC);

        if(batch_start >= batch || out_start >= output_dim || k_start >= nC)
            return;

        // Setup shared memory
        __shared__ char smem_ptr[GetSmemSize()];

        // Setup X (A) tensor and window
        auto x_tensor_full = make_naive_tensor_view<address_space_enum::global>(
            p_x, make_tuple(batch, nC), make_tuple(nC, 1), number<1>{}, number<1>{});
        auto x_tensor_padded = pad_tensor_view(x_tensor_full,
                                               make_tuple(number<kMTile>{}, number<kKTile>{}),
                                               sequence<false, Problem::kPadK>{});

        auto x_dram_window = make_tile_window(x_tensor_padded,
                                              make_tuple(number<kMTile>{}, number<kKTile>{}),
                                              {batch_start, k_start});

        // Setup Phi (B) tensor and window
        // Note: We pad to kNTile (not kNTileLogical) for pipeline compatibility
        auto phi_tensor_full = make_naive_tensor_view<address_space_enum::global>(
            p_phi, make_tuple(output_dim, nC), make_tuple(1, output_dim), number<1>{}, number<1>{});
        auto phi_tensor_padded = pad_tensor_view(phi_tensor_full,
                                                 make_tuple(number<kNTile>{}, number<kKTile>{}),
                                                 sequence<Problem::kPadN, Problem::kPadK>{});

        auto phi_dram_window = make_tile_window(phi_tensor_padded,
                                                make_tuple(number<kNTile>{}, number<kKTile>{}),
                                                {out_start, k_start});

        // Element-wise functions (pass-through for now)
        constexpr auto PassThrough = [](auto& e, const auto& x) { e = x; };

        // Create the fused pipeline
        Pipeline pipeline;

        // Simpler approach: Use full kMTile array but accept the register pressure
        // For M=64, this is 64 floats = 256 bytes in VGPRs
        // Trade-off: 20% performance loss but correct norm computation
        ComputeDataType norm_accum[kMTile];
        for(index_t m = 0; m < kMTile; ++m)
            norm_accum[m] = 0.0f;

        // Track which K tile we're on for bounds checking
        index_t current_k_tile = 0;

        auto fusion_func = [&](auto& a_tile) {
            using ATileType           = ck_tile::remove_cvref_t<decltype(a_tile)>;
            constexpr auto tile_spans = ATileType::get_distributed_spans();

            // Calculate the actual K range for this tile
            const index_t tile_k_start = k_start + current_k_tile * kKTile;
            const index_t tile_k_end   = ck_tile::min(tile_k_start + kKTile, k_end);
            const bool is_last_tile    = (tile_k_end < tile_k_start + kKTile);

            // Fast path: If not the last tile, no padding - accumulate everything
            if(!is_last_tile)
            {
                sweep_tile_span(tile_spans[number<0>{}], [&](auto idx_m) {
                    sweep_tile_span(tile_spans[number<1>{}], [&](auto idx_k) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            a_tile.get_tile_distribution(), make_tuple(idx_m, idx_k));

                        const index_t local_m  = tile_idx.at(number<0>{});
                        constexpr auto i_j_idx = make_tuple(idx_m, idx_k);
                        ComputeDataType val    = type_convert<ComputeDataType>(a_tile[i_j_idx]);
                        norm_accum[local_m] += val * val;
                    });
                });
            }
            else
            {
                // Slow path: Last tile may have padding - check bounds
                sweep_tile_span(tile_spans[number<0>{}], [&](auto idx_m) {
                    sweep_tile_span(tile_spans[number<1>{}], [&](auto idx_k) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            a_tile.get_tile_distribution(), make_tuple(idx_m, idx_k));

                        const index_t local_m  = tile_idx.at(number<0>{});
                        const index_t local_k  = tile_idx.at(number<1>{});
                        const index_t global_k = tile_k_start + local_k;

                        // Only accumulate if within actual data range
                        if(global_k < tile_k_end)
                        {
                            constexpr auto i_j_idx = make_tuple(idx_m, idx_k);
                            ComputeDataType val    = type_convert<ComputeDataType>(a_tile[i_j_idx]);
                            norm_accum[local_m] += val * val;
                        }
                    });
                });
            }

            // Increment tile counter for next iteration
            current_k_tile++;
        };

        auto c_block_tile = pipeline(x_dram_window,
                                     PassThrough,
                                     phi_dram_window,
                                     PassThrough,
                                     k_tiles_per_block,
                                     smem_ptr,
                                     fusion_func);

        // After all tiles, reduce norms and store
        __shared__ ComputeDataType norm_sums[kMTile];
        const index_t thread_id = get_thread_id();
        for(index_t i = thread_id; i < kMTile; i += kBlockSize)
            norm_sums[i] = 0.0f;
        block_sync_lds();

        // Two-level reduction: warp-level shuffle + cross-warp atomic
        // This is critical for multi-warp blocks (kBlockSize=128 = 2 or 4 warps)
        for(index_t m = 0; m < kMTile; ++m)
        {
            ComputeDataType partial_sum = norm_accum[m];

// Level 1: Warp-level reduction using shuffle
#pragma unroll
            for(index_t offset = get_warp_size() / 2; offset > 0; offset >>= 1)
                partial_sum += __shfl_down(partial_sum, offset);

            // Level 2: Cross-warp reduction using atomics
            // Only lane 0 of each warp writes to shared memory
            if((thread_id % get_warp_size()) == 0 && partial_sum != 0.0f)
                atomicAdd(&norm_sums[m], partial_sum);
        }
        block_sync_lds();

        // Store partial norms to global memory
        for(index_t i = thread_id; i < kMTile; i += kBlockSize)
        {
            const index_t global_m = batch_start + i;
            if(global_m < batch)
                p_partial_norms[block_k * batch + global_m] = norm_sums[i];
        }

        // Store partial GEMM results to workspace
        const index_t grid_k_total = (nC + k_per_block - 1) / k_per_block;

        constexpr auto result_spans = decltype(c_block_tile)::get_distributed_spans();
        sweep_tile_span(result_spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(result_spans[number<1>{}], [&](auto idx1) {
                const auto tile_idx = get_x_indices_from_distributed_indices(
                    c_block_tile.get_tile_distribution(), make_tuple(idx0, idx1));

                const index_t local_m  = tile_idx.at(number<0>{});
                const index_t local_n  = tile_idx.at(number<1>{});
                const index_t global_m = batch_start + local_m;
                const index_t global_n = out_start + local_n;

                if(global_m < batch && global_n < output_dim && local_n < kNTileLogical)
                {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    ComputeDataType value  = c_block_tile[i_j_idx];
                    const index_t workspace_idx =
                        global_m * (grid_k_total * output_dim) + block_k * output_dim + global_n;
                    p_workspace[workspace_idx] = value;
                }
            });
        });
    }
};

} // namespace ck_tile
