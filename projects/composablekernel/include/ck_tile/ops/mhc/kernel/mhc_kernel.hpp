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
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/mhc/kernel/block_sinkhorn_reduce.hpp"

// Manifold Constrained Hyper Connection Kernel V5:
// =====================================================================
// Split-K implementation with 2D grid (B, C):
// - Grid dimension 0: Batch tiles (B / kMTile)
// - Grid dimension 1: C tiles (nC / kKTile) - split-K dimension
// - Each block computes partial GEMM for its C-tile
// - Results stored to workspace buffer (no atomics!)
// - Separate reduction kernel combines partial results

namespace ck_tile {

template <typename Problem_,
          typename Policy_     = MHCDefaultPolicy,
          typename Activation_ = element_wise::Sigmoid>
struct MHCKernelV5
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

    // Adaptive K-tiles per block based on C dimension
    CK_TILE_HOST_DEVICE static constexpr index_t GetKTilesPerBlock(index_t nC)
    {
        // OPTIMIZED: Adaptive selection based on C size to minimize split-K overhead
        // - Very Large C (≥8192): 16 tiles/block (1024 elements) - minimize split-K overhead
        // - Large C (≥4096): 12 tiles/block (768 elements) - balance overhead and work
        // - Medium C (≥1024): 8 tiles/block (512 elements) - good MFMA utilization
        // - Small C (≥256): 4 tiles/block (256 elements) - reduce overhead
        // - Tiny C (<256): 2 tiles/block (128 elements) - minimize overhead
        return (nC >= 8192) ? 16 : (nC >= 4096) ? 12 : (nC >= 1024) ? 8 : (nC >= 256) ? 4 : 2;
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return kBlockSize; }

    // Padding to avoid LDS bank conflicts
    // AMD GPUs have 32 LDS banks, 4-byte bank width
    // For bf16 (2 bytes), we need padding to avoid stride being multiple of 32
    static constexpr index_t kKTilePadded = kKTile + 8; // Add 8 elements padding

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        using GemmPolicy = GemmPipelineAGmemBGmemCRegV1DefaultPolicy;
        return GemmPolicy::GetSmemSizeA<Problem>() + GemmPolicy::GetSmemSizeB<Problem>();
    }

    // Grid configuration: 3D grid (M, N, K) for split-K with output tiling
    // Each block processes adaptive number of K-tiles (hierarchical split-K)
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
        // This reduces LDS usage and eliminates wasted GEMM work
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
        // Use kNTileLogical for LDS window to match actual loaded data
        auto phi_lds_window = make_tile_window(
            phi_lds_tensor, make_tuple(number<kNTileLogical>{}, number<kKTile>{}), {0, 0});

        // Step 1: Dedicated norm phase — one pass over p_x, then K-loop (best for large C).
        __shared__ ComputeDataType norm_sums[kMTile];
        const index_t thread_id = get_thread_id();
        for(index_t i = thread_id; i < kMTile; i += kBlockSize)
            norm_sums[i] = 0.0f;
        block_sync_lds();

        const index_t k_range       = k_end - k_start;
        constexpr index_t kVecSize  = 4;
        constexpr index_t kVecSize8 = 8;
        for(index_t local_m = 0; local_m < kMTile; ++local_m)
        {
            const index_t global_m = batch_start + local_m;
            if(global_m >= batch)
                continue;
            ComputeDataType partial_sum = 0.0f;
            const XDataType* row_ptr    = p_x + global_m * nC + k_start;
            index_t k                   = thread_id * kVecSize8;
            for(; k + kVecSize8 <= k_range; k += kBlockSize * kVecSize8)
            {
                using VecType8 = ext_vector_t<XDataType, kVecSize8>;
                VecType8 vec   = *c_style_pointer_cast<const VecType8*>(row_ptr + k);
#pragma unroll
                for(index_t i = 0; i < kVecSize8; ++i)
                {
                    ComputeDataType val = type_convert<ComputeDataType>(vec[i]);
                    partial_sum += val * val;
                }
            }
            for(; k + kVecSize <= k_range; k += kBlockSize * kVecSize)
            {
                using VecType = ext_vector_t<XDataType, kVecSize>;
                VecType vec   = *c_style_pointer_cast<const VecType*>(row_ptr + k);
#pragma unroll
                for(index_t i = 0; i < kVecSize; ++i)
                {
                    ComputeDataType val = type_convert<ComputeDataType>(vec[i]);
                    partial_sum += val * val;
                }
            }
            for(; k < k_range; ++k)
            {
                ComputeDataType val = type_convert<ComputeDataType>(row_ptr[k]);
                partial_sum += val * val;
            }
#pragma unroll
            for(index_t offset = get_warp_size() / 2; offset > 0; offset >>= 1)
                partial_sum += __shfl_down(partial_sum, offset);
            if((thread_id % get_warp_size()) == 0 && partial_sum != 0.0f)
                atomicAdd(&norm_sums[local_m], partial_sum);
        }
        block_sync_lds();
        for(index_t i = thread_id; i < kMTile; i += kBlockSize)
        {
            const index_t global_m = batch_start + i;
            if(global_m < batch)
                p_partial_norms[block_k * batch + global_m] = norm_sums[i];
        }
        block_sync_lds();

        // Step 2: K-loop — GEMM over tiles with GEMM pipeline scheduling pattern
        // Pattern: load next → sync → GEMM current → sync → store next → move
        // This overlaps "load next" with "GEMM current" at instruction level

        // Prologue: Load tile 0 into registers and store to LDS
        auto x_tile   = make_static_distributed_tensor<XDataType>(x_load_tile_dist);
        auto phi_tile = make_static_distributed_tensor<PhiDataType>(phi_load_tile_dist);

        if(k_tiles_per_block > 0 && k_start < k_end)
        {
            // Load tile 0
            load_tile(x_tile, x_dram_window);
            load_tile(phi_tile, phi_dram_window);

            // Store tile 0 to LDS
            if constexpr(NumWarps == 1)
                store_tile(x_lds_window, x_tile);
            store_tile(phi_lds_window, phi_tile);

            // Move to tile 1
            move_tile_window(x_dram_window, {0, kKTile});
            move_tile_window(phi_dram_window, {0, kKTile});
        }

        // Main loop: Process tiles 0 to k_tiles_per_block-1
        for(index_t k_tile_idx = 0; k_tile_idx < k_tiles_per_block; ++k_tile_idx)
        {
            const index_t k_current = k_start + k_tile_idx * kKTile;
            if(k_current >= k_end)
                break;

            const bool has_next_tile = (k_tile_idx + 1 < k_tiles_per_block) &&
                                       (k_start + (k_tile_idx + 1) * kKTile < k_end);

            // Single-warp path: load next → sync → GEMM → sync → store next
            if constexpr(NumWarps == 1)
            {
                if(has_next_tile)
                {
                    // Load next tile (tile i+1) into registers FIRST
                    load_tile(x_tile, x_dram_window);
                    load_tile(phi_tile, phi_dram_window);
                }

                // Sync before GEMM
                block_sync_lds();

                // GEMM on current tile (tile i) from LDS
                BlockGemm{}(result_tile, x_lds_window, phi_lds_window);

                if(has_next_tile)
                {
                    // Sync before storing next tile
                    block_sync_lds();

                    // Store next tile (tile i+1) to LDS
                    store_tile(x_lds_window, x_tile);
                    store_tile(phi_lds_window, phi_tile);

                    // Move to tile i+2
                    move_tile_window(x_dram_window, {0, kKTile});
                    move_tile_window(phi_dram_window, {0, kKTile});
                }
            }
            else
            {
                // Multi-warp path: x_tile holds current data from previous iteration
                // Save current x_tile before potentially loading next
                auto x_tile_current = x_tile;

                if(has_next_tile)
                {
                    // Load next tile into x_tile (overwrites it)
                    load_tile(x_tile, x_dram_window);
                    load_tile(phi_tile, phi_dram_window);
                }

                // Sync before GEMM
                block_sync_lds();

                // GEMM uses the saved current tile from registers
                BlockGemm{}(result_tile, x_tile_current, phi_lds_window);

                if(has_next_tile)
                {
                    // Sync before storing next tile
                    block_sync_lds();

                    // Store next tile to LDS (x_tile already has next data)
                    store_tile(phi_lds_window, phi_tile);

                    // Move to tile i+2
                    move_tile_window(x_dram_window, {0, kKTile});
                    move_tile_window(phi_dram_window, {0, kKTile});
                }
            }
        }

        // Store partial results to workspace buffer
        // OPTIMIZED LAYOUT: [batch][grid_k][output_dim] for coalesced reduction access
        // For N=24: only store the first kNTileLogical columns (skip padding)
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

// Optimized reduction kernel with block-level shared memory reduction
template <typename Problem_, typename Activation_ = element_wise::Sigmoid>
struct MHCReductionKernel
{
    using Activation      = ck_tile::remove_cvref_t<Activation_>;
    using Problem         = ck_tile::remove_cvref_t<Problem_>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;

    static constexpr index_t kBlockSize =
        256; // 512 can help large shapes but hurts small (fewer blocks)
    static constexpr index_t kVecSize     = 4;
    static constexpr index_t kVecSize8    = 8;   // 8-wide for workspace when grid_k allows
    static constexpr index_t kMaxGridKLDS = 128; // max grid_k for LDS-backed norm load

    CK_TILE_HOST static constexpr auto BlockSize() { return kBlockSize; }

    CK_TILE_DEVICE void operator()(const ComputeDataType* p_workspace,
                                   const ComputeDataType* p_partial_norms,
                                   YDataType* p_output,
                                   index_t batch,
                                   index_t nC,
                                   index_t output_dim,
                                   [[maybe_unused]] index_t n,
                                   index_t grid_k,
                                   [[maybe_unused]] float alpha_pre,
                                   [[maybe_unused]] float alpha_post,
                                   [[maybe_unused]] float alpha_res,
                                   [[maybe_unused]] float bias,
                                   [[maybe_unused]] index_t sinkhorn_iters = 0) const
    {
        const index_t tid        = get_thread_id();
        const index_t block_id   = get_block_id();
        const index_t block_size = get_block_size();

        const index_t elements_per_block = block_size;
        const index_t global_start       = block_id * elements_per_block;
        const index_t total_elements     = batch * output_dim;

        const index_t global_idx = global_start + tid;

        if(global_idx >= total_elements)
            return;

        const index_t global_m = global_idx / output_dim;
        const index_t global_n = global_idx % output_dim;

        // Step 2: Compute final norm from partial norms
        // Block-cooperative load: one thread per distinct global_m loads grid_k norms into LDS,
        // then all threads with that global_m read from LDS. Cuts partial_norms global reads
        // by ~output_dim (was 256*grid_k, now ~(256/output_dim)*grid_k).
        ComputeDataType norm = 1.0f;
        if(grid_k <= kMaxGridKLDS)
        {
            __shared__ ComputeDataType norms_lds[kMaxGridKLDS];
            const index_t global_m_start = (block_id * block_size) / output_dim;
            const index_t global_m_end   = (block_id * block_size + block_size - 1) / output_dim;

            for(index_t gm = global_m_start; gm <= global_m_end; ++gm)
            {
                // First thread in this block with global_m == gm (may be tid 0 when gm*output_dim <
                // block_start)
                const index_t first_global_idx_for_gm =
                    ck_tile::max(gm * output_dim, block_id * block_size);
                const index_t tid_loader = first_global_idx_for_gm - block_id * block_size;
                if(tid_loader < block_size && tid == tid_loader)
                {
                    for(index_t k = 0; k < grid_k; ++k)
                        norms_lds[k] = p_partial_norms[k * batch + gm];
                }
                block_sync_lds();

                if(global_m == gm)
                {
                    ComputeDataType sum_squares = 0.0f;
                    for(index_t k = 0; k < grid_k; ++k)
                        sum_squares += norms_lds[k];
                    const ComputeDataType sqrt_nC = ck_tile::sqrt(static_cast<ComputeDataType>(nC));
                    norm                          = ck_tile::sqrt(sum_squares) / sqrt_nC;
                    norm                          = (norm > 1e-12f) ? norm : 1.0f;
                }
                block_sync_lds();
            }
        }
        else
        {
            ComputeDataType sum_squares = 0.0f;
            index_t k                   = 0;
            for(; k + kVecSize <= grid_k; k += kVecSize)
            {
                using VecType = ext_vector_t<ComputeDataType, kVecSize>;
                VecType vec_norms;
#pragma unroll
                for(index_t i = 0; i < kVecSize; ++i)
                    vec_norms[i] = p_partial_norms[(k + i) * batch + global_m];
#pragma unroll
                for(index_t i = 0; i < kVecSize; ++i)
                    sum_squares += vec_norms[i];
            }
            for(; k < grid_k; ++k)
                sum_squares += p_partial_norms[k * batch + global_m];
            const ComputeDataType sqrt_nC = ck_tile::sqrt(static_cast<ComputeDataType>(nC));
            norm                          = ck_tile::sqrt(sum_squares) / sqrt_nC;
            norm                          = (norm > 1e-12f) ? norm : 1.0f;
        }

        // Step 3: Reduce partial GEMM results (workspace) - 8-wide then 4-wide then scalar
        const index_t workspace_base = global_m * (grid_k * output_dim) + global_n;
        ComputeDataType value        = 0.0f;
        index_t k                    = 0;

        for(; k + kVecSize8 <= grid_k; k += kVecSize8)
        {
            using VecType8 = ext_vector_t<ComputeDataType, kVecSize8>;
            VecType8 vec_values;
#pragma unroll
            for(index_t i = 0; i < kVecSize8; ++i)
                vec_values[i] = p_workspace[workspace_base + (k + i) * output_dim];
#pragma unroll
            for(index_t i = 0; i < kVecSize8; ++i)
                value += vec_values[i];
        }
        for(; k + kVecSize <= grid_k; k += kVecSize)
        {
            using VecType = ext_vector_t<ComputeDataType, kVecSize>;
            VecType vec_values;
#pragma unroll
            for(index_t i = 0; i < kVecSize; ++i)
                vec_values[i] = p_workspace[workspace_base + (k + i) * output_dim];
#pragma unroll
            for(index_t i = 0; i < kVecSize; ++i)
                value += vec_values[i];
        }
        for(; k < grid_k; ++k)
            value += p_workspace[workspace_base + k * output_dim];

        // Step 4: Apply formulas - write H^pre and H^post directly, buffer H^res
        if(global_n < n)
        {
            // Section 1: H^pre [0:n] -> (alpha_pre/norm) * σ(GEMM) + bias
            YDataType activated;
            Activation{}(activated, type_convert<YDataType>(value));
            YDataType final_output = type_convert<YDataType>(
                (alpha_pre / norm) * type_convert<ComputeDataType>(activated) + bias);
            p_output[global_idx] = final_output;
        }
        else if(global_n < 2 * n)
        {
            // Section 2: H^post [n:2n] -> (alpha_post/norm) * 2*σ(GEMM) + bias
            YDataType activated;
            Activation{}(activated, type_convert<YDataType>(value));
            YDataType final_output = type_convert<YDataType>(
                (alpha_post / norm) * 2.0f * type_convert<ComputeDataType>(activated) + bias);
            p_output[global_idx] = final_output;
        }
        else
        {
            // Section 3: H^res [2n:2n+n²] -> normalize, bias, then optionally apply Sinkhorn
            ComputeDataType h_res_value = (alpha_res / norm) * value + bias;

            // NOTE: Sinkhorn is now handled by a separate kernel (MHCSinkhornKernel)
            // This allows better control over parallelization and avoids issues
            // when batch elements span multiple blocks

            // Just write the normalized value (Sinkhorn will be applied separately if needed)
            p_output[global_idx] = type_convert<YDataType>(h_res_value);
        }
    }
};

// Dedicated Sinkhorn kernel for MHC H^res normalization
// Direct memory access version - each thread independently processes one 4×4 matrix
template <typename YDataType, typename ComputeDataType = float>
struct MHCSinkhornKernel
{
    static constexpr index_t kN         = 4;
    static constexpr index_t kNSquared  = kN * kN;
    static constexpr index_t kBlockSize = 64; // For __launch_bounds__

    CK_TILE_HOST static constexpr index_t BlockSize() { return is_wave32() ? 32 : 64; }

    CK_TILE_DEVICE void operator()(YDataType* p_output,
                                   index_t batch,
                                   index_t output_dim,
                                   index_t n,
                                   index_t sinkhorn_iters) const
    {
        // Each thread processes one batch element's 4×4 matrix independently
        const index_t batch_idx = get_block_id() * get_block_size() + get_thread_id();

        if(batch_idx >= batch)
            return;

        // H^res starts at index 2*n for this batch element
        const index_t h_res_start = batch_idx * output_dim + 2 * n;

        // Direct memory access: load 4×4 matrix into registers with vectorization
        ComputeDataType matrix[kNSquared];

        // Vectorized load from global memory (row-major layout)
        // Load 4 rows of 4 elements each using vector loads for coalescing
        constexpr index_t kVecSize = 4;
        using VecType              = ext_vector_t<YDataType, kVecSize>;

        for(index_t row = 0; row < kN; ++row)
        {
            VecType vec = *c_style_pointer_cast<const VecType*>(p_output + h_res_start + row * kN);
            for(index_t col = 0; col < kN; ++col)
            {
                matrix[row * kN + col] = type_convert<ComputeDataType>(vec[col]);
            }
        }

        // Apply Sinkhorn-Knopp algorithm
        // Find max for numerical stability
        ComputeDataType max_val = matrix[0];
        for(index_t i = 1; i < kNSquared; ++i)
        {
            max_val = ck_tile::max(max_val, matrix[i]);
        }

        // Exponentiate with max subtracted
        for(index_t i = 0; i < kNSquared; ++i)
        {
            matrix[i] = ck_tile::exp(matrix[i] - max_val);
        }

        // Sinkhorn iterations
        for(index_t iter = 0; iter < sinkhorn_iters; ++iter)
        {
            // Normalize rows
            for(index_t row = 0; row < kN; ++row)
            {
                ComputeDataType row_sum = 0.0;
                for(index_t col = 0; col < kN; ++col)
                {
                    row_sum += matrix[row * kN + col];
                }

                if(row_sum > 1e-12)
                {
                    for(index_t col = 0; col < kN; ++col)
                    {
                        matrix[row * kN + col] /= row_sum;
                    }
                }
                else
                {
                    // Set to uniform distribution if row sum is too small
                    ComputeDataType uniform_val = 1.0 / static_cast<ComputeDataType>(kN);
                    for(index_t col = 0; col < kN; ++col)
                    {
                        matrix[row * kN + col] = uniform_val;
                    }
                }
            }

            // Normalize columns
            for(index_t col = 0; col < kN; ++col)
            {
                ComputeDataType col_sum = 0.0;
                for(index_t row = 0; row < kN; ++row)
                {
                    col_sum += matrix[row * kN + col];
                }

                if(col_sum > 1e-12)
                {
                    for(index_t row = 0; row < kN; ++row)
                    {
                        matrix[row * kN + col] /= col_sum;
                    }
                }
                else
                {
                    // Set to uniform distribution if column sum is too small
                    ComputeDataType uniform_val = 1.0 / static_cast<ComputeDataType>(kN);
                    for(index_t row = 0; row < kN; ++row)
                    {
                        matrix[row * kN + col] = uniform_val;
                    }
                }
            }
        }

        // Vectorized write back to global memory
        for(index_t row = 0; row < kN; ++row)
        {
            VecType vec{};
            for(index_t col = 0; col < kN; ++col)
            {
                vec[col] = type_convert<YDataType>(matrix[row * kN + col]);
            }
            *c_style_pointer_cast<VecType*>(p_output + h_res_start + row * kN) = vec;
        }
    }
};

// Tile-based version using 2D tensor view per thread
// NOTE: Tile distributions don't work when each thread processes a different batch
// because the distribution assumes all threads work on contiguous data
// Kept for reference - use MHCSinkhornKernel (hybrid version) instead
/*template <typename YDataType, typename ComputeDataType = float>
struct MHCSinkhornKernel_Tile
{
    static constexpr index_t kN = 4;
    static constexpr index_t kNSquared = kN * kN;
    static constexpr index_t kBlockSize = 64;

    CK_TILE_HOST static constexpr index_t BlockSize()
    {
        return is_wave32() ? 32 : 64;
    }

    CK_TILE_DEVICE void operator()(YDataType* p_output,
                                   index_t batch,
                                   index_t output_dim,
                                   index_t n,
                                   index_t sinkhorn_iters) const
    {
        // FMHA-style approach: Each thread processes ONE batch element independently
        const index_t batch_idx = get_block_id() * get_block_size() + get_thread_id();

        if(batch_idx >= batch)
            return;

        // H^res starts at index 2*n for this batch element
        const index_t h_res_start = batch_idx * output_dim + 2 * n;

        constexpr auto tile_dist = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<1, kN>,
                      sequence<1, kN>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<0, 0>>,
                sequence<1, 2>,
                sequence<1, 1>>{});

        auto input_tile = make_static_distributed_tensor<ComputeDataType>(tile_dist);
        auto output_tile = make_static_distributed_tensor<YDataType>(tile_dist);

        // Create tensor view for THIS batch's 4×4 H^res matrix
        // Memory layout: 16 consecutive elements in row-major order
        // Row stride is kN (4 elements per row)
        auto h_res_tensor = make_naive_tensor_view<address_space_enum::global>(
            p_output + h_res_start,
            make_tuple(kN, kN),      // [4, 4] matrix
            make_tuple(kN, 1),       // Row-major: stride is 4 for rows, 1 for columns
            number<kN>{}, number<1>{});

        // Window at origin - tile distribution will map to the 4×4 matrix
        auto h_res_window = make_tile_window(h_res_tensor,
                                            make_tuple(number<kN>{}, number<kN>{}),
                                            {0, 0},
                                            tile_dist);

        load_tile(input_tile, h_res_window);

        sinkhorn_knopp_naive_full<decltype(input_tile), decltype(output_tile), ComputeDataType>(
            input_tile, output_tile, sinkhorn_iters);

        store_tile(h_res_window, output_tile);
    }
};
*/

// Hybrid Sinkhorn kernel: Direct vectorized load + tile-based Sinkhorn processing
// Combines the benefits of vectorized memory access with tile-based computation
/*template <typename YDataType, typename ComputeDataType = float>
struct MHCSinkhornKernel
{
    static constexpr index_t kN = 4;
    static constexpr index_t kNSquared = kN * kN;
    static constexpr index_t kBlockSize = 64;

    CK_TILE_HOST static constexpr index_t BlockSize()
    {
        return is_wave32() ? 32 : 64;
    }

    CK_TILE_DEVICE void operator()(YDataType* p_output,
                                   index_t batch,
                                   index_t output_dim,
                                   index_t n,
                                   index_t sinkhorn_iters) const
    {
        // Each thread processes one batch element's 4×4 matrix independently
        const index_t batch_idx = get_block_id() * get_block_size() + get_thread_id();

        if(batch_idx >= batch)
            return;

        // H^res starts at index 2*n for this batch element
        const index_t h_res_start = batch_idx * output_dim + 2 * n;

        // Step 1: Direct vectorized load from global memory into VGPR buffer
        ComputeDataType matrix[kNSquared];
        constexpr index_t kVecSize = 4;
        using VecType = ext_vector_t<YDataType, kVecSize>;

        for(index_t row = 0; row < kN; ++row)
        {
            VecType vec = *c_style_pointer_cast<const VecType*>(p_output + h_res_start + row * kN);
            for(index_t col = 0; col < kN; ++col)
            {
                matrix[row * kN + col] = type_convert<ComputeDataType>(vec[col]);
            }
        }

        // Step 2: Create VGPR tensor view for the matrix
        auto matrix_tensor = make_naive_tensor_view<address_space_enum::vgpr>(
            matrix,
            make_tuple(kN, kN),
            make_tuple(kN, 1),
            number<kN>{}, number<1>{});

        // Step 3: Create tile distribution and distributed tensors
        constexpr auto tile_dist = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<1, kN>,
                      sequence<1, kN>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<0, 0>>,
                sequence<1, 2>,
                sequence<1, 1>>{});

        auto input_tile = make_static_distributed_tensor<ComputeDataType>(tile_dist);
        auto output_tile = make_static_distributed_tensor<ComputeDataType>(tile_dist);

        // Step 4: Create tile window on the VGPR tensor and load into input_tile
        auto matrix_window = make_tile_window(matrix_tensor,
                                             make_tuple(number<kN>{}, number<kN>{}),
                                             {0, 0},
                                             tile_dist);

        load_tile(input_tile, matrix_window);

        // Step 5: Apply Sinkhorn using the tile-based function
        sinkhorn_knopp_naive_full<decltype(input_tile), decltype(output_tile), ComputeDataType>(
            input_tile, output_tile, sinkhorn_iters);

        // Step 6: Store output_tile back to the VGPR buffer via tile window
        store_tile(matrix_window, output_tile);

        // Step 7: Vectorized write back to global memory
        for(index_t row = 0; row < kN; ++row)
        {
            VecType vec{};
            for(index_t col = 0; col < kN; ++col)
            {
                vec[col] = type_convert<YDataType>(matrix[row * kN + col]);
            }
            *c_style_pointer_cast<VecType*>(p_output + h_res_start + row * kN) = vec;
        }
    }
};*/

} // namespace ck_tile
