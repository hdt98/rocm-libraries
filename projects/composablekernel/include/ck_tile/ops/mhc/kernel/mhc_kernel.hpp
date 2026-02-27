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

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    // Adaptive K-tiles per block based on C dimension
    CK_TILE_HOST_DEVICE static constexpr index_t GetKTilesPerBlock(index_t nC)
    {
        // Adaptive selection based on C size:
        // - Large C (≥4096): 8 tiles/block (512 elements) - maximize MFMA utilization
        // - Medium C (≥1024): 4 tiles/block (256 elements) - balance overhead and work
        // - Small C (≥256): 2 tiles/block (128 elements) - reduce overhead
        // - Tiny C (<256): 1 tile/block (64 elements) - minimize overhead
        return (nC >= 4096) ? 8 : (nC >= 1024) ? 4 : (nC >= 256) ? 2 : 1;
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return kBlockSize; }

    // Padding to avoid LDS bank conflicts
    // AMD GPUs have 32 LDS banks, 4-byte bank width
    // For bf16 (2 bytes), we need padding to avoid stride being multiple of 32
    static constexpr index_t kKTilePadded = kKTile + 8; // Add 8 elements padding

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        // Use GEMM pipeline's smem size calculation (handles swizzled layout correctly)
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

        constexpr index_t smem_size_a = GemmPolicy::GetSmemSizeA<Problem>();
        constexpr index_t smem_size_b = GemmPolicy::GetSmemSizeB<Problem>();

        __shared__ char smem_ptr[smem_size_a + smem_size_b];
        XDataType* x_lds     = reinterpret_cast<XDataType*>(smem_ptr);
        PhiDataType* phi_lds = reinterpret_cast<PhiDataType*>(smem_ptr + smem_size_a);

        // Use BlockGemmARegBSmemCRegV2 for multi-warp (like FMHA), ASmemBSmem for single-warp
        constexpr index_t NumWarps = Problem::BlockGemmShape::NumWarps;

        using BlockGemm = std::conditional_t<NumWarps == 1,
                                             BlockGemmASmemBSmemCRegV1<Problem, Policy>,
                                             BlockGemmARegBSmemCRegV2<Problem, Policy>>;

        auto result_tile = BlockGemm::MakeCBlockTile();
        set_tile(result_tile, 0.0f);

        // Create tensor views for X and Phi
        auto x_tensor_full = make_naive_tensor_view<address_space_enum::global>(
            p_x, make_tuple(batch, nC), make_tuple(nC, 1), number<1>{}, number<1>{});

        auto x_tensor_padded = pad_tensor_view(x_tensor_full,
                                               make_tuple(number<kMTile>{}, number<kKTile>{}),
                                               sequence<false, Problem::kPadK>{});

        // CRITICAL: For multi-warp, use BlockGemm's expected distribution (like FMHA)
        // For single-warp, use Problem's load distribution
        constexpr auto x_load_tile_dist = []() {
            if constexpr(NumWarps == 1)
            {
                return Problem::MakeXLoadTileDistribution();
            }
            else
            {
                return BlockGemm::template MakeABlockTileDistribution<kMTile, kKTile>();
            }
        }();

        auto x_dram_window = make_tile_window(x_tensor_padded,
                                              make_tuple(number<kMTile>{}, number<kKTile>{}),
                                              {batch_start, k_start},
                                              x_load_tile_dist);

        // Use GEMM's swizzled LDS descriptor for X
        auto x_lds_tensor = make_tensor_view<address_space_enum::lds>(x_lds, a_lds_block_desc);
        auto x_lds_window =
            make_tile_window(x_lds_tensor, make_tuple(number<kMTile>{}, number<kKTile>{}), {0, 0});

        // Create Phi tensor view and window
        auto phi_tensor_full = make_naive_tensor_view<address_space_enum::global>(
            p_phi, make_tuple(output_dim, nC), make_tuple(1, output_dim), number<1>{}, number<1>{});

        auto phi_tensor_padded = pad_tensor_view(phi_tensor_full,
                                                 make_tuple(number<kNTile>{}, number<kKTile>{}),
                                                 sequence<false, Problem::kPadK>{});

        constexpr auto phi_load_tile_dist = Problem::MakePhiLoadTileDistribution();
        auto phi_dram_window              = make_tile_window(phi_tensor_padded,
                                                make_tuple(number<kNTile>{}, number<kKTile>{}),
                                                             {out_start, k_start},
                                                phi_load_tile_dist);

        // Use GEMM's swizzled LDS descriptor for Phi
        auto phi_lds_tensor = make_tensor_view<address_space_enum::lds>(phi_lds, b_lds_block_desc);
        auto phi_lds_window = make_tile_window(
            phi_lds_tensor, make_tuple(number<kNTile>{}, number<kKTile>{}), {0, 0});

        // Step 1: Compute partial norms - one per batch element in this tile
        __shared__ ComputeDataType norm_sums[kMTile];
        const index_t thread_id = get_thread_id();

        // Initialize
        for(index_t i = thread_id; i < kMTile; i += kBlockSize)
        {
            norm_sums[i] = 0.0f;
        }
        block_sync_lds();

        // Each batch element gets its own norm
        const index_t k_range      = k_end - k_start;
        constexpr index_t kVecSize = 4;

        // Compute norms for each batch element
        for(index_t local_m = 0; local_m < kMTile; ++local_m)
        {
            const index_t global_m = batch_start + local_m;
            if(global_m >= batch)
                continue;

            ComputeDataType partial_sum = 0.0f;
            const XDataType* row_ptr    = p_x + global_m * nC + k_start;

            for(index_t k = thread_id * kVecSize; k < k_range; k += kBlockSize * kVecSize)
            {
                if(k + kVecSize <= k_range)
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
                else
                {
                    for(index_t i = 0; i < kVecSize && k + i < k_range; ++i)
                    {
                        ComputeDataType val = type_convert<ComputeDataType>(row_ptr[k + i]);
                        partial_sum += val * val;
                    }
                }
            }

// Use warp-level reduction (AMD warp size = 64)
#pragma unroll
            for(index_t offset = 32; offset > 0; offset >>= 1)
            {
                partial_sum += __shfl_down(partial_sum, offset);
            }

            // First thread in warp writes to shared memory (only 1 atomic per warp!)
            if((thread_id % get_warp_size()) == 0 && partial_sum != 0.0f)
            {
                atomicAdd(&norm_sums[local_m], partial_sum);
            }
        }

        block_sync_lds();

        // Write partial norms
        for(index_t i = thread_id; i < kMTile; i += kBlockSize)
        {
            const index_t global_m = batch_start + i;
            if(global_m < batch)
            {
                p_partial_norms[block_k * batch + global_m] = norm_sums[i];
            }
        }

        block_sync_lds();

        // Loop over K-tiles within this block's K-range (adaptive count)
        for(index_t k_tile_idx = 0; k_tile_idx < k_tiles_per_block; ++k_tile_idx)
        {
            const index_t k_current = k_start + k_tile_idx * kKTile;
            if(k_current >= k_end)
                break;

            // SCOPED SECTION 1: Load and process X tile
            // Limit x_tile lifetime to reduce register pressure
            {
                auto x_tile = make_static_distributed_tensor<XDataType>(x_load_tile_dist);
                load_tile(x_tile, x_dram_window);

                // For multi-warp (BlockGemmARegBSmemCRegV2): X stays in registers
                // For single-warp (BlockGemmASmemBSmemCRegV1): X goes to LDS
                if constexpr(NumWarps == 1)
                {
                    store_tile(x_lds_window, x_tile);
                    // x_tile goes out of scope here for single-warp, freeing registers
                }
                // For multi-warp, x_tile must stay alive for GEMM below

                // SCOPED SECTION 2: Load and process Phi tile
                // Limit phi_tile lifetime to reduce register pressure
                {
                    auto phi_tile = make_static_distributed_tensor<PhiDataType>(phi_load_tile_dist);
                    load_tile(phi_tile, phi_dram_window);
                    store_tile(phi_lds_window, phi_tile);
                    // phi_tile goes out of scope here, freeing registers
                }

                block_sync_lds();

                // Accumulate GEMM
                if constexpr(NumWarps == 1)
                {
                    // Single-warp: both from LDS (x_tile already freed)
                    BlockGemm{}(result_tile, x_lds_window, phi_lds_window);
                }
                else
                {
                    // Multi-warp: X from registers, Phi from LDS
                    BlockGemm{}(result_tile, x_tile, phi_lds_window);
                }
                // x_tile goes out of scope here for multi-warp, freeing registers
            }

            // Move windows to next K-tile
            move_tile_window(x_dram_window, {0, kKTile});
            move_tile_window(phi_dram_window, {0, kKTile});

            // NOTE: No sync needed here! Next iteration will sync before GEMM anyway
            // Removing this sync cuts multi-warp overhead in half
        }

        // Store partial results to workspace buffer: p_workspace[block_k, batch, output_dim]
        // Layout: [grid_k][batch][output_dim]
        constexpr auto result_spans = decltype(result_tile)::get_distributed_spans();
        sweep_tile_span(result_spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(result_spans[number<1>{}], [&](auto idx1) {
                const auto tile_idx = get_x_indices_from_distributed_indices(
                    result_tile.get_tile_distribution(), make_tuple(idx0, idx1));

                const index_t local_m  = tile_idx.at(number<0>{});
                const index_t local_n  = tile_idx.at(number<1>{});
                const index_t global_m = batch_start + local_m;
                const index_t global_n = out_start + local_n;

                if(global_m < batch && global_n < output_dim)
                {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    ComputeDataType value  = result_tile[i_j_idx];

                    // Store to workspace: [block_k][global_m][global_n]
                    const index_t workspace_idx =
                        block_k * (batch * output_dim) + global_m * output_dim + global_n;
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

    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kVecSize   = 4; // Vectorized loads

    CK_TILE_HOST static constexpr auto BlockSize() { return kBlockSize; }

    CK_TILE_DEVICE void operator()(const ComputeDataType* p_workspace,
                                   [[maybe_unused]] const ComputeDataType* p_partial_norms,
                                   YDataType* p_output,
                                   index_t batch,
                                   [[maybe_unused]] index_t nC,
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

        // Step 2: Compute final norm from partial norms
        const index_t global_m = global_idx / output_dim;

        ComputeDataType sum_squares = 0.0f;
        const index_t norm_base     = global_m;

        index_t k = 0;
        for(; k + kVecSize <= grid_k; k += kVecSize)
        {
            using VecType = ext_vector_t<ComputeDataType, kVecSize>;
            VecType vec_norms;

#pragma unroll
            for(index_t i = 0; i < kVecSize; ++i)
            {
                vec_norms[i] = p_partial_norms[(k + i) * batch + norm_base];
            }

#pragma unroll
            for(index_t i = 0; i < kVecSize; ++i)
            {
                sum_squares += vec_norms[i];
            }
        }

        for(; k < grid_k; ++k)
        {
            sum_squares += p_partial_norms[k * batch + norm_base];
        }

        const ComputeDataType sqrt_nC = ck_tile::sqrt(static_cast<ComputeDataType>(nC));
        ComputeDataType norm          = ck_tile::sqrt(sum_squares) / sqrt_nC;
        norm                          = (norm > 1e-12f) ? norm : 1.0f;

        // Step 3: Reduce partial GEMM results and apply normalization (no activation yet)
        ComputeDataType value          = 0.0f;
        const index_t workspace_stride = batch * output_dim;

        // Vectorized reduction for workspace
        k = 0;
        for(; k + kVecSize <= grid_k; k += kVecSize)
        {
            using VecType = ext_vector_t<ComputeDataType, kVecSize>;
            VecType vec_values;

#pragma unroll
            for(index_t i = 0; i < kVecSize; ++i)
            {
                const index_t workspace_idx = (k + i) * workspace_stride + global_idx;
                vec_values[i]               = p_workspace[workspace_idx];
            }

#pragma unroll
            for(index_t i = 0; i < kVecSize; ++i)
            {
                value += vec_values[i];
            }
        }

        // Handle remaining elements
        for(; k < grid_k; ++k)
        {
            const index_t workspace_idx = k * workspace_stride + global_idx;
            value += p_workspace[workspace_idx];
        }

        // Step 4: Apply formulas - write H^pre and H^post directly, buffer H^res
        const index_t global_n = global_idx % output_dim;

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
