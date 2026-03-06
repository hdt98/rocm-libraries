// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <mint/mint.h>
#include "../block_level/block_gemm_pipeline_agmem_bgmem_creg.hpp"

namespace mint {

// Grid-level GEMM orchestration
// Each block computes a tile of the output matrix
template <typename Problem, typename Policy>
struct GridGemm
{
    using ADataType = typename Problem::ADataType;
    using BDataType = typename Problem::BDataType;
    using CDataType = typename Problem::CDataType;
    using AccDataType = typename Problem::AccDataType;

    static constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;
    static constexpr index_t kBlockSize = Problem::kBlockSize;

    // Warp configuration
    using WarpPolicy = BlockGemmASmemBSmemCRegPolicy;
    static constexpr index_t kMPerWarp = WarpPolicy::kMPerWarp;
    static constexpr index_t kNPerWarp = WarpPolicy::kNPerWarp;
    static constexpr index_t kKPerWarp = WarpPolicy::kKPerWarp;
    static constexpr index_t kMWarp = WarpPolicy::kMWarp;
    static constexpr index_t kNWarp = WarpPolicy::kNWarp;

    template <typename AGmemView, typename BGmemView, typename CGmemView>
    MINT_DEVICE void operator()(const AGmemView& a_gmem_view,
                                const BGmemView& b_gmem_view,
                                CGmemView& c_gmem_view,
                                index_t M,
                                index_t N,
                                index_t K) const
    {
        // Block index calculation
        const index_t num_tile_n = (N + kNPerBlock - 1) / kNPerBlock;
        const index_t block_id = blockIdx.x;
        const index_t m_block_idx = block_id / num_tile_n;
        const index_t n_block_idx = block_id % num_tile_n;

        const index_t m_block_start = m_block_idx * kMPerBlock;
        const index_t n_block_start = n_block_idx * kNPerBlock;

        // Early exit if out of bounds
        if(m_block_start >= M || n_block_start >= N)
            return;

        // Create C accumulator distributed tensor
        constexpr auto warp_matmul_dstr =
            BlockGemmASmemBSmemCReg<Problem>::make_warp_matmul_distribution();

        constexpr auto c_warp_dstr = extract_dimension_distribution(
            warp_matmul_dstr, sequence<alias_t, alias_t{"M"}, alias_t{"N"}>{});

        constexpr auto c_warp_elem_layout =
            make_aliased_naive_packed_tensor_descriptor(
                make_index_sequence<c_warp_dstr.element_ndim()>{},
                index_constant<-1>{},
                c_warp_dstr.element_lengths());

        // Create accumulator in registers
        auto c_acc_dist_tensor = make_distributed_tensor_vgpr<AccDataType>(
            constant<c_warp_dstr>{}, constant<c_warp_elem_layout>{});

        // Initialize accumulator to zero
        c_acc_dist_tensor.fill(static_cast<AccDataType>(0));

        // Create block pipeline
        BlockGemmPipelineAGmemBGmemCReg<Problem, Policy> block_pipeline;

        // Execute block-level GEMM
        block_pipeline(a_gmem_view, b_gmem_view, c_acc_dist_tensor,
                      m_block_start, n_block_start, K);

        // Reduce across K dimension if using warp matmul
        tile::simt::warp::reduce_sync(
            c_acc_dist_tensor,
            warp_matmul_dstr,
            sequence<alias_t, alias_t{"K"}>{},
            [](auto a, auto b) { return a + b; });

        // Convert accumulator to output type
        auto c_warp_dist_tensor = make_distributed_tensor_vgpr<CDataType>(
            constant<c_warp_dstr>{}, constant<c_warp_elem_layout>{});

        // Type conversion using element-wise operation
        static_for_n<c_warp_elem_layout.bottom_lengths()[0]>()([&](auto i) {
            c_warp_dist_tensor.memory().template at<i>() =
                static_cast<CDataType>(c_acc_dist_tensor.memory().template at<i>());
        });

        // Create window for C output
        const index_t warp_id = threadIdx.x / MINT_WARP_SIZE;
        const index_t m_warp = warp_id / kNWarp;
        const index_t n_warp = warp_id % kNWarp;

        auto c_warp_gmem_window = make_distributed_window(
            c_gmem_view,
            nd_index<2>{m_block_start + m_warp * kMPerWarp,
                       n_block_start + n_warp * kNPerWarp},
            constant<c_warp_dstr>{},
            constant<c_warp_elem_layout>{},
            constant<thread_in_this_warp{}>{});

        // Store C to global memory
        const auto gmem_mask = tuple<>{};
        store(c_warp_gmem_window, c_warp_dist_tensor, gmem_mask);
    }
};

} // namespace mint
