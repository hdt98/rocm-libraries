// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <mint/mint.h>
#include "block_gemm_pipeline_agmem_bgmem_creg_policy.hpp"
#include "../warp_level/block_gemm_asmem_bsmem_creg.hpp"

namespace mint {

// Block-level GEMM pipeline: A and B from global memory, C in registers
// Implements the pipeline: load from global -> store to shared -> compute
//
// This is analogous to CK_Tile's BlockGemmPipelineAGmemBGmemCReg
template <typename Problem, typename Policy = BlockGemmPipelineAGmemBGmemCRegPolicy>
struct BlockGemmPipelineAGmemBGmemCReg
{
    using ADataType = remove_cvref_t<typename Problem::ADataType>;
    using BDataType = remove_cvref_t<typename Problem::BDataType>;
    using CDataType = remove_cvref_t<typename Problem::CDataType>;

    static constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;
    static constexpr index_t kBlockSize = Problem::kBlockSize;

    // Warp GEMM configuration
    using WarpGemm = BlockGemmASmemBSmemCReg<Problem, BlockGemmASmemBSmemCRegPolicy>;
    using WarpPolicy = BlockGemmASmemBSmemCRegPolicy;
    static constexpr index_t kMPerWarp = WarpPolicy::kMPerWarp;
    static constexpr index_t kNPerWarp = WarpPolicy::kNPerWarp;
    static constexpr index_t kKPerWarp = WarpPolicy::kKPerWarp;
    static constexpr index_t kMWarp = WarpPolicy::kMWarp;
    static constexpr index_t kNWarp = WarpPolicy::kNWarp;

    // Main operator: performs block-level GEMM
    template <typename AGmemTensorView,
              typename BGmemTensorView,
              typename CAccDistTensor>
    MINT_DEVICE void operator()(const AGmemTensorView& a_gmem_view,
                                const BGmemTensorView& b_gmem_view,
                                CAccDistTensor& c_acc_dist_tensor,
                                index_t m_block_start,
                                index_t n_block_start,
                                index_t k_size) const
    {
        // Create block copy distributions
        constexpr auto a_block_copy_dstr = Policy::template MakeABlockCopyDistribution<Problem>();
        constexpr auto b_block_copy_dstr = Policy::template MakeBBlockCopyDistribution<Problem>();

        // Element layouts for block copies
        constexpr auto a_block_copy_elem_layout =
            make_aliased_naive_packed_tensor_descriptor(
                make_index_sequence<a_block_copy_dstr.element_ndim()>{},
                index_constant<-1>{},
                a_block_copy_dstr.element_lengths());

        constexpr auto b_block_copy_elem_layout =
            make_aliased_naive_packed_tensor_descriptor(
                make_index_sequence<b_block_copy_dstr.element_ndim()>{},
                index_constant<-1>{},
                b_block_copy_dstr.element_lengths());

        // Shared memory descriptors for block tiles (compile-time packed layout)
        constexpr auto a_smem_desc = tensor::make_aliased_naive_packed_tensor_descriptor(
            make_index_sequence<2>{},
            index_constant<-1>{},
            nd_index<2>{kMPerBlock, kKPerBlock});

        constexpr auto b_smem_desc = tensor::make_aliased_naive_packed_tensor_descriptor(
            make_index_sequence<2>{},
            index_constant<-1>{},
            nd_index<2>{kNPerBlock, kKPerBlock});

        // Allocate shared memory
        constexpr index_t a_smem_size = a_smem_desc.bottom_lengths()[0] * sizeof(ADataType);
        constexpr index_t b_smem_size = b_smem_desc.bottom_lengths()[0] * sizeof(BDataType);
        constexpr index_t smem_size = a_smem_size + b_smem_size;

        __shared__ char p_shared[smem_size];
        ADataType* p_a_shared = reinterpret_cast<ADataType*>(p_shared);
        BDataType* p_b_shared = reinterpret_cast<BDataType*>(p_shared + a_smem_size);

        // Create shared memory tensor views
        const auto a_smem_view = make_tensor_view(
            a_smem_desc,
            make_shared_memory_view(p_a_shared, a_smem_desc.bottom_lengths()[0]));

        const auto b_smem_view = make_tensor_view(
            b_smem_desc,
            make_shared_memory_view(p_b_shared, b_smem_desc.bottom_lengths()[0]));

        // Create windows for block copies (global -> shared)
        using ThreadBlock = thread_in_this_block<kBlockSize>;

        auto a_block_gmem_window = make_distributed_window(
            a_gmem_view,
            nd_index<2>{m_block_start, 0},
            constant<a_block_copy_dstr>{},
            constant<a_block_copy_elem_layout>{},
            constant<ThreadBlock{}>{});

        auto b_block_gmem_window = make_distributed_window(
            b_gmem_view,
            nd_index<2>{n_block_start, 0},
            constant<b_block_copy_dstr>{},
            constant<b_block_copy_elem_layout>{},
            constant<ThreadBlock{}>{});

        auto a_block_smem_window = make_distributed_window(
            a_smem_view,
            nd_index<2>{0, 0},
            constant<a_block_copy_dstr>{},
            constant<a_block_copy_elem_layout>{},
            constant<ThreadBlock{}>{});

        auto b_block_smem_window = make_distributed_window(
            b_smem_view,
            nd_index<2>{0, 0},
            constant<b_block_copy_dstr>{},
            constant<b_block_copy_elem_layout>{},
            constant<ThreadBlock{}>{});

        // Warp ID within block
        const index_t warp_id = threadIdx.x / MINT_WARP_SIZE;
        const index_t m_warp = warp_id / kNWarp;
        const index_t n_warp = warp_id % kNWarp;

        // Create warp-level distributions for matmul
        constexpr auto warp_matmul_dstr = WarpGemm::make_warp_matmul_distribution();

        constexpr auto a_warp_dstr = extract_dimension_distribution(
            warp_matmul_dstr, sequence<alias_t, alias_t{"M"}, alias_t{"K"}>{});

        constexpr auto b_warp_dstr = extract_dimension_distribution(
            warp_matmul_dstr, sequence<alias_t, alias_t{"N"}, alias_t{"K"}>{});

        constexpr auto a_warp_elem_layout =
            make_aliased_naive_packed_tensor_descriptor(
                make_index_sequence<a_warp_dstr.element_ndim()>{},
                index_constant<-1>{},
                a_warp_dstr.element_lengths());

        constexpr auto b_warp_elem_layout =
            make_aliased_naive_packed_tensor_descriptor(
                make_index_sequence<b_warp_dstr.element_ndim()>{},
                index_constant<-1>{},
                b_warp_dstr.element_lengths());

        // Create warp windows into shared memory
        auto a_warp_smem_window = make_distributed_window(
            a_smem_view,
            nd_index<2>{m_warp * kMPerWarp, 0},
            constant<a_warp_dstr>{},
            constant<a_warp_elem_layout>{},
            constant<thread_in_this_warp{}>{});

        auto b_warp_smem_window = make_distributed_window(
            b_smem_view,
            nd_index<2>{n_warp * kNPerWarp, 0},
            constant<b_warp_dstr>{},
            constant<b_warp_elem_layout>{},
            constant<thread_in_this_warp{}>{});

        // Masks (empty for simplicity - no boundary checking)
        const auto gmem_mask = tuple<>{};
        const auto smem_mask = tuple<>{};

        // Main K-loop: iterate over K dimension in blocks
        for(index_t k_block = 0; k_block < k_size; k_block += kKPerBlock)
        {
            // Block-level: load from global memory to VGPRs
            const auto a_block_tile = load(a_block_gmem_window, gmem_mask);
            const auto b_block_tile = load(b_block_gmem_window, gmem_mask);

            // Block-level: store from VGPRs to shared memory
            store(a_block_smem_window, a_block_tile, smem_mask);
            store(b_block_smem_window, b_block_tile, smem_mask);

            __syncthreads();

            // Warp-level: iterate over K within the block
            static_for_n<kKPerBlock / kKPerWarp>()([&](auto k_iter) {
                // Create warp GEMM operator
                WarpGemm warp_gemm;

                // Perform warp-level GEMM: C += A * B
                warp_gemm(c_acc_dist_tensor, a_warp_smem_window, b_warp_smem_window);

                // Move windows to next K position
                move_window(a_warp_smem_window, nd_index<2>{0, kKPerWarp});
                move_window(b_warp_smem_window, nd_index<2>{0, kKPerWarp});
            });

            // Reset warp windows for next block
            move_window(a_warp_smem_window, nd_index<2>{0, -kKPerBlock});
            move_window(b_warp_smem_window, nd_index<2>{0, -kKPerBlock});

            __syncthreads();

            // Move block windows to next K block
            move_window(a_block_gmem_window, nd_index<2>{0, kKPerBlock});
            move_window(b_block_gmem_window, nd_index<2>{0, kKPerBlock});
        }
    }
};

} // namespace mint
