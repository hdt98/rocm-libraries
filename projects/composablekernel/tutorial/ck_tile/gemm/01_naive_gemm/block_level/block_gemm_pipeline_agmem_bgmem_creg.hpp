// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "block_gemm_pipeline_agmem_bgmem_creg_policy.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"

namespace ck_tile {

// BlockGemmPipelineAGmemBGmemCReg: non-prefetch GEMM pipeline.
//
// A (input):  tile window over global memory
// B (input):  tile window over global memory
// C (output): static distributed tensor in registers (VGPRs)
//
// This is the simplest correct pipeline: no double-buffering, no prefetching.
// For each K iteration: load A and B from DRAM -> registers, store to LDS,
// synchronize, call block GEMM (MFMA), synchronize.
template <typename Problem, typename Policy = BlockGemmPipelineAGmemBGmemCRegPolicy>
struct BlockGemmPipelineAGmemBGmemCReg
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    // BlockGemm type is inferred from the policy at compile time.
    // This is the warp-coordination layer (BlockGemmASmemBSmemCReg).
    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;

    // GetStaticLdsSize: compile-time constant, called by the host to determine how much
    // __shared__ memory to allocate before launching the kernel.
    //
    // Layout: [A tile | padding to 16-byte boundary | B tile]
    // The 16-byte alignment between A and B ensures that p_b_lds is 16-byte aligned,
    // satisfying ds_write_b128 alignment requirements for B.
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetStaticLdsSize()
    {
        return integer_divide_ceil(
                   sizeof(ADataType) *
                       Policy::template MakeALdsBlockDescriptor<Problem>().get_element_space_size(),
                   16) *
                   16 +
               sizeof(BDataType) *
                   Policy::template MakeBLdsBlockDescriptor<Problem>().get_element_space_size();
    }

    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                        const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                        index_t num_loop,
                                        void* p_smem) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                std::is_same_v<BDataType, remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kNPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        // -----------------------------------------------------------------------------------------
        // Definitions of all needed tiles

        // A tile in LDS: pointer to the A region of shared memory.
        ADataType* p_a_lds = static_cast<ADataType*>(p_smem);

        // a_lds_block_desc: compile-time LDS layout descriptor built via an intermediate 3D
        // form (explicit tile sub-dimensions) that is then merged into a 2D (K, M) view.
        // This same descriptor governs both the store path (DRAM->LDS) and the load path
        // (LDS->MFMA). Both must use the same descriptor to ensure physical address agreement.
        constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();

        // Create a tensor view over the A region of LDS using the LDS descriptor.
        auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

        // a_lds_block_space_size_aligned: byte size of A's LDS region, rounded up to 16 bytes.
        // B is placed immediately after, so alignment ensures p_b_lds is 16-byte aligned.
        constexpr index_t a_lds_block_space_size_aligned =
            integer_divide_ceil(sizeof(ADataType) * a_lds_block_desc.get_element_space_size(), 16) *
            16;

        // B tile in LDS: pointer placed after A's aligned region.
        BDataType* p_b_lds = static_cast<BDataType*>(
            static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));

        constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();

        auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);

        // A DRAM tile window for load (carries tile distribution for coalesced+vectorized loads).
        // The tile distribution tells each thread which elements to load from DRAM.
        auto a_copy_dram_window =
            make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             a_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeADramTileDistribution<Problem>());

        // A LDS tile window for store (uses the same distribution as the DRAM window).
        // Using the same distribution ensures that the element thread T loads from DRAM
        // is the same element thread T stores to LDS -- preserving the physical layout.
        auto a_copy_lds_window =
            make_tile_window(a_lds_block,
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             {0, 0},
                             a_copy_dram_window.get_tile_distribution());

        // B DRAM tile window for load (same structure as A).
        auto b_copy_dram_window =
            make_tile_window(b_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                             b_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeBDramTileDistribution<Problem>());

        // B LDS tile window for store (same distribution as B DRAM window).
        auto b_copy_lds_window =
            make_tile_window(b_lds_block,
                             make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                             {0, 0},
                             b_copy_dram_window.get_tile_distribution());

        // A LDS tile window for block GEMM (no tile distribution).
        // The block GEMM (BlockGemmASmemBSmemCReg) applies its own warp-level distribution
        // internally -- it does not need the DRAM distribution here.
        auto a_lds_gemm_window = make_tile_window(
            a_lds_block, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // B LDS tile window for block GEMM (no tile distribution, same reasoning).
        auto b_lds_gemm_window = make_tile_window(
            b_lds_block, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // Instantiate the block GEMM object.
        auto block_gemm = BlockGemm();

        // c_block_tile type is inferred at compile time from block_gemm's init-form operator().
        // This avoids having to specify the C distribution encoding explicitly here.
        auto c_block_tile = decltype(block_gemm(a_lds_gemm_window, b_lds_gemm_window)){};

        // DRAM tile types for A and B, sized to hold one full block tile per thread.
        using ABlockTileDistr = decltype(a_copy_dram_window.get_tile_distribution());
        using BBlockTileDistr = decltype(b_copy_dram_window.get_tile_distribution());

        using ABlockTile = decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
        using BBlockTile = decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));

        ABlockTile a_block_tile;
        BBlockTile b_block_tile;

        // Step sizes for sliding the DRAM windows along the K dimension each iteration.
        // (0, kKPerBlock): advance K by kKPerBlock, hold M/N fixed.
        using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
        using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;
        constexpr ADramTileWindowStep a_dram_tile_window_step = make_array(0, kKPerBlock);
        constexpr BDramTileWindowStep b_dram_tile_window_step = make_array(0, kKPerBlock);

        // -------------------------------------------------------------------------------------
        // GEMM pipeline start

        // Initialize C accumulator to zero in registers before the K loop.
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        // Non-prefetch pipeline: loads and compute do NOT overlap.
        // Each iteration follows: DRAM load -> advance windows -> LDS store -> sync -> MFMA -> sync
        index_t iCounter = num_loop;

        while(iCounter > 0)
        {
            // Step 1: Load A and B tiles from DRAM into per-thread registers (VGPRs).
            // Each thread loads its assigned elements using vectorized global loads.
            a_block_tile = load_tile(a_copy_dram_window);
            b_block_tile = load_tile(b_copy_dram_window);

            // Step 2: Advance DRAM windows to the next K slice for the NEXT iteration.
            // Done here (before the LDS store) so the compiler can schedule the address
            // computation while the store is in flight.
            move_tile_window(a_copy_dram_window, a_dram_tile_window_step);
            move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

            // Step 3: Store A and B tiles from registers (VGPRs) to LDS.
            // Uses the same tile distribution as the DRAM load, so thread T's elements
            // land at the correct LDS address for the MFMA to read.
            store_tile(a_copy_lds_window, a_block_tile);
            store_tile(b_copy_lds_window, b_block_tile);

            // Step 4: Barrier -- wait for ALL threads to finish writing LDS.
            // Without this, threads that finish early could start reading stale LDS data
            // from the previous iteration while other threads are still writing.
            block_sync_lds();

            // Step 5: Block GEMM: c_block_tile += A_lds * B_lds using MFMA instructions.
            // The block GEMM reads A and B from LDS and accumulates into c_block_tile (VGPRs).
            block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

            // Step 6: Barrier -- wait for ALL threads to finish reading LDS (MFMA complete).
            // Without this, the next iteration's LDS store could overwrite data that another
            // thread is still consuming for MFMA.
            block_sync_lds();

            iCounter--;
        }

        return c_block_tile;
    }
};

} // namespace ck_tile
