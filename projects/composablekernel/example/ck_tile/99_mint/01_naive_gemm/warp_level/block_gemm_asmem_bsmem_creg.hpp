// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <mint/mint.h>
#include "block_gemm_asmem_bsmem_creg_policy.hpp"

namespace mint {

// Warp-level GEMM: A and B from shared memory, C in registers
// This is analogous to CK_Tile's BlockGemmASmemBSmemCReg
//
// Computes: C[M, N] += A[M, K] * B[K, N]
// where A and B are in shared memory, C is distributed across warp registers
template <typename Problem, typename Policy = BlockGemmASmemBSmemCRegPolicy>
struct BlockGemmASmemBSmemCReg
{
    using ADataType = remove_cvref_t<typename Problem::ADataType>;
    using BDataType = remove_cvref_t<typename Problem::BDataType>;
    using CDataType = remove_cvref_t<typename Problem::CDataType>;

    static constexpr index_t kMPerWarp = Policy::kMPerWarp;
    static constexpr index_t kNPerWarp = Policy::kNPerWarp;
    static constexpr index_t kKPerWarp = Policy::kKPerWarp;
    static constexpr index_t kMWarp = Policy::kMWarp;
    static constexpr index_t kNWarp = Policy::kNWarp;

    // Create warp-level distribution for matmul
    // This distributes work among threads in a warp
    // NOTE: Simplified version - MINT's make_simple_distribution cannot be called
    //       from device code due to STL usage. In production, manually construct
    //       the distribution using polymorphers like in mint/test/kernel/simt/test_kernel_simt_gemm.cu
    MINT_HOST_DEVICE static constexpr auto make_warp_matmul_distribution()
    {
        // For now, return a dummy value - this will be handled differently
        // In actual implementation, we'll avoid using this in device code
        return index_constant<0>{};
    }

    // Operator: C += A * B
    // A and B are windows into shared memory
    // C is a distributed tensor in registers
    template <typename CDistTensor, typename AWindow, typename BWindow>
    MINT_DEVICE void operator()(CDistTensor& c_dist_tensor,
                                const AWindow& a_smem_window,
                                const BWindow& b_smem_window) const
    {
        constexpr auto warp_matmul_dstr = make_warp_matmul_distribution();

        // Extract M, N distributions from the full M-N-K distribution
        constexpr auto a_warp_dstr = extract_dimension_distribution(
            warp_matmul_dstr, sequence<alias_t, alias_t{"M"}, alias_t{"K"}>{});

        constexpr auto b_warp_dstr = extract_dimension_distribution(
            warp_matmul_dstr, sequence<alias_t, alias_t{"K"}, alias_t{"N"}>{});

        // Element layouts for warp tiles
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

        // Load A warp tile from shared memory
        const auto a_warp_tile = load(a_smem_window, tuple<>{});

        // Load B warp tile from shared memory
        const auto b_warp_tile = load(b_smem_window, tuple<>{});

        // Perform warp-level matrix multiplication
        // C[M,N] += A[M,K] * B[K,N]
        tile::simt::warp::matmul_mn_mk_kn_no_shuffle(
            c_dist_tensor, a_warp_tile, b_warp_tile);
    }
};

} // namespace mint
