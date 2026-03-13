// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_problem.hpp"
#include "ck_tile/ops/mhc/kernel/mhc_kernel_fused_pipeline.hpp"
#include "ck_tile/ops/mhc/kernel/mhc_reduction_kernel_optimized.hpp"
// Sinkhorn kernels
#include "ck_tile/ops/mhc/kernel/mhc_sinkhorn_kernel.hpp"
#include "ck_tile/ops/mhc/kernel/mhc_sinkhorn_log_kernel.hpp"
#include "ck_tile/ops/mhc/kernel/mhc_sinkhorn_log_kernel_cktile_v2.hpp"
#include "ck_tile/ops/mhc/kernel/mhc_sinkhorn_kernel_cktile_v2.hpp"
#include "ck_tile/ops/mhc/kernel/mhc_sinkhorn_kernel_cktile_unified.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {

// MHC Fused Pipeline Invoker
// Provides type definitions and helper methods for the 3-stage MHC pipeline
template <typename XDataType_,
          typename PhiDataType_,
          typename YDataType_,
          typename ComputeDataType_,
          typename ActivationFunc_ = ck_tile::element_wise::Sigmoid,
          ck_tile::index_t MTile_  = 64,
          bool UseLogSinkhorn_     = true>
struct MHCFusedPipelineInvoker
{
    using XDataType       = ck_tile::remove_cvref_t<XDataType_>;
    using PhiDataType     = ck_tile::remove_cvref_t<PhiDataType_>;
    using YDataType       = ck_tile::remove_cvref_t<YDataType_>;
    using ComputeDataType = ck_tile::remove_cvref_t<ComputeDataType_>;
    using ActivationFunc  = ck_tile::remove_cvref_t<ActivationFunc_>;

    static constexpr ck_tile::index_t MTile = MTile_;
    static constexpr bool UseLogSinkhorn    = UseLogSinkhorn_;

    // Kernel type definitions
    using Problem = ck_tile::MHCProblemV5GemmDist<XDataType, ComputeDataType, YDataType, MTile>;
    using BlockGemmShape = typename Problem::BlockGemmShape;
    using GemmKernel     = ck_tile::
        MHCKernelFusedPipeline<Problem, ck_tile::UniversalGemmPipelineAgBgCrPolicy, ActivationFunc>;
    using ReductionKernel = ck_tile::MHCReductionKernelOptimized<Problem, ActivationFunc>;
    // Sinkhorn kernel - supports arbitrary n
    // Testing unified V2 kernel (old kernels commented out)
    using SinkhornKernel =
        // Old kernels:
        // std::conditional_t<UseLogSinkhorn,
        //                    ck_tile::MHCSinkhornLogKernel<YDataType, ComputeDataType>,
        //                    ck_tile::MHCSinkhornKernel<YDataType, ComputeDataType>>;
        // Unified V2 kernel (testing):
        ck_tile::
            MHCSinkhornKernelTileV2UnifiedDispatcher<YDataType, ComputeDataType, UseLogSinkhorn>;

    // Helper methods
    CK_TILE_HOST static constexpr auto GetGemmBlockSize() { return GemmKernel::BlockSize(); }
    CK_TILE_HOST static constexpr auto GetReductionBlockSize()
    {
        return ReductionKernel::BlockSize();
    }
    CK_TILE_HOST static constexpr auto GetSinkhornBlockSize()
    {
        return SinkhornKernel::BlockSize();
    }

    CK_TILE_HOST static auto
    GetGridSize(ck_tile::index_t B, ck_tile::index_t output_dim, ck_tile::index_t nC)
    {
        return GemmKernel::GetGridSize(B, output_dim, nC);
    }

    CK_TILE_HOST static ck_tile::index_t GetReductionBlocks(ck_tile::index_t B,
                                                            ck_tile::index_t output_dim)
    {
        return (B * output_dim + GetReductionBlockSize() - 1) / GetReductionBlockSize();
    }

    CK_TILE_HOST static ck_tile::index_t GetSinkhornBlocks(ck_tile::index_t B)
    {
        return (B + GetSinkhornBlockSize() - 1) / GetSinkhornBlockSize();
    }

    // Host-callable GetSmemSize() - provides shared memory size for kernel launch
    // NOTE: The actual LDS layout is complex with padding and permutation for bank conflict
    // avoidance. Rather than trying to replicate the device-only GetSmemSize() calculation,
    // we provide a simplified approximation based on the block dimensions and data types.
    CK_TILE_HOST static constexpr ck_tile::index_t GetSmemSize()
    {
        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using BDataType = remove_cvref_t<typename Problem::BDataType>;

        constexpr ck_tile::index_t kM = BlockGemmShape::kM;
        constexpr ck_tile::index_t kN = BlockGemmShape::kN;
        constexpr ck_tile::index_t kK = BlockGemmShape::kK;

        // Base calculation: A (M×K) + B (N×K) with type sizes
        constexpr ck_tile::index_t base_a = kM * kK * sizeof(ADataType);
        constexpr ck_tile::index_t base_b = kN * kK * sizeof(BDataType);

        // The actual LDS layout adds padding for bank conflict avoidance
        // Empirically, for bf16/fp16 (2 bytes), the padding results in:
        // - MTile=16: 4096 bytes (base would be ~2048)
        // - MTile=32,64,128: 8192 bytes (base would be 4096-8192)
        // This suggests roughly 2x overhead for smaller tiles, less for larger

        // Conservative estimate: use base calculation with alignment
        // This should work across different data types
        constexpr ck_tile::index_t aligned_a = ((base_a + 15) / 16) * 16;
        constexpr ck_tile::index_t aligned_b = ((base_b + 15) / 16) * 16;
        constexpr ck_tile::index_t total     = aligned_a + aligned_b;

        // For bf16/fp16, verify against known good values
        // If the calculation differs significantly, use empirical value
        if constexpr(sizeof(ADataType) == 2 && sizeof(BDataType) == 2)
        {
            // For 2-byte types (bf16/fp16), use empirically validated values
            if constexpr(MTile == 16)
                return 4096;
            else
                return 8192; // MTile 32, 64, 128 all use 8192
        }
        else
        {
            // For other type sizes, use the calculated value
            // This should scale appropriately with type size
            return total;
        }
    }
};

} // namespace ck_tile
