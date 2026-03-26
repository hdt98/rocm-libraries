// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_problem.hpp"
#include "ck_tile/ops/mhc/kernel/mhc_kernel_fused.hpp"
#include "ck_tile/ops/mhc/kernel/mhc_reduction_kernel.hpp"
#include "ck_tile/ops/mhc/kernel/mhc_sinkhorn_kernel.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {

// MHC Invoker
// Provides type definitions and helper methods for the 3-stage MHC pipeline
template <typename XDataType_, // X and Phi input matrices type
          typename YDataType_,
          typename ComputeDataType_,
          typename ActivationFunc_ = ck_tile::element_wise::Sigmoid,
          ck_tile::index_t MTile_  = 64,
          bool UseLogSinkhorn_     = true>
struct MHCInvoker
{
    using XDataType       = ck_tile::remove_cvref_t<XDataType_>;
    using YDataType       = ck_tile::remove_cvref_t<YDataType_>;
    using ComputeDataType = ck_tile::remove_cvref_t<ComputeDataType_>;
    using ActivationFunc  = ck_tile::remove_cvref_t<ActivationFunc_>;

    static constexpr ck_tile::index_t MTile = MTile_;
    static constexpr bool UseLogSinkhorn    = UseLogSinkhorn_;

    // Kernel type definitions
    using Problem = ck_tile::MHCProblemGemmDist<XDataType, ComputeDataType, YDataType, MTile>;
    using BlockGemmShape = typename Problem::BlockGemmShape;
    using GemmKernel     = ck_tile::
        MHCKernelFused<Problem, ck_tile::UniversalGemmPipelineAgBgCrPolicy, ActivationFunc>;
    using ReductionKernel = ck_tile::MHCReductionKernel<Problem, ActivationFunc>;

    using SinkhornKernel =
        ck_tile::MHCSinkhornKernelDispatcher<YDataType, ComputeDataType, UseLogSinkhorn>;

    CK_TILE_HOST static constexpr bool IsSupportedArgument(index_t n)
    {
        bool supported = SinkhornKernel::IsSupportedArgument(n);

        return supported;
    }

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
};

} // namespace ck_tile
