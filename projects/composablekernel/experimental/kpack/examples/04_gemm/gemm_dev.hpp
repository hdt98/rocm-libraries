// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Device-side CK Tile GEMM type wiring for kpack.
//
// Maps GemmKernel (our schema) → CK Tile template stack (7 types).
// Each .hip variant file includes this header and instantiates runGemm<K>
// with a specific constexpr GemmKernel.
//
// CkTypeMap:   DataType enum → CK Tile C++ type (float, half_t, bf16_t)
// CkLayoutMap: Layout enum   → CK Tile layout tag (RowMajor, ColumnMajor)
// runGemm<K>:  wires the full CK Tile GEMM pipeline from K's types/layouts
//
// Tile geometry is parameterized through GemmKernel fields, validated by
// make_kernel() against CK Tile's WarpGemmDispatcher table.

#pragma once

#include "gemm_api.hpp"
#include "gemm_args.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"

namespace rocm_ck {

// ============================================================================
// CkTypeMap: DataType → CK Tile C++ type
// ============================================================================

template <DataType>
struct CkTypeMap;

template <>
struct CkTypeMap<DataType::FP32>
{
    using type = float;
};

template <>
struct CkTypeMap<DataType::FP16>
{
    using type = ck_tile::half_t;
};

template <>
struct CkTypeMap<DataType::BF16>
{
    using type = ck_tile::bf16_t;
};

// ============================================================================
// CkLayoutMap: Layout → CK Tile layout tag
// ============================================================================

template <Layout>
struct CkLayoutMap;

template <>
struct CkLayoutMap<Layout::Row>
{
    using type = ck_tile::tensor_layout::gemm::RowMajor;
};

template <>
struct CkLayoutMap<Layout::Col>
{
    using type = ck_tile::tensor_layout::gemm::ColumnMajor;
};

// ============================================================================
// runGemm<K>: assemble CK Tile GEMM pipeline from GemmKernel descriptor
// ============================================================================

/// Device-side GEMM bridge: GemmArgs → CK Tile template stack → ck_tile::GemmKernel.
///
/// Wires the 7-type CK Tile GEMM stack (shape, traits, problem, pipeline,
/// partitioner, epilogue, kernel) with types and layouts from the GemmKernel
/// NTTP. Tile geometry comes from GemmKernel fields (validated at compile time).
template <GemmKernel K>
__device__ void runGemm(GemmArgs args)
{
    // --- Map schema types to CK Tile types ---
    using AType   = typename CkTypeMap<K.a_dtype>::type;
    using BType   = typename CkTypeMap<K.b_dtype>::type;
    using AccType = typename CkTypeMap<K.acc_dtype>::type;
    using OType   = typename CkTypeMap<K.c_dtype>::type;

    using ALayout = typename CkLayoutMap<K.a_layout>::type;
    using BLayout = typename CkLayoutMap<K.b_layout>::type;
    using CLayout = typename CkLayoutMap<K.c_layout>::type;

    // --- Step 1: Tile geometry (from GemmKernel, validated by make_kernel) ---
    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<K.block_tile.m, K.block_tile.n, K.block_tile.k>,
                               ck_tile::sequence<K.block_warps.m, K.block_warps.n, K.block_warps.k>,
                               ck_tile::sequence<K.warp_tile.m, K.warp_tile.n, K.warp_tile.k>>;

    // --- Step 2: Traits (no padding, layouts from kernel descriptor) ---
    using GemmTraits = ck_tile::TileGemmTraits<false, false, false, ALayout, BLayout, CLayout>;

    // --- Step 3: Pipeline problem (types from kernel descriptor) ---
    using PipelineProblem =
        ck_tile::GemmPipelineProblem<AType, BType, AccType, GemmShape, GemmTraits>;

    // --- Step 4: Pipeline (simplest: A/B from global memory, C in registers) ---
    using GemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<PipelineProblem>;

    // --- Step 5: Partitioner (1D blockIdx → 2D tile coordinates) ---
    using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;

    // --- Step 6: Epilogue (shuffle accumulator through LDS, store to global) ---
    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<AType,
                                         BType,
                                         ck_tile::tuple<>, // DsDataType (no bias/fused)
                                         AccType,
                                         OType,            // output (may differ from acc)
                                         ck_tile::tuple<>, // DsLayout
                                         CLayout,
                                         ck_tile::element_wise::PassThrough,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         K.block_warps.m,
                                         K.block_warps.n,
                                         K.warp_tile.m,
                                         K.warp_tile.n,
                                         K.warp_tile.k,
                                         PipelineProblem::TransposeC>>;

    // --- Step 7: Kernel (ties everything together) ---
    using CkKernel   = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
    using KernelArgs = typename CkKernel::UniversalGemmKernel::KernelArgs;

    // Convert clean ABI struct to CK Tile's internal args
    const KernelArgs kargs{{args.a},        // as_ptr
                           {args.b},        // bs_ptr
                           {},              // ds_ptr  (empty — no D tensors)
                           args.c,          // e_ptr
                           args.M,          // M
                           args.N,          // N
                           args.K,          // K
                           {args.stride_A}, // stride_As
                           {args.stride_B}, // stride_Bs
                           {},              // stride_Ds (empty)
                           args.stride_C,   // stride_E
                           1};              // k_batch (no split-K)

    CkKernel{}(kargs);
}

} // namespace rocm_ck
