// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Device-side CK Tile GEMM type wiring for kpack.
//
// Maps GemmKernel (our schema) → CK Tile template stack (7 types).
// Each .hip variant file includes this header and instantiates runGemm<K>
// with a specific constexpr GemmKernel.
//
// CkLayoutMap: Layout enum   → CK Tile layout tag (RowMajor, ColumnMajor)
// runGemm<K>:  wires the full CK Tile GEMM pipeline from K's types/layouts
//
// Tile geometry is parameterized through GemmKernel fields, validated by
// make_kernel() against CK Tile's WarpGemmDispatcher table.

#pragma once

#include "gemm_api.hpp"
#include "gemm_args.hpp"

#include <rocm_ck/ck_type_map.hpp>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

namespace rocm_ck {

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
// ComposedCDEOp: CombineOp × Activation → single CK Tile elementwise functor
// ============================================================================

/// Composed epilogue functor: combine D tensors in float, activate, cast to output.
///
/// Replaces the old CkEpilogueOpMap 1:1 enum mapping with a two-axis composition.
/// All arithmetic is in float to avoid precision loss between combine and activation.
/// Delegates activation math to CK Tile's existing functor implementations.
template <CombineOp Combine, Activation Act>
struct ComposedCDEOp
{
    template <typename E, typename C, typename... Ds>
    CK_TILE_HOST_DEVICE void operator()(E& e, const C& c, const Ds&... ds) const
    {
        float result = ck_tile::type_convert<float>(c);

        // Combine D tensors with matmul result
        if constexpr(Combine == CombineOp::Add)
        {
            ((result += ck_tile::type_convert<float>(ds)), ...);
        }
        else if constexpr(Combine == CombineOp::Multiply)
        {
            ((result *= ck_tile::type_convert<float>(ds)), ...);
        }

        // Apply activation (delegating to CK Tile's optimized implementations)
        if constexpr(Act == Activation::Relu)
        {
            ck_tile::element_wise::Relu{}(result, result);
        }
        else if constexpr(Act == Activation::FastGelu)
        {
            ck_tile::element_wise::FastGelu{}(result, result);
        }
        else if constexpr(Act == Activation::Gelu)
        {
            ck_tile::element_wise::Gelu{}(result, result);
        }
        else if constexpr(Act == Activation::Silu)
        {
            ck_tile::element_wise::Silu{}(result, result);
        }
        else if constexpr(Act == Activation::Sigmoid)
        {
            ck_tile::element_wise::Sigmoid{}(result, result);
        }

        e = ck_tile::type_convert<E>(result);
    }
};

// ============================================================================
// EpilogueTypes<K>: derive DsDataType, DsLayout, and Op tuples from GemmKernel
// ============================================================================

template <GemmKernel K>
struct EpilogueTypes
{
    using Op = ComposedCDEOp<K.combine, K.activation>;

    // Always resolved (GemmKernel fields have valid defaults even when unused).
    // std::conditional_t below selects which ones go into the tuples.
    using D0Type   = typename CkTypeMap<K.d0_dtype>::type;
    using D1Type   = typename CkTypeMap<K.d1_dtype>::type;
    using D0Layout = typename CkLayoutMap<K.d0_layout>::type;
    using D1Layout = typename CkLayoutMap<K.d1_layout>::type;

    using DsDataType = std::conditional_t<K.num_d_tensors == 0,
                                          ck_tile::tuple<>,
                                          std::conditional_t<K.num_d_tensors == 1,
                                                             ck_tile::tuple<D0Type>,
                                                             ck_tile::tuple<D0Type, D1Type>>>;

    using DsLayout = std::conditional_t<K.num_d_tensors == 0,
                                        ck_tile::tuple<>,
                                        std::conditional_t<K.num_d_tensors == 1,
                                                           ck_tile::tuple<D0Layout>,
                                                           ck_tile::tuple<D0Layout, D1Layout>>>;
};

// ============================================================================
// runGemm<K>: assemble CK Tile GEMM pipeline from GemmKernel descriptor
// ============================================================================

/// Device-side GEMM bridge: args → CK Tile template stack → ck_tile::GemmKernel.
///
/// Wires the 7-type CK Tile GEMM stack (shape, traits, problem, pipeline,
/// partitioner, epilogue, kernel) with types and layouts from the GemmKernel
/// NTTP. Tile geometry comes from GemmKernel fields (validated at compile time).
///
/// ArgsType is deduced from the extern "C" function parameter — GemmArgs for
/// plain GEMM, GemmArgs1D for fused epilogues with one D tensor.
template <GemmKernel K, typename ArgsType>
__device__ void runGemm(ArgsType args)
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

    // --- Epilogue types from kernel descriptor ---
    using EpiOp      = typename EpilogueTypes<K>::Op;
    using DsDataType = typename EpilogueTypes<K>::DsDataType;
    using DsLayout   = typename EpilogueTypes<K>::DsLayout;

    // --- Step 6: Epilogue (shuffle accumulator through LDS, store to global) ---
    using GemmEpilogue =
        ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<AType,
                                                                   BType,
                                                                   DsDataType,
                                                                   AccType,
                                                                   OType,
                                                                   DsLayout,
                                                                   CLayout,
                                                                   EpiOp,
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

    // Convert clean ABI struct to CK Tile's internal args.
    // Branch on D tensor count to construct the right KernelArgs initializer.
    if constexpr(K.num_d_tensors == 0)
    {
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
    else if constexpr(K.num_d_tensors == 1)
    {
        const KernelArgs kargs{{args.a},         // as_ptr
                               {args.b},         // bs_ptr
                               {args.d0},        // ds_ptr (1 D tensor)
                               args.e,           // e_ptr
                               args.M,           // M
                               args.N,           // N
                               args.K,           // K
                               {args.stride_A},  // stride_As
                               {args.stride_B},  // stride_Bs
                               {args.stride_D0}, // stride_Ds (1 stride)
                               args.stride_E,    // stride_E
                               1};               // k_batch (no split-K)
        CkKernel{}(kargs);
    }
}

} // namespace rocm_ck
