// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

#include "block_level/block_gemm_pipeline_agmem_bgmem_creg.hpp"
#include "host_level/grid_gemm.hpp"

namespace ck_tile {

// GridGemmProblem: type-tag struct that bundles all data types for the GEMM computation.
// This pattern (empty struct with type aliases) is pervasive in CK Tile. It allows the
// compiler to deduce all data type information from a single type parameter rather than
// threading multiple separate type parameters through the template hierarchy.
template <typename ADataType_,
          typename BDataType_,
          typename AccDataType_,
          typename CDataType_,
          typename CElementFunction_>
struct GridGemmProblem
{
    using ADataType   = ADataType_;
    using BDataType   = BDataType_;
    using AccDataType = AccDataType_;
    using CDataType   = CDataType_;

    // CElementFunction: a callable applied to each output element after the GEMM.
    // In this tutorial it is the identity function (passes through unchanged).
    // In production kernels it can implement bias addition, activation (ReLU, GELU, etc.),
    // or any other per-element epilogue without rewriting the GEMM core.
    using CElementFunction = CElementFunction_;
};

// TileGemmShape: compile-time struct encoding the tile dimensions.
// All values are baked into the type system (static constexpr), so the compiler
// can compute loop bounds, register counts, and LDS sizes entirely at compile time.
template <index_t kMPerTile, index_t kNPerTile, index_t kKPerTile>
struct TileGemmShape
{
    static constexpr index_t kM = kMPerTile;
    static constexpr index_t kN = kNPerTile;
    static constexpr index_t kK = kKPerTile;
};

// BlockGemmPipelineProblem: specification type that bundles the information needed
// by the block-level pipeline. Distinct from GridGemmProblem because the block pipeline
// does not need AccDataType or CElementFunction -- those are handled at the grid level.
// kBlockSize is a compile-time constant that determines LDS size and DRAM distribution.
template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          index_t kBlockSize_,
          typename BlockGemmShape_>
struct BlockGemmPipelineProblem
{
    using ADataType      = remove_cvref_t<ADataType_>;
    using BDataType      = remove_cvref_t<BDataType_>;
    using CDataType      = remove_cvref_t<CDataType_>;
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    static constexpr index_t kBlockSize = kBlockSize_;
};

// Gemm: top-level kernel functor. Combines all configuration into a single struct
// that exposes operator() as the GPU kernel entry point.
//
// Template parameters mirror the GEMM configuration in practice_gemm.cpp:
//   ADataType, BDataType, AccDataType, CDataType: element types
//   CElementFunction: epilogue function type
//   kAAlignment, kBAlignment, kCAlignment: guaranteed vectorizable element count for loads/stores
//   kBlockSize_: threads per block
//   kMPerBlock_, kNPerBlock_, kKPerBlock_: block tile dimensions
//
// C = CElementFunction(type_convert<CDataType>(A * B))
template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename CElementFunction,
          index_t kAAlignment,
          index_t kBAlignment,
          index_t kCAlignment,
          index_t kBlockSize_,
          index_t kMPerBlock_,
          index_t kNPerBlock_,
          index_t kKPerBlock_>
struct Gemm
{
    static constexpr index_t kBlockSize = kBlockSize_;

    // GridGemmProblem_: bundles all data type info, threaded down to GridGemm.
    using GridGemmProblem_ =
        GridGemmProblem<ADataType, BDataType, AccDataType, CDataType, CElementFunction>;

    // GridGemmPolicy: nested policy struct that encodes tile sizes and maps block IDs to tiles.
    // Separating Policy from Problem follows the CK Tile "what vs how" design:
    //   Problem = what data types and shapes
    //   Policy  = how to partition and schedule work
    struct GridGemmPolicy
    {
        static constexpr index_t kBlockSize = kBlockSize_;
        static constexpr index_t kMPerBlock = kMPerBlock_;
        static constexpr index_t kNPerBlock = kNPerBlock_;
        static constexpr index_t kKPerBlock = kKPerBlock_;

        // MakeBlock2TileMap: returns a lambda mapping linear block_id to 2D tile (iM, iN).
        //
        // Implementation uses make_merge_transform with (N0, M0) -- N is the fast-moving
        // (inner) dimension. This means consecutive block IDs map to consecutive N tiles:
        //   block 0 -> (n=0, m=0), block 1 -> (n=1, m=0), block 2 -> (n=0, m=1), ...
        //
        // N-first ordering improves L2 cache reuse: adjacent blocks load the same A rows
        // (same M strip) but different B rows. Since B is read once per block while A rows
        // are reused across all N tiles in the same M strip, keeping adjacent blocks on the
        // same M strip allows the A data to remain in cache longer.
        template <typename Problem>
        CK_TILE_HOST_DEVICE static constexpr auto MakeBlock2TileMap(index_t M0, index_t N0)
        {
            const auto unmerge = make_merge_transform(make_tuple(N0, M0));

            return [unmerge](index_t block_id) {
                multi_index<2> unmerged;
                unmerge.calculate_lower_index(unmerged, make_multi_index(block_id));

                // unmerged[0] = n index (fast dimension), unmerged[1] = m index (slow dimension)
                // Return (m, n) as the tile coordinate.
                return make_multi_index(unmerged.at(number<1>{}), unmerged.at(number<0>{}));
            };
        }

        // GetBlockGemmPipeline: constructs and returns the block pipeline object.
        // BlockGemmPipelineProblem_ bundles the types and block size for the pipeline.
        // Returns BlockGemmPipelineAGmemBGmemCReg, the non-prefetch pipeline.
        template <typename Problem>
        CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemmPipeline()
        {
            using BlockGemmPipelineProblem_ =
                BlockGemmPipelineProblem<ADataType,
                                         BDataType,
                                         AccDataType,
                                         kBlockSize,
                                         TileGemmShape<kMPerBlock, kNPerBlock, kKPerBlock>>;
            return BlockGemmPipelineAGmemBGmemCReg<BlockGemmPipelineProblem_>{};
        }
    };

    using GridGemm_ = GridGemm<GridGemmProblem_, GridGemmPolicy>;

    // operator(): the GPU kernel entry point.
    // Receives raw device pointers and problem dimensions, wraps them in tensor views,
    // then delegates all computation to GridGemm_.
    CK_TILE_DEVICE void operator()(const ADataType* p_a,
                                   const BDataType* p_b,
                                   CDataType* p_c,
                                   const index_t M,
                                   const index_t N,
                                   const index_t K,
                                   const index_t Lda,
                                   const index_t Ldb,
                                   const index_t Ldc,
                                   const CElementFunction& c_element_func) const
    {
        // make_naive_tensor_view wraps a raw pointer in a structured multi-dimensional view.
        // Parameters:
        //   address_space_enum::global -- data is in GPU DRAM (not LDS or registers)
        //   p_a                        -- raw device pointer from hipMalloc
        //   make_tuple(M, K)           -- logical shape [M rows, K cols]
        //   make_tuple(Lda, 1)         -- strides: Lda elements per row, 1 per column (row-major)
        //   number<kAAlignment>{}      -- guaranteed vector length: kAAlignment fp16 elements
        //                                 can always be loaded in one instruction (128-bit load)
        //   number<1>{}                -- guaranteed vector stride: consecutive elements in last
        //                                 dimension are contiguous in memory (stride=1)
        const auto a_dram = [&] {
            return make_naive_tensor_view<address_space_enum::global>(
                p_a, make_tuple(M, K), make_tuple(Lda, 1), number<kAAlignment>{}, number<1>{});
        }();

        // B is stored as [N, K] (transposed): N as the leading dimension, K contiguous.
        // This layout enables both coalesced (K is innermost) and vectorized loads simultaneously.
        // The GEMM computes C = A * B^T conceptually, but B is stored as B^T in memory already.
        const auto b_dram = [&] {
            return make_naive_tensor_view<address_space_enum::global>(
                p_b, make_tuple(N, K), make_tuple(Ldb, 1), number<kBAlignment>{}, number<1>{});
        }();

        const auto c_dram = [&] {
            return make_naive_tensor_view<address_space_enum::global>(
                p_c, make_tuple(M, N), make_tuple(Ldc, 1), number<kCAlignment>{}, number<1>{});
        }();

        GridGemm_{}(a_dram, b_dram, c_dram, c_element_func);
    }
};

} // namespace ck_tile
