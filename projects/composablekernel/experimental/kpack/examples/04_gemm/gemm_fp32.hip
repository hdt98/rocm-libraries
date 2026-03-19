// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Device bridge for fp32 GEMM through the kpack pattern.
//
// This file assembles CK Tile's GEMM template stack (7 types) and bridges
// it to our clean GemmArgs ABI struct. The host never includes CK Tile
// headers — it constructs GemmArgs and launches this kernel via
// hipModuleLaunchKernel.
//
// Template assembly follows example/ck_tile/03_gemm/gemm_basic_invoker.hpp
// pattern, hardcoded to fp32 with 128x128x32 tiles and 16x16x16 warp tile:
//
//   1. Shape        — tile geometry: block tile, warp layout, warp tile
//   2. Traits       — padding flags, memory layouts (A=Row, B=Col, C=Row)
//   3. Problem      — combines data types, shape, and traits for the pipeline
//   4. Pipeline     — the GEMM compute pipeline (AGmem→BGmem→CReg, V1 basic)
//   5. Partitioner  — maps 1D blockIdx.x to 2D (M,N) tile coordinates
//   6. Epilogue     — shuffles accumulator to output via LDS (CShuffleEpilogue)
//   7. Kernel       — top-level GemmKernel that orchestrates everything
//
// The bridge function receives GemmArgs (our ABI), converts to CK Tile's
// UniversalGemmKernelArgs internally, then delegates to GemmKernel::operator().
// This is one-time-per-workgroup overhead (trivial) and keeps the host
// completely free of CK Tile headers.

#include "gemm_args.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"

using namespace kpack_gemm;

// ============================================================================
// Step 1: Tile geometry — 128x128x32 block tile, 2x2x1 warp layout, 16x16x16 warp tile
//
// fp32 MFMA constraints:
//   - 32x32 warp tiles only support K=4 or K=8 → LDS too large at 256x256 block
//   - 16x16 warp tiles support K=4, K=8, or K=16 → 128x128 block fits in LDS
// ============================================================================
using GemmShape = ck_tile::TileGemmShape<
    ck_tile::sequence<128, 128, 32>,  // BlockTile (M, N, K)
    ck_tile::sequence<2, 2, 1>,       // BlockWarps (M, N, K) → 4 warps = 256 threads
    ck_tile::sequence<16, 16, 16>>;   // WarpTile — K=16 is max for fp32 16x16 MFMA

// ============================================================================
// Step 2: Traits — no padding, standard BLAS layouts
// ============================================================================
using GemmTraits = ck_tile::TileGemmTraits<
    false,                                     // kPadM — require M aligned to M_TILE
    false,                                     // kPadN — require N aligned to N_TILE
    false,                                     // kPadK — require K aligned to K_TILE
    ck_tile::tensor_layout::gemm::RowMajor,    // A layout
    ck_tile::tensor_layout::gemm::ColumnMajor, // B layout
    ck_tile::tensor_layout::gemm::RowMajor>;   // C layout

// ============================================================================
// Step 3: Pipeline problem — fp32 data types + shape + traits
// ============================================================================
using PipelineProblem = ck_tile::GemmPipelineProblem<
    float,      // ADataType
    float,      // BDataType
    float,      // AccDataType (accumulation precision)
    GemmShape,
    GemmTraits>;

// ============================================================================
// Step 4: Pipeline — simplest GEMM pipeline (A/B from global memory, C in registers)
// ============================================================================
using GemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<PipelineProblem>;

// ============================================================================
// Step 5: Partitioner — flattens 2D (M,N) tile grid into 1D blockIdx.x
// ============================================================================
using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;

// ============================================================================
// Step 6: Epilogue — shuffle accumulator through LDS, then store to global
// ============================================================================
using GemmEpilogue = ck_tile::CShuffleEpilogue<
    ck_tile::CShuffleEpilogueProblem<
        float,                                     // ADataType
        float,                                     // BDataType
        ck_tile::tuple<>,                           // DsDataType (no bias/fused tensors)
        float,                                     // AccDataType
        float,                                     // ODataType (output = fp32)
        ck_tile::tuple<>,                           // DsLayout (no bias/fused tensors)
        ck_tile::tensor_layout::gemm::RowMajor,    // C/E layout
        ck_tile::element_wise::PassThrough,        // element-wise op (identity)
        TilePartitioner::MPerBlock,                // = 256
        TilePartitioner::NPerBlock,                // = 256
        2,                                         // M_Warp
        2,                                         // N_Warp
        16,                                        // M_Warp_Tile
        16,                                        // N_Warp_Tile
        16,                                        // K_Warp_Tile
        PipelineProblem::TransposeC>>;             // = false

// ============================================================================
// Step 7: Kernel — ties partitioner, pipeline, and epilogue together
// ============================================================================
using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

// KernelArgs = UniversalGemmKernelArgs<1, 1, 0> (1 A tensor, 1 B tensor, 0 D tensors)
using KernelArgs = typename Kernel::UniversalGemmKernel::KernelArgs;

// ============================================================================
// Bridge function: clean GemmArgs ABI → CK Tile KernelArgs → GemmKernel
// ============================================================================
extern "C" __global__ void gemm_fp32(GemmArgs args)
{
    // Convert our clean ABI struct to CK Tile's internal kernel args.
    // We build KernelArgs on the device side so the host never needs CK headers.
    //
    // Must use aggregate initialization because as_ptr/bs_ptr/ds_ptr are const.
    // Zero-sized arrays (ds_ptr, stride_Ds) are initialized empty.
    // PersistentAsyncInputScheduler uses default member initializer (nullptr signals).
    const KernelArgs kargs{{args.a},        // as_ptr
                           {args.b},        // bs_ptr
                           {},              // ds_ptr  (empty — no D tensors)
                           args.c,          // e_ptr
                           args.M,          // M
                           args.N,          // N
                           args.K,          // K
                           {args.stride_A}, // stride_As
                           {args.stride_B}, // stride_Bs
                           {},              // stride_Ds (empty — no D tensors)
                           args.stride_C,   // stride_E
                           1};              // k_batch (no split-K)

    // Delegate to CK Tile's GemmKernel, which handles:
    //   - tile partitioning (blockIdx.x → M,N tile)
    //   - pipeline execution (load A,B → MFMA → accumulate)
    //   - epilogue (shuffle → store C)
    Kernel{}(kargs);
}
