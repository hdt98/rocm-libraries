// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: device — CK Tile GEMM type wiring for kpack.
//
// DEVICE ONLY: compiled via --cuda-device-only in .hip files.
// This header must NOT be included from host-only .cpp files.
//
// Maps GemmSpec (our schema) -> CK Tile template stack.
// Each .hip variant file includes this header and instantiates run<S>
// with a specific constexpr GemmSpec.
//
// Supports:
//   - Pipeline selection: V1, V3, V4, Memory, Preshuffle
//   - Tile partitioning: Linear (standard), StreamK (work-balanced)
//   - Batched GEMM: batch dimension via blockIdx.y (runtime, in Args)
//
// Terminology mapping: rocm_ck → CK Tile
//   block_waves    → BlockWarps (sequence<MWarp, NWarp, KWarp>)
//   wave_tile      → WarpTile (sequence<MPerXdl, NPerXdl, KPerXdl>)
//   block_tile     → BlockTile (sequence<MPerBlock, NPerBlock, KPerBlock>)
//   workgroup_size → BlockSize (product of BlockWarps × wavefront_size)
//
// Compilation boundary:
//   _spec.hpp — schema types + consteval factory (both passes)
//   _dev.hpp (this) — CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#ifndef __HIP_DEVICE_COMPILE__
#error "gemm_dev.hpp requires device compilation (--cuda-device-only)."
#endif

#include <rocm_ck/gemm_spec.hpp>

#include <rocm_ck/args.hpp>
#include <rocm_ck/ck_type_map.hpp>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
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
// unpackDTensor: extract a D tensor's pointer and stride from Args
// ============================================================================

/// Unpack a D tensor from Args: typed pointer (with batch offset) and leading-dim stride.
/// Centralizes the per-D-tensor boilerplate that was previously duplicated per NumDTensors branch.
template <PhysicalTensor PT>
__device__ auto unpackDTensor(const Args& args, index_t i_batch) -> std::pair<const void*, index_t>
{
    using DType        = typename CkTypeMap<PT.dtype>::type;
    const TensorArg& t = args.tensors[PT.args_slot];
    index_t stride     = static_cast<index_t>(leadingDimStride(PT.layout, t.strides));
    const void* ptr    = static_cast<const DType*>(t.ptr) +
                      i_batch * static_cast<index_t>(args.batch_strides[PT.args_slot]);
    return {ptr, stride};
}

// ============================================================================
// ComposedCDEOp: epilogue op chain → single CK Tile elementwise functor
// ============================================================================

/// Composed epilogue functor: applies K's epilogue_ops chain in sequence.
///
/// Binary ops (Add, Mul) fold D tensors into the accumulator via parameter pack.
/// Unary ops (Relu, etc.) transform the accumulator in place, delegating to
/// CK Tile's optimized implementations.
///
/// All arithmetic is in float. The 4 if-constexpr lines are bounded by
/// kMaxEpilogueOps and fully resolved at compile time — zero runtime overhead.
template <GemmSpec S>
struct ComposedCDEOp
{
    template <typename E, typename C, typename... Ds>
    CK_TILE_HOST_DEVICE void operator()(E& e, const C& c, const Ds&... ds) const
    {
        float result = ck_tile::type_convert<float>(c);
        if constexpr(S.num_epilogue_ops > 0)
            apply_op<S.epilogue_ops[0]>(result, ds...);
        if constexpr(S.num_epilogue_ops > 1)
            apply_op<S.epilogue_ops[1]>(result, ds...);
        if constexpr(S.num_epilogue_ops > 2)
            apply_op<S.epilogue_ops[2]>(result, ds...);
        if constexpr(S.num_epilogue_ops > 3)
            apply_op<S.epilogue_ops[3]>(result, ds...);
        e = ck_tile::type_convert<E>(result);
    }

    private:
    template <EpilogueOp Op, typename... Ds>
    CK_TILE_HOST_DEVICE static void apply_op(float& result, const Ds&... ds)
    {
        if constexpr(Op == EpilogueOp::Add)
            ((result += ck_tile::type_convert<float>(ds)), ...);
        else if constexpr(Op == EpilogueOp::Mul)
            ((result *= ck_tile::type_convert<float>(ds)), ...);
        else if constexpr(Op == EpilogueOp::Relu)
            ck_tile::element_wise::Relu{}(result, result);
        else if constexpr(Op == EpilogueOp::FastGelu)
            ck_tile::element_wise::FastGelu{}(result, result);
        else if constexpr(Op == EpilogueOp::Gelu)
            ck_tile::element_wise::Gelu{}(result, result);
        else if constexpr(Op == EpilogueOp::Silu)
            ck_tile::element_wise::Silu{}(result, result);
        else if constexpr(Op == EpilogueOp::Sigmoid)
            ck_tile::element_wise::Sigmoid{}(result, result);
    }
};

// ============================================================================
// EpilogueTypes<S>: derive DsDataType, DsLayout, and Op tuples from GemmSpec
// ============================================================================

template <GemmSpec S>
struct EpilogueTypes
{
    // PassThrough for no epilogue ops (avoids float round-trip that corrupts integer types).
    // ComposedCDEOp for epilogue chains (Add, Relu, etc.) — operates in float.
    using Op = std::conditional_t<S.num_epilogue_ops == 0,
                                  ck_tile::element_wise::PassThrough,
                                  ComposedCDEOp<S>>;

    // D tensor count: auxiliary tensors (bias, etc.) — excludes scale tensor
    static constexpr int NumDTensors = S.numDTensors();

    // D0/D1 types from physical tensor table via named accessors.
    // Default to float/Row when no D tensors (unused but satisfies template resolution).
    using D0Type =
        std::conditional_t<(NumDTensors >= 1), typename CkTypeMap<S.d0().dtype>::type, float>;
    using D1Type =
        std::conditional_t<(NumDTensors >= 2), typename CkTypeMap<S.d1().dtype>::type, float>;
    using D0Layout = std::conditional_t<(NumDTensors >= 1),
                                        typename CkLayoutMap<S.d0().layout>::type,
                                        typename CkLayoutMap<Layout::Row>::type>;
    using D1Layout = std::conditional_t<(NumDTensors >= 2),
                                        typename CkLayoutMap<S.d1().layout>::type,
                                        typename CkLayoutMap<Layout::Row>::type>;

    using DsDataType = std::conditional_t<NumDTensors == 0,
                                          ck_tile::tuple<>,
                                          std::conditional_t<NumDTensors == 1,
                                                             ck_tile::tuple<D0Type>,
                                                             ck_tile::tuple<D0Type, D1Type>>>;

    using DsLayout = std::conditional_t<NumDTensors == 0,
                                        ck_tile::tuple<>,
                                        std::conditional_t<NumDTensors == 1,
                                                           ck_tile::tuple<D0Layout>,
                                                           ck_tile::tuple<D0Layout, D1Layout>>>;
};

// ============================================================================
// run<S>: assemble CK Tile GEMM pipeline from GemmSpec descriptor
// ============================================================================

/// Device-side GEMM bridge: Args → CK Tile template stack → ck_tile::GemmKernel.
///
/// Tensor slot mapping comes from the spec's role-based accessors:
///   lhs() = left operand,  rhs() = right operand,  output() = final output
///   lengths[0] = first dim, lengths[1] = second dim
///
/// Supports batched GEMM (batch dimension via blockIdx.y when batch_count > 0),
/// multiple pipeline strategies (V1, V3, V4, Memory, Preshuffle), scheduling
/// modes (Intrawave, Interwave), tile partitioners (Linear, StreamK), and
/// block-quantized GEMM (BQuant: INT4 weight with per-group scale tensors).
template <GemmSpec S>
__device__ void run(Args args)
{
    // Device-side validation — catches invalid manual construction.
    static_assert(S.num_physical_tensors >= 3,
                  "kernel must have at least lhs, rhs, and output tensors");
    static_assert(S.workgroup_size > 0, "workgroup_size must be positive");
    static_assert(EpilogueTypes<S>::NumDTensors <= 2,
                  "at most 2 D tensors supported in this example");
    static_assert(S.tile_partitioner == TilePartitioner::Linear,
                  "only Linear tile partitioning supported in run<S>(); "
                  "Stream-K requires a separate entry point");

    // --- Physical tensor table: role-based access (compile-time constants) ---
    static constexpr auto PT_LHS = S.lhs();    // GEMM left operand
    static constexpr auto PT_RHS = S.rhs();    // GEMM right operand
    static constexpr auto PT_OUT = S.output(); // final output

    // --- Map schema types to CK Tile types ---
    using AType   = typename CkTypeMap<PT_LHS.dtype>::type;
    using BType   = typename CkTypeMap<PT_RHS.dtype>::type;
    using AccType = typename CkTypeMap<S.acc_dtype>::type;
    using OType   = typename CkTypeMap<PT_OUT.dtype>::type;

    using ALayout = typename CkLayoutMap<PT_LHS.layout>::type;
    using BLayout = typename CkLayoutMap<PT_RHS.layout>::type;
    using CLayout = typename CkLayoutMap<PT_OUT.layout>::type;

    // --- Unpack generic Args — compiler generates s_load at fixed offsets ---
    const TensorArg& t_lhs = args.tensors[PT_LHS.args_slot];
    const TensorArg& t_rhs = args.tensors[PT_RHS.args_slot];
    const TensorArg& t_out = args.tensors[PT_OUT.args_slot];

    index_t M     = t_lhs.lengths[0];
    index_t N     = t_rhs.lengths[1];
    index_t K_dim = t_lhs.lengths[1];

    index_t stride_lhs = static_cast<index_t>(leadingDimStride(PT_LHS.layout, t_lhs.strides));
    index_t stride_rhs = static_cast<index_t>(leadingDimStride(PT_RHS.layout, t_rhs.strides));
    index_t stride_out = static_cast<index_t>(leadingDimStride(PT_OUT.layout, t_out.strides));

    // --- Batch offset (runtime) ---
    // When batch_count > 0, blockIdx.y indexes into the batch dimension.
    // batch_strides are in elements — typed pointer arithmetic handles scaling.
    // When batch_count == 0 (unbatched), i_batch is 0 and offsets are zero.
    const index_t i_batch = args.batch_count > 0 ? static_cast<index_t>(blockIdx.y) : 0;

    const AType* lhs_ptr = static_cast<const AType*>(t_lhs.ptr) +
                           i_batch * static_cast<index_t>(args.batch_strides[PT_LHS.args_slot]);
    const BType* rhs_ptr = static_cast<const BType*>(t_rhs.ptr) +
                           i_batch * static_cast<index_t>(args.batch_strides[PT_RHS.args_slot]);
    OType* out_typed = static_cast<OType*>(const_cast<void*>(t_out.ptr)) +
                       i_batch * static_cast<index_t>(args.batch_strides[PT_OUT.args_slot]);
    void* out_ptr = out_typed;

    // --- Tile geometry (from GemmSpec, validated by makeSpec) ---
    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<S.block_tile.m, S.block_tile.n, S.block_tile.k>,
                               ck_tile::sequence<S.block_waves.m, S.block_waves.n, S.block_waves.k>,
                               ck_tile::sequence<S.wave_tile.m, S.wave_tile.n, S.wave_tile.k>>;

    // TransposeC: true when output is column-major (matches CK Tile convention)
    static constexpr bool TransposeC = (PT_OUT.layout == Layout::Col);

    // --- Partitioner (1D blockIdx → 2D tile coordinates) ---
    using Partitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;

    // =========================================================================
    // BQuant path: block-quantized GEMM (INT4 weight with per-group scales)
    // =========================================================================
    if constexpr(S.group_size > 0)
    {
        static_assert(S.numDTensors() == 0,
                      "BQuant GEMM with epilogue D tensors not yet supported");

        // Scale tensor from physical tensor table
        static constexpr auto PT_SCALE = S.scale();
        using BQType                   = typename CkTypeMap<PT_SCALE.dtype>::type;

        // QuantGroupShape: kM=1 (no M grouping), kN=1 (standard per-group), kK=group_size
        using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, S.group_size>>;

        // BQuant traits (no preshuffle for MVP)
        using BQuantTraits = ck_tile::TileGemmQuantTraits<S.pad_m,
                                                          S.pad_n,
                                                          false, // kPadK
                                                          false, // APreshuffleQuant
                                                          false, // BPreshuffleQuant
                                                          false, // PreshuffleB
                                                          ALayout,
                                                          BLayout,
                                                          CLayout,
                                                          ck_tile::QuantType::BQuantGrouped>;

        // BQuant problem → pipeline → kernel
        using BQuantProblem = ck_tile::GemmBQuantPipelineProblem<AType,
                                                                 BType,
                                                                 BQType,
                                                                 AccType,
                                                                 GemmShape,
                                                                 BQuantTraits,
                                                                 QuantGroupSize>;

        using BQuantPipeline = ck_tile::BQuantGemmPipelineAgBgCrCompV3<BQuantProblem>;

        // Epilogue: CShuffle with PassThrough (no D tensors)
        // BQuant dequantizes B to compute type (AType), so epilogue sees AType for both A and B.
        using BQuantEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<AType,
                                             AType,
                                             ck_tile::tuple<>,
                                             AccType,
                                             OType,
                                             ck_tile::tuple<>,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
                                             Partitioner::MPerBlock,
                                             Partitioner::NPerBlock,
                                             S.block_waves.m,
                                             S.block_waves.n,
                                             S.wave_tile.m,
                                             S.wave_tile.n,
                                             S.wave_tile.k,
                                             TransposeC>>;

        using BQuantKernel = ck_tile::QuantGemmKernel<Partitioner,
                                                      BQuantPipeline,
                                                      BQuantEpilogue,
                                                      ck_tile::QuantType::BQuantGrouped>;

        // Extract scale tensor
        const TensorArg& t_scale = args.tensors[PT_SCALE.args_slot];
        const BQType* scale_ptr  = static_cast<const BQType*>(t_scale.ptr);
        index_t stride_scale = static_cast<index_t>(leadingDimStride(Layout::Row, t_scale.strides));
        index_t QK_B         = t_scale.lengths[0]; // K / group_size

        const ck_tile::QuantGemmKernelArgs kargs{
            lhs_ptr,      // a_ptr
            rhs_ptr,      // b_ptr
            nullptr,      // aq_ptr (unused for B-only quantization)
            scale_ptr,    // bq_ptr (scale tensor)
            out_ptr,      // c_ptr
            M,            // M
            N,            // N
            K_dim,        // K
            0,            // QK_A (unused)
            QK_B,         // QK_B = K / group_size
            stride_lhs,   // stride_A
            stride_rhs,   // stride_B
            stride_out,   // stride_C
            0,            // stride_AQ (unused)
            stride_scale, // stride_BQ
            S.k_batch     // k_batch
        };
        BQuantKernel{}(kargs);
    }
    // =========================================================================
    // Standard GEMM path (non-quantized)
    // =========================================================================
    else
    {

        // --- Pipeline selection (S.pipeline + S.pipeline_scheduler) ---

        // Map rocm_ck::PipelineScheduler → ck_tile::GemmPipelineScheduler
        static constexpr auto CkScheduler = S.pipeline_scheduler == PipelineScheduler::Interwave
                                                ? ck_tile::GemmPipelineScheduler::Interwave
                                                : ck_tile::GemmPipelineScheduler::Intrawave;

        // V1: simple pipeline (A/B from global, C in registers)
        using V1Pipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<ck_tile::GemmPipelineProblem<
            AType,
            BType,
            AccType,
            GemmShape,
            ck_tile::TileGemmTraits<S.pad_m, S.pad_n, false, ALayout, BLayout, CLayout>>>;

        // V3: compute-optimized (software-pipelined loads, double LDS buffer)
        using V3Pipeline = ck_tile::GemmPipelineAgBgCrCompV3<ck_tile::UniversalGemmPipelineProblem<
            AType,
            BType,
            AccType,
            GemmShape,
            ck_tile::
                TileGemmUniversalTraits<S.pad_m, S.pad_n, false, true, ALayout, BLayout, CLayout>>>;

        // V4: compute double-buffer (ping-pong LDS layout)
        using V4Pipeline = ck_tile::GemmPipelineAgBgCrCompV4<ck_tile::UniversalGemmPipelineProblem<
            AType,
            BType,
            AccType,
            GemmShape,
            ck_tile::
                TileGemmUniversalTraits<S.pad_m, S.pad_n, false, true, ALayout, BLayout, CLayout>>>;

        // Memory: memory-optimized pipeline (A/B through LDS, supports Intrawave/Interwave)
        using MemoryPipeline = ck_tile::GemmPipelineAgBgCrMem<ck_tile::UniversalGemmPipelineProblem<
            AType,
            BType,
            AccType,
            GemmShape,
            ck_tile::
                TileGemmUniversalTraits<S.pad_m, S.pad_n, false, true, ALayout, BLayout, CLayout>,
            CkScheduler>>;

        // Preshuffle: weight preshuffle pipeline (A=Row, B=Col, DoubleSmemBuffer=true,
        // Preshuffle=true)
        using PreshufflePipeline = ck_tile::WeightPreshufflePipelineAGmemBGmemCRegV2<
            ck_tile::UniversalGemmPipelineProblem<AType,
                                                  BType,
                                                  AccType,
                                                  GemmShape,
                                                  ck_tile::TileGemmUniversalTraits<S.pad_m,
                                                                                   S.pad_n,
                                                                                   false,
                                                                                   true,
                                                                                   ALayout,
                                                                                   BLayout,
                                                                                   CLayout>>>;

        // Pipeline dispatch (one line per variant, reads as a flat table):
        using GemmPipeline = std::conditional_t<
            S.pipeline == Pipeline::V1,
            V1Pipeline,
            std::conditional_t<
                S.pipeline == Pipeline::V3,
                V3Pipeline,
                std::conditional_t<S.pipeline == Pipeline::V4,
                                   V4Pipeline,
                                   std::conditional_t<S.pipeline == Pipeline::Memory,
                                                      MemoryPipeline,
                                                      /* Preshuffle */ PreshufflePipeline>>>>;

        // --- Epilogue types from kernel descriptor ---
        using EpiOp      = typename EpilogueTypes<S>::Op;
        using DsDataType = typename EpilogueTypes<S>::DsDataType;
        using DsLayout   = typename EpilogueTypes<S>::DsLayout;

        // --- Epilogue strategy ---
        using CShuffleEpi =
            ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<AType,
                                                                       BType,
                                                                       DsDataType,
                                                                       AccType,
                                                                       OType,
                                                                       DsLayout,
                                                                       CLayout,
                                                                       EpiOp,
                                                                       Partitioner::MPerBlock,
                                                                       Partitioner::NPerBlock,
                                                                       S.block_waves.m, // MWave
                                                                       S.block_waves.n, // NWave
                                                                       S.wave_tile.m,   // MPerXdl
                                                                       S.wave_tile.n,   // NPerXdl
                                                                       S.wave_tile.k,   // KPerXdl
                                                                       TransposeC>>;

        // Direct2D: no LDS shuffle, direct 2D store. Uses DefaultGemm2DEpilogue
        // (not Default2DEpilogue — that one lacks the UniversalGemmKernel interface).
        using Direct2DEpi = ck_tile::DefaultGemm2DEpilogue<
            ck_tile::DefaultGemm2DEpilogueProblem<AType,
                                                  BType,
                                                  DsDataType,
                                                  AccType,
                                                  OType,
                                                  DsLayout,
                                                  CLayout,
                                                  EpiOp,
                                                  Partitioner::MPerBlock,
                                                  Partitioner::NPerBlock,
                                                  S.pad_m,
                                                  S.pad_n,
                                                  S.wave_tile.m,
                                                  S.wave_tile.n,
                                                  S.wave_tile.k,
                                                  TransposeC>>;

        using GemmEpilogue =
            std::conditional_t<S.epilogue == EpilogueStrategy::Direct2D, Direct2DEpi, CShuffleEpi>;

        // --- Kernel assembly + launch ---
        using CkKernel   = ck_tile::GemmKernel<Partitioner, GemmPipeline, GemmEpilogue>;
        using KernelArgs = typename CkKernel::UniversalGemmKernel::KernelArgs;

        // Convert generic Args to CK Tile's internal args.
        // Branch on D tensor count for the right tuple-sized initializer.
        static constexpr int NumDTensors = EpilogueTypes<S>::NumDTensors;

        if constexpr(NumDTensors == 0)
        {
            const KernelArgs kargs{{lhs_ptr},
                                   {rhs_ptr},
                                   {},
                                   out_ptr,
                                   M,
                                   N,
                                   K_dim,
                                   {stride_lhs},
                                   {stride_rhs},
                                   {},
                                   stride_out,
                                   S.k_batch};
            CkKernel{}(kargs);
        }
        else if constexpr(NumDTensors == 1)
        {
            auto [d0_ptr, stride_d0] = unpackDTensor<S.d0()>(args, i_batch);
            const KernelArgs kargs{{lhs_ptr},
                                   {rhs_ptr},
                                   {d0_ptr},
                                   out_ptr,
                                   M,
                                   N,
                                   K_dim,
                                   {stride_lhs},
                                   {stride_rhs},
                                   {stride_d0},
                                   stride_out,
                                   S.k_batch};
            CkKernel{}(kargs);
        }
        else if constexpr(NumDTensors == 2)
        {
            auto [d0_ptr, stride_d0] = unpackDTensor<S.d0()>(args, i_batch);
            auto [d1_ptr, stride_d1] = unpackDTensor<S.d1()>(args, i_batch);
            const KernelArgs kargs{{lhs_ptr},
                                   {rhs_ptr},
                                   {d0_ptr, d1_ptr},
                                   out_ptr,
                                   M,
                                   N,
                                   K_dim,
                                   {stride_lhs},
                                   {stride_rhs},
                                   {stride_d0, stride_d1},
                                   stride_out,
                                   S.k_batch};
            CkKernel{}(kargs);
        }

    } // end standard GEMM path
}

} // namespace rocm_ck
