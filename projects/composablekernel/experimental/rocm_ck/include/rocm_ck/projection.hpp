// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: meta — project rocm_ck GemmSpec to CK dispatcher KernelKey.
//
// Pure mapping between two representations of the same kernel space.
// No runtime dependencies. Header-only.

#pragma once

#include <rocm_ck/gemm_spec.hpp>

#include <ck_tile/dispatcher/kernel_key.hpp>

#include <stdexcept>
#include <string>
#include <utility>

namespace rocm_ck {

// ============================================================================
// Individual enum projections
// ============================================================================

/// Map rocm_ck DataType to dispatcher DataType.
/// FP8_FNUZ/FP8_OCP both collapse to FP8 (arch determines physical format).
inline ck_tile::dispatcher::DataType projectDtype(DataType dt)
{
    switch(dt)
    {
    case DataType::FP64: return ck_tile::dispatcher::DataType::FP64;
    case DataType::FP32: return ck_tile::dispatcher::DataType::FP32;
    case DataType::FP16: return ck_tile::dispatcher::DataType::FP16;
    case DataType::BF16: return ck_tile::dispatcher::DataType::BF16;
    case DataType::FP8_FNUZ:
    case DataType::FP8_OCP: return ck_tile::dispatcher::DataType::FP8;
    case DataType::BF8_FNUZ:
    case DataType::BF8_OCP: return ck_tile::dispatcher::DataType::BF8;
    case DataType::I8: return ck_tile::dispatcher::DataType::INT8;
    case DataType::I4: return ck_tile::dispatcher::DataType::INT4;
    case DataType::I32: return ck_tile::dispatcher::DataType::INT32;
    default: return ck_tile::dispatcher::DataType::UNKNOWN;
    }
}

/// Map rocm_ck Layout to dispatcher LayoutTag.
/// Only Row and Col are valid for GEMM matrices. Auto is a resolve-time
/// placeholder and Contiguous is for rank-1 tensors — neither should reach here.
inline ck_tile::dispatcher::LayoutTag projectLayout(Layout l)
{
    switch(l)
    {
    case Layout::Row: return ck_tile::dispatcher::LayoutTag::RowMajor;
    case Layout::Col: return ck_tile::dispatcher::LayoutTag::ColMajor;
    default:
        throw std::runtime_error("projectLayout: unsupported layout (Auto/Contiguous are not "
                                 "valid physical layouts for GEMM projection)");
    }
}

/// Map rocm_ck Pipeline to dispatcher Pipeline.
inline ck_tile::dispatcher::Pipeline projectPipeline(Pipeline p)
{
    switch(p)
    {
    case Pipeline::V1: return ck_tile::dispatcher::Pipeline::CompV1;
    case Pipeline::V3: return ck_tile::dispatcher::Pipeline::CompV3;
    case Pipeline::V4: return ck_tile::dispatcher::Pipeline::CompV4;
    case Pipeline::Memory: return ck_tile::dispatcher::Pipeline::Mem;
    case Pipeline::Preshuffle: return ck_tile::dispatcher::Pipeline::PreShuffleV2;
    }
    return ck_tile::dispatcher::Pipeline::CompV1;
}

/// Map rocm_ck PipelineScheduler to dispatcher Scheduler.
inline ck_tile::dispatcher::Scheduler projectScheduler(PipelineScheduler s)
{
    switch(s)
    {
    case PipelineScheduler::Intrawave: return ck_tile::dispatcher::Scheduler::Intrawave;
    case PipelineScheduler::Interwave: return ck_tile::dispatcher::Scheduler::Interwave;
    }
    return ck_tile::dispatcher::Scheduler::Intrawave;
}

/// Map rocm_ck epilogue chain to dispatcher Epilogue enum + elementwise_op string.
///
/// The dispatcher conflates two concerns that rocm-ck separates:
///   - WHAT ops execute → rocm_ck epilogue_ops[] sequence
///   - HOW results are stored → rocm_ck StoreStrategy
///
/// This function flattens the composable chain into the dispatcher's encoding.
inline std::pair<ck_tile::dispatcher::Epilogue, std::string> projectEpilogue(const GemmSpec& spec)
{
    namespace disp = ck_tile::dispatcher;

    bool has_binary = false;
    bool has_unary  = false;
    EpilogueOp binary_op{};
    EpilogueOp unary_op{};

    for(int i = 0; i < spec.num_epilogue_ops; ++i)
    {
        EpilogueOp op = spec.epilogue_ops[i];
        if(op == EpilogueOp::Add || op == EpilogueOp::Mul)
        {
            has_binary = true;
            binary_op  = op;
        }
        else
        {
            has_unary = true;
            unary_op  = op;
        }
    }

    // Map unary activation name to dispatcher's string convention
    auto unaryName = [](EpilogueOp op) -> std::string {
        switch(op)
        {
        case EpilogueOp::Relu: return "Relu";
        case EpilogueOp::FastGelu: return "FastGelu";
        case EpilogueOp::Gelu: return "Gelu";
        case EpilogueOp::Silu: return "Swish"; // dispatcher calls SiLU "Swish"
        case EpilogueOp::Sigmoid: return "Sigmoid";
        default: throw std::runtime_error("projectEpilogue: unsupported unary EpilogueOp");
        }
    };

    // D tensors require a binary op — if present without one, the spec is malformed
    if(spec.numDTensors() > 0 && !has_binary)
        throw std::runtime_error("projectEpilogue: GemmSpec has D tensors but no binary "
                                 "epilogue op (Add/Mul) — schema mismatch");

    if(!has_binary && !has_unary)
    {
        // No epilogue ops
        if(spec.store_strategy == StoreStrategy::Direct2D)
            return {disp::Epilogue::Default, "PassThrough"};
        return {disp::Epilogue::CShuffle, "PassThrough"};
    }

    if(has_binary && has_unary)
    {
        // Binary + unary: e.g., Add + Relu
        return {disp::Epilogue::BiasActivation, unaryName(unary_op)};
    }

    if(has_binary)
    {
        // Binary only: Add or Mul
        if(spec.numDTensors() > 1)
        {
            // Multiple D tensors: MultiDAdd or MultiDMultiply
            std::string op_name = (binary_op == EpilogueOp::Add) ? "MultiDAdd" : "MultiDMultiply";
            return {disp::Epilogue::Bias, op_name};
        }
        std::string op_name = (binary_op == EpilogueOp::Add) ? "Add" : "Multiply";
        return {disp::Epilogue::Bias, op_name};
    }

    // Unary only: activation without bias
    return {disp::Epilogue::Activation, unaryName(unary_op)};
}

// ============================================================================
// Main projection: GemmSpec → KernelKey
// ============================================================================

/// Project a rocm_ck GemmSpec to a CK dispatcher KernelKey.
///
/// The mapping is an isomorphism for the kernel space both systems target
/// (documented in memory/dispatcher-rocmck-mapping.md).
///
/// The arch string (e.g., "gfx942") is stored in the KernelKey but not used
/// for projection decisions. FP8/BF8 variant selection is handled upstream:
/// kpack archives are pre-filtered by TargetSet, so a gfx942 archive only
/// contains FP8_FNUZ kernels and a gfx950 archive only FP8_OCP.
inline ck_tile::dispatcher::KernelKey projectToDispatcher(const GemmSpec& spec,
                                                          const std::string& arch)
{
    namespace disp = ck_tile::dispatcher;

    auto [epilogue, elementwise_op] = projectEpilogue(spec);

    disp::KernelKey key{};

    // Signature — WHAT operation
    key.signature.dtype_a     = projectDtype(spec.lhs().dtype);
    key.signature.dtype_b     = projectDtype(spec.rhs().dtype);
    key.signature.dtype_c     = projectDtype(spec.output().dtype);
    key.signature.dtype_acc   = projectDtype(spec.acc_dtype);
    key.signature.layout_a    = projectLayout(spec.lhs().layout);
    key.signature.layout_b    = projectLayout(spec.rhs().layout);
    key.signature.layout_c    = projectLayout(spec.output().layout);
    key.signature.transpose_a = (spec.lhs().layout == Layout::Col);
    key.signature.transpose_b = (spec.rhs().layout == Layout::Col);
    key.signature.grouped     = false; // runtime property, not in spec
    key.signature.split_k     = static_cast<std::uint8_t>(spec.k_batch);

    key.signature.elementwise_op      = elementwise_op;
    key.signature.num_d_tensors       = static_cast<std::uint8_t>(spec.numDTensors());
    key.signature.structured_sparsity = false; // not in rocm-ck yet

    // Algorithm — HOW it's implemented
    key.algorithm.tile_shape = {static_cast<std::uint16_t>(spec.block_tile.m),
                                static_cast<std::uint16_t>(spec.block_tile.n),
                                static_cast<std::uint16_t>(spec.block_tile.k)};

    key.algorithm.wave_shape = {static_cast<std::uint8_t>(spec.block_waves.m),
                                static_cast<std::uint8_t>(spec.block_waves.n),
                                static_cast<std::uint8_t>(spec.block_waves.k)};

    key.algorithm.warp_tile_shape = {static_cast<std::uint8_t>(spec.wave_tile.m),
                                     static_cast<std::uint8_t>(spec.wave_tile.n),
                                     static_cast<std::uint8_t>(spec.wave_tile.k)};

    key.algorithm.pipeline  = projectPipeline(spec.pipeline);
    key.algorithm.scheduler = projectScheduler(spec.pipeline_scheduler);
    key.algorithm.epilogue  = epilogue;

    key.algorithm.block_size = static_cast<std::uint16_t>(spec.workgroup_size);
    key.algorithm.double_buffer =
        (spec.pipeline == Pipeline::V4 || spec.pipeline == Pipeline::Preshuffle);
    key.algorithm.persistent  = (spec.tile_partitioner == TilePartitioner::StreamK);
    key.algorithm.preshuffle  = (spec.pipeline == Pipeline::Preshuffle);
    key.algorithm.transpose_c = false; // CK internal, not exposed
    key.algorithm.num_wave_groups =
        static_cast<std::uint8_t>(spec.block_waves.m * spec.block_waves.n * spec.block_waves.k);

    key.gfx_arch = arch;

    return key;
}

} // namespace rocm_ck
