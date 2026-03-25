// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Vector add host-side API for rocm_ck elementwise example.
//
// Host-only: make_kernel() validates a Signature + ElementwiseAlgorithm into
// a VectorAddKernel descriptor. This header is NOT included by device code
// (.hip files). Device code includes rocm_vector_add_kernel.hpp (the pure
// structural types) via rocm_vector_add_dev.hpp.

#pragma once

#include "rocm_vector_add_kernel.hpp"

#include <rocm_ck/resolve.hpp>
#include <rocm_ck/tensor_desc.hpp>

namespace rocm_ck {

// ============================================================================
// make_kernel: operator-centric Signature -> VectorAddKernel
// ============================================================================

/// Resolve and validate a vector add using the operator-centric Signature.
///
/// Pattern-matches: first op must be AddOp. Tensors get rank=1,
/// Layout::Contiguous as elementwise defaults.
///
/// Validates:
///   - Input types match (a_dtype == b_dtype)
///   - block_tile, block_warps, warp_tile are positive
///   - block_warps is power of 2 (CK Tile reduce_on_sequence requirement)
///   - warp_tile >= warp_size (64)
///   - kVectorM >= 1 and block_tile divisibility
consteval VectorAddKernel make_kernel(Signature sig, ElementwiseAlgorithm algo)
{
    ResolvedSignature resolved = resolve(sig);

    // First op must be AddOp
    if(!std::holds_alternative<AddOp>(sig.ops[0]))
        throw "vector add make_kernel requires AddOp as first operator";
    const AddOp& add = std::get<AddOp>(sig.ops[0]);

    // Remaining ops must be empty
    for(int i = 1; i < kMaxOps; ++i)
        if(!std::holds_alternative<std::monostate>(sig.ops[i]))
            throw "vector add make_kernel only supports a single AddOp";

    TensorDesc a_td   = resolved.tensor(add.lhs);
    TensorDesc b_td   = resolved.tensor(add.rhs);
    TensorDesc out_td = resolved.tensor(add.out);

    if(a_td.dtype != b_td.dtype)
        throw "vector add requires matching input dtypes";

    if(algo.block_tile <= 0)
        throw "block_tile must be positive";
    if(algo.block_warps <= 0)
        throw "block_warps must be positive";
    if((algo.block_warps & (algo.block_warps - 1)) != 0)
        throw "block_warps must be a power of 2 (required by CK Tile reduce_on_sequence)";
    if(algo.warp_tile <= 0)
        throw "warp_tile must be positive";
    if(algo.warp_tile < warp_size)
        throw "warp_tile must be >= warp_size (64)";

    int in_bits    = data_type_bits(a_td.dtype);
    int out_bits   = data_type_bits(out_td.dtype);
    int max_bits   = in_bits > out_bits ? in_bits : out_bits;
    int kVectorM_a = 128 / max_bits;
    int kVectorM_b = algo.warp_tile / warp_size;
    int kVectorM   = kVectorM_a < kVectorM_b ? kVectorM_a : kVectorM_b;

    if(kVectorM < 1)
        throw "computed kVectorM must be >= 1 (warp_tile too small for type)";

    int elements_per_iter = algo.block_warps * kVectorM * warp_size;
    if(algo.block_tile % elements_per_iter != 0)
        throw "block_tile must be divisible by (block_warps * kVectorM * warp_size)";

    int kRepeatM = algo.block_tile / elements_per_iter;
    if(kRepeatM < 1)
        throw "computed kRepeatM must be >= 1 (block_tile too small for given warps)";

    return {algo.block_tile,
            warp_size * algo.block_warps,
            a_td.dtype,
            out_td.dtype,
            algo.block_warps,
            algo.warp_tile,
            algo.pad};
}

// ============================================================================
// Compile-time tests
// ============================================================================
// clang-format off

// --- Homogeneous: dtype sets all tensors ---
static_assert(make_kernel(
    Signature{.dtype = DataType::FP32,
              .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    ElementwiseAlgorithm{1024, 1, 1024, true}).in_dtype == DataType::FP32);
static_assert(make_kernel(
    Signature{.dtype = DataType::FP32,
              .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    ElementwiseAlgorithm{1024, 1, 1024, true}).out_dtype == DataType::FP32);

// --- Mixed types via Tensor override ---
static_assert(make_kernel(
    Signature{.dtype = DataType::FP16,
              .tensors = {Tensor{.name = "C", .dtype = DataType::FP32}},
              .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    ElementwiseAlgorithm{1024, 1, 1024, true}).in_dtype == DataType::FP16);
static_assert(make_kernel(
    Signature{.dtype = DataType::FP16,
              .tensors = {Tensor{.name = "C", .dtype = DataType::FP32}},
              .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    ElementwiseAlgorithm{1024, 1, 1024, true}).out_dtype == DataType::FP32);

// Error cases (uncommenting any would produce consteval compile errors):
// No dtype set:
// make_kernel(Signature{.ops = {AddOp{}}}, ElementwiseAlgorithm{1024, 1, 1024, true})
//
// No AddOp:
// make_kernel(Signature{.dtype = DataType::FP32, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
//             ElementwiseAlgorithm{1024, 1, 1024, true})

// clang-format on

} // namespace rocm_ck
