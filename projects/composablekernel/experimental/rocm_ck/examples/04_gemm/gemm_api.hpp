// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// GEMM schema for kpack GEMM example.
//
// Defines the "WHAT" and "HOW" of a GEMM through operator-centric Signature
// and GemmAlgorithm, validated by make_kernel() into a GemmKernel NTTP with
// consteval checks for MFMA warp tile validity and tile divisibility.
//
// This header has NO CK Tile dependency. It is included by both host code
// (main.cpp) and device code (.hip files via gemm_dev.hpp).

#pragma once

#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/layout.hpp>
#include <rocm_ck/resolve.hpp>
#include <rocm_ck/tensor_desc.hpp>
#include <rocm_ck/types.hpp>

namespace rocm_ck {

// ============================================================================
// Epilogue: CombineOp × Activation (internal representation)
// ============================================================================

/// How to combine D tensors with the matmul result.
///
///   None     — E = A * B                    (no D tensors)
///   Add      — E = A * B + D0 [+ D1]       (bias addition)
///   Multiply — E = A * B * D0 [* D1]       (scaling)
enum class CombineOp
{
    None,
    Add,
    Multiply
};

/// Fused activation applied after the combine step.
///
///   None     — identity (no activation)
///   Relu     — max(0, x)
///   FastGelu — approximate GELU: x * sigmoid(1.702 * x)
///   Gelu     — exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
///   Silu     — x * sigmoid(x)  (aka Swish with beta=1)
///   Sigmoid  — 1 / (1 + exp(-x))
enum class Activation
{
    None,
    Relu,
    FastGelu,
    Gelu,
    Silu,
    Sigmoid
};

// ============================================================================
// Tile geometry types
// ============================================================================

/// M×N×K dimension triple for tile geometry specification.
struct Dim3
{
    int m, n, k;
};

/// Algorithm: describes HOW a GEMM executes (tile geometry).
/// Independent of data types — paired with Signature in make_kernel().
struct GemmAlgorithm
{
    Dim3 block_tile;  // Elements per thread block {M, N, K}
    Dim3 block_warps; // Warp distribution {M, N, K}
    Dim3 warp_tile;   // Elements per warp per MFMA step {M, N, K}
};

// ============================================================================
// Warp tile validation
// ============================================================================

/// Check if (a_dtype, warp_m, warp_n, warp_k) is a valid MFMA warp gemm
/// configuration. Based on CK Tile's WarpGemmDispatcher specializations
/// for gfx9 (MFMA). Only covers standard symmetric tile shapes.
consteval bool is_valid_warp_gemm(DataType a_dtype, int m, int n, int k)
{
    if(a_dtype == DataType::FP32)
    {
        if(m == 16 && n == 16 && (k == 4 || k == 8 || k == 16))
            return true;
        if(m == 32 && n == 32 && (k == 4 || k == 8))
            return true;
    }
    if(a_dtype == DataType::FP16)
    {
        if(m == 16 && n == 16 && (k == 16 || k == 32))
            return true;
        if(m == 32 && n == 32 && (k == 8 || k == 16))
            return true;
    }
    if(a_dtype == DataType::BF16)
    {
        if(m == 16 && n == 16 && (k == 16 || k == 32))
            return true;
        if(m == 32 && n == 32 && (k == 8 || k == 16))
            return true;
    }
    return false;
}

// ============================================================================
// GemmKernel — structural NTTP for template instantiation
// ============================================================================

/// Validated kernel descriptor with all types, layouts, and tile geometry resolved.
/// All members are structural types (enums, ints, aggregates) so this works as NTTP.
struct GemmKernel
{
    DataType a_dtype;
    DataType b_dtype;
    DataType c_dtype;
    DataType acc_dtype;
    Layout a_layout;
    Layout b_layout;
    Layout c_layout;
    Dim3 block_tile;
    Dim3 block_warps;
    Dim3 warp_tile;
    int thread_block_size;

    // Epilogue fields
    CombineOp combine;
    Activation activation;
    int num_d_tensors; // 0, 1, or 2
    DataType d0_dtype; // valid when num_d_tensors >= 1
    Layout d0_layout;
    DataType d1_dtype; // valid when num_d_tensors >= 2
    Layout d1_layout;
};

// ============================================================================
// make_kernel: operator-centric Signature → GemmKernel
// ============================================================================

/// Resolve and validate a GEMM using the operator-centric Signature.
///
/// Pattern-matches the ops array to determine epilogue:
///   {GemmOp}                          -> no epilogue
///   {GemmOp, AddOp}                   -> CombineOp::Add, 1 D tensor
///   {GemmOp, AddOp, ReluOp/...}       -> CombineOp::Add + activation
///   {GemmOp, MulOp}                   -> CombineOp::Multiply, 1 D tensor
///   {GemmOp, ReluOp/...}              -> activation only, no D tensors
///
/// The D tensor is the rhs of the AddOp/MulOp (the "bias" or "scale" operand).
/// c_dtype comes from the GemmOp output tensor.
///
/// Validates:
///   - All tile dimensions are positive
///   - block_warps.k == 1 (CShuffleEpilogue requires warps_m × warps_n block size)
///   - Warp tile matches MFMA dispatcher table for the input dtype
///   - Block tile is divisible by (block_warps × warp_tile) in each dimension
///
/// Derives thread_block_size = block_warps.m × block_warps.n × block_warps.k × 64.
consteval GemmKernel make_kernel(Signature sig, GemmAlgorithm algo)
{
    ResolvedSignature resolved = resolve(sig);

    // First op must be GemmOp
    if(!std::holds_alternative<GemmOp>(sig.ops[0]))
        throw "GEMM make_kernel requires GemmOp as first operator";
    const GemmOp& gemm = std::get<GemmOp>(sig.ops[0]);

    TensorDesc a_td = resolved.tensor(gemm.lhs);
    TensorDesc b_td = resolved.tensor(gemm.rhs);
    TensorDesc c_td = resolved.tensor(gemm.out);
    DataType acc    = gemm.acc_dtype;

    // Pattern-match epilogue from remaining ops
    CombineOp combine     = CombineOp::None;
    Activation activation = Activation::None;
    int num_d_tensors     = 0;
    DataType d0_dtype     = DataType::FP32;
    Layout d0_layout      = Layout::Row;
    DataType d1_dtype     = DataType::FP32;
    Layout d1_layout      = Layout::Row;

    int next_op = 1;

    // Check for combine op (AddOp or MulOp after GemmOp)
    if(next_op < kMaxOps && std::holds_alternative<AddOp>(sig.ops[next_op]))
    {
        const AddOp& add = std::get<AddOp>(sig.ops[next_op]);
        combine          = CombineOp::Add;
        num_d_tensors    = 1;
        TensorDesc d0_td = resolved.tensor(add.rhs);
        d0_dtype         = d0_td.dtype;
        d0_layout        = d0_td.layout != Layout::Auto ? d0_td.layout : Layout::Row;
        next_op++;
    }
    else if(next_op < kMaxOps && std::holds_alternative<MulOp>(sig.ops[next_op]))
    {
        const MulOp& mul = std::get<MulOp>(sig.ops[next_op]);
        combine          = CombineOp::Multiply;
        num_d_tensors    = 1;
        TensorDesc d0_td = resolved.tensor(mul.rhs);
        d0_dtype         = d0_td.dtype;
        d0_layout        = d0_td.layout != Layout::Auto ? d0_td.layout : Layout::Row;
        next_op++;
    }

    // Check for activation (unary op after combine or directly after GemmOp)
    if(next_op < kMaxOps)
    {
        if(std::holds_alternative<ReluOp>(sig.ops[next_op]))
        {
            activation = Activation::Relu;
            next_op++;
        }
        else if(std::holds_alternative<FastGeluOp>(sig.ops[next_op]))
        {
            activation = Activation::FastGelu;
            next_op++;
        }
        else if(std::holds_alternative<GeluOp>(sig.ops[next_op]))
        {
            activation = Activation::Gelu;
            next_op++;
        }
        else if(std::holds_alternative<SiluOp>(sig.ops[next_op]))
        {
            activation = Activation::Silu;
            next_op++;
        }
        else if(std::holds_alternative<SigmoidOp>(sig.ops[next_op]))
        {
            activation = Activation::Sigmoid;
            next_op++;
        }
    }

    // Remaining ops must be empty
    for(int i = next_op; i < kMaxOps; ++i)
        if(!std::holds_alternative<std::monostate>(sig.ops[i]))
            throw "unexpected operator after GEMM epilogue chain";

    // Tile validation
    if(algo.block_tile.m <= 0 || algo.block_tile.n <= 0 || algo.block_tile.k <= 0)
        throw "block_tile dimensions must be positive";
    if(algo.block_warps.m <= 0 || algo.block_warps.n <= 0 || algo.block_warps.k <= 0)
        throw "block_warps dimensions must be positive";
    if(algo.warp_tile.m <= 0 || algo.warp_tile.n <= 0 || algo.warp_tile.k <= 0)
        throw "warp_tile dimensions must be positive";

    if(algo.block_warps.k != 1)
        throw "block_warps.k must be 1 (CShuffleEpilogue constraint)";

    if(!is_valid_warp_gemm(a_td.dtype, algo.warp_tile.m, algo.warp_tile.n, algo.warp_tile.k))
        throw "warp_tile is not a valid MFMA configuration for this dtype";

    if(algo.block_tile.m % (algo.block_warps.m * algo.warp_tile.m) != 0)
        throw "block_tile.m must be divisible by (block_warps.m * warp_tile.m)";
    if(algo.block_tile.n % (algo.block_warps.n * algo.warp_tile.n) != 0)
        throw "block_tile.n must be divisible by (block_warps.n * warp_tile.n)";
    if(algo.block_tile.k % (algo.block_warps.k * algo.warp_tile.k) != 0)
        throw "block_tile.k must be divisible by (block_warps.k * warp_tile.k)";

    int thread_block_size =
        algo.block_warps.m * algo.block_warps.n * algo.block_warps.k * warp_size;

    return {a_td.dtype,
            b_td.dtype,
            c_td.dtype,
            acc,
            a_td.layout,
            b_td.layout,
            c_td.layout,
            algo.block_tile,
            algo.block_warps,
            algo.warp_tile,
            thread_block_size,
            combine,
            activation,
            num_d_tensors,
            d0_dtype,
            d0_layout,
            d1_dtype,
            d1_layout};
}

// ============================================================================
// Compile-time tests
// ============================================================================
// clang-format off

// --- is_valid_warp_gemm ---

// FP32: supports 16×16×{4,8,16} and 32×32×{4,8}
static_assert( is_valid_warp_gemm(DataType::FP32, 16, 16, 4));
static_assert( is_valid_warp_gemm(DataType::FP32, 16, 16, 16));
static_assert( is_valid_warp_gemm(DataType::FP32, 32, 32, 8));
static_assert(!is_valid_warp_gemm(DataType::FP32, 32, 32, 16)); // fp32 only k∈{4,8} at 32×32

// FP16: supports 16×16×{16,32} and 32×32×{8,16}
static_assert( is_valid_warp_gemm(DataType::FP16, 16, 16, 16));
static_assert( is_valid_warp_gemm(DataType::FP16, 32, 32, 16));
static_assert(!is_valid_warp_gemm(DataType::FP16, 32, 32, 4)); // fp16 only k∈{8,16} at 32×32

// BF16: same valid set as FP16
static_assert( is_valid_warp_gemm(DataType::BF16, 16, 16, 16));
static_assert( is_valid_warp_gemm(DataType::BF16, 32, 32, 16));

// --- make_kernel: plain GEMM ---

static_assert(make_kernel(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).a_dtype == DataType::FP16);
static_assert(make_kernel(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).thread_block_size == 256);
static_assert(make_kernel(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).combine == CombineOp::None);

// --- make_kernel: GEMM + Add (epilogue pattern match) ---

static_assert(make_kernel(
    Signature{.dtype = DataType::FP16,
              .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                      AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).combine == CombineOp::Add);
static_assert(make_kernel(
    Signature{.dtype = DataType::FP16,
              .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                      AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).num_d_tensors == 1);
static_assert(make_kernel(
    Signature{.dtype = DataType::FP16,
              .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                      AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).d0_dtype == DataType::FP16);

// --- make_kernel: GEMM + Add + Relu (full epilogue chain) ---

static_assert(make_kernel(
    Signature{.dtype = DataType::FP16,
              .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                      AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                      ReluOp{.in = "D", .out = "E"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).activation == Activation::Relu);
static_assert(make_kernel(
    Signature{.dtype = DataType::FP16,
              .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                      AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                      ReluOp{.in = "D", .out = "E"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).combine == CombineOp::Add);

// --- make_kernel: fp16 32×32×16 warp tile ---

static_assert(make_kernel(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {32, 32, 16}}).thread_block_size == 256);

// --- Layout defaults (A=Row, B=Col, C=Row) ---

static_assert(make_kernel(
    Signature{.dtype = DataType::FP32, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).a_layout == Layout::Row);
static_assert(make_kernel(
    Signature{.dtype = DataType::FP32, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).b_layout == Layout::Col);

// Error cases (uncommenting any would produce consteval compile errors):
// fp32 32×32×16 — not in MFMA dispatcher table:
// make_kernel(Signature{.dtype = DataType::FP32, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
//             GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {32, 32, 16}})
//
// Bad divisibility — 128 % (3 × 16) = 32 ≠ 0:
// make_kernel(Signature{.dtype = DataType::FP32, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
//             GemmAlgorithm{{128, 128, 32}, {3, 2, 1}, {16, 16, 16}})
//
// block_warps.k != 1 — violates CShuffleEpilogue constraint:
// make_kernel(Signature{.dtype = DataType::FP32, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
//             GemmAlgorithm{{128, 128, 32}, {2, 2, 2}, {16, 16, 16}})

// clang-format on

} // namespace rocm_ck
