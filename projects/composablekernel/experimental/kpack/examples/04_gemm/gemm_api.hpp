// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// GEMM schema for kpack GEMM example.
//
// Defines the "WHAT" and "HOW" of a GEMM through two independent structs:
//
//   GemmSignature — data types and memory layouts
//   GemmAlgorithm — tile geometry (block tile, warp distribution, warp tile)
//
// Combined in GemmConfig and validated by make_kernel() into a GemmKernel
// NTTP with consteval checks for MFMA warp tile validity and tile divisibility.
//
// This header has NO CK Tile dependency. It is included by both host code
// (main.cpp) and device code (.hip files via gemm_dev.hpp).

#pragma once

#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/layout.hpp>
#include <rocm_ck/resolve.hpp>
#include <rocm_ck/tensor_desc.hpp>
#include <rocm_ck/types.hpp>

#include <array>
#include <optional>

namespace rocm_ck {

// ============================================================================
// Epilogue: CombineOp × Activation
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
// GemmSignature — the "WHAT" of a GEMM
// ============================================================================

/// Describes data types and memory layouts for a GEMM operation.
///
/// Optional dtype hierarchy (two levels, unlike elementwise's three):
///
///     dtype                    (kernel-level default)
///     ├── a_dtype              (A override)
///     ├── b_dtype              (B override)
///     ├── c_dtype              (C output override)
///     └── acc_dtype            (accumulator, defaults to FP32)
///
/// Specify only what differs — use {.dtype = FP32} for homogeneous kernels,
/// {.dtype = FP16, .c_dtype = FP32} for widening, or override individual
/// fields for asymmetric inputs.
struct GemmSignature
{
    std::optional<DataType> dtype;     // kernel-level default
    std::optional<DataType> a_dtype;   // A override
    std::optional<DataType> b_dtype;   // B override
    std::optional<DataType> c_dtype;   // C output override
    std::optional<DataType> acc_dtype; // accumulator (default: FP32)
    Layout a_layout = Layout::Row;
    Layout b_layout = Layout::Col;
    Layout c_layout = Layout::Row;

    // Fused epilogue: E = activation(combine(A*B, D0, D1, ...))
    CombineOp combine     = CombineOp::None;
    Activation activation = Activation::None;
    std::optional<DataType> d0_dtype; // first D tensor type
    std::optional<Layout> d0_layout;  // defaults to Row if d0_dtype set
    std::optional<DataType> d1_dtype; // second D tensor type
    std::optional<Layout> d1_layout;  // defaults to Row if d1_dtype set

    // Operation: matrix multiply with named tensor slots
    struct Gemm
    {
        std::string_view lhs = "A";
        std::string_view rhs = "B";
        std::string_view out = "C";
    };
    Gemm op{};
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
/// Independent of data types — paired with GemmSignature in GemmConfig.
struct GemmAlgorithm
{
    Dim3 block_tile;  // Elements per thread block {M, N, K}
    Dim3 block_warps; // Warp distribution {M, N, K}
    Dim3 warp_tile;   // Elements per warp per MFMA step {M, N, K}
};

/// Compile-time configuration for a GEMM kernel variant.
/// Combines a Signature (what types/layouts) with an Algorithm (what tile geometry).
struct GemmConfig
{
    GemmSignature signature;
    GemmAlgorithm algorithm;
};

// ============================================================================
// Tensor resolution
// ============================================================================

/// Resolved tensor descriptors and accumulator type from a GemmSignature.
/// acc_dtype is not a tensor — it has no rank, no pointer, no layout.
struct ResolvedGemmTensors
{
    std::array<TensorDesc, 3> tensors; // A, B, C
    DataType acc_dtype;
};

/// Resolve the optional dtype hierarchy into concrete TensorDesc entries.
///
/// Resolution chains:
///   a_dtype   = a_dtype   ?? dtype ?? error
///   b_dtype   = b_dtype   ?? dtype ?? error
///   c_dtype   = c_dtype   ?? dtype ?? error
///   acc_dtype = acc_dtype  ?? FP32
consteval ResolvedGemmTensors resolve_tensors(GemmSignature sig)
{
    DataType a = sig.a_dtype ? *sig.a_dtype
                 : sig.dtype ? *sig.dtype
                             : throw "a_dtype unresolvable: set a_dtype or dtype";

    DataType b = sig.b_dtype ? *sig.b_dtype
                 : sig.dtype ? *sig.dtype
                             : throw "b_dtype unresolvable: set b_dtype or dtype";

    DataType c = sig.c_dtype ? *sig.c_dtype
                 : sig.dtype ? *sig.dtype
                             : throw "c_dtype unresolvable: set c_dtype or dtype";

    DataType acc = sig.acc_dtype ? *sig.acc_dtype : DataType::FP32;

    return {{TensorDesc{"A", a, 2, sig.a_layout},
             TensorDesc{"B", b, 2, sig.b_layout},
             TensorDesc{"C", c, 2, sig.c_layout}},
            acc};
}

// ============================================================================
// Epilogue resolution
// ============================================================================

/// Resolved epilogue from a GemmSignature. Concrete types for D tensors.
struct ResolvedEpilogue
{
    CombineOp combine;
    Activation activation;
    int num_d_tensors; // 0, 1, or 2
    DataType d0_dtype; // valid when num_d_tensors >= 1
    Layout d0_layout;
    DataType d1_dtype; // valid when num_d_tensors >= 2
    Layout d1_layout;
};

/// Resolve the epilogue fields into concrete D tensor types and count.
///
/// Rules:
///   - CombineOp::None forbids D tensors (d0_dtype/d1_dtype must not be set)
///   - CombineOp::Add/Multiply implies D0 exists; d0_dtype cascades from dtype
///   - D1 exists only when d1_dtype is explicitly set
///   - Activation is always valid (independent of D tensors)
///   - d0_layout defaults to Row, d1_layout defaults to Row
consteval ResolvedEpilogue resolve_epilogue(GemmSignature sig)
{
    bool has_d0 = sig.d0_dtype.has_value();
    bool has_d1 = sig.d1_dtype.has_value();

    if(sig.combine == CombineOp::None)
    {
        if(has_d0 || has_d1)
            throw "CombineOp::None must not have D tensors";
        return {CombineOp::None,
                sig.activation,
                0,
                DataType::FP32,
                Layout::Row,
                DataType::FP32,
                Layout::Row};
    }

    // Add or Multiply: D0 exists, dtype cascades from d0_dtype ?? dtype
    DataType d0_dt = sig.d0_dtype ? *sig.d0_dtype
                     : sig.dtype  ? *sig.dtype
                                  : throw "d0_dtype unresolvable: set d0_dtype or dtype";
    Layout d0_ly   = sig.d0_layout ? *sig.d0_layout : Layout::Row;

    int num_d      = has_d1 ? 2 : 1;
    DataType d1_dt = has_d1 ? *sig.d1_dtype : DataType::FP32;
    Layout d1_ly   = sig.d1_layout ? *sig.d1_layout : Layout::Row;

    return {sig.combine, sig.activation, num_d, d0_dt, d0_ly, d1_dt, d1_ly};
}

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

/// Resolve and validate a GemmConfig into a GemmKernel (consteval).
///
/// Validates:
///   - All tile dimensions are positive
///   - block_warps.k == 1 (CShuffleEpilogue requires warps_m × warps_n block size)
///   - Warp tile matches MFMA dispatcher table for the input dtype
///   - Block tile is divisible by (block_warps × warp_tile) in each dimension
///
/// Derives thread_block_size = block_warps.m × block_warps.n × block_warps.k × 64.
consteval GemmKernel make_kernel(GemmConfig cfg)
{
    ResolvedGemmTensors resolved = resolve_tensors(cfg.signature);
    ResolvedEpilogue epilogue    = resolve_epilogue(cfg.signature);
    GemmAlgorithm algo           = cfg.algorithm;

    if(algo.block_tile.m <= 0 || algo.block_tile.n <= 0 || algo.block_tile.k <= 0)
        throw "block_tile dimensions must be positive";
    if(algo.block_warps.m <= 0 || algo.block_warps.n <= 0 || algo.block_warps.k <= 0)
        throw "block_warps dimensions must be positive";
    if(algo.warp_tile.m <= 0 || algo.warp_tile.n <= 0 || algo.warp_tile.k <= 0)
        throw "warp_tile dimensions must be positive";

    if(algo.block_warps.k != 1)
        throw "block_warps.k must be 1 (CShuffleEpilogue constraint)";

    if(!is_valid_warp_gemm(
           resolved.tensors[0].dtype, algo.warp_tile.m, algo.warp_tile.n, algo.warp_tile.k))
        throw "warp_tile is not a valid MFMA configuration for this dtype";

    if(algo.block_tile.m % (algo.block_warps.m * algo.warp_tile.m) != 0)
        throw "block_tile.m must be divisible by (block_warps.m * warp_tile.m)";
    if(algo.block_tile.n % (algo.block_warps.n * algo.warp_tile.n) != 0)
        throw "block_tile.n must be divisible by (block_warps.n * warp_tile.n)";
    if(algo.block_tile.k % (algo.block_warps.k * algo.warp_tile.k) != 0)
        throw "block_tile.k must be divisible by (block_warps.k * warp_tile.k)";

    int thread_block_size =
        algo.block_warps.m * algo.block_warps.n * algo.block_warps.k * warp_size;

    return {resolved.tensors[0].dtype,
            resolved.tensors[1].dtype,
            resolved.tensors[2].dtype,
            resolved.acc_dtype,
            resolved.tensors[0].layout,
            resolved.tensors[1].layout,
            resolved.tensors[2].layout,
            algo.block_tile,
            algo.block_warps,
            algo.warp_tile,
            thread_block_size,
            epilogue.combine,
            epilogue.activation,
            epilogue.num_d_tensors,
            epilogue.d0_dtype,
            epilogue.d0_layout,
            epilogue.d1_dtype,
            epilogue.d1_layout};
}

// ============================================================================
// make_kernel overload for operator-centric Signature
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

    // Same tile validation as old make_kernel
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
// resolve_tensors compile-time tests
// ============================================================================
// clang-format off

// --- Homogeneous: dtype sets everything ---
static_assert(resolve_tensors({.dtype = DataType::FP32}).tensors[0].dtype == DataType::FP32);
static_assert(resolve_tensors({.dtype = DataType::FP32}).tensors[1].dtype == DataType::FP32);
static_assert(resolve_tensors({.dtype = DataType::FP32}).tensors[2].dtype == DataType::FP32);

static_assert(resolve_tensors({.dtype = DataType::FP16}).tensors[0].dtype == DataType::FP16);
static_assert(resolve_tensors({.dtype = DataType::FP16}).tensors[2].dtype == DataType::FP16);

static_assert(resolve_tensors({.dtype = DataType::BF16}).tensors[0].dtype == DataType::BF16);
static_assert(resolve_tensors({.dtype = DataType::BF16}).tensors[2].dtype == DataType::BF16);

// --- acc_dtype defaults to FP32, regardless of dtype ---
static_assert(resolve_tensors({.dtype = DataType::FP16}).acc_dtype == DataType::FP32);
static_assert(resolve_tensors({.dtype = DataType::BF16}).acc_dtype == DataType::FP32);
static_assert(resolve_tensors({.dtype = DataType::FP32}).acc_dtype == DataType::FP32);

// --- acc_dtype override ---
static_assert(resolve_tensors({.dtype = DataType::FP16, .acc_dtype = DataType::FP16}).acc_dtype == DataType::FP16);

// --- Per-operand overrides ---
static_assert(resolve_tensors({.dtype = DataType::FP32, .a_dtype = DataType::FP16}).tensors[0].dtype == DataType::FP16);
static_assert(resolve_tensors({.dtype = DataType::FP32, .a_dtype = DataType::FP16}).tensors[1].dtype == DataType::FP32);

// --- Mixed-type widening: fp16 inputs, fp32 output ---
static_assert(resolve_tensors({.dtype = DataType::FP16, .c_dtype = DataType::FP32}).tensors[0].dtype == DataType::FP16);
static_assert(resolve_tensors({.dtype = DataType::FP16, .c_dtype = DataType::FP32}).tensors[2].dtype == DataType::FP32);
static_assert(resolve_tensors({.dtype = DataType::FP16, .c_dtype = DataType::FP32}).acc_dtype == DataType::FP32);

// --- TensorDesc metadata: name, rank, layout ---
static_assert(resolve_tensors({.dtype = DataType::FP32}).tensors[0].name == "A");
static_assert(resolve_tensors({.dtype = DataType::FP32}).tensors[1].name == "B");
static_assert(resolve_tensors({.dtype = DataType::FP32}).tensors[2].name == "C");
static_assert(resolve_tensors({.dtype = DataType::FP32}).tensors[0].rank == 2);
static_assert(resolve_tensors({.dtype = DataType::FP32}).tensors[2].rank == 2);

// --- Operation slots match tensor names ---
static_assert(GemmSignature{}.op.lhs == "A");
static_assert(GemmSignature{}.op.rhs == "B");
static_assert(GemmSignature{}.op.out == "C");
// --- Layout propagation through resolve_tensors ---
static_assert(resolve_tensors({.dtype = DataType::FP32}).tensors[0].layout == Layout::Row);
static_assert(resolve_tensors({.dtype = DataType::FP32}).tensors[1].layout == Layout::Col);
static_assert(resolve_tensors({.dtype = DataType::FP32}).tensors[2].layout == Layout::Row);

// --- Layout override flows through resolve_tensors ---
static_assert(resolve_tensors({.dtype = DataType::FP32, .a_layout = Layout::Col}).tensors[0].layout == Layout::Col);

// Error cases (uncommenting any would produce consteval compile errors):
// resolve_tensors({})                                    — nothing resolvable
// resolve_tensors({.a_dtype = DataType::FP16})           — b, c unknown
// resolve_tensors({.c_dtype = DataType::FP32})           — a, b unknown

// ============================================================================
// is_valid_warp_gemm compile-time tests
// ============================================================================

// --- FP32: supports 16×16×{4,8,16} and 32×32×{4,8} ---
static_assert( is_valid_warp_gemm(DataType::FP32, 16, 16, 4));
static_assert( is_valid_warp_gemm(DataType::FP32, 16, 16, 16));
static_assert( is_valid_warp_gemm(DataType::FP32, 32, 32, 8));
static_assert(!is_valid_warp_gemm(DataType::FP32, 32, 32, 16)); // fp32 only k∈{4,8} at 32×32

// --- FP16: supports 16×16×{16,32} and 32×32×{8,16} ---
static_assert( is_valid_warp_gemm(DataType::FP16, 16, 16, 16));
static_assert( is_valid_warp_gemm(DataType::FP16, 32, 32, 16));
static_assert(!is_valid_warp_gemm(DataType::FP16, 32, 32, 4)); // fp16 only k∈{8,16} at 32×32

// --- BF16: same valid set as FP16 ---
static_assert( is_valid_warp_gemm(DataType::BF16, 16, 16, 16));
static_assert( is_valid_warp_gemm(DataType::BF16, 32, 32, 16));

// ============================================================================
// make_kernel compile-time tests
// ============================================================================

// --- Valid: fp32 16×16×16 warp tile ---
static_assert(make_kernel(
    {.signature = {.dtype = DataType::FP32},
     .algorithm = {{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}}).thread_block_size == 256);

// --- Valid: fp16 32×32×16 warp tile ---
static_assert(make_kernel(
    {.signature = {.dtype = DataType::FP16},
     .algorithm = {{128, 128, 32}, {2, 2, 1}, {32, 32, 16}}}).thread_block_size == 256);

// --- Layout defaults ---
static_assert(make_kernel(
    {.signature = {.dtype = DataType::FP32},
     .algorithm = {{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}}).a_layout == Layout::Row);
static_assert(make_kernel(
    {.signature = {.dtype = DataType::FP32},
     .algorithm = {{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}}).b_layout == Layout::Col);

// --- Layout override ---
static_assert(make_kernel(
    {.signature = {.dtype = DataType::FP32, .a_layout = Layout::Col},
     .algorithm = {{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}}).a_layout == Layout::Col);

// Error cases (uncommenting any would produce consteval compile errors):
// fp32 32×32×16 — not in MFMA dispatcher table:
// make_kernel({.signature = {.dtype = DataType::FP32},
//              .algorithm = {{128, 128, 32}, {2, 2, 1}, {32, 32, 16}}})
//
// Bad divisibility — 128 % (3 × 16) = 32 ≠ 0:
// make_kernel({.signature = {.dtype = DataType::FP32},
//              .algorithm = {{128, 128, 32}, {3, 2, 1}, {16, 16, 16}}})
//
// block_warps.k != 1 — violates CShuffleEpilogue constraint:
// make_kernel({.signature = {.dtype = DataType::FP32},
//              .algorithm = {{128, 128, 32}, {2, 2, 2}, {16, 16, 16}}})

// ============================================================================
// resolve_epilogue compile-time tests
// ============================================================================

// --- Default: no combine, no activation ---
static_assert(resolve_epilogue({.dtype = DataType::FP16}).num_d_tensors == 0);
static_assert(resolve_epilogue({.dtype = DataType::FP16}).combine == CombineOp::None);
static_assert(resolve_epilogue({.dtype = DataType::FP16}).activation == Activation::None);

// --- Add with d0 (dtype cascades to d0) ---
static_assert(resolve_epilogue(
    {.dtype = DataType::FP16, .combine = CombineOp::Add})
    .num_d_tensors == 1);
static_assert(resolve_epilogue(
    {.dtype = DataType::FP16, .combine = CombineOp::Add})
    .d0_dtype == DataType::FP16);
static_assert(resolve_epilogue(
    {.dtype = DataType::FP16, .combine = CombineOp::Add})
    .d0_layout == Layout::Row);

// --- d0_dtype override (overrides cascade) ---
static_assert(resolve_epilogue(
    {.dtype = DataType::FP16, .combine = CombineOp::Add, .d0_dtype = DataType::FP32})
    .d0_dtype == DataType::FP32);

// --- Add with d0 + d1 ---
static_assert(resolve_epilogue(
    {.dtype = DataType::FP16, .combine = CombineOp::Add, .d1_dtype = DataType::FP16})
    .num_d_tensors == 2);

// --- Layout override for d0 ---
static_assert(resolve_epilogue(
    {.dtype = DataType::FP16, .combine = CombineOp::Add, .d0_layout = Layout::Col})
    .d0_layout == Layout::Col);

// --- Activation only (no D tensors): valid ---
static_assert(resolve_epilogue(
    {.dtype = DataType::FP16, .activation = Activation::Relu})
    .activation == Activation::Relu);
static_assert(resolve_epilogue(
    {.dtype = DataType::FP16, .activation = Activation::Relu})
    .combine == CombineOp::None);
static_assert(resolve_epilogue(
    {.dtype = DataType::FP16, .activation = Activation::Relu})
    .num_d_tensors == 0);

// --- Composed: Add + Relu (d0 cascaded from dtype) ---
static_assert(resolve_epilogue(
    {.dtype = DataType::FP16, .combine = CombineOp::Add, .activation = Activation::Relu})
    .combine == CombineOp::Add);
static_assert(resolve_epilogue(
    {.dtype = DataType::FP16, .combine = CombineOp::Add, .activation = Activation::Relu})
    .activation == Activation::Relu);
static_assert(resolve_epilogue(
    {.dtype = DataType::FP16, .combine = CombineOp::Add, .activation = Activation::Relu})
    .d0_dtype == DataType::FP16);

// --- make_kernel with defaults (no combine, no activation) ---
static_assert(make_kernel(
    {.signature = {.dtype = DataType::FP16},
     .algorithm = {{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}}).combine == CombineOp::None);
static_assert(make_kernel(
    {.signature = {.dtype = DataType::FP16},
     .algorithm = {{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}}).activation == Activation::None);
static_assert(make_kernel(
    {.signature = {.dtype = DataType::FP16},
     .algorithm = {{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}}).num_d_tensors == 0);

// --- make_kernel with Add (d0 cascaded from dtype) ---
static_assert(make_kernel(
    {.signature = {.dtype = DataType::FP16, .combine = CombineOp::Add},
     .algorithm = {{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}}).num_d_tensors == 1);
static_assert(make_kernel(
    {.signature = {.dtype = DataType::FP16, .combine = CombineOp::Add},
     .algorithm = {{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}}).d0_dtype == DataType::FP16);

// --- make_kernel with Add + Relu ---
static_assert(make_kernel(
    {.signature = {.dtype = DataType::FP16, .combine = CombineOp::Add,
                   .activation = Activation::Relu},
     .algorithm = {{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}}).activation == Activation::Relu);

// Error cases (uncommenting any would produce consteval compile errors):
// None with D tensor:
// resolve_epilogue({.dtype = DataType::FP16, .d0_dtype = DataType::FP16})
//
// Add without dtype or d0_dtype:
// resolve_epilogue({.combine = CombineOp::Add})

// ============================================================================
// Signature-based make_kernel equivalence tests
// ============================================================================

// --- Plain GEMM: new Signature produces same kernel as old GemmConfig ---
static_assert(make_kernel(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).a_dtype == DataType::FP16);
static_assert(make_kernel(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).thread_block_size == 256);
static_assert(make_kernel(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).combine == CombineOp::None);

// --- GEMM + Add: epilogue pattern match ---
static_assert(make_kernel(
    Signature{.dtype = DataType::FP16,
              .ops = {GemmOp{.out = "C"},
                      AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).combine == CombineOp::Add);
static_assert(make_kernel(
    Signature{.dtype = DataType::FP16,
              .ops = {GemmOp{.out = "C"},
                      AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).num_d_tensors == 1);
static_assert(make_kernel(
    Signature{.dtype = DataType::FP16,
              .ops = {GemmOp{.out = "C"},
                      AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).d0_dtype == DataType::FP16);

// --- GEMM + Add + Relu: full epilogue chain ---
static_assert(make_kernel(
    Signature{.dtype = DataType::FP16,
              .ops = {GemmOp{.out = "C"},
                      AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                      ReluOp{.in = "D", .out = "E"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).activation == Activation::Relu);
static_assert(make_kernel(
    Signature{.dtype = DataType::FP16,
              .ops = {GemmOp{.out = "C"},
                      AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                      ReluOp{.in = "D", .out = "E"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).combine == CombineOp::Add);

// --- Equivalence: new path matches old path for plain GEMM ---
static_assert(make_kernel(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).thread_block_size ==
    make_kernel(GemmConfig{
        .signature = {.dtype = DataType::FP16},
        .algorithm = {{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}}).thread_block_size);

// --- Equivalence: new path matches old path for GEMM + Add ---
static_assert(make_kernel(
    Signature{.dtype = DataType::FP16,
              .ops = {GemmOp{.out = "C"},
                      AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}).d0_dtype ==
    make_kernel(GemmConfig{
        .signature = {.dtype = DataType::FP16, .combine = CombineOp::Add},
        .algorithm = {{128, 128, 32}, {2, 2, 1}, {16, 16, 16}}}).d0_dtype);

// clang-format on

} // namespace rocm_ck
