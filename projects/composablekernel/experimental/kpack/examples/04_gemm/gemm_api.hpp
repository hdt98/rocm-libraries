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

#include <optional>

namespace rocm_ck {

// ============================================================================
// Layout
// ============================================================================

/// Memory layout for GEMM matrices.
/// Row = row-major (stride is number of columns).
/// Col = column-major (stride is number of rows).
enum class Layout
{
    Row,
    Col
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
// Type resolution
// ============================================================================

/// Resolved types from a GemmSignature. All concrete, no optionals.
struct ResolvedGemmTypes
{
    DataType a_dtype;
    DataType b_dtype;
    DataType c_dtype;
    DataType acc_dtype;
};

/// Resolve the optional dtype hierarchy into concrete types.
///
/// Resolution chains:
///   a_dtype   = a_dtype   ?? dtype ?? error
///   b_dtype   = b_dtype   ?? dtype ?? error
///   c_dtype   = c_dtype   ?? dtype ?? error
///   acc_dtype = acc_dtype  ?? FP32
consteval ResolvedGemmTypes resolve_types(GemmSignature sig)
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

    return {a, b, c, acc};
}

// ============================================================================
// Warp tile validation
// ============================================================================

/// AMD CDNA wavefront size.
constexpr int warp_size = 64;

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
    ResolvedGemmTypes types = resolve_types(cfg.signature);
    GemmAlgorithm algo      = cfg.algorithm;

    if(algo.block_tile.m <= 0 || algo.block_tile.n <= 0 || algo.block_tile.k <= 0)
        throw "block_tile dimensions must be positive";
    if(algo.block_warps.m <= 0 || algo.block_warps.n <= 0 || algo.block_warps.k <= 0)
        throw "block_warps dimensions must be positive";
    if(algo.warp_tile.m <= 0 || algo.warp_tile.n <= 0 || algo.warp_tile.k <= 0)
        throw "warp_tile dimensions must be positive";

    if(algo.block_warps.k != 1)
        throw "block_warps.k must be 1 (CShuffleEpilogue constraint)";

    if(!is_valid_warp_gemm(types.a_dtype, algo.warp_tile.m, algo.warp_tile.n, algo.warp_tile.k))
        throw "warp_tile is not a valid MFMA configuration for this dtype";

    if(algo.block_tile.m % (algo.block_warps.m * algo.warp_tile.m) != 0)
        throw "block_tile.m must be divisible by (block_warps.m * warp_tile.m)";
    if(algo.block_tile.n % (algo.block_warps.n * algo.warp_tile.n) != 0)
        throw "block_tile.n must be divisible by (block_warps.n * warp_tile.n)";
    if(algo.block_tile.k % (algo.block_warps.k * algo.warp_tile.k) != 0)
        throw "block_tile.k must be divisible by (block_warps.k * warp_tile.k)";

    int thread_block_size =
        algo.block_warps.m * algo.block_warps.n * algo.block_warps.k * warp_size;

    return {types.a_dtype,
            types.b_dtype,
            types.c_dtype,
            types.acc_dtype,
            cfg.signature.a_layout,
            cfg.signature.b_layout,
            cfg.signature.c_layout,
            algo.block_tile,
            algo.block_warps,
            algo.warp_tile,
            thread_block_size};
}

// ============================================================================
// resolve_types compile-time tests
// ============================================================================
// clang-format off

// --- Homogeneous: dtype sets everything ---
static_assert(resolve_types({.dtype = DataType::FP32}).a_dtype == DataType::FP32);
static_assert(resolve_types({.dtype = DataType::FP32}).b_dtype == DataType::FP32);
static_assert(resolve_types({.dtype = DataType::FP32}).c_dtype == DataType::FP32);

static_assert(resolve_types({.dtype = DataType::FP16}).a_dtype == DataType::FP16);
static_assert(resolve_types({.dtype = DataType::FP16}).c_dtype == DataType::FP16);

static_assert(resolve_types({.dtype = DataType::BF16}).a_dtype == DataType::BF16);
static_assert(resolve_types({.dtype = DataType::BF16}).c_dtype == DataType::BF16);

// --- acc_dtype defaults to FP32, regardless of dtype ---
static_assert(resolve_types({.dtype = DataType::FP16}).acc_dtype == DataType::FP32);
static_assert(resolve_types({.dtype = DataType::BF16}).acc_dtype == DataType::FP32);
static_assert(resolve_types({.dtype = DataType::FP32}).acc_dtype == DataType::FP32);

// --- acc_dtype override ---
static_assert(resolve_types({.dtype = DataType::FP16, .acc_dtype = DataType::FP16}).acc_dtype == DataType::FP16);

// --- Per-operand overrides ---
static_assert(resolve_types({.dtype = DataType::FP32, .a_dtype = DataType::FP16}).a_dtype == DataType::FP16);
static_assert(resolve_types({.dtype = DataType::FP32, .a_dtype = DataType::FP16}).b_dtype == DataType::FP32);

// --- Mixed-type widening: fp16 inputs, fp32 output ---
static_assert(resolve_types({.dtype = DataType::FP16, .c_dtype = DataType::FP32}).a_dtype == DataType::FP16);
static_assert(resolve_types({.dtype = DataType::FP16, .c_dtype = DataType::FP32}).c_dtype == DataType::FP32);
static_assert(resolve_types({.dtype = DataType::FP16, .c_dtype = DataType::FP32}).acc_dtype == DataType::FP32);

// Error cases (uncommenting any would produce consteval compile errors):
// resolve_types({})                                    — nothing resolvable
// resolve_types({.a_dtype = DataType::FP16})           — b, c unknown
// resolve_types({.c_dtype = DataType::FP32})           — a, b unknown

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
// clang-format on

} // namespace rocm_ck
