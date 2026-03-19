// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// GemmSignature schema for kpack GEMM example.
//
// Defines the "WHAT" of a GEMM: data types and memory layouts. Extends
// example 03's optional dtype hierarchy to GEMM's asymmetric, multi-type
// domain:
//
//   - No in_dtype level. GEMM's A and B are asymmetric (M×K vs K×N), so
//     a shared "input default" suggests false symmetry. Two levels
//     (dtype → per-operand) is cleaner.
//   - acc_dtype defaults to FP32, not to dtype. Every practical GEMM
//     accumulates in fp32 (even fp16/bf16 MFMA).
//   - Layouts are non-optional with sensible BLAS defaults.
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
// GemmKernel — structural NTTP for template instantiation
// ============================================================================

/// Validated kernel descriptor with all types and layouts resolved.
/// All members are enum classes (structural types) so this works as NTTP.
/// Future: tile geometry fields (block tile, warp tile) will live here too.
struct GemmKernel
{
    DataType a_dtype;
    DataType b_dtype;
    DataType c_dtype;
    DataType acc_dtype;
    Layout a_layout;
    Layout b_layout;
    Layout c_layout;
};

/// Resolve a GemmSignature into a GemmKernel (consteval).
/// Takes GemmSignature directly — a future GemmConfig will combine
/// {.signature, .algorithm} to add tile geometry control.
consteval GemmKernel make_kernel(GemmSignature sig)
{
    auto types = resolve_types(sig);
    return {types.a_dtype,
            types.b_dtype,
            types.c_dtype,
            types.acc_dtype,
            sig.a_layout,
            sig.b_layout,
            sig.c_layout};
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

// --- make_kernel: layout defaults ---
static_assert(make_kernel({.dtype = DataType::FP32}).a_layout == Layout::Row);
static_assert(make_kernel({.dtype = DataType::FP32}).b_layout == Layout::Col);
static_assert(make_kernel({.dtype = DataType::FP32}).c_layout == Layout::Row);

// --- make_kernel: layout override ---
static_assert(make_kernel({.dtype = DataType::FP32, .a_layout = Layout::Col}).a_layout == Layout::Col);

// Error cases (uncommenting any would produce consteval compile errors):
// resolve_types({})                                    — nothing resolvable
// resolve_types({.a_dtype = DataType::FP16})           — b, c unknown
// resolve_types({.c_dtype = DataType::FP32})           — a, b unknown
// clang-format on

} // namespace rocm_ck
