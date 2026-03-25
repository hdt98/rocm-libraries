// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// GemmKernel — structural NTTP descriptor for GEMM template instantiation.
//
// Device-safe: no std::string_view, no throw, no host-only dependencies.
// Included by both device code (gemm_dev.hpp) and host code (gemm_api.hpp).
// Host-only logic (make_kernel, validation, named accessors) is in gemm_api.hpp.

#pragma once

#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/layout.hpp>
#include <rocm_ck/physical_tensor.hpp>
#include <rocm_ck/types.hpp>

namespace rocm_ck {

// ============================================================================
// Epilogue operations (composable chain)
// ============================================================================

/// Epilogue operations applied after the GEMM matmul result.
///
/// Binary ops (Add, Mul) fold over D tensors via parameter pack:
///   Add — result += D0 [+ D1]     (bias addition)
///   Mul — result *= D0 [* D1]     (scaling)
///
/// Unary ops transform the accumulator in place:
///   Relu     — max(0, x)
///   FastGelu — approximate GELU: x * sigmoid(1.702 * x)
///   Gelu     — exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
///   Silu     — x * sigmoid(x)  (aka Swish with beta=1)
///   Sigmoid  — 1 / (1 + exp(-x))
///
/// Operations compose as an ordered sequence in GemmKernel::epilogue_ops[].
/// The Signature's operator chain (AddOp -> ReluOp) maps directly to this
/// array, minus the string names (which aren't structural types for NTTP).
enum class EpilogueOp
{
    Add,
    Mul,
    Relu,
    FastGelu,
    Gelu,
    Silu,
    Sigmoid
};

/// Maximum epilogue ops in a chain. 4 covers combine + activation + future ops.
inline constexpr int kMaxEpilogueOps = 4;

// ============================================================================
// Tile geometry types
// ============================================================================

/// M x N x K dimension triple for tile geometry specification.
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
// GemmKernel — structural NTTP for template instantiation
// ============================================================================

/// Validated kernel descriptor with all types, layouts, and tile geometry resolved.
/// All members are structural types (enums, ints, aggregates) so this works as NTTP.
///
/// Physical tensor table layout (ordered by args_slot):
///   [0] = A (GEMM lhs input)
///   [1] = B (GEMM rhs input)
///   [2] = output (final output — name varies: "C", "D", or "E" depending on epilogue)
///   [3] = D0 (optional — first auxiliary tensor, e.g., "bias")
///   [4] = D1 (optional — second auxiliary tensor)
struct GemmKernel
{
    // Physical tensor table — the kernel's view of Args::tensors[]
    int num_physical_tensors;
    std::array<PhysicalTensor, kMaxPhysicalTensors> physical_tensors;

    // Accumulator type (register-only, not a physical tensor)
    DataType acc_dtype;

    // Tile geometry
    Dim3 block_tile;
    Dim3 block_warps;
    Dim3 warp_tile;
    int thread_block_size;

    // Epilogue: composable op chain applied after matmul
    int num_epilogue_ops;
    std::array<EpilogueOp, kMaxEpilogueOps> epilogue_ops;

    /// Check if the epilogue chain contains a specific op.
    constexpr bool has_epilogue_op(EpilogueOp op) const
    {
        for(int i = 0; i < num_epilogue_ops; ++i)
            if(epilogue_ops[i] == op)
                return true;
        return false;
    }

    /// The GEMM output tensor (always at position 2 in the physical tensor table).
    /// Name varies by epilogue chain: "C" (plain), "D" (with combine), "E" (with activation).
    constexpr PhysicalTensor output() const { return physical_tensors[2]; }
};

} // namespace rocm_ck
