// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// VectorAddKernel — structural NTTP descriptor for elementwise template instantiation.
//
// Device-safe: no std::string_view, no throw, no host-only dependencies.
// Included by both device code (rocm_vector_add_dev.hpp) and host code
// (rocm_vector_add_api.hpp). Host-only logic (make_kernel, validation) is
// in rocm_vector_add_api.hpp.

#pragma once

#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/types.hpp>

namespace rocm_ck {

/// Algorithm: describes HOW the kernel executes (tile geometry, pipeline).
/// Independent of data types — paired with Signature in make_kernel().
struct ElementwiseAlgorithm
{
    int block_tile;  // Elements processed per thread block (BlockTile)
    int block_warps; // Number of warps per thread block (BlockWarps)
    int warp_tile;   // Warp tile size for vector width calculation (WarpTile)
    bool pad;        // Enable padding for unaligned problem sizes
};

/// Validated kernel descriptor used as NTTP and host launch info.
/// All members are structural types (no std::optional, no pointers, etc.).
struct VectorAddKernel
{
    int block_tile;        // Elements per thread block (for grid calculation)
    int thread_block_size; // Threads per block (= warp_size * block_warps)
    DataType in_dtype;     // Input storage type (a, b)
    DataType out_dtype;    // Output storage type (c)
    int block_warps;       // Warps per block
    int warp_tile;         // Warp tile size
    bool pad;              // Padding enabled
};

/// Check if problem size N is aligned to a variant's block_tile (no padding needed).
constexpr bool isAligned(VectorAddKernel k, int n) { return n > 0 && n % k.block_tile == 0; }

} // namespace rocm_ck
