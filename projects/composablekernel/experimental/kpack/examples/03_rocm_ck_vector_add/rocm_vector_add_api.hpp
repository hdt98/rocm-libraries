// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Shared argument struct and compile-time configuration for rocm_ck vector add.
//
// This header has NO CK Tile dependency. It is included by both host code
// (main.cpp) and device code (.hip files) to share the kernel ABI and
// compile-time-validated configuration.

#pragma once

#include <rocm_ck/datatype_utils.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rocm_ck {

/// Fixed-width integer type for kernel index/size arguments.
/// Matches ck_tile::index_t but avoids pulling in CK Tile headers.
using index_t = std::int32_t;

/// Kernel arguments passed by value through hipModuleLaunchKernel.
/// Layout must match exactly between host and device.
///
/// Always computes c = alpha * a + beta * b. For plain addition, pass
/// alpha = 1.0f and beta = 1.0f. This matches the BLAS convention where
/// scalar parameters are always present.
struct VectorAddArgs
{
    index_t n;
    float alpha;
    float beta;
    // 4 bytes implicit padding (pointers need 8-byte alignment)
    const void* a;
    const void* b;
    void* c;
};

static_assert(std::is_trivially_copyable_v<VectorAddArgs>,
              "VectorAddArgs must be trivially copyable for kernarg passing");
static_assert(sizeof(VectorAddArgs) == 40, "unexpected VectorAddArgs size");
static_assert(alignof(VectorAddArgs) == 8, "unexpected VectorAddArgs alignment");
static_assert(offsetof(VectorAddArgs, n) == 0, "unexpected offset for n");
static_assert(offsetof(VectorAddArgs, alpha) == 4, "unexpected offset for alpha");
static_assert(offsetof(VectorAddArgs, beta) == 8, "unexpected offset for beta");
static_assert(offsetof(VectorAddArgs, a) == 16, "unexpected offset for a");
static_assert(offsetof(VectorAddArgs, b) == 24, "unexpected offset for b");
static_assert(offsetof(VectorAddArgs, c) == 32, "unexpected offset for c");

/// Signature: describes WHAT the kernel computes (types).
/// In CK Tile's builder pattern, this is the "what" — independent of
/// how the kernel is tiled, scheduled, or padded.
///
/// Each operation defines its own signature struct. Vector add uses
/// {in_type, out_type}. GEMM will use {a_type, b_type, c_type}, etc.
struct ElementwiseSignature
{
    DataType in_type;
    DataType out_type;
};

/// Algorithm: describes HOW the kernel executes (tile geometry, pipeline).
/// In CK Tile's builder pattern, this is the "how" — independent of
/// the data types being computed.
struct ElementwiseAlgorithm
{
    int block_tile;  // Elements processed per thread block (BlockTile)
    int block_warps; // Number of warps per thread block (BlockWarps)
    int warp_tile;   // Warp tile size for vector width calculation (WarpTile)
    bool pad;        // Enable padding for unaligned problem sizes
};

/// Compile-time configuration for an elementwise kernel variant.
/// This is the user-facing API — contains a Signature (what) and Algorithm (how).
struct ElementwiseConfig
{
    ElementwiseSignature signature;
    ElementwiseAlgorithm algorithm;
};

/// Validated kernel descriptor used as NTTP and host launch info.
/// All members are structural types (no std::optional, no pointers, etc.).
struct VectorAddKernel
{
    int block_tile;        // Elements per thread block (for grid calculation)
    int thread_block_size; // Threads per block (= warp_size * block_warps)
    DataType in_type;      // Input storage type (a, b)
    DataType out_type;     // Output storage type (c)
    int block_warps;       // Warps per block
    int warp_tile;         // Warp tile size
    bool pad;              // Padding enabled
};

/// AMD GCN/CDNA wavefront size.
constexpr int warp_size = 64;

/// Validate a signature + algorithm pair and produce a structural kernel descriptor.
///
/// Validates CK Tile ElementWiseShape compatibility:
///   kVectorM = min(128 / max_type_bits, warp_tile / warp_size) — must be >= 1
///   kRepeatM = block_tile / (block_warps * kVectorM * warp_size) — must be >= 1, integer
///   block_warps must be power of 2 (required by CK Tile reduce_on_sequence)
///   thread_block_size = warp_size * block_warps (NOT block_tile)
///
/// For mixed types, kVectorM is constrained by the wider type (fewer elements
/// per 128-bit register).
///
/// Invalid configs produce a compile-time error (consteval).
consteval VectorAddKernel make_kernel(ElementwiseSignature sig, ElementwiseAlgorithm algo)
{
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

    int in_bits    = data_type_bits(sig.in_type);
    int out_bits   = data_type_bits(sig.out_type);
    int max_bits   = in_bits > out_bits ? in_bits : out_bits;
    int kVectorM_a = 128 / max_bits;             // max vector from wider type
    int kVectorM_b = algo.warp_tile / warp_size; // max vector from warp tile
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
            sig.in_type,
            sig.out_type,
            algo.block_warps,
            algo.warp_tile,
            algo.pad};
}

/// Check if problem size N is aligned to a variant's block_tile (no padding needed).
constexpr bool isAligned(VectorAddKernel k, int n) { return n > 0 && n % k.block_tile == 0; }

/// Convenience: extract signature and algorithm from config.
consteval VectorAddKernel make_kernel(ElementwiseConfig cfg)
{
    return make_kernel(cfg.signature, cfg.algorithm);
}

} // namespace rocm_ck