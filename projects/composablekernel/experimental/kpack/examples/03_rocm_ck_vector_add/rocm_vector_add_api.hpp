// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Shared argument struct and compile-time configuration for rocm_ck vector add.
//
// This header has NO CK Tile dependency. It is included by both host code
// (main.cpp) and device code (.hip files) to share the kernel ABI and
// compile-time-validated configuration.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace rocm_ck {

/// Fixed-width integer type for kernel index/size arguments.
/// Matches ck_tile::index_t but avoids pulling in CK Tile headers.
using index_t = std::int32_t;

/// Data type tag for compile-time kernel configuration.
/// Modeled on ck_tile::builder::DataType but independent of CK Tile headers.
/// No UNDEFINED — every config must specify a valid type.
enum class DataType
{
    FP32,
    FP16,
    BF16,
    FP8
};

/// Returns the bit-width of a DataType. Uses bits (not bytes) so future
/// sub-byte types (fp4, fp6, int4) are clean integers.
/// No default case — lets -Wswitch catch unhandled enum values.
constexpr int data_type_bits(DataType dt)
{
    switch(dt)
    {
    case DataType::FP32: return 32;
    case DataType::FP16: return 16;
    case DataType::BF16: return 16;
    case DataType::FP8: return 8;
    }
    return 0;
}

/// Kernel arguments passed by value through hipModuleLaunchKernel.
/// Layout must match exactly between host and device.
struct VectorAddArgs
{
    index_t n;
    const void* a;
    const void* b;
    void* c;
};

static_assert(std::is_trivially_copyable_v<VectorAddArgs>,
              "VectorAddArgs must be trivially copyable for kernarg passing");
static_assert(sizeof(VectorAddArgs) == 32, "unexpected VectorAddArgs size");
static_assert(alignof(VectorAddArgs) == 8, "unexpected VectorAddArgs alignment");
static_assert(offsetof(VectorAddArgs, n) == 0, "unexpected offset for n");
static_assert(offsetof(VectorAddArgs, a) == 8, "unexpected offset for a");
static_assert(offsetof(VectorAddArgs, b) == 16, "unexpected offset for b");
static_assert(offsetof(VectorAddArgs, c) == 24, "unexpected offset for c");

/// Signature: describes WHAT the kernel computes (types).
/// In CK Tile's builder pattern, this is the "what" — independent of
/// how the kernel is tiled, scheduled, or padded.
struct elementwise_signature
{
    DataType compute_type;
};

/// Algorithm: describes HOW the kernel executes (tile geometry, pipeline).
/// In CK Tile's builder pattern, this is the "how" — independent of
/// the data types being computed.
struct elementwise_algorithm
{
    int block_tile;  // Elements processed per thread block (BlockTile)
    int block_warps; // Number of warps per thread block (BlockWarps)
    int warp_tile;   // Warp tile size for vector width calculation (WarpTile)
    bool pad;        // Enable padding for unaligned problem sizes
};

/// Compile-time configuration for a vector add variant.
/// This is the user-facing input — it may contain non-structural members
/// (e.g. std::optional) that prevent direct use as an NTTP.
/// Backward-compatible: constructs sig+algo with defaults (block_warps=1,
/// warp_tile=block_size, pad=true).
struct vector_add_config
{
    int block_size; // Elements processed per thread block (BlockTile)
    DataType compute_type          = DataType::FP32; // Data type for granularity validation
    std::optional<int> unroll_hint = std::nullopt;   // Proof-of-concept non-structural member
};

/// Structural output used as NTTP and host launch info.
/// All members must be structural types (no std::optional, no pointers, etc.).
struct vector_add_struct
{
    int block_tile;        // Elements per thread block (for grid calculation)
    int thread_block_size; // Threads per block (= warp_size * block_warps)
    DataType compute_type; // Data type (always set by make_kernel)
    int block_warps;       // Warps per block
    int warp_tile;         // Warp tile size
    bool pad;              // Padding enabled
};

/// AMD GCN/CDNA wavefront size.
constexpr int warp_size = 64;

/// Validate a signature + algorithm pair and produce a structural kernel descriptor.
///
/// Validates CK Tile ElementWiseShape compatibility:
///   kVectorM = min(128 / type_bits, warp_tile / warp_size) — must be >= 1
///   kRepeatM = block_tile / (block_warps * kVectorM * warp_size) — must be >= 1, integer
///   thread_block_size = warp_size * block_warps (NOT block_tile)
///
/// Invalid configs produce a compile-time error (consteval).
consteval vector_add_struct make_kernel(elementwise_signature sig, elementwise_algorithm algo)
{
    if(algo.block_tile <= 0)
        throw "block_tile must be positive";
    if(algo.block_warps <= 0)
        throw "block_warps must be positive";
    if(algo.warp_tile <= 0)
        throw "warp_tile must be positive";
    if(algo.warp_tile < warp_size)
        throw "warp_tile must be >= warp_size (64)";

    int type_bits  = data_type_bits(sig.compute_type);
    int kVectorM_a = 128 / type_bits;            // max vector from type width
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
            sig.compute_type,
            algo.block_warps,
            algo.warp_tile,
            algo.pad};
}

/// Backward-compatible: vector_add_config → sig + algo with defaults.
/// Defaults: block_warps=1, warp_tile=block_size, pad=true.
consteval vector_add_struct make_kernel(vector_add_config cfg)
{
    return make_kernel(elementwise_signature{cfg.compute_type},
                       elementwise_algorithm{cfg.block_size, 1, cfg.block_size, true});
}

} // namespace rocm_ck