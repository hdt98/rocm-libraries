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

/// Kernel arguments passed by value through hipModuleLaunchKernel.
/// Layout must match exactly between host and device.
struct VectorAddArgs
{
    index_t n;
    const float* a;
    const float* b;
    float* c;
};

static_assert(std::is_trivially_copyable_v<VectorAddArgs>,
              "VectorAddArgs must be trivially copyable for kernarg passing");
static_assert(sizeof(VectorAddArgs) == 32, "unexpected VectorAddArgs size");
static_assert(alignof(VectorAddArgs) == 8, "unexpected VectorAddArgs alignment");
static_assert(offsetof(VectorAddArgs, n) == 0, "unexpected offset for n");
static_assert(offsetof(VectorAddArgs, a) == 8, "unexpected offset for a");
static_assert(offsetof(VectorAddArgs, b) == 16, "unexpected offset for b");
static_assert(offsetof(VectorAddArgs, c) == 24, "unexpected offset for c");

/// Compile-time configuration for a vector add variant.
/// This is the user-facing input — it may contain non-structural members
/// (e.g. std::optional) that prevent direct use as an NTTP.
struct vector_add_config
{
    int block_size; // Elements processed per thread block (BlockTile)
    std::optional<int> unroll_hint = std::nullopt; // Proof-of-concept non-structural member
};

/// Structural output used as NTTP and host launch info.
/// All members must be structural types (no std::optional, no pointers, etc.).
struct vector_add_struct
{
    int block_size;        // Elements per thread block (for grid calculation)
    int thread_block_size; // Threads per block (for launch config)
};

/// Validate a vector_add_config and produce a structural kernel descriptor.
///
/// block_size must be a positive multiple of 256, which is the CK Tile
/// granularity for BlockWarps=1, float (vector_width=4), warp_size=64:
///   256 = 1 warp × 64 threads/warp × 4 elements/thread
///
/// Invalid configs produce a compile-time error (consteval).
consteval vector_add_struct make_kernel(vector_add_config cfg)
{
    if(cfg.block_size <= 0 || cfg.block_size % 256 != 0)
        throw "block_size must be a positive multiple of 256";
    return {cfg.block_size, cfg.block_size};
}

} // namespace rocm_ck