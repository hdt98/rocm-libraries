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
    DataType compute_type          = DataType::FP32; // Data type for granularity validation
    std::optional<int> unroll_hint = std::nullopt;   // Proof-of-concept non-structural member
};

/// Structural output used as NTTP and host launch info.
/// All members must be structural types (no std::optional, no pointers, etc.).
struct vector_add_struct
{
    int block_size;        // Elements per thread block (for grid calculation)
    int thread_block_size; // Threads per block (for launch config)
    DataType compute_type; // Data type (always set by make_kernel)
};

/// Validate a vector_add_config and produce a structural kernel descriptor.
///
/// block_size must be a positive multiple of the type-dependent granularity:
///   min_block = 8192 / type_bits
///   FP32(32b)→256, FP16/BF16(16b)→512, FP8(8b)→1024
///
/// This comes from CK Tile ElementWiseShape:
///   kVectorM = min(128 / type_bits, block_size / 64)
///   min_block = 64 * (128 / type_bits) = 8192 / type_bits
///
/// Invalid configs produce a compile-time error (consteval).
consteval vector_add_struct make_kernel(vector_add_config cfg)
{
    if(cfg.block_size <= 0)
        throw "block_size must be positive";

    int min_block = 8192 / data_type_bits(cfg.compute_type);

    if(cfg.block_size % min_block != 0)
        throw "block_size must be a multiple of the type-dependent granularity "
              "(256 for FP32, 512 for FP16/BF16, 1024 for FP8)";

    return {cfg.block_size, cfg.block_size, cfg.compute_type};
}

} // namespace rocm_ck