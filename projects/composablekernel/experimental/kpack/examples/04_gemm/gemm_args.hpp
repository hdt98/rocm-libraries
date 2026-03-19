// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Shared argument struct and compile-time configuration for the kpack GEMM example.
//
// This header has NO CK Tile dependency. It is included by both host code
// (main.cpp) and device code (.hip files) to share the kernel ABI and
// grid launch constants.

#pragma once

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
/// Computes C = A * B (basic GEMM, no alpha/beta scaling).
///
/// Memory layout conventions (matching BLAS lda/ldb/ldc):
///   A [M x K] RowMajor:    stride_A = K  (distance between rows)
///   B [K x N] ColumnMajor: stride_B = K  (distance between columns)
///   C [M x N] RowMajor:    stride_C = N  (distance between rows)
struct GemmArgs
{
    index_t M;        // rows of A and C
    index_t N;        // columns of B and C
    index_t K;        // columns of A / rows of B
    index_t stride_A; // leading dimension of A (= K for RowMajor)
    index_t stride_B; // leading dimension of B (= K for ColumnMajor)
    index_t stride_C; // leading dimension of C (= N for RowMajor)
    const void* a;    // device pointer to A [M x K]
    const void* b;    // device pointer to B [K x N]
    void* c;          // device pointer to C [M x N]
};

// --- ABI stability assertions ---
// Six 32-bit integers (24 bytes) followed by three 8-byte pointers (24 bytes).
// 24 is divisible by 8, so no padding between the integer and pointer groups.
static_assert(std::is_trivially_copyable_v<GemmArgs>,
              "GemmArgs must be trivially copyable for kernarg passing");
static_assert(sizeof(GemmArgs) == 48, "unexpected GemmArgs size");
static_assert(alignof(GemmArgs) == 8, "unexpected GemmArgs alignment");
static_assert(offsetof(GemmArgs, M) == 0, "unexpected offset for M");
static_assert(offsetof(GemmArgs, N) == 4, "unexpected offset for N");
static_assert(offsetof(GemmArgs, K) == 8, "unexpected offset for K");
static_assert(offsetof(GemmArgs, stride_A) == 12, "unexpected offset for stride_A");
static_assert(offsetof(GemmArgs, stride_B) == 16, "unexpected offset for stride_B");
static_assert(offsetof(GemmArgs, stride_C) == 20, "unexpected offset for stride_C");
static_assert(offsetof(GemmArgs, a) == 24, "unexpected offset for a");
static_assert(offsetof(GemmArgs, b) == 32, "unexpected offset for b");
static_assert(offsetof(GemmArgs, c) == 40, "unexpected offset for c");

// --- Grid launch constants ---
// These match the hardcoded tile configuration in gemm_fp32.hip so the host
// can compute grid dimensions without including CK Tile headers.
//
// Tile shape: 128x128x32 (M_Tile x N_Tile x K_Tile)
// Block warps: 2x2x1 = 4 warps = 256 threads (64 threads/warp on CDNA)
constexpr int M_TILE     = 128;
constexpr int N_TILE     = 128;
constexpr int BLOCK_SIZE = 256; // = 2 * 2 * 1 * 64

} // namespace rocm_ck
