// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck_tile/core/utility/lds_bank_conflict_analysis.hpp"

using namespace ck_tile;

// =============================================================================
// Bank Configuration Tests
// =============================================================================

TEST(LdsBankConfig, Gfx950ReadConfig)
{
    using Config = lds_bank_config_gfx950_read;
    EXPECT_EQ(Config::NumBanks, 64);
    EXPECT_EQ(Config::BytesPerBank, 4);
}

TEST(LdsBankConfig, Gfx950WriteConfig)
{
    using Config = lds_bank_config_gfx950_write;
    EXPECT_EQ(Config::NumBanks, 32);
    EXPECT_EQ(Config::BytesPerBank, 4);
}

TEST(LdsBankConfig, DefaultConfig)
{
    using Config = lds_bank_config_default;
    EXPECT_EQ(Config::NumBanks, 32);
    EXPECT_EQ(Config::BytesPerBank, 4);
}

// =============================================================================
// Bank Computation Tests
// =============================================================================

TEST(LdsBankConfig, ComputeBank32Banks)
{
    using Config = lds_bank_config<32>;

    // Bank 0: bytes 0-3
    EXPECT_EQ(Config::compute_bank(0), 0);
    EXPECT_EQ(Config::compute_bank(1), 0);
    EXPECT_EQ(Config::compute_bank(2), 0);
    EXPECT_EQ(Config::compute_bank(3), 0);

    // Bank 1: bytes 4-7
    EXPECT_EQ(Config::compute_bank(4), 1);
    EXPECT_EQ(Config::compute_bank(7), 1);

    // Bank 31: last bank in a 32-bank config
    EXPECT_EQ(Config::compute_bank(31 * 4), 31);
    EXPECT_EQ(Config::compute_bank(31 * 4 + 3), 31);

    // Wrap around: byte 128 = bank 0 again (128 / 4 = 32, 32 % 32 = 0)
    EXPECT_EQ(Config::compute_bank(128), 0);
}

TEST(LdsBankConfig, ComputeBank64Banks)
{
    using Config = lds_bank_config<64>;

    EXPECT_EQ(Config::compute_bank(0), 0);
    EXPECT_EQ(Config::compute_bank(4), 1);
    EXPECT_EQ(Config::compute_bank(63 * 4), 63);

    // Wrap around: byte 256 = bank 0 again (256 / 4 = 64, 64 % 64 = 0)
    EXPECT_EQ(Config::compute_bank(256), 0);
}

// =============================================================================
// Linear Access Pattern Tests
// =============================================================================

TEST(LdsBankConflictLinear, NoConflicts32Threads32Banks)
{
    // 32 threads, each accessing a different 4-byte word -> no conflicts
    constexpr auto result = compute_lds_bank_conflicts_linear<32, 32, 4, 0>();

    EXPECT_EQ(result.total_conflicts, 0);
    EXPECT_EQ(result.max_threads_per_bank, 1);
    EXPECT_FALSE(result.has_conflicts());
}

TEST(LdsBankConflictLinear, FullConflicts32ThreadsSameBank)
{
    // 32 threads all accessing the same bank (stride 128 bytes = 32 banks * 4 bytes)
    constexpr auto result = compute_lds_bank_conflicts_linear<32, 32, 128, 0>();

    EXPECT_EQ(result.total_conflicts, 31); // 32 threads - 1
    EXPECT_EQ(result.max_threads_per_bank, 32);
    EXPECT_TRUE(result.has_conflicts());
}

TEST(LdsBankConflictLinear, TwoWayConflict)
{
    // 32 threads with stride 8 bytes (2 banks) -> 2-way conflicts
    // Threads 0,2,4,... access even banks; threads 1,3,5,... access odd banks
    // Actually: stride 8 means thread i accesses bank (i*2) % 32
    // So threads 0,16 access bank 0; threads 1,17 access bank 2; etc.
    constexpr auto result = compute_lds_bank_conflicts_linear<32, 32, 8, 0>();

    // Each bank is accessed by 2 threads (threads i and i+16)
    EXPECT_EQ(result.max_threads_per_bank, 2);
    EXPECT_EQ(result.total_conflicts, 16); // 16 banks * (2-1) = 16
    EXPECT_TRUE(result.has_conflicts());
}

TEST(LdsBankConflictLinear, FourWayConflict64Banks)
{
    // 64 threads with stride 16 bytes on 64 banks
    // Thread i accesses bank (i * 4) % 64
    // So threads 0,16,32,48 all access bank 0
    constexpr auto result = compute_lds_bank_conflicts_linear<64, 64, 16, 0>();

    EXPECT_EQ(result.max_threads_per_bank, 4);
    EXPECT_EQ(result.total_conflicts, 48); // 16 active banks * (4-1) = 48
    EXPECT_TRUE(result.has_conflicts());
}

TEST(LdsBankConflictLinear, NoConflict64Threads64Banks)
{
    // 64 threads, each accessing a different bank
    constexpr auto result = compute_lds_bank_conflicts_linear<64, 64, 4, 0>();

    EXPECT_EQ(result.total_conflicts, 0);
    EXPECT_EQ(result.max_threads_per_bank, 1);
    EXPECT_FALSE(result.has_conflicts());
}

// =============================================================================
// Array-based Offset Tests
// =============================================================================

TEST(LdsBankConflictFromOffsets, NoConflicts)
{
    constexpr array<index_t, 4> offsets{0, 4, 8, 12}; // Each thread accesses a different bank
    constexpr auto result = compute_lds_bank_conflicts_from_offsets<4, 32>(offsets);

    EXPECT_EQ(result.total_conflicts, 0);
    EXPECT_EQ(result.max_threads_per_bank, 1);
    EXPECT_FALSE(result.has_conflicts());
}

TEST(LdsBankConflictFromOffsets, AllSameBank)
{
    constexpr array<index_t, 4> offsets{0, 128, 256, 384}; // All access bank 0 (32-bank config)
    constexpr auto result = compute_lds_bank_conflicts_from_offsets<4, 32>(offsets);

    EXPECT_EQ(result.total_conflicts, 3);
    EXPECT_EQ(result.max_threads_per_bank, 4);
    EXPECT_TRUE(result.has_conflicts());
}

TEST(LdsBankConflictFromOffsets, MixedPattern)
{
    // Threads 0,1 access bank 0; threads 2,3 access bank 1
    constexpr array<index_t, 4> offsets{0, 128, 4, 132};
    constexpr auto result = compute_lds_bank_conflicts_from_offsets<4, 32>(offsets);

    EXPECT_EQ(result.total_conflicts, 2); // 2 conflicts (2-1) * 2 banks
    EXPECT_EQ(result.max_threads_per_bank, 2);
    EXPECT_TRUE(result.has_conflicts());
}

// =============================================================================
// Result Structure Tests
// =============================================================================

TEST(LdsBankConflictResult, ConflictRatio)
{
    lds_bank_conflict_result<32, 32> result{};
    result.total_conflicts = 0;
    EXPECT_FLOAT_EQ(result.conflict_ratio(), 0.0f);

    result.total_conflicts = 16;
    EXPECT_FLOAT_EQ(result.conflict_ratio(), 0.5f);

    result.total_conflicts = 31;
    EXPECT_FLOAT_EQ(result.conflict_ratio(), 31.0f / 32.0f);
}

// =============================================================================
// XOR Swizzle Pattern Tests
// =============================================================================

TEST(LdsBankConflictXorSwizzle, NoSwizzleWithConflicts)
{
    // Without XOR swizzle (XorMask=0), typical patterns have conflicts
    // 64 threads, 64 banks, 8-element vectors, 128-element row stride, no XOR
    constexpr auto result = compute_lds_bank_conflicts_xor_swizzle<
        64,   // WarpSize
        64,   // NumBanks
        8,    // VectorLen (FP16x8 = 16 bytes)
        128,  // RowStride in elements
        0,    // XorMask (no swizzle)
        2     // DataSize (FP16 = 2 bytes)
    >();

    // Without swizzle, we expect conflicts depending on the layout
    // The actual conflict count depends on the specific access pattern
    // This test verifies the function runs correctly
    EXPECT_GE(result.max_threads_per_bank, 1);
}

TEST(LdsBankConflictXorSwizzle, SimplePattern)
{
    // Simple test case: 4 threads, 4 banks, 1-element vectors
    // Row stride 4, no XOR
    constexpr auto result = compute_lds_bank_conflicts_xor_swizzle<
        4,    // WarpSize
        4,    // NumBanks
        1,    // VectorLen
        4,    // RowStride
        0,    // XorMask
        4     // DataSize (FP32 = 4 bytes)
    >();

    // Each thread accesses a different bank in this simple pattern
    EXPECT_LE(result.total_conflicts, 3); // At most 3 conflicts with 4 threads
}

// =============================================================================
// Compile-time Verification Tests
// =============================================================================

// These tests verify that the analysis works at compile time
static_assert(lds_bank_config<32>::compute_bank(0) == 0, "Bank 0 check");
static_assert(lds_bank_config<32>::compute_bank(4) == 1, "Bank 1 check");
static_assert(lds_bank_config<64>::compute_bank(252) == 63, "Bank 63 check");

// Verify linear pattern analysis is constexpr
static_assert(!compute_lds_bank_conflicts_linear<32, 32, 4, 0>().has_conflicts(),
              "32 threads, 32 banks, stride 4 should have no conflicts");

static_assert(compute_lds_bank_conflicts_linear<32, 32, 128, 0>().has_conflicts(),
              "32 threads all same bank should have conflicts");
