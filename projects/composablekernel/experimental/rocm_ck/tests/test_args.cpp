// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/args.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>

using namespace rocm_ck;

// ============================================================================
// TensorArg ABI
// ============================================================================

TEST(TensorArg, TriviallyCopyable) { EXPECT_TRUE(std::is_trivially_copyable_v<TensorArg>); }

TEST(TensorArg, StandardLayout) { EXPECT_TRUE(std::is_standard_layout_v<TensorArg>); }

TEST(TensorArg, Size)
{
    // ptr(8) + lengths(6*4=24) + strides(6*8=48) = 80
    EXPECT_EQ(sizeof(TensorArg), 80);
}

TEST(TensorArg, Alignment) { EXPECT_EQ(alignof(TensorArg), 8); }

TEST(TensorArg, FieldOffsets)
{
    EXPECT_EQ(offsetof(TensorArg, ptr), 0);
    EXPECT_EQ(offsetof(TensorArg, lengths), 8);
    EXPECT_EQ(offsetof(TensorArg, strides), 32);
}

// ============================================================================
// ScalarValue ABI
// ============================================================================

TEST(ScalarValue, TriviallyCopyable) { EXPECT_TRUE(std::is_trivially_copyable_v<ScalarValue>); }

TEST(ScalarValue, Size)
{
    // Union of float(4), int32(4), uint32(4), double(8) -> 8 bytes
    EXPECT_EQ(sizeof(ScalarValue), 8);
}

// ============================================================================
// Args ABI
// ============================================================================

TEST(Args, TriviallyCopyable) { EXPECT_TRUE(std::is_trivially_copyable_v<Args>); }

TEST(Args, StandardLayout) { EXPECT_TRUE(std::is_standard_layout_v<Args>); }

TEST(Args, Size)
{
    // 16 tensors * 80 + 16 scalars * 8 = 1280 + 128 = 1408
    EXPECT_EQ(sizeof(Args), 1408);
}

TEST(Args, Alignment) { EXPECT_EQ(alignof(Args), 8); }

TEST(Args, FitsInKernargBudget)
{
    // HSA minimum kernarg size is 4096 bytes
    EXPECT_LE(sizeof(Args), 4096);
}

// ============================================================================
// Capacity constants
// ============================================================================

TEST(Args, Constants)
{
    EXPECT_EQ(kMaxRank, 6);
    EXPECT_EQ(kMaxTensors, 16);
    EXPECT_EQ(kMaxScalars, 16);
}

// ============================================================================
// ScalarValue union access
// ============================================================================

TEST(ScalarValue, FloatAccess)
{
    ScalarValue sv{};
    sv.f32 = 3.14f;
    EXPECT_FLOAT_EQ(sv.f32, 3.14f);
}

TEST(ScalarValue, Int32Access)
{
    ScalarValue sv{};
    sv.i32 = -42;
    EXPECT_EQ(sv.i32, -42);
}

TEST(ScalarValue, DoubleAccess)
{
    ScalarValue sv{};
    sv.f64 = 2.718281828;
    EXPECT_DOUBLE_EQ(sv.f64, 2.718281828);
}
