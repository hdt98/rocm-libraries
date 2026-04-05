// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/elementwise_spec.hpp>

#include <gtest/gtest.h>

using namespace rocm_ck;

// ============================================================================
// Valid specs
// ============================================================================

TEST(ElementwiseMakeSpec, BasicFP32)
{
    constexpr auto spec = makeSpec(
        Signature{.dtype = DataType::FP32, .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        ElementwiseAlgorithm{.block_tile = 256, .block_waves = 1, .wave_tile = 256},
        TargetSet::cdna());

    EXPECT_EQ(spec.num_physical_tensors, 3);
    EXPECT_EQ(spec.block_tile, 256);
    EXPECT_EQ(spec.workgroup_size, 64);
}

TEST(ElementwiseMakeSpec, BasicFP16)
{
    constexpr auto spec = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        ElementwiseAlgorithm{.block_tile = 512, .block_waves = 1, .wave_tile = 512},
        TargetSet::cdna());

    EXPECT_EQ(spec.num_physical_tensors, 3);
    EXPECT_EQ(spec.block_tile, 512);
}

TEST(ElementwiseMakeSpec, MultiWave)
{
    constexpr auto spec = makeSpec(
        Signature{.dtype = DataType::FP32, .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        ElementwiseAlgorithm{.block_tile = 1024, .block_waves = 4, .wave_tile = 256},
        TargetSet::cdna());

    EXPECT_EQ(spec.workgroup_size, 64 * 4);
    EXPECT_EQ(spec.block_waves, 4);
}

TEST(ElementwiseMakeSpec, WithPadding)
{
    constexpr auto spec = makeSpec(
        Signature{.dtype = DataType::FP32, .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        ElementwiseAlgorithm{.block_tile = 256, .block_waves = 1, .wave_tile = 256, .pad = true},
        TargetSet::cdna());

    EXPECT_TRUE(spec.pad);
}

TEST(ElementwiseMakeSpec, PhysicalTensorTable)
{
    constexpr auto spec = makeSpec(
        Signature{.dtype = DataType::BF16, .ops = {AddOp{.lhs = "X", .rhs = "Y", .out = "Z"}}},
        ElementwiseAlgorithm{.block_tile = 512, .block_waves = 1, .wave_tile = 512},
        TargetSet::cdna());

    EXPECT_TRUE(spec.physical_tensors[0].name == "X");
    EXPECT_TRUE(spec.physical_tensors[1].name == "Y");
    EXPECT_TRUE(spec.physical_tensors[2].name == "Z");
    EXPECT_EQ(spec.physical_tensors[0].dtype, DataType::BF16);
    EXPECT_EQ(spec.physical_tensors[0].args_slot, 0);
    EXPECT_EQ(spec.physical_tensors[1].args_slot, 1);
    EXPECT_EQ(spec.physical_tensors[2].args_slot, 2);
}

// ============================================================================
// isAligned
// ============================================================================

TEST(ElementwiseIsAligned, AlignedProblemSize)
{
    constexpr auto spec = makeSpec(
        Signature{.dtype = DataType::FP32, .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        ElementwiseAlgorithm{.block_tile = 256, .block_waves = 1, .wave_tile = 256},
        TargetSet::cdna());

    EXPECT_TRUE(isAligned(spec, 256));
    EXPECT_TRUE(isAligned(spec, 512));
    EXPECT_TRUE(isAligned(spec, 1024));
    EXPECT_FALSE(isAligned(spec, 255));
    EXPECT_FALSE(isAligned(spec, 100));
    EXPECT_FALSE(isAligned(spec, 0));
    EXPECT_FALSE(isAligned(spec, -1));
}

// ============================================================================
// Validation errors (consteval throws → runtime test via try/catch not possible,
// but we verify the valid cases compile and invalid ones are documented in
// compile_fail/ tests)
// ============================================================================

// Verify that mixed input types (mismatched lhs/rhs) would be caught:
// This is tested via compile_fail tests; here we verify the valid path.
TEST(ElementwiseMakeSpec, MixedOutputDtype)
{
    // Mixed precision: FP16 inputs, FP32 output — this is valid
    constexpr auto spec =
        makeSpec(Signature{.dtype   = DataType::FP16,
                           .tensors = {Tensor{.name = "C", .dtype = DataType::FP32}},
                           .ops     = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                 ElementwiseAlgorithm{.block_tile = 512, .block_waves = 1, .wave_tile = 512},
                 TargetSet::cdna());

    EXPECT_EQ(spec.physical_tensors[0].dtype, DataType::FP16); // A
    EXPECT_EQ(spec.physical_tensors[1].dtype, DataType::FP16); // B
    EXPECT_EQ(spec.physical_tensors[2].dtype, DataType::FP32); // C (overridden)
}
