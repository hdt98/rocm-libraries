// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/gemm_spec.hpp>

#include <gtest/gtest.h>

using namespace rocm_ck;

// ============================================================================
// is_valid_warp_gemm
// ============================================================================

TEST(WarpGemm, AcceptsFP32With16x16Tile)
{
    EXPECT_TRUE(is_valid_warp_gemm(DataType::FP32, 16, 16, 4));
    EXPECT_TRUE(is_valid_warp_gemm(DataType::FP32, 16, 16, 8));
    EXPECT_TRUE(is_valid_warp_gemm(DataType::FP32, 16, 16, 16));
}

TEST(WarpGemm, AcceptsFP32With32x32OnlyForSmallK)
{
    EXPECT_TRUE(is_valid_warp_gemm(DataType::FP32, 32, 32, 4));
    EXPECT_TRUE(is_valid_warp_gemm(DataType::FP32, 32, 32, 8));
    EXPECT_FALSE(is_valid_warp_gemm(DataType::FP32, 32, 32, 16)); // k=16 invalid at 32x32 for fp32
}

TEST(WarpGemm, AcceptsFP16With16x16Tile)
{
    EXPECT_TRUE(is_valid_warp_gemm(DataType::FP16, 16, 16, 16));
    EXPECT_TRUE(is_valid_warp_gemm(DataType::FP16, 16, 16, 32));
    EXPECT_FALSE(is_valid_warp_gemm(DataType::FP16, 16, 16, 4));
}

TEST(WarpGemm, AcceptsFP16With32x32Tile)
{
    EXPECT_TRUE(is_valid_warp_gemm(DataType::FP16, 32, 32, 8));
    EXPECT_TRUE(is_valid_warp_gemm(DataType::FP16, 32, 32, 16));
    EXPECT_FALSE(is_valid_warp_gemm(DataType::FP16, 32, 32, 4)); // k=4 invalid at 32x32 for fp16
}

TEST(WarpGemm, AcceptsSameTilesForBF16AsFP16)
{
    EXPECT_TRUE(is_valid_warp_gemm(DataType::BF16, 16, 16, 16));
    EXPECT_TRUE(is_valid_warp_gemm(DataType::BF16, 32, 32, 16));
    EXPECT_FALSE(is_valid_warp_gemm(DataType::BF16, 32, 32, 4));
}

TEST(WarpGemm, RejectsAsymmetricAndIntegerConfigs)
{
    // Asymmetric tiles not supported
    EXPECT_FALSE(is_valid_warp_gemm(DataType::FP32, 16, 32, 8));
    EXPECT_FALSE(is_valid_warp_gemm(DataType::FP16, 32, 16, 16));

    // Integer types not supported in warp gemm validation
    EXPECT_FALSE(is_valid_warp_gemm(DataType::I32, 16, 16, 4));
}

// ============================================================================
// make_spec: plain GEMM
// ============================================================================

TEST(MakeSpec, ProducesThreePhysicalTensorsForPlainGemm)
{
    constexpr auto k = make_spec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(k.num_physical_tensors, 3);
}

TEST(MakeSpec, MapsGemmTensorsToSequentialSlots)
{
    constexpr auto k = make_spec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(slot(k, "A"), 0);
    EXPECT_EQ(slot(k, "B"), 1);
    EXPECT_EQ(slot(k, "C"), 2);
}

TEST(MakeSpec, PropagatesDtypeToAllGemmTensors)
{
    constexpr auto k = make_spec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(dtype(k, "A"), DataType::FP16);
    EXPECT_EQ(dtype(k, "B"), DataType::FP16);
    EXPECT_EQ(dtype(k, "C"), DataType::FP16);
}

TEST(MakeSpec, ComputesThreadBlockSizeFromWarps)
{
    constexpr auto k = make_spec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    // 2 * 2 * 1 * 64 = 256
    EXPECT_EQ(k.thread_block_size, 256);
}

TEST(MakeSpec, ReportsZeroEpilogueOpsForPlainGemm)
{
    constexpr auto k = make_spec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(k.num_epilogue_ops, 0);
}

// ============================================================================
// make_spec: GEMM + Add
// ============================================================================

TEST(MakeSpec, RegistersAddAsEpilogueOp)
{
    constexpr auto k = make_spec(Signature{.dtype = DataType::FP16,
                                           .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                     AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                                 GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(k.num_epilogue_ops, 1);
    EXPECT_EQ(k.epilogue_ops[0], EpilogueOp::Add);
    EXPECT_EQ(k.num_physical_tensors, 4); // A, B, D(output), bias(D0)
}

TEST(MakeSpec, PlacesBiasInD0SlotForGemmAdd)
{
    constexpr auto k = make_spec(Signature{.dtype = DataType::FP16,
                                           .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                     AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                                 GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(slot(k, "D"), 2);    // output slot
    EXPECT_EQ(slot(k, "bias"), 3); // D0 slot
}

TEST(MakeSpec, PropagatesDtypeToBiasTensor)
{
    constexpr auto k = make_spec(Signature{.dtype = DataType::FP16,
                                           .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                     AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                                 GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(dtype(k, "bias"), DataType::FP16);
}

// ============================================================================
// make_spec: GEMM + Add + Relu
// ============================================================================

TEST(MakeSpec, RegistersAddAndReluAsEpilogueOps)
{
    constexpr auto k = make_spec(Signature{.dtype = DataType::FP16,
                                           .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                     AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                                                     ReluOp{.in = "D", .out = "E"}}},
                                 GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(k.num_epilogue_ops, 2);
    EXPECT_TRUE(k.has_epilogue_op(EpilogueOp::Add));
    EXPECT_TRUE(k.has_epilogue_op(EpilogueOp::Relu));
}

TEST(MakeSpec, UsesFinalOutputSlotForGemmAddRelu)
{
    constexpr auto k = make_spec(Signature{.dtype = DataType::FP16,
                                           .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                     AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                                                     ReluOp{.in = "D", .out = "E"}}},
                                 GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(slot(k, "E"), 2);           // final output in slot 2
    EXPECT_EQ(k.num_physical_tensors, 4); // A, B, E(output), bias(D0)
}

// ============================================================================
// make_spec: 32x32 warp tile
// ============================================================================

TEST(MakeSpec, Accepts32x32WarpTileWithCorrectBlockSize)
{
    constexpr auto k = make_spec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {32, 32, 16}});

    EXPECT_EQ(k.thread_block_size, 256);
    EXPECT_EQ(k.warp_tile.m, 32);
    EXPECT_EQ(k.warp_tile.n, 32);
    EXPECT_EQ(k.warp_tile.k, 16);
}

// ============================================================================
// make_spec: layout defaults
// ============================================================================

TEST(MakeSpec, AssignsRowColRowLayoutByDefault)
{
    constexpr auto k = make_spec(
        Signature{.dtype = DataType::FP32, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(layout(k, "A"), Layout::Row);
    EXPECT_EQ(layout(k, "B"), Layout::Col);
    EXPECT_EQ(layout(k, "C"), Layout::Row);
}

// ============================================================================
// GemmSpec named accessors
// ============================================================================

TEST(GemmSpec, ProvidesLhsRhsOutputNamedAccessors)
{
    constexpr auto k = make_spec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(k.lhs().args_slot, 0);
    EXPECT_EQ(k.rhs().args_slot, 1);
    EXPECT_EQ(k.output().args_slot, 2);
    EXPECT_EQ(k.lhs().dtype, DataType::FP16);
}

// ============================================================================
// Accumulator dtype
// ============================================================================

TEST(MakeSpec, DefaultsAccDtypeToFP32)
{
    constexpr auto k = make_spec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(k.acc_dtype, DataType::FP32); // GemmOp default acc_dtype
}

// ============================================================================
// Multiple data types
// ============================================================================

TEST(MakeSpec, ProducesFP32GemmWithMatchingAccDtype)
{
    constexpr auto k = make_spec(
        Signature{.dtype = DataType::FP32, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(dtype(k, "A"), DataType::FP32);
    EXPECT_EQ(k.acc_dtype, DataType::FP32);
}

TEST(MakeSpec, ProducesBF16GemmWithCorrectDtype)
{
    constexpr auto k = make_spec(
        Signature{.dtype = DataType::BF16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(dtype(k, "A"), DataType::BF16);
}
