// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/gemm_spec.hpp>

#include <gtest/gtest.h>

using namespace rocm_ck;

// ============================================================================
// isValidWaveTile
// ============================================================================

TEST(WaveTileValidation, AcceptsFP32With16x16Tile)
{
    EXPECT_TRUE(isValidWaveTile(DataType::FP32, 16, 16, 4));
    EXPECT_TRUE(isValidWaveTile(DataType::FP32, 16, 16, 8));
    EXPECT_TRUE(isValidWaveTile(DataType::FP32, 16, 16, 16));
}

TEST(WaveTileValidation, AcceptsFP32With32x32OnlyForSmallK)
{
    EXPECT_TRUE(isValidWaveTile(DataType::FP32, 32, 32, 4));
    EXPECT_TRUE(isValidWaveTile(DataType::FP32, 32, 32, 8));
    EXPECT_FALSE(isValidWaveTile(DataType::FP32, 32, 32, 16)); // k=16 invalid at 32x32 for fp32
}

TEST(WaveTileValidation, AcceptsFP16With16x16Tile)
{
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 16, 16, 16));
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 16, 16, 32));
    EXPECT_FALSE(isValidWaveTile(DataType::FP16, 16, 16, 4));
}

TEST(WaveTileValidation, AcceptsFP16With32x32Tile)
{
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 32, 32, 8));
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 32, 32, 16));
    EXPECT_FALSE(isValidWaveTile(DataType::FP16, 32, 32, 4)); // k=4 invalid at 32x32 for fp16
}

TEST(WaveTileValidation, AcceptsSameTilesForBF16AsFP16)
{
    EXPECT_TRUE(isValidWaveTile(DataType::BF16, 16, 16, 16));
    EXPECT_TRUE(isValidWaveTile(DataType::BF16, 32, 32, 16));
    EXPECT_FALSE(isValidWaveTile(DataType::BF16, 32, 32, 4));
}

TEST(WaveTileValidation, RejectsAsymmetricAndIntegerConfigs)
{
    // Asymmetric tiles not supported
    EXPECT_FALSE(isValidWaveTile(DataType::FP32, 16, 32, 8));
    EXPECT_FALSE(isValidWaveTile(DataType::FP16, 32, 16, 16));

    // Integer types not yet in wave tile validation table
    EXPECT_FALSE(isValidWaveTile(DataType::I32, 16, 16, 4));
}

// ============================================================================
// makeSpec: plain GEMM
// ============================================================================

TEST(MakeSpec, ProducesThreePhysicalTensorsForPlainGemm)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(k.num_physical_tensors, 3);
}

TEST(MakeSpec, MapsGemmTensorsToSequentialSlots)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(slot(k, "A"), 0);
    EXPECT_EQ(slot(k, "B"), 1);
    EXPECT_EQ(slot(k, "C"), 2);
}

TEST(MakeSpec, PropagatesDtypeToAllGemmTensors)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(dtype(k, "A"), DataType::FP16);
    EXPECT_EQ(dtype(k, "B"), DataType::FP16);
    EXPECT_EQ(dtype(k, "C"), DataType::FP16);
}

TEST(MakeSpec, ComputesThreadBlockSizeFromWaves)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    // 2 * 2 * 1 * 64 = 256
    EXPECT_EQ(k.workgroup_size, 256);
}

TEST(MakeSpec, ReportsZeroEpilogueOpsForPlainGemm)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(k.num_epilogue_ops, 0);
}

// ============================================================================
// makeSpec: GEMM + Add
// ============================================================================

TEST(MakeSpec, RegistersAddAsEpilogueOp)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(k.num_epilogue_ops, 1);
    EXPECT_EQ(k.epilogue_ops[0], EpilogueOp::Add);
    EXPECT_EQ(k.num_physical_tensors, 4); // A, B, D(output), bias(D0)
}

TEST(MakeSpec, PlacesBiasInD0SlotForGemmAdd)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(slot(k, "D"), 2);    // output slot
    EXPECT_EQ(slot(k, "bias"), 3); // D0 slot
}

TEST(MakeSpec, PropagatesDtypeToBiasTensor)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(dtype(k, "bias"), DataType::FP16);
}

// ============================================================================
// makeSpec: GEMM + Add + Relu
// ============================================================================

TEST(MakeSpec, RegistersAddAndReluAsEpilogueOps)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                                                    ReluOp{.in = "D", .out = "E"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(k.num_epilogue_ops, 2);
    EXPECT_TRUE(k.hasEpilogueOp(EpilogueOp::Add));
    EXPECT_TRUE(k.hasEpilogueOp(EpilogueOp::Relu));
}

TEST(MakeSpec, UsesFinalOutputSlotForGemmAddRelu)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                                                    ReluOp{.in = "D", .out = "E"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(slot(k, "E"), 2);           // final output in slot 2
    EXPECT_EQ(k.num_physical_tensors, 4); // A, B, E(output), bias(D0)
}

// ============================================================================
// makeSpec: 32x32 wave tile
// ============================================================================

TEST(MakeSpec, Accepts32x32WaveTileWithCorrectBlockSize)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {32, 32, 16}});

    EXPECT_EQ(k.workgroup_size, 256);
    EXPECT_EQ(k.wave_tile.m, 32);
    EXPECT_EQ(k.wave_tile.n, 32);
    EXPECT_EQ(k.wave_tile.k, 16);
}

// ============================================================================
// makeSpec: layout defaults
// ============================================================================

TEST(MakeSpec, AssignsRowColRowLayoutByDefault)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP32, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(layout(k, "A"), Layout::Row);
    EXPECT_EQ(layout(k, "B"), Layout::Col);
    EXPECT_EQ(layout(k, "C"), Layout::Row);
}

TEST(MakeSpec, OverridesBLayoutToRowForRR)
{
    constexpr auto k = makeSpec(Signature{.dtype   = DataType::FP16,
                                          .tensors = {Tensor{.name = "B", .layout = Layout::Row}},
                                          .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(layout(k, "A"), Layout::Row);
    EXPECT_EQ(layout(k, "B"), Layout::Row);
    EXPECT_EQ(layout(k, "C"), Layout::Row);
}

TEST(MakeSpec, OverridesBothLayoutsForCC)
{
    constexpr auto k = makeSpec(Signature{.dtype   = DataType::FP16,
                                          .tensors = {Tensor{.name = "A", .layout = Layout::Col},
                                                      Tensor{.name = "B", .layout = Layout::Col}},
                                          .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(layout(k, "A"), Layout::Col);
    EXPECT_EQ(layout(k, "B"), Layout::Col);
    EXPECT_EQ(layout(k, "C"), Layout::Row);
}

TEST(MakeSpec, OverridesALayoutForCR)
{
    constexpr auto k = makeSpec(Signature{.dtype   = DataType::FP16,
                                          .tensors = {Tensor{.name = "A", .layout = Layout::Col},
                                                      Tensor{.name = "B", .layout = Layout::Row}},
                                          .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(layout(k, "A"), Layout::Col);
    EXPECT_EQ(layout(k, "B"), Layout::Row);
    EXPECT_EQ(layout(k, "C"), Layout::Row);
}

TEST(MakeSpec, LayoutOverrideFlowsToPhysicalTensorTable)
{
    constexpr auto k = makeSpec(Signature{.dtype   = DataType::FP16,
                                          .tensors = {Tensor{.name = "B", .layout = Layout::Row}},
                                          .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    // Verify the physical tensor table (what the device code sees)
    EXPECT_EQ(k.lhs().layout, Layout::Row);
    EXPECT_EQ(k.rhs().layout, Layout::Row);
    EXPECT_EQ(k.output().layout, Layout::Row);
}

// ============================================================================
// GemmSpec named accessors
// ============================================================================

TEST(GemmSpec, ProvidesLhsRhsOutputNamedAccessors)
{
    constexpr auto k = makeSpec(
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
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(k.acc_dtype, DataType::FP32); // GemmOp default acc_dtype
}

// ============================================================================
// Multiple data types
// ============================================================================

TEST(MakeSpec, ProducesFP32GemmWithMatchingAccDtype)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP32, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(dtype(k, "A"), DataType::FP32);
    EXPECT_EQ(k.acc_dtype, DataType::FP32);
}

TEST(MakeSpec, ProducesBF16GemmWithCorrectDtype)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::BF16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(dtype(k, "A"), DataType::BF16);
}

// ============================================================================
// Split-K (k_batch)
// ============================================================================

TEST(MakeSpec, DefaultsKBatchToOne)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(k.k_batch, 1);
}

TEST(MakeSpec, AcceptsExplicitKBatch)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{.block_tile  = {128, 128, 32},
                      .block_waves = {2, 2, 1},
                      .wave_tile   = {16, 16, 16},
                      .k_batch     = 4});

    EXPECT_EQ(k.k_batch, 4);
}

TEST(MakeSpec, KBatchPreservesOtherFields)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{.block_tile  = {128, 128, 32},
                      .block_waves = {2, 2, 1},
                      .wave_tile   = {16, 16, 16},
                      .k_batch     = 4});

    EXPECT_EQ(k.num_physical_tensors, 3);
    EXPECT_EQ(k.workgroup_size, 256);
    EXPECT_EQ(k.block_tile.k, 32);
}

TEST(MakeSpec, KBatchWorksWithEpilogueOps)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                                GemmAlgorithm{.block_tile  = {128, 128, 32},
                                              .block_waves = {2, 2, 1},
                                              .wave_tile   = {16, 16, 16},
                                              .k_batch     = 2});

    EXPECT_EQ(k.k_batch, 2);
    EXPECT_EQ(k.num_epilogue_ops, 1);
    EXPECT_TRUE(k.hasEpilogueOp(EpilogueOp::Add));
}

// ============================================================================
// isValidWaveTile: GpuTarget-specific validation
// ============================================================================

TEST(WaveTileValidation, AcceptsFP8TilesForGfx942)
{
    // gfx942 MFMA: 32x32x16, 16x16x32
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 16, GpuTarget::gfx942));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 16, 16, 32, GpuTarget::gfx942));
    // gfx950-only tiles rejected on gfx942
    EXPECT_FALSE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 32, GpuTarget::gfx942));
    EXPECT_FALSE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 64, GpuTarget::gfx942));
    EXPECT_FALSE(isValidWaveTile(DataType::FP8_FNUZ, 16, 16, 64, GpuTarget::gfx942));
}

TEST(WaveTileValidation, AcceptsFP8TilesForGfx950)
{
    // gfx950 MFMA: 32x32x{16,32,64}, 16x16x{32,64}
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 16, GpuTarget::gfx950));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 32, GpuTarget::gfx950));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 64, GpuTarget::gfx950));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 16, 16, 32, GpuTarget::gfx950));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 16, 16, 64, GpuTarget::gfx950));
}

TEST(WaveTileValidation, AnyTargetRejectsFP8ArchSpecificTiles)
{
    // Any = intersection: only tiles valid on all CDNA targets
    // 32x32x16 and 16x16x32 are valid on both gfx942 and gfx950
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 16, GpuTarget::Any));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 16, 16, 32, GpuTarget::Any));
    // gfx950-only tiles rejected under Any
    EXPECT_FALSE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 32, GpuTarget::Any));
    EXPECT_FALSE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 64, GpuTarget::Any));
    EXPECT_FALSE(isValidWaveTile(DataType::FP8_FNUZ, 16, 16, 64, GpuTarget::Any));
}

TEST(WaveTileValidation, DefaultTargetIsAny)
{
    // 4-arg overload (no target) behaves like Any
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 16));
    EXPECT_FALSE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 64));
}

TEST(WaveTileValidation, Gfx90aAcceptsSameTilesAsCDNABaseline)
{
    // gfx90a has same MFMA tile set as the baseline (no FP8)
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 16, 16, 16, GpuTarget::gfx90a));
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 32, 32, 16, GpuTarget::gfx90a));
    // gfx90a has no FP8 MFMA support
    EXPECT_FALSE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 16, GpuTarget::gfx90a));
}

TEST(WaveTileValidation, BF8HasSameTilesAsFP8)
{
    EXPECT_TRUE(isValidWaveTile(DataType::BF8_FNUZ, 32, 32, 16, GpuTarget::gfx942));
    EXPECT_TRUE(isValidWaveTile(DataType::BF8_FNUZ, 32, 32, 32, GpuTarget::gfx950));
    EXPECT_FALSE(isValidWaveTile(DataType::BF8_FNUZ, 32, 32, 32, GpuTarget::gfx942));
}

// ============================================================================
// makeSpec: GpuTarget parameter
// ============================================================================

TEST(MakeSpec, AcceptsGpuTargetParameter)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        GpuTarget::gfx942);

    EXPECT_EQ(k.workgroup_size, 256);
}

TEST(MakeSpec, DefaultsToGpuTargetAny)
{
    // 2-arg makeSpec still works (defaults to GpuTarget::Any)
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(k.workgroup_size, 256);
}

// ============================================================================
// makeSpec: Pipeline::Memory + Scheduling
// ============================================================================

TEST(MakeSpec, AcceptsMemoryPipelineWithIntrawaveScheduling)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{.block_tile         = {128, 128, 32},
                      .block_waves        = {2, 2, 1},
                      .wave_tile          = {16, 16, 16},
                      .pipeline           = Pipeline::Memory,
                      .pipeline_scheduler = PipelineScheduler::Intrawave});

    EXPECT_EQ(k.pipeline, Pipeline::Memory);
    EXPECT_EQ(k.pipeline_scheduler, PipelineScheduler::Intrawave);
}

TEST(MakeSpec, AcceptsMemoryPipelineWithInterwaveScheduling)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{.block_tile         = {128, 128, 32},
                      .block_waves        = {2, 2, 1},
                      .wave_tile          = {16, 16, 16},
                      .pipeline           = Pipeline::Memory,
                      .pipeline_scheduler = PipelineScheduler::Interwave});

    EXPECT_EQ(k.pipeline, Pipeline::Memory);
    EXPECT_EQ(k.pipeline_scheduler, PipelineScheduler::Interwave);
}

TEST(MakeSpec, DefaultsSchedulingToIntrawave)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

    EXPECT_EQ(k.pipeline_scheduler, PipelineScheduler::Intrawave);
}
