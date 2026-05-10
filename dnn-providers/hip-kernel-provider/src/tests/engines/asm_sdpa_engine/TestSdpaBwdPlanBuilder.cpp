// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "GraphTest.hpp"
#include "HipKernelHandle.hpp"
#include "HipKernelSettings.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaBwdPlanBuilder.hpp"
#include "hip_kernel_provider_common/HipDeviceUtils.hpp"

namespace asm_sdpa_engine
{
namespace
{

class TestSdpaBwdPlanBuilder : public ::testing::Test
{
protected:
    SdpaBwdPlanBuilder _planBuilder;
    HipKernelHandle _handle;
};

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableReturnsFalseForNonSdpaBwdGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

auto createSdpaBwdGraph(const std::vector<int64_t>& dims = {4, 8, 256, 128},
                        hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                        = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                        bool withScale = false,
                        bool alibiMask = false,
                        bool paddingMask = false,
                        bool causalMask = false)
{
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    return hipdnn_test_sdk::utilities::createValidSdpaBwdGraph(dims,
                                                               strides,
                                                               dims,
                                                               strides,
                                                               dims,
                                                               strides,
                                                               dims,
                                                               strides,
                                                               dataType,
                                                               withScale,
                                                               alibiMask,
                                                               paddingMask,
                                                               causalMask);
}

// Build a Q/K/V graph with arbitrary head dim D_qk = D_v across all tensors.
// Mirrors createSdpaBwdGraph but allows independent control over head dim.
auto createSdpaBwdGraphWithHdim(int64_t batch,
                                int64_t numHeads,
                                int64_t seqLen,
                                int64_t headDim,
                                hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                                = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                                bool causalMask = false)
{
    return createSdpaBwdGraph({batch, numHeads, seqLen, headDim},
                              dataType,
                              /*withScale=*/false,
                              /*alibiMask=*/false,
                              /*paddingMask=*/false,
                              causalMask);
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableSdpaBwdVariations)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
    {
        GTEST_SKIP();
    }

    std::vector<std::pair<GraphTest, bool>> applicabilityTests = {
        // Valid backward graph: BF16, HD=128, FP32 stats, no masking — only
        // configuration with a calibrated CPU reference today.
        {GraphTest{createSdpaBwdGraph(), "Valid BF16 HD128 backward"}, true},

        // HD=64: rejected — registry has only pddv=0 rows for hd64, and the
        // CPU reference is not calibrated for the kernel anyway.
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 64}), "Head dimension 64"}, false},

        // HD=192: dispatch infrastructure exists but CPU reference correctness
        // has not been verified. Gated until I8.4.x.
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 192}), "Head dimension 192"}, false},

        // FP16: dispatch infrastructure exists but CPU reference correctness
        // has not been verified. Gated until I8.4.x.
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 128}, DataType::HALF), "FP16 tensors"}, false},

        // Causal mask not currently dispatched.
        {GraphTest{
             createSdpaBwdGraph({4, 8, 256, 128}, DataType::BFLOAT16, false, false, false, true),
             "causal_mask = true"},
         false},

        // Alibi mask not supported.
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 128}, DataType::BFLOAT16, false, true),
                   "alibi_mask = true"},
         false},

        // Padding mask not supported.
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 128}, DataType::BFLOAT16, false, false, true),
                   "padding_mask = true"},
         false},

        // With scale tensor (should still be accepted)
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 128}, DataType::BFLOAT16, true),
                   "with scale tensor"},
         true},
    };

    for(const auto& [test, applicability] : applicabilityTests)
    {
        EXPECT_EQ(_planBuilder.isApplicable(_handle, test.graphWrapper()), applicability)
            << test.message;
    }
}

TEST_F(TestSdpaBwdPlanBuilder, BackwardWorkspaceSizeSmallUnaligned)
{
    // B=1, H=3, S=255, D=128 — chosen so D buffer raw size (3060) is NOT a multiple of 64,
    // exercising the alignUp() rounding: 3060 → 3072
    auto builder = createSdpaBwdGraph({1, 3, 255, 128});
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    HipKernelSettings settings;
    size_t workspaceSize = _planBuilder.getMaxWorkspaceSize(_handle, graphWrapper, settings);

    // D buffer: 1*3*255*4 = 3060, aligned to 64 → 3072  (exercises alignment rounding)
    // dq_acc:   1*3*255*128*4 = 391680, aligned to 64 → 391680 (already aligned)
    EXPECT_EQ(workspaceSize, 3072u + 391680u);
}

TEST_F(TestSdpaBwdPlanBuilder, BackwardWorkspaceSizeMedium)
{
    // B=2, H=8, S=512, D=128
    auto builder = createSdpaBwdGraph({2, 8, 512, 128});
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    HipKernelSettings settings;
    size_t workspaceSize = _planBuilder.getMaxWorkspaceSize(_handle, graphWrapper, settings);

    // D buffer: 2*8*512*4 = 32768, aligned to 64 → 32768
    // dq_acc:   2*8*512*128*4 = 4194304, aligned to 64 → 4194304
    EXPECT_EQ(workspaceSize, 32768u + 4194304u);
}

TEST_F(TestSdpaBwdPlanBuilder, BackwardWorkspaceSizeLarge)
{
    // B=4, H=16, S=1024, D=128
    auto builder = createSdpaBwdGraph({4, 16, 1024, 128});
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    HipKernelSettings settings;
    size_t workspaceSize = _planBuilder.getMaxWorkspaceSize(_handle, graphWrapper, settings);

    // D buffer: 4*16*1024*4 = 262144, aligned to 64 → 262144
    // dq_acc:   4*16*1024*128*4 = 33554432, aligned to 64 → 33554432
    EXPECT_EQ(workspaceSize, 262144u + 33554432u);
}

// CSV-derived constants used by the registry-lookup tests.
namespace
{
constexpr int MASK_NONE = 0; // MaskType::NO_MASK
constexpr int MASK_TOP_LEFT_CAUSAL = 1; // MaskType::TOP_LEFT_CAUSAL
constexpr int MODE_BATCH = 0; // BatchMode::BATCH
constexpr int ATOMIC_A32 = 1; // AccumulatorMode::A32
constexpr int ATOMIC_NONE = 0; // odo / dq_convert use atomic32=0
constexpr int PSSK_ON = 1;
constexpr int PDDV_ON = 1;
constexpr int PSSK_OFF = 0;
constexpr int PDDV_OFF = 0;
constexpr int BF16_CVT_RTNE = 0; // RoundingMode::RTNE
constexpr int BF16_CVT_FP16_SENTINEL = 3; // CSV sentinel for fp16 rows
} // namespace

// POC config: hd128 / bf16 / NO_MASK / BATCH must still resolve across all
// three pipeline-stage registries.
TEST(SdpaBwdRegistryLookup, RegistryLookup_Hd128Bf16NoMaskBatch)
{
    using namespace bwd_dispatch;

    auto odoKey = lookupKernelNameKey(PipelineStage::ODO,
                                      "gfx942",
                                      "bf16",
                                      /*hdimQ=*/128,
                                      /*hdimV=*/128,
                                      MASK_NONE,
                                      ATOMIC_NONE,
                                      PSSK_OFF,
                                      PDDV_OFF,
                                      MODE_BATCH,
                                      BF16_CVT_FP16_SENTINEL);
    EXPECT_FALSE(odoKey.empty()) << "odo lookup should resolve";

    auto dqdkdvKey = lookupKernelNameKey(PipelineStage::DQDKDV,
                                         "gfx942",
                                         "bf16",
                                         /*hdimQ=*/128,
                                         /*hdimV=*/128,
                                         MASK_NONE,
                                         ATOMIC_A32,
                                         PSSK_ON,
                                         PDDV_ON,
                                         MODE_BATCH,
                                         BF16_CVT_RTNE);
    EXPECT_FALSE(dqdkdvKey.empty()) << "dqdkdv lookup should resolve";

    auto dqConvertKey = lookupKernelNameKey(PipelineStage::DQ_CONVERT,
                                            "gfx942",
                                            "bf16",
                                            /*hdimQ=*/128,
                                            /*hdimV=*/128,
                                            MASK_NONE,
                                            ATOMIC_NONE,
                                            PSSK_OFF,
                                            PDDV_OFF,
                                            MODE_BATCH,
                                            BF16_CVT_RTNE);
    EXPECT_FALSE(dqConvertKey.empty()) << "dq_convert lookup should resolve";
}

// FP16 / hd64 row resolves via the unpadded (pssk=1, pddv=0) layout that the
// registry actually carries for that head-dim. This exercises the lookup for
// a hd64 entry even though isApplicable's day-one matrix forces pddv=1 and
// therefore rejects hd64 at dispatch time.
TEST(SdpaBwdRegistryLookup, RegistryLookup_Hd64Fp16NoMaskBatch)
{
    using namespace bwd_dispatch;

    auto odoKey = lookupKernelNameKey(PipelineStage::ODO,
                                      "gfx942",
                                      "fp16",
                                      /*hdimQ=*/64,
                                      /*hdimV=*/64,
                                      MASK_NONE,
                                      ATOMIC_NONE,
                                      PSSK_OFF,
                                      PDDV_OFF,
                                      MODE_BATCH,
                                      BF16_CVT_FP16_SENTINEL);
    EXPECT_FALSE(odoKey.empty());

    // hd64 dqdkdv is registered with pssk=1, pddv=0 (no padded-D variant).
    auto dqdkdvKey = lookupKernelNameKey(PipelineStage::DQDKDV,
                                         "gfx942",
                                         "fp16",
                                         /*hdimQ=*/64,
                                         /*hdimV=*/64,
                                         MASK_NONE,
                                         ATOMIC_A32,
                                         PSSK_ON,
                                         PDDV_OFF,
                                         MODE_BATCH,
                                         BF16_CVT_FP16_SENTINEL);
    EXPECT_FALSE(dqdkdvKey.empty());

    auto dqConvertKey = lookupKernelNameKey(PipelineStage::DQ_CONVERT,
                                            "gfx942",
                                            "fp16",
                                            /*hdimQ=*/64,
                                            /*hdimV=*/64,
                                            MASK_NONE,
                                            ATOMIC_NONE,
                                            PSSK_OFF,
                                            PDDV_OFF,
                                            MODE_BATCH,
                                            BF16_CVT_FP16_SENTINEL);
    EXPECT_FALSE(dqConvertKey.empty());
}

// hd192 / bf16 / TOP_LEFT_CAUSAL row resolves via the registry. isApplicable
// rejects causal day-one but the lookup itself must work.
TEST(SdpaBwdRegistryLookup, RegistryLookup_Hd192Bf16CausalBatch)
{
    using namespace bwd_dispatch;

    // Causal hd192/bf16 dqdkdv batch kernels are pssk=1, pddv=1 in the registry.
    auto dqdkdvKey = lookupKernelNameKey(PipelineStage::DQDKDV,
                                         "gfx942",
                                         "bf16",
                                         /*hdimQ=*/192,
                                         /*hdimV=*/192,
                                         MASK_TOP_LEFT_CAUSAL,
                                         ATOMIC_A32,
                                         PSSK_ON,
                                         PDDV_ON,
                                         MODE_BATCH,
                                         BF16_CVT_RTNE);
    EXPECT_FALSE(dqdkdvKey.empty()) << "hd192 bf16 causal_tl dqdkdv lookup should resolve";

    // odo is mask-agnostic.
    auto odoKey = lookupKernelNameKey(PipelineStage::ODO,
                                      "gfx942",
                                      "bf16",
                                      /*hdimQ=*/192,
                                      /*hdimV=*/192,
                                      MASK_NONE,
                                      ATOMIC_NONE,
                                      PSSK_OFF,
                                      PDDV_OFF,
                                      MODE_BATCH,
                                      BF16_CVT_FP16_SENTINEL);
    EXPECT_FALSE(odoKey.empty()) << "hd192 bf16 odo lookup should resolve";
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicable_RejectsHd96)
{
    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaBwdGraphWithHdim(/*batch=*/2, /*numHeads=*/8, /*seqLen=*/256, 96);
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicable_RejectsFp8)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaBwdGraphWithHdim(
        /*batch=*/2, /*numHeads=*/8, /*seqLen=*/256, 128, DataType::FP8_E4M3);
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicable_RejectsGfx950)
{
    // Cannot synthesise a different device string from the test harness, so
    // this test only meaningfully runs on a non-gfx942 device. On gfx942 it
    // is skipped (the positive case is covered by IsApplicableSdpaBwdVariations).
    auto deviceString = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString == "gfx942")
    {
        GTEST_SKIP() << "Test requires non-gfx942 device to assert rejection";
    }

    auto builder = createSdpaBwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicable_RejectsAsymmetricHdim)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
    {
        GTEST_SKIP();
    }

    // V tensor has D_v = 64 while Q/K have D_qk = 128 — backward kernels
    // require square head dimensions (D_qk == D_v).
    std::vector<int64_t> qDims = {2, 8, 256, 128};
    std::vector<int64_t> kDims = {2, 8, 256, 128};
    std::vector<int64_t> vDims = {2, 8, 256, 64};
    std::vector<int64_t> oDims = {2, 8, 256, 64};

    auto builder = hipdnn_test_sdk::utilities::createValidSdpaBwdGraph(
        qDims,
        hipdnn_data_sdk::utilities::generateStrides(qDims),
        kDims,
        hipdnn_data_sdk::utilities::generateStrides(kDims),
        vDims,
        hipdnn_data_sdk::utilities::generateStrides(vDims),
        oDims,
        hipdnn_data_sdk::utilities::generateStrides(oDims),
        DataType::BFLOAT16);

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

class PsskPddvComputationTest : public ::testing::TestWithParam<
                                    std::tuple<unsigned int, unsigned int, unsigned int, int, int>>
{
};

// Parameters: (seqLenKv, tsKv, headDimV, expectedPssk, expectedPddv).
TEST_P(PsskPddvComputationTest, PaddingFlagComputation_PsskPddv)
{
    auto [seqLenKv, tsKv, headDimV, expectedPssk, expectedPddv] = GetParam();

    EXPECT_EQ(bwd_dispatch::computePssk(seqLenKv, tsKv), expectedPssk);
    EXPECT_EQ(bwd_dispatch::computePddv(headDimV), expectedPddv);
}

INSTANTIATE_TEST_SUITE_P(PsskPddvCases,
                         PsskPddvComputationTest,
                         ::testing::Values(
                             // seqLenKv exactly divides tsKv -> pssk=0; headDimV==128 -> pddv=0
                             std::make_tuple(256u, 64u, 128u, 0, 0),
                             // seqLenKv not divisible -> pssk=1; headDimV==128 -> pddv=0
                             std::make_tuple(257u, 64u, 128u, 1, 0),
                             // seqLenKv divisible -> pssk=0; headDimV!=128 -> pddv=1
                             std::make_tuple(192u, 64u, 64u, 0, 1),
                             // seqLenKv not divisible -> pssk=1; headDimV!=128 -> pddv=1
                             std::make_tuple(200u, 192u, 192u, 1, 1),
                             // tsKv == 0 (unresolved tile) -> pssk defaults to 1
                             std::make_tuple(256u, 0u, 128u, 1, 0)));

} // namespace
} // namespace asm_sdpa_engine
