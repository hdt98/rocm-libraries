// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GpuConvBwdRefTestFixture.hpp"

#include "ConvShapeCatalog.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gpu_conv_bwd_ref_test;
using namespace gpu_conv_ref_test;

// Alias to avoid verbose braced-init-list issues inside EXPECT_THROW macros
using Vec = std::vector<int64_t>;

// ============================================================================
// TestGpuConvBwdRefValidation — validateInput throw paths (via GpuFpReferenceConvolution::dgrad)
// ============================================================================

TEST(TestGpuConvBwdRefValidation, ThrowsOnInvalidDimCount)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dx({8, 8});
    Tensor<float> w({8, 8});
    Tensor<float> dy({8, 8});

    EXPECT_THROW(GpuFpReferenceConvolution::dgrad<float>(dx, w, dy, Vec{1}, Vec{1}, Vec{0}, Vec{0}),
                 std::invalid_argument);
}

TEST(TestGpuConvBwdRefValidation, ThrowsOnWeightDimMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dx({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::dgrad<float>(
                     dx, w, dy, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvBwdRefValidation, ThrowsOnOutputDimMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dx({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::dgrad<float>(
                     dx, w, dy, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

// ============================================================================
// Standalone tests (TEST, not TEST_P)
// ============================================================================

// ============================================================================
// TestGpuConvBwdRefAsymPad — asymmetric (pre != post) padding tests
// ============================================================================

TEST(TestGpuConvBwdRefAsymPadFp32, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    // x=[1,1,3,3], w=[1,1,3,3], padding=(1,0)/(0,1) -> y=[1,1,2,2]
    runGpuVsCpuConvBwd<float>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 1e-5f);
}

TEST(TestGpuConvBwdRefAsymPadFp16, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvBwd<half>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 5e-2f);
}

TEST(TestGpuConvBwdRefAsymPadBfp16, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvBwd<bfloat16>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 0.1f);
}

// ============================================================================
// TestGpuConvBwdRefAlphaBeta — alpha/beta scaling tests
// ============================================================================

TEST(TestGpuConvBwdRefAlphaBeta, AlphaOnly)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> dyTensor({1, 1, 2, 2});
    Tensor<float> dxRef({1, 1, 4, 4});
    Tensor<float> dxScaled({1, 1, 4, 4});

    const unsigned int seed = 42;
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Compute with alpha=1.0
    GpuFpReferenceConvolution::dgrad<float>(dxRef, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0});

    // Compute with alpha=2.0
    GpuFpReferenceConvolution::dgrad<float>(
        dxScaled, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0}, 2.0);

    const auto* refData = dxRef.memory().hostData();
    const auto* scaledData = dxScaled.memory().hostData();
    auto numElements = dxRef.elementCount();

    for(size_t i = 0; i < numElements; ++i)
    {
        ASSERT_NEAR(scaledData[i], 2.0f * refData[i], 1e-5f) << "Alpha scaling failed at " << i;
    }
}

TEST(TestGpuConvBwdRefAlphaBeta, BetaAccumulate)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> dyTensor({1, 1, 2, 2});
    Tensor<float> dxTensor({1, 1, 4, 4});

    const unsigned int seed = 42;
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    dxTensor.fillWithValue(1.0f);

    // Pre-fill dx with 1.0, then compute with alpha=1.0, beta=1.0
    // Result should be dgrad(dy,w) + 1.0
    Tensor<float> dxNoAccum({1, 1, 4, 4});
    GpuFpReferenceConvolution::dgrad<float>(dxNoAccum, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0});
    GpuFpReferenceConvolution::dgrad<float>(
        dxTensor, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0}, 1.0, 1.0);

    const auto* noAccumData = dxNoAccum.memory().hostData();
    const auto* accumData = dxTensor.memory().hostData();
    auto numElements = dxTensor.elementCount();

    for(size_t i = 0; i < numElements; ++i)
    {
        ASSERT_NEAR(accumData[i], noAccumData[i] + 1.0f, 1e-5f)
            << "Beta accumulation failed at " << i;
    }
}

TEST(TestGpuConvBwdRefAlphaBeta, BetaZeroSkipsRead)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> dyTensor({1, 1, 2, 2});
    Tensor<float> dxBetaZero({1, 1, 4, 4});
    Tensor<float> dxDefault({1, 1, 4, 4});

    const unsigned int seed = 42;
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Pre-fill with garbage — should be ignored when beta=0
    dxBetaZero.fillWithValue(999.0f);

    GpuFpReferenceConvolution::dgrad<float>(dxDefault, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0});
    GpuFpReferenceConvolution::dgrad<float>(
        dxBetaZero, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0}, 1.0, 0.0);

    const auto* defaultData = dxDefault.memory().hostData();
    const auto* betaZeroData = dxBetaZero.memory().hostData();
    auto numElements = dxDefault.elementCount();

    for(size_t i = 0; i < numElements; ++i)
    {
        ASSERT_NEAR(betaZeroData[i], defaultData[i], 1e-5f)
            << "Beta=0 should ignore pre-filled data at " << i;
    }
}

// ============================================================================
// TestGpuConvBwdRefStridedFp32 — non-packed (strided) tensor tests
// ============================================================================

TEST(TestGpuConvBwdRefStridedFp32, NonPackedOutput)
{
    SKIP_IF_NO_DEVICES();

    // dx: [1, 2, 4, 4] with inter-channel gap (stride[1]=32 vs packed 16)
    const std::vector<int64_t> dxDims = {1, 2, 4, 4};
    const std::vector<int64_t> dxStrides = {64, 32, 4, 1};

    Tensor<float> dxCpu(dxDims, dxStrides);
    Tensor<float> dxGpu(dxDims, dxStrides);
    Tensor<float> wTensor({1, 2, 3, 3});
    Tensor<float> dyTensor({1, 1, 2, 2});

    const unsigned int seed = 42;
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::dgrad<float, float, float, double>(
        dxCpu, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0});

    GpuFpReferenceConvolution::dgrad<float, float, float, double>(
        dxGpu, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0});

    assertAllClose(dxCpu, dxGpu, 1e-5f);
}

TEST(TestGpuConvBwdRefStridedFp32, NonPackedInput)
{
    SKIP_IF_NO_DEVICES();

    // dy: [1, 1, 4, 4] with inter-row gap (stride[2]=8 vs packed 4)
    const std::vector<int64_t> dyDims = {1, 1, 4, 4};
    const std::vector<int64_t> dyStrides = {32, 32, 8, 1};

    Tensor<float> dxCpu({1, 1, 6, 6});
    Tensor<float> dxGpu({1, 1, 6, 6});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> dyTensor(dyDims, dyStrides);

    const unsigned int seed = 42;
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::dgrad<float, float, float, double>(
        dxCpu, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0});

    GpuFpReferenceConvolution::dgrad<float, float, float, double>(
        dxGpu, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0});

    assertAllClose(dxCpu, dxGpu, 1e-5f);
}

// ============================================================================
// TestGpuConvBwdRefMixedType — separate DX/DY type tests
// ============================================================================

TEST(TestGpuConvBwdRefMixedType, FloatDxHalfWeight)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> dxCpu({1, 1, 4, 4});
    Tensor<float> dxGpu({1, 1, 4, 4});
    Tensor<half> wTensor({1, 1, 3, 3});
    Tensor<float> dyTensor({1, 1, 2, 2});

    compareGpuVsCpuConvBwd<float, half, float, double>(
        dxCpu, dxGpu, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0}, {0, 0}, 5e-2f, 1.0f);
}

TEST(TestGpuConvBwdRefMixedType, HalfDxFloatWeight)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> dxCpu({1, 1, 4, 4});
    Tensor<half> dxGpu({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<half> dyTensor({1, 1, 2, 2});

    compareGpuVsCpuConvBwd<half, float, half, double>(
        dxCpu, dxGpu, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0}, {0, 0}, 5e-2f, 1.0f);
}

// ============================================================================
// TestGpuConvBwdRefShapes — parameterized shape coverage across types
// TEST_P definitions for the fixture classes from the shared header.
// ============================================================================

TEST_P(TestGpuConvBwdRefShapesFp32, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}
TEST_P(TestGpuConvBwdRefShapesFp16, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}
TEST_P(TestGpuConvBwdRefShapesBfp16, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}

// ============================================================================
// Default layout (NCHW/NCDHW/NCL) instantiations.
// Small shapes run on every PR; Medium/Large shapes filtered via --gtest_filter for nightly.
// ============================================================================

// fp32: small shapes
INSTANTIATE_TEST_SUITE_P(Small2d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(getSmall2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small3d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(getSmall3dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small1d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(getSmall1dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });

// fp16: small shapes
INSTANTIATE_TEST_SUITE_P(Small2d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(getSmall2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small1d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(getSmall1dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small3d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(getSmall3dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });

// bfp16: small shapes
INSTANTIATE_TEST_SUITE_P(Small2d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(getSmall2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small1d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(getSmall1dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small3d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(getSmall3dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });

// ============================================================================
// Channel-last (NLC/NHWC/NDHWC) instantiations — small shapes.
// ============================================================================

// fp32 NLC/NHWC/NDHWC
INSTANTIATE_TEST_SUITE_P(SmallNlc1d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall1dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(SmallNhwc2d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(SmallNdhwc3d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });

// fp16 NLC/NHWC/NDHWC
INSTANTIATE_TEST_SUITE_P(SmallNlc1d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall1dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(SmallNhwc2d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(SmallNdhwc3d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });

// bfp16 NLC/NHWC/NDHWC
INSTANTIATE_TEST_SUITE_P(SmallNlc1d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall1dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(SmallNhwc2d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(SmallNdhwc3d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });

// ============================================================================
// Default layout (NCL/NCHW/NCDHW) — medium/large shapes.
// Filter with --gtest_filter="Medium*:Large*" or --gtest_filter="-*Medium*-*Large*"
// ============================================================================

// fp32
INSTANTIATE_TEST_SUITE_P(Medium1d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(getMedium1dDgradCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Medium2d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(getMedium2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Medium3d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(getMedium3dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large1d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(getLarge1dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large2d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(getLarge2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large3d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(getLarge3dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });

// fp16
INSTANTIATE_TEST_SUITE_P(Medium1d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(getMedium1dDgradCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Medium2d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(getMedium2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Medium3d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(getMedium3dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large1d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(getLarge1dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large2d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(getLarge2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large3d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(getLarge3dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });

// bfp16
INSTANTIATE_TEST_SUITE_P(Medium1d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(getMedium1dDgradCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Medium2d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(getMedium2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Medium3d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(getMedium3dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large1d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(getLarge1dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large2d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(getLarge2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large3d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(getLarge3dConvCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });

// ============================================================================
// Channel-last (NLC/NHWC/NDHWC) — medium/large shapes.
// ============================================================================

// fp32
INSTANTIATE_TEST_SUITE_P(MediumNlc1d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(MediumNhwc2d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(MediumNdhwc3d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(LargeNlc1d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge1dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(LargeNhwc2d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(LargeNdhwc3d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge3dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });

// fp16
INSTANTIATE_TEST_SUITE_P(MediumNlc1d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(MediumNhwc2d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(MediumNdhwc3d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(LargeNlc1d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge1dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(LargeNhwc2d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(LargeNdhwc3d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge3dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });

// bfp16
INSTANTIATE_TEST_SUITE_P(MediumNlc1d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(MediumNhwc2d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(MediumNdhwc3d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(LargeNlc1d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge1dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(LargeNhwc2d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(LargeNdhwc3d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge3dConvCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
