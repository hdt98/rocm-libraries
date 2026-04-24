// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GpuConvFwdRefTestFixture.hpp"

#include <hipdnn_gpu_ref/detail/GpuRefHipError.hpp>
#include <hipdnn_gpu_ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn_test_sdk/utilities/ConvolutionValidation.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gpu_conv_fwd_ref_test;

// Alias to avoid verbose braced-init-list issues inside EXPECT_THROW macros
using Vec = std::vector<int64_t>;

// ============================================================================
// TestConvolutionValidation — direct tests for validateConvolutionParams
// (no GPU needed, tests the standalone validation function)
// ============================================================================

TEST(TestConvolutionValidation, AcceptsValidParams)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_NO_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
        x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}));
}

TEST(TestConvolutionValidation, ThrowsOnWeightDimMismatch)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnOutputDimMismatch)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnStridesSizeMismatch)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnDilationsSizeMismatch)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnPrePaddingSizeMismatch)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnPostPaddingSizeMismatch)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnZeroStride)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{0, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnNegativeDilation)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, -1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnNegativePrePadding)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{-1, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnNegativePostPadding)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, -1}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnOutputDimValueMismatch)
{
    // Input [1,1,4,4], kernel [1,1,3,3], no padding, stride 1 -> expected output [1,1,2,2]
    // Provide wrong output dims [1,1,3,3]
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 3, 3});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

// ============================================================================
// TestGpuConvFwdRefValidation — validateInput throw paths (via GpuFpReferenceConvolution)
// ============================================================================

TEST(TestGpuConvFwdRefValidation, ThrowsOnInvalidDimCount)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({8, 8});
    Tensor<float> w({8, 8});
    Tensor<float> y({8, 8});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(x, w, y, Vec{1}, Vec{1}, Vec{0}, Vec{0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnWeightDimMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnOutputDimMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnStridesSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(
        GpuFpReferenceConvolution::fprop<float>(x, w, y, Vec{1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
        std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnDilationsSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnPrePaddingSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(
        GpuFpReferenceConvolution::fprop<float>(x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0}, Vec{0, 0}),
        std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnPostPaddingSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnZeroStride)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{0, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnNegativeDilation)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, -1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnNegativePrePadding)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{-1, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnNegativePostPadding)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, -1}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnOutputDimValueMismatch)
{
    SKIP_IF_NO_DEVICES();
    // Input [1,1,4,4], kernel [1,1,3,3], no padding, stride 1 -> expected output [1,1,2,2]
    // Provide wrong output dims [1,1,3,3]
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 3, 3});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

// ============================================================================
// Standalone tests (TEST, not TEST_P)
// ============================================================================

// ============================================================================
// TestGpuConvFwdRefAsymPad — asymmetric (pre != post) padding tests
// ============================================================================

TEST(TestGpuConvFwdRefAsymPadFp32, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 1e-5f);
}

TEST(TestGpuConvFwdRefAsymPadFp16, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<half>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 5e-2f);
}

TEST(TestGpuConvFwdRefAsymPadBfp16, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<bfloat16>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 0.1f);
}

// ============================================================================
// TestGpuConvFwdRefAlphaBeta — alpha/beta scaling tests
// ============================================================================

TEST(TestGpuConvFwdRefAlphaBeta, AlphaOnly)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> yRef({1, 1, 2, 2});
    Tensor<float> yScaled({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Compute with alpha=1.0
    GpuFpReferenceConvolution::fprop<float>(xTensor, wTensor, yRef, {1, 1}, {1, 1}, {0, 0});

    // Compute with alpha=2.0
    GpuFpReferenceConvolution::fprop<float>(xTensor, wTensor, yScaled, {1, 1}, {1, 1}, {0, 0}, 2.0);

    const auto* refData = yRef.memory().hostData();
    const auto* scaledData = yScaled.memory().hostData();

    for(size_t i = 0; i < 4; ++i)
    {
        ASSERT_NEAR(scaledData[i], 2.0f * refData[i], 1e-5f) << "Alpha scaling failed at " << i;
    }
}

TEST(TestGpuConvFwdRefAlphaBeta, BetaAccumulate)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> yTensor({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    yTensor.fillWithValue(1.0f);

    // Pre-fill y with 1.0, then compute with alpha=1.0, beta=1.0
    // Result should be conv(x,w) + 1.0
    Tensor<float> yNoAccum({1, 1, 2, 2});
    GpuFpReferenceConvolution::fprop<float>(xTensor, wTensor, yNoAccum, {1, 1}, {1, 1}, {0, 0});
    GpuFpReferenceConvolution::fprop<float>(
        xTensor, wTensor, yTensor, {1, 1}, {1, 1}, {0, 0}, 1.0, 1.0);

    const auto* noAccumData = yNoAccum.memory().hostData();
    const auto* accumData = yTensor.memory().hostData();

    for(size_t i = 0; i < 4; ++i)
    {
        ASSERT_NEAR(accumData[i], noAccumData[i] + 1.0f, 1e-5f)
            << "Beta accumulation failed at " << i;
    }
}

TEST(TestGpuConvFwdRefAlphaBeta, BetaZeroSkipsRead)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> yBetaZero({1, 1, 2, 2});
    Tensor<float> yDefault({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Pre-fill with garbage — should be ignored when beta=0
    yBetaZero.fillWithValue(999.0f);

    GpuFpReferenceConvolution::fprop<float>(xTensor, wTensor, yDefault, {1, 1}, {1, 1}, {0, 0});
    GpuFpReferenceConvolution::fprop<float>(
        xTensor, wTensor, yBetaZero, {1, 1}, {1, 1}, {0, 0}, 1.0, 0.0);

    const auto* defaultData = yDefault.memory().hostData();
    const auto* betaZeroData = yBetaZero.memory().hostData();

    for(size_t i = 0; i < 4; ++i)
    {
        ASSERT_NEAR(betaZeroData[i], defaultData[i], 1e-5f)
            << "Beta=0 should ignore pre-filled data at " << i;
    }
}

// ============================================================================
// TestGpuConvFwdRefStridedFp32 — non-packed (strided) tensor tests
// Verifies stride-based indexing with memory gaps between elements.
// ============================================================================

TEST(TestGpuConvFwdRefStridedFp32, NonPackedInput)
{
    SKIP_IF_NO_DEVICES();

    // x: [1, 2, 4, 4] with inter-channel gap (stride[1]=32 vs packed 16)
    const std::vector<int64_t> xDims = {1, 2, 4, 4};
    const std::vector<int64_t> xStrides = {64, 32, 4, 1}; // packed would be {32, 16, 4, 1}

    Tensor<float> xTensor(xDims, xStrides);
    Tensor<float> wTensor({1, 2, 3, 3});
    Tensor<float> yCpu({1, 1, 2, 2});
    Tensor<float> yGpu({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yCpu, {1, 1}, {1, 1}, {0, 0});

    GpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {0, 0});

    assertAllClose(yCpu, yGpu, 1e-5f);
}

TEST(TestGpuConvFwdRefStridedFp32, NonPackedOutput)
{
    SKIP_IF_NO_DEVICES();

    // y: [1, 1, 4, 4] with inter-row gap (stride[2]=8 vs packed 4)
    const std::vector<int64_t> yDims = {1, 1, 4, 4};
    const std::vector<int64_t> yStrides = {32, 32, 8, 1}; // packed would be {16, 16, 4, 1}

    Tensor<float> xTensor({1, 1, 6, 6});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> yCpu(yDims, yStrides);
    Tensor<float> yGpu(yDims, yStrides);

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yCpu, {1, 1}, {1, 1}, {0, 0});

    GpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {0, 0});

    assertAllClose(yCpu, yGpu, 1e-5f);
}

TEST(TestGpuConvFwdRefStridedFp32, NonPackedInputAndOutput)
{
    SKIP_IF_NO_DEVICES();

    // Both x and y have non-packed strides with inter-row gaps
    const std::vector<int64_t> xDims = {1, 2, 4, 4};
    const std::vector<int64_t> xStrides = {64, 32, 6, 1}; // packed would be {32, 16, 4, 1}

    const std::vector<int64_t> yDims = {1, 1, 2, 2};
    const std::vector<int64_t> yStrides = {8, 8, 4, 1}; // packed would be {4, 4, 2, 1}

    Tensor<float> xTensor(xDims, xStrides);
    Tensor<float> wTensor({1, 2, 3, 3});
    Tensor<float> yCpu(yDims, yStrides);
    Tensor<float> yGpu(yDims, yStrides);

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yCpu, {1, 1}, {1, 1}, {0, 0});

    GpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {0, 0});

    assertAllClose(yCpu, yGpu, 1e-5f);
}

TEST(TestGpuConvFwdRefStridedFp32, NonPackedWithPadding)
{
    SKIP_IF_NO_DEVICES();

    // Non-packed input with padding to exercise both features together
    const std::vector<int64_t> xDims = {1, 2, 3, 3};
    const std::vector<int64_t> xStrides = {36, 18, 3, 1}; // packed would be {18, 9, 3, 1}

    Tensor<float> xTensor(xDims, xStrides);
    Tensor<float> wTensor({1, 2, 3, 3});
    Tensor<float> yCpu({1, 1, 3, 3});
    Tensor<float> yGpu({1, 1, 3, 3});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yCpu, {1, 1}, {1, 1}, {1, 1});

    GpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {1, 1});

    assertAllClose(yCpu, yGpu, 1e-5f);
}

// ============================================================================
// TestGpuConvFwdRefInt8 — int8 input with int32 or float output
// ============================================================================

TEST(TestGpuConvFwdRefInt8, Int8ToInt32)
{
    SKIP_IF_NO_DEVICES();

    Tensor<int8_t> xTensor({1, 1, 4, 4});
    Tensor<int8_t> wTensor({1, 1, 3, 3});
    Tensor<int32_t> yGpu({1, 1, 2, 2});

    // Fill with small values that won't overflow
    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(static_cast<int8_t>(-3), static_cast<int8_t>(3), seed);
    wTensor.fillWithRandomValues(static_cast<int8_t>(-2), static_cast<int8_t>(2), seed + 1);

    GpuFpReferenceConvolution::fprop<int8_t, int8_t, int32_t, int32_t>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {0, 0});

    // Verify manually: compute expected output
    const auto* x = xTensor.memory().hostData();
    const auto* w = wTensor.memory().hostData();
    const auto* y = yGpu.memory().hostData();

    // Each output element should be the sum of element-wise products of a 3x3 patch
    for(int ho = 0; ho < 2; ++ho)
    {
        for(int wo = 0; wo < 2; ++wo)
        {
            int32_t expected = 0;
            for(int kh = 0; kh < 3; ++kh)
            {
                for(int kw = 0; kw < 3; ++kw)
                {
                    expected += static_cast<int32_t>(x[(ho + kh) * 4 + (wo + kw)])
                                * static_cast<int32_t>(w[kh * 3 + kw]);
                }
            }
            ASSERT_EQ(y[ho * 2 + wo], expected)
                << "Int8->Int32 mismatch at (" << ho << "," << wo << ")";
        }
    }
}

TEST(TestGpuConvFwdRefInt8, Int8ToFloat)
{
    SKIP_IF_NO_DEVICES();

    Tensor<int8_t> xTensor({1, 1, 4, 4});
    Tensor<int8_t> wTensor({1, 1, 3, 3});
    Tensor<float> yGpu({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(static_cast<int8_t>(-3), static_cast<int8_t>(3), seed);
    wTensor.fillWithRandomValues(static_cast<int8_t>(-2), static_cast<int8_t>(2), seed + 1);

    GpuFpReferenceConvolution::fprop<int8_t, int8_t, float, float>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {0, 0});

    const auto* x = xTensor.memory().hostData();
    const auto* w = wTensor.memory().hostData();
    const auto* y = yGpu.memory().hostData();

    for(int ho = 0; ho < 2; ++ho)
    {
        for(int wo = 0; wo < 2; ++wo)
        {
            float expected = 0.0f;
            for(int kh = 0; kh < 3; ++kh)
            {
                for(int kw = 0; kw < 3; ++kw)
                {
                    expected += static_cast<float>(x[(ho + kh) * 4 + (wo + kw)])
                                * static_cast<float>(w[kh * 3 + kw]);
                }
            }
            ASSERT_NEAR(y[ho * 2 + wo], expected, 1e-5f)
                << "Int8->Float mismatch at (" << ho << "," << wo << ")";
        }
    }
}

// ============================================================================
// TestGpuConvFwdRefTf32 — TF32 truncation test
// ============================================================================

TEST(TestGpuConvFwdRefTf32, DiffersFromNonTf32)
{
    SKIP_IF_NO_DEVICES();

    // Use values with enough mantissa bits to show TF32 truncation
    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> yNoTf32({1, 1, 2, 2});
    Tensor<float> yTf32({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Regular computation with float accumulation
    GpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yNoTf32, {1, 1}, {1, 1}, {0, 0}, 1.0, 0.0, false);

    // TF32 computation
    GpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yTf32, {1, 1}, {1, 1}, {0, 0}, 1.0, 0.0, true);

    const auto* noTf32Data = yNoTf32.memory().hostData();
    const auto* tf32Data = yTf32.memory().hostData();

    // TF32 results should be close but not identical to full-precision
    bool hasDifference = false;
    for(size_t i = 0; i < 4; ++i)
    {
        if(std::abs(noTf32Data[i] - tf32Data[i]) > 1e-10f)
        {
            hasDifference = true;
        }
        // But should still be close
        ASSERT_NEAR(noTf32Data[i], tf32Data[i], 0.1f)
            << "TF32 too far from full precision at " << i;
    }
    ASSERT_TRUE(hasDifference) << "TF32 should produce different results from full precision";
}

// ============================================================================
// TestGpuConvFwdRefMixedType — separate WEI_TYPE tests
// ============================================================================

TEST(TestGpuConvFwdRefMixedType, FloatInputHalfWeight)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<half> wTensor({1, 1, 3, 3});
    Tensor<float> yCpu({1, 1, 2, 2});
    Tensor<float> yGpu({1, 1, 2, 2});

    compareGpuVsCpuConvFwd<float, half, float, double>(
        xTensor, wTensor, yCpu, yGpu, {1, 1}, {1, 1}, {0, 0}, {0, 0}, 5e-2f, 1.0f);
}

TEST(TestGpuConvFwdRefMixedType, HalfInputFloatWeight)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> xTensor({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<half> yCpu({1, 1, 2, 2});
    Tensor<half> yGpu({1, 1, 2, 2});

    compareGpuVsCpuConvFwd<half, float, half, double>(
        xTensor, wTensor, yCpu, yGpu, {1, 1}, {1, 1}, {0, 0}, {0, 0}, 5e-2f, 1.0f);
}

// ============================================================================
// TestGpuConvFwdRefPerformance — timing comparisons
// ============================================================================

TEST(TestGpuConvFwdRefPerformance, MediumTensorTimingComparison)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> xDims = {2, 64, 14, 14};
    const std::vector<int64_t> wDims = {128, 64, 3, 3};
    const std::vector<int64_t> yDims = {2, 128, 12, 12};
    const std::vector<int64_t> strides = {1, 1};
    const std::vector<int64_t> dilations = {1, 1};
    const std::vector<int64_t> padding = {0, 0};

    Tensor<float> xTensor(xDims);
    Tensor<float> wTensor(wDims);
    Tensor<float> yCpu(yDims);
    Tensor<float> yGpu(yDims);

    const unsigned int seed = 123;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Warm-up run (includes HipRTC compilation on first call)
    GpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yGpu, strides, dilations, padding);

    // Time CPU
    auto cpuStart = std::chrono::high_resolution_clock::now();
    CpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yCpu, strides, dilations, padding);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuUs = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count();
    auto cpuMs = static_cast<double>(cpuUs) / 1000.0;

    // Time GPU (kernel already compiled from warm-up)
    hipEvent_t gpuStart = nullptr;
    hipEvent_t gpuStop = nullptr;
    static_cast<void>(hipEventCreate(&gpuStart));
    static_cast<void>(hipEventCreate(&gpuStop));

    static_cast<void>(hipEventRecord(gpuStart, nullptr));
    GpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yGpu, strides, dilations, padding);
    static_cast<void>(hipEventRecord(gpuStop, nullptr));
    static_cast<void>(hipEventSynchronize(gpuStop));

    float gpuMs = 0.0f;
    static_cast<void>(hipEventElapsedTime(&gpuMs, gpuStart, gpuStop));

    static_cast<void>(hipEventDestroy(gpuStart));
    static_cast<void>(hipEventDestroy(gpuStop));

    RecordProperty("cpu_ms", std::to_string(cpuMs));
    RecordProperty("gpu_ms", std::to_string(static_cast<double>(gpuMs)));

    assertAllClose(yCpu, yGpu, 1e-3f);
}

TEST(TestGpuConvFwdRefPerformance, DISABLED_LargeTensorTimingComparison)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> xDims = {8, 128, 28, 28};
    const std::vector<int64_t> wDims = {256, 128, 3, 3};
    const std::vector<int64_t> yDims = {8, 256, 26, 26};
    const std::vector<int64_t> strides = {1, 1};
    const std::vector<int64_t> dilations = {1, 1};
    const std::vector<int64_t> padding = {0, 0};

    Tensor<float> xTensor(xDims);
    Tensor<float> wTensor(wDims);
    Tensor<float> yCpu(yDims);
    Tensor<float> yGpu(yDims);

    const unsigned int seed = 456;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Warm-up run
    GpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yGpu, strides, dilations, padding);

    // Time CPU
    auto cpuStart = std::chrono::high_resolution_clock::now();
    CpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yCpu, strides, dilations, padding);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuUs = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count();
    auto cpuMs = static_cast<double>(cpuUs) / 1000.0;

    // Time GPU
    hipEvent_t gpuStart = nullptr;
    hipEvent_t gpuStop = nullptr;
    static_cast<void>(hipEventCreate(&gpuStart));
    static_cast<void>(hipEventCreate(&gpuStop));

    static_cast<void>(hipEventRecord(gpuStart, nullptr));
    GpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yGpu, strides, dilations, padding);
    static_cast<void>(hipEventRecord(gpuStop, nullptr));
    static_cast<void>(hipEventSynchronize(gpuStop));

    float gpuMs = 0.0f;
    static_cast<void>(hipEventElapsedTime(&gpuMs, gpuStart, gpuStop));

    static_cast<void>(hipEventDestroy(gpuStart));
    static_cast<void>(hipEventDestroy(gpuStop));

    RecordProperty("cpu_ms", std::to_string(cpuMs));
    RecordProperty("gpu_ms", std::to_string(static_cast<double>(gpuMs)));

    assertAllClose(yCpu, yGpu, 1e-2f);
}

// ============================================================================
// TestGpuConvFwdRefShapes — parameterized shape coverage across types
// TEST_P definitions for the fixture classes from the shared header.
// ============================================================================

TEST_P(TestGpuConvFwdRefShapesFp32, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRefShapesFp16, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRefShapesBfp16, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}

// ============================================================================
// Default layout (NCHW/NCDHW/NCW) instantiations — packed strides, no layout set.
// Only small/fast shapes here. Medium/large shapes are in the slow binary.
// ============================================================================

// fp32 NCHW/NCDHW
INSTANTIATE_TEST_SUITE_P(Small2d,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(getSmall2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small3d,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(getSmall3dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Medium3d,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(getMedium3dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });

// fp32 NCW: 1D shapes
INSTANTIATE_TEST_SUITE_P(Small1d,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(getSmall1dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });

// fp16 NCHW/NCDHW/NCW
INSTANTIATE_TEST_SUITE_P(Small2d,
                         TestGpuConvFwdRefShapesFp16,
                         ::testing::ValuesIn(getSmall2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small1d,
                         TestGpuConvFwdRefShapesFp16,
                         ::testing::ValuesIn(getSmall1dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small3d,
                         TestGpuConvFwdRefShapesFp16,
                         ::testing::ValuesIn(getSmall3dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });

// bfp16 NCHW/NCDHW/NCW
INSTANTIATE_TEST_SUITE_P(Small2d,
                         TestGpuConvFwdRefShapesBfp16,
                         ::testing::ValuesIn(getSmall2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small1d,
                         TestGpuConvFwdRefShapesBfp16,
                         ::testing::ValuesIn(getSmall1dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small3d,
                         TestGpuConvFwdRefShapesBfp16,
                         ::testing::ValuesIn(getSmall3dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });

// ============================================================================
// Channel-last (NHWC/NDHWC) instantiations — small shapes only.
// Medium/large channel-last shapes are in the slow binary.
// ============================================================================

// fp32 NHWC/NDHWC
INSTANTIATE_TEST_SUITE_P(Nhwc2dSmall,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dSmall,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dMedium,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });

// fp16 NHWC/NDHWC
INSTANTIATE_TEST_SUITE_P(Nhwc2dSmall,
                         TestGpuConvFwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dSmall,
                         TestGpuConvFwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });

// bfp16 NHWC/NDHWC
INSTANTIATE_TEST_SUITE_P(Nhwc2dSmall,
                         TestGpuConvFwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dSmall,
                         TestGpuConvFwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
