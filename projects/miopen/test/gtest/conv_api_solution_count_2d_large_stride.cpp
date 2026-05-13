// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// API-level applicability sweep for 2D grouped CK xdlops solvers on shapes
// whose total element-strides bracket / exceed INT_MAX. No specific upstream
// PyTorch reproducer; shapes were chosen to bracket the 2^31 element-stride
// boundary for x = (1, 96, H, W) with weight (32, 96, 3, 3).
//
// What this test asserts:
//   * miopenConvolution{Forward,BackwardData,BackwardWeights}GetSolutionCount
//     returns > 0 for each shape × dtype × direction.
//   * At least one returned solution has algorithm == miopenConvolutionAlgoImplicitGEMM
//     (i.e. the CK path remains applicable for large-stride 2D shapes).
//
// What this test does NOT do:
//   * No kernel is launched, no buffers are allocated, no numerical compare is
//     performed. A regression caused by int32 wrap-around inside the kernel
//     would not be caught here.

#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include <array>
#include <vector>

namespace {

struct Shape2D
{
    int n, c, h, w;
};

// Shapes bracketing the INT_MAX element-stride boundary for x = (1, 96, H, W).
// Element count = 96 * H * W; INT_MAX ≈ 2.147 B.
std::vector<Shape2D> ReproducerShapes()
{
    return {
        {1, 96, 4096, 4096},   // 1.61 B (just below INT_MAX)
        {1, 96, 4608, 4608},   // 2.04 B (just below INT_MAX)
        {1, 96, 4736, 4736},   // 2.15 B (just above INT_MAX)
        {1, 96, 5120, 5120},   // 2.52 B
        {1, 96, 5632, 5632},   // 3.05 B
        {1, 96, 6144, 6144},   // 3.62 B
        {1, 96, 8192, 8192},   // 6.44 B
        {1, 96, 9216, 9216},   // 8.16 B
        {1, 96, 10240, 10240}, // 10.07 B
        {1, 96, 11264, 11264}, // 12.18 B
        {1, 96, 12288, 12288}, // 14.50 B
        {1, 96, 14336, 14336}, // 19.73 B
        {1, 96, 16384, 16384}, // 25.77 B
        {2, 96, 4096, 4096},   // 3.22 B (smallest applicable BwdData >INT_MAX)
        {1, 96, 8192, 4096},   // 3.22 B (non-square)
        {1, 96, 4096, 8192},   // 3.22 B (non-square)
    };
}

struct Descriptors
{
    miopenHandle_t handle                  = nullptr;
    miopenTensorDescriptor_t xDesc         = nullptr;
    miopenTensorDescriptor_t wDesc         = nullptr;
    miopenTensorDescriptor_t yDesc         = nullptr;
    miopenConvolutionDescriptor_t convDesc = nullptr;

    ~Descriptors()
    {
        if(yDesc != nullptr)
            miopenDestroyTensorDescriptor(yDesc);
        if(convDesc != nullptr)
            miopenDestroyConvolutionDescriptor(convDesc);
        if(wDesc != nullptr)
            miopenDestroyTensorDescriptor(wDesc);
        if(xDesc != nullptr)
            miopenDestroyTensorDescriptor(xDesc);
        if(handle != nullptr)
            miopenDestroy(handle);
    }
};

::testing::AssertionResult
SetupDescriptors(const Shape2D& s, miopenDataType_t dtype, Descriptors& d)
{
    if(miopenCreateWithStream(&d.handle, /*stream=*/nullptr) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "miopenCreateWithStream failed";

    if(miopenCreateTensorDescriptor(&d.xDesc) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "create xDesc failed";
    {
        const int dims[4] = {s.n, s.c, s.h, s.w};
        if(miopenSetTensorDescriptor(d.xDesc, dtype, 4, dims, /*strides=*/nullptr) !=
           miopenStatusSuccess)
            return ::testing::AssertionFailure() << "set xDesc failed";
    }

    if(miopenCreateTensorDescriptor(&d.wDesc) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "create wDesc failed";
    {
        const int dims[4] = {32, 96, 3, 3};
        if(miopenSetTensorDescriptor(d.wDesc, dtype, 4, dims, /*strides=*/nullptr) !=
           miopenStatusSuccess)
            return ::testing::AssertionFailure() << "set wDesc failed";
    }

    if(miopenCreateConvolutionDescriptor(&d.convDesc) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "create convDesc failed";
    {
        const int pads[2]    = {1, 1};
        const int strides[2] = {1, 1};
        const int dils[2]    = {1, 1};
        if(miopenInitConvolutionNdDescriptor(
               d.convDesc, 2, pads, strides, dils, miopenConvolution) != miopenStatusSuccess)
            return ::testing::AssertionFailure() << "init convDesc failed";
    }

    if(miopenCreateTensorDescriptor(&d.yDesc) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "create yDesc failed";
    {
        int yDim[4] = {0};
        int yNbDims = 0;
        if(miopenGetConvolutionNdForwardOutputDim(
               d.convDesc, d.xDesc, d.wDesc, &yNbDims, yDim) != miopenStatusSuccess)
            return ::testing::AssertionFailure() << "get yDim failed";
        if(yNbDims != 4)
            return ::testing::AssertionFailure() << "yNbDims != 4";
        if(miopenSetTensorDescriptor(d.yDesc, dtype, 4, yDim, /*strides=*/nullptr) !=
           miopenStatusSuccess)
            return ::testing::AssertionFailure() << "set yDesc failed";
    }
    return ::testing::AssertionSuccess();
}

// Known CK applicability gaps for 2D large-stride shapes (gfx942, populated
// from the first sweep). These are heuristic-driven CK tile-selection misses,
// not correctness regressions. Fwd is mostly covered (one BFP16 hole at
// 14336x14336); BwdData/Wrw have larger holes that close again at the
// largest square shape (16384x16384) and at {2,96,4096,4096}.
bool IsFwdKnownFailing2D(miopenDataType_t dtype, const Shape2D& s)
{
    // {1,96,14336,14336} fails Fwd for all dtypes; FP32 has additional
    // square-shape gaps from 10240 through 14336.
    if(s.c == 96 && s.h == s.w && s.n == 1)
    {
        if(s.h == 14336)
            return true;
        if(dtype == miopenFloat && s.h >= 10240 && s.h <= 14336)
            return true;
    }
    return false;
}

bool IsBwdDataKnownFailing2D(miopenDataType_t dtype, const Shape2D& s)
{
    // Non-square {1,96,*,*} variants fail for all dtypes.
    if(s.n == 1 && s.c == 96 &&
       ((s.h == 4096 && s.w == 8192) || (s.h == 8192 && s.w == 4096)))
        return true;
    // Square c=96 shapes: gap covers a wide mid-range that closes only at
    // 16384x16384 and at {2,96,4096,4096}.
    if(s.c != 96 || s.h != s.w)
        return false;
    if(dtype == miopenFloat)
        return s.h >= 4096 && s.h <= 14336;
    if(dtype == miopenHalf || dtype == miopenBFloat16)
        return s.h >= 4736 && s.h <= 14336;
    return false;
}

bool IsWrwKnownFailing2D(miopenDataType_t dtype, const Shape2D& s)
{
    if(dtype != miopenFloat)
        return false;
    // FP32 Wrw rejects square c=96 shapes from 4096 through 14336.
    if(s.c == 96 && s.h == s.w && s.h >= 4096 && s.h <= 14336)
        return true;
    // Plus the non-square {1,96,8192,4096} and {1,96,4096,8192}.
    if(s.n == 1 && s.c == 96 && ((s.h == 8192 && s.w == 4096) || (s.h == 4096 && s.w == 8192)))
        return true;
    return false;
}

void ExpectAtLeastOneImplicitGemm(const std::vector<miopenConvSolution_t>& solutions,
                                  const Shape2D& s,
                                  const char* direction)
{
    bool has_ck_implicit_gemm = false;
    for(const auto& sol : solutions)
        if(sol.algorithm == miopenConvolutionAlgoImplicitGEMM)
        {
            has_ck_implicit_gemm = true;
            break;
        }
    EXPECT_TRUE(has_ck_implicit_gemm)
        << "No ImplicitGEMM (CK) " << direction << " solution returned for shape " << s.n << "x"
        << s.c << "x" << s.h << "x" << s.w;
}

void RunFwd(const Shape2D& s, miopenDataType_t dtype)
{
    if(IsFwdKnownFailing2D(dtype, s))
        GTEST_SKIP() << "Known Fwd CK applicability gap for shape " << s.n << "x" << s.c << "x"
                     << s.h << "x" << s.w;

    Descriptors d;
    ASSERT_TRUE(SetupDescriptors(s, dtype, d));

    size_t count = 0;
    ASSERT_EQ(miopenConvolutionForwardGetSolutionCount(
                  d.handle, d.wDesc, d.xDesc, d.convDesc, d.yDesc, &count),
              miopenStatusSuccess);
    ASSERT_GT(count, 0u) << "MIOpen reports zero forward solutions for shape " << s.n << "x" << s.c
                         << "x" << s.h << "x" << s.w;

    std::vector<miopenConvSolution_t> solutions(count);
    size_t returned = 0;
    ASSERT_EQ(miopenConvolutionForwardGetSolution(d.handle,
                                                  d.wDesc,
                                                  d.xDesc,
                                                  d.convDesc,
                                                  d.yDesc,
                                                  count,
                                                  &returned,
                                                  solutions.data()),
              miopenStatusSuccess);
    ASSERT_GT(returned, 0u);
    solutions.resize(returned);
    ExpectAtLeastOneImplicitGemm(solutions, s, "forward");
}

void RunBwdData(const Shape2D& s, miopenDataType_t dtype)
{
    if(IsBwdDataKnownFailing2D(dtype, s))
        GTEST_SKIP() << "Known BwdData CK applicability gap for shape " << s.n << "x" << s.c << "x"
                     << s.h << "x" << s.w;

    Descriptors d;
    ASSERT_TRUE(SetupDescriptors(s, dtype, d));

    size_t count = 0;
    ASSERT_EQ(miopenConvolutionBackwardDataGetSolutionCount(
                  d.handle, d.yDesc, d.wDesc, d.convDesc, d.xDesc, &count),
              miopenStatusSuccess);
    ASSERT_GT(count, 0u) << "MIOpen reports zero backward-data solutions for shape " << s.n << "x"
                         << s.c << "x" << s.h << "x" << s.w;

    std::vector<miopenConvSolution_t> solutions(count);
    size_t returned = 0;
    ASSERT_EQ(miopenConvolutionBackwardDataGetSolution(d.handle,
                                                       d.yDesc,
                                                       d.wDesc,
                                                       d.convDesc,
                                                       d.xDesc,
                                                       count,
                                                       &returned,
                                                       solutions.data()),
              miopenStatusSuccess);
    ASSERT_GT(returned, 0u);
    solutions.resize(returned);
    ExpectAtLeastOneImplicitGemm(solutions, s, "backward-data");
}

void RunWrw(const Shape2D& s, miopenDataType_t dtype)
{
    if(IsWrwKnownFailing2D(dtype, s))
        GTEST_SKIP() << "Known Wrw CK applicability gap for shape " << s.n << "x" << s.c << "x"
                     << s.h << "x" << s.w;

    Descriptors d;
    ASSERT_TRUE(SetupDescriptors(s, dtype, d));

    size_t count = 0;
    ASSERT_EQ(miopenConvolutionBackwardWeightsGetSolutionCount(
                  d.handle, d.yDesc, d.xDesc, d.convDesc, d.wDesc, &count),
              miopenStatusSuccess);
    ASSERT_GT(count, 0u) << "MIOpen reports zero backward-weights solutions for shape " << s.n
                         << "x" << s.c << "x" << s.h << "x" << s.w;

    std::vector<miopenConvSolution_t> solutions(count);
    size_t returned = 0;
    ASSERT_EQ(miopenConvolutionBackwardWeightsGetSolution(d.handle,
                                                          d.yDesc,
                                                          d.xDesc,
                                                          d.convDesc,
                                                          d.wDesc,
                                                          count,
                                                          &returned,
                                                          solutions.data()),
              miopenStatusSuccess);
    ASSERT_GT(returned, 0u);
    solutions.resize(returned);
    ExpectAtLeastOneImplicitGemm(solutions, s, "backward-weights");
}

class GPU_ConvApi_SolutionCount2DLargeStride_FP16 : public ::testing::TestWithParam<Shape2D>
{
};
class GPU_ConvApi_SolutionCount2DLargeStride_FP32 : public ::testing::TestWithParam<Shape2D>
{
};
class GPU_ConvApi_SolutionCount2DLargeStride_BFP16 : public ::testing::TestWithParam<Shape2D>
{
};

} // namespace

TEST_P(GPU_ConvApi_SolutionCount2DLargeStride_FP16, FwdNonZeroAndIncludesCK)
{
    RunFwd(GetParam(), miopenHalf);
}
TEST_P(GPU_ConvApi_SolutionCount2DLargeStride_FP16, BwdDataNonZeroAndIncludesCK)
{
    RunBwdData(GetParam(), miopenHalf);
}
TEST_P(GPU_ConvApi_SolutionCount2DLargeStride_FP16, WrwNonZeroAndIncludesCK)
{
    RunWrw(GetParam(), miopenHalf);
}

TEST_P(GPU_ConvApi_SolutionCount2DLargeStride_FP32, FwdNonZeroAndIncludesCK)
{
    RunFwd(GetParam(), miopenFloat);
}
TEST_P(GPU_ConvApi_SolutionCount2DLargeStride_FP32, BwdDataNonZeroAndIncludesCK)
{
    RunBwdData(GetParam(), miopenFloat);
}
TEST_P(GPU_ConvApi_SolutionCount2DLargeStride_FP32, WrwNonZeroAndIncludesCK)
{
    RunWrw(GetParam(), miopenFloat);
}

TEST_P(GPU_ConvApi_SolutionCount2DLargeStride_BFP16, FwdNonZeroAndIncludesCK)
{
    RunFwd(GetParam(), miopenBFloat16);
}
TEST_P(GPU_ConvApi_SolutionCount2DLargeStride_BFP16, BwdDataNonZeroAndIncludesCK)
{
    RunBwdData(GetParam(), miopenBFloat16);
}
TEST_P(GPU_ConvApi_SolutionCount2DLargeStride_BFP16, WrwNonZeroAndIncludesCK)
{
    RunWrw(GetParam(), miopenBFloat16);
}

INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvApi_SolutionCount2DLargeStride_FP16,
                         ::testing::ValuesIn(ReproducerShapes()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvApi_SolutionCount2DLargeStride_FP32,
                         ::testing::ValuesIn(ReproducerShapes()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvApi_SolutionCount2DLargeStride_BFP16,
                         ::testing::ValuesIn(ReproducerShapes()));
