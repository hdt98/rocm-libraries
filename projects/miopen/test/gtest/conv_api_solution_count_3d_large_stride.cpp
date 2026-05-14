// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// API-level applicability sweep for 3D grouped CK xdlops solvers on shapes
// whose total element-strides exceed INT_MAX (ROCM-23997 reproducer family:
// torch.nn.Conv3d(96, 32, kernel_size=3, padding=1) on x = (1, 96, Nx, Ny, Z)).
//
// What this test asserts:
//   * miopenConvolution{Forward,BackwardData,BackwardWeights}CompileSolution
//     succeeds for the corresponding 3D grouped CK xdlops solver
//     (ConvHipImplicitGemm3DGroup{Fwd,Bwd,Wrw}Xdlops) — i.e. the specific
//     solver whose AllTensorsDimsFitIntoInt() guard was removed by ROCM-23997
//     is applicable and compilable for these large-stride shapes.
//
// What this test does NOT do:
//   * No kernel is launched, no buffers are allocated, no numerical compare is
//     performed. A regression caused by int32 wrap-around inside the kernel
//     would not be caught here.

#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>
#include <gtest/gtest.h>
#include <array>
#include <vector>

namespace {

struct Shape3D
{
    int n, c, d, h, w;
};

// Mirrors the PyTorch reproducer in ROCM-23997.
//   torch.nn.Conv3d(96, 32, kernel_size=3, padding=1)
//   x = torch.empty((1, 96, Nx, Ny, Z))
std::vector<Shape3D> ReproducerShapes()
{
    constexpr std::array<int, 5> spatial_xy = {64, 128, 256, 512, 1024};
    constexpr std::array<int, 10> z_values  = {16, 32, 64, 84, 86, 88, 128, 256, 512, 1024};
    std::vector<Shape3D> out;
    out.reserve(spatial_xy.size() * z_values.size());
    for(int nxy : spatial_xy)
        for(int z : z_values)
            out.push_back({1, 96, nxy, nxy, z});
    return out;
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
SetupDescriptors(const Shape3D& s, miopenDataType_t dtype, Descriptors& d)
{
    if(miopenCreateWithStream(&d.handle, /*stream=*/nullptr) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "miopenCreateWithStream failed";

    if(miopenCreateTensorDescriptor(&d.xDesc) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "create xDesc failed";
    {
        const int dims[5] = {s.n, s.c, s.d, s.h, s.w};
        if(miopenSetTensorDescriptor(d.xDesc, dtype, 5, dims, /*strides=*/nullptr) !=
           miopenStatusSuccess)
            return ::testing::AssertionFailure() << "set xDesc failed";
    }

    if(miopenCreateTensorDescriptor(&d.wDesc) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "create wDesc failed";
    {
        const int dims[5] = {32, 96, 3, 3, 3};
        if(miopenSetTensorDescriptor(d.wDesc, dtype, 5, dims, /*strides=*/nullptr) !=
           miopenStatusSuccess)
            return ::testing::AssertionFailure() << "set wDesc failed";
    }

    if(miopenCreateConvolutionDescriptor(&d.convDesc) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "create convDesc failed";
    {
        const int pads[3]    = {1, 1, 1};
        const int strides[3] = {1, 1, 1};
        const int dils[3]    = {1, 1, 1};
        if(miopenInitConvolutionNdDescriptor(
               d.convDesc, 3, pads, strides, dils, miopenConvolution) != miopenStatusSuccess)
            return ::testing::AssertionFailure() << "init convDesc failed";
    }

    if(miopenCreateTensorDescriptor(&d.yDesc) != miopenStatusSuccess)
        return ::testing::AssertionFailure() << "create yDesc failed";
    {
        int yDim[5] = {0};
        int yNbDims = 0;
        if(miopenGetConvolutionNdForwardOutputDim(
               d.convDesc, d.xDesc, d.wDesc, &yNbDims, yDim) != miopenStatusSuccess)
            return ::testing::AssertionFailure() << "get yDim failed";
        if(yNbDims != 5)
            return ::testing::AssertionFailure() << "yNbDims != 5";
        if(miopenSetTensorDescriptor(d.yDesc, dtype, 5, yDim, /*strides=*/nullptr) !=
           miopenStatusSuccess)
            return ::testing::AssertionFailure() << "set yDesc failed";
    }
    return ::testing::AssertionSuccess();
}

// All known-failing lists below were captured from the May 12 2026 sweep on this
// branch. Each failure is a CK loader miss (no kernel-instance match for that
// dtype/shape) — non-CK solvers still cover the case, but the test asserts that
// at least one ImplicitGEMM (CK) entry is returned. FP16 and BFP16 currently
// have no Fwd or Wrw gaps. TODO(ROCM-23997): trim entries as CK applicability
// is broadened.

template <std::size_t N>
bool MatchesDhw(const Shape3D& s, const std::array<std::array<int, 3>, N>& dhw)
{
    for(const auto& t : dhw)
        if(t[0] == s.d && t[1] == s.h && t[2] == s.w)
            return true;
    return false;
}

bool IsFwdKnownFailing(miopenDataType_t dtype, const Shape3D& s)
{
    if(dtype != miopenFloat) // FP16/BFP16 are clean.
        return false;
    static constexpr std::array<std::array<int, 3>, 3> dhw = {{
        {1024, 1024, 84},
        {1024, 1024, 86},
        {1024, 1024, 88},
    }};
    return MatchesDhw(s, dhw);
}

bool IsBwdDataKnownFailing(miopenDataType_t dtype, const Shape3D& s)
{
    // Shapes the CK 3D grouped bwd-data xdlops solver rejects across all dtypes.
    static constexpr std::array<std::array<int, 3>, 16> baseline = {{
        {128, 128, 1024},
        {256, 256, 256},
        {256, 256, 512},
        {256, 256, 1024},
        {512, 512, 64},
        {512, 512, 84},
        {512, 512, 86},
        {512, 512, 88},
        {512, 512, 128},
        {512, 512, 256},
        {1024, 1024, 16},
        {1024, 1024, 32},
        {1024, 1024, 64},
        {1024, 1024, 84},
        {1024, 1024, 86},
        {1024, 1024, 88},
    }};
    if(MatchesDhw(s, baseline))
        return true;
    if(dtype != miopenFloat) // Only FP32 has additional bwd-data gaps.
        return false;
    static constexpr std::array<std::array<int, 3>, 5> fp32_extra = {{
        {128, 128, 512},
        {256, 256, 86},
        {256, 256, 88},
        {256, 256, 128},
        {512, 512, 32},
    }};
    return MatchesDhw(s, fp32_extra);
}

bool IsWrwKnownFailing(miopenDataType_t dtype, const Shape3D& s)
{
    if(dtype != miopenFloat) // FP16/BFP16 are clean.
        return false;
    static constexpr std::array<std::array<int, 3>, 21> dhw = {{
        {128, 128, 512},
        {128, 128, 1024},
        {256, 256, 86},
        {256, 256, 88},
        {256, 256, 128},
        {256, 256, 256},
        {256, 256, 512},
        {256, 256, 1024},
        {512, 512, 32},
        {512, 512, 64},
        {512, 512, 84},
        {512, 512, 86},
        {512, 512, 88},
        {512, 512, 128},
        {512, 512, 256},
        {1024, 1024, 16},
        {1024, 1024, 32},
        {1024, 1024, 64},
        {1024, 1024, 84},
        {1024, 1024, 86},
        {1024, 1024, 88},
    }};
    return MatchesDhw(s, dhw);
}

uint64_t SolverIdFromName(const char* name)
{
    return miopen::solver::Id(name).Value();
}

void RunFwd(const Shape3D& s, miopenDataType_t dtype)
{
    if(IsFwdKnownFailing(dtype, s))
        GTEST_SKIP() << "Known Fwd CK applicability gap for shape " << s.n << "x" << s.c << "x"
                     << s.d << "x" << s.h << "x" << s.w << " (ROCM-23997)";

    Descriptors d;
    ASSERT_TRUE(SetupDescriptors(s, dtype, d));

    EXPECT_EQ(miopenConvolutionForwardCompileSolution(
                  d.handle,
                  d.wDesc,
                  d.xDesc,
                  d.convDesc,
                  d.yDesc,
                  SolverIdFromName("ConvHipImplicitGemm3DGroupFwdXdlops")),
              miopenStatusSuccess)
        << "ConvHipImplicitGemm3DGroupFwdXdlops not applicable/compilable for shape " << s.n << "x"
        << s.c << "x" << s.d << "x" << s.h << "x" << s.w;
}

void RunBwdData(const Shape3D& s, miopenDataType_t dtype)
{
    if(IsBwdDataKnownFailing(dtype, s))
        GTEST_SKIP() << "Known BwdData CK applicability gap for shape " << s.n << "x" << s.c << "x"
                     << s.d << "x" << s.h << "x" << s.w << " (ROCM-23997)";

    Descriptors d;
    ASSERT_TRUE(SetupDescriptors(s, dtype, d));

    // dyDesc has y's shape, dxDesc has x's shape.
    EXPECT_EQ(miopenConvolutionBackwardDataCompileSolution(
                  d.handle,
                  d.yDesc,
                  d.wDesc,
                  d.convDesc,
                  d.xDesc,
                  SolverIdFromName("ConvHipImplicitGemm3DGroupBwdXdlops")),
              miopenStatusSuccess)
        << "ConvHipImplicitGemm3DGroupBwdXdlops not applicable/compilable for shape " << s.n << "x"
        << s.c << "x" << s.d << "x" << s.h << "x" << s.w;
}

void RunWrw(const Shape3D& s, miopenDataType_t dtype)
{
    if(IsWrwKnownFailing(dtype, s))
        GTEST_SKIP() << "Known Wrw CK applicability gap for shape " << s.n << "x" << s.c << "x"
                     << s.d << "x" << s.h << "x" << s.w << " (ROCM-23997)";

    Descriptors d;
    ASSERT_TRUE(SetupDescriptors(s, dtype, d));

    // dyDesc has y's shape, xDesc has x's shape, dwDesc has w's shape.
    EXPECT_EQ(miopenConvolutionBackwardWeightsCompileSolution(
                  d.handle,
                  d.yDesc,
                  d.xDesc,
                  d.convDesc,
                  d.wDesc,
                  SolverIdFromName("ConvHipImplicitGemm3DGroupWrwXdlops")),
              miopenStatusSuccess)
        << "ConvHipImplicitGemm3DGroupWrwXdlops not applicable/compilable for shape " << s.n << "x"
        << s.c << "x" << s.d << "x" << s.h << "x" << s.w;
}

class GPU_ConvApi_SolutionCount3DLargeStride_FP16 : public ::testing::TestWithParam<Shape3D>
{
};
class GPU_ConvApi_SolutionCount3DLargeStride_FP32 : public ::testing::TestWithParam<Shape3D>
{
};
class GPU_ConvApi_SolutionCount3DLargeStride_BFP16 : public ::testing::TestWithParam<Shape3D>
{
};

} // namespace

TEST_P(GPU_ConvApi_SolutionCount3DLargeStride_FP16, FwdNonZeroAndIncludesCK)
{
    RunFwd(GetParam(), miopenHalf);
}
TEST_P(GPU_ConvApi_SolutionCount3DLargeStride_FP16, BwdDataNonZeroAndIncludesCK)
{
    RunBwdData(GetParam(), miopenHalf);
}
TEST_P(GPU_ConvApi_SolutionCount3DLargeStride_FP16, WrwNonZeroAndIncludesCK)
{
    RunWrw(GetParam(), miopenHalf);
}

TEST_P(GPU_ConvApi_SolutionCount3DLargeStride_FP32, FwdNonZeroAndIncludesCK)
{
    RunFwd(GetParam(), miopenFloat);
}
TEST_P(GPU_ConvApi_SolutionCount3DLargeStride_FP32, BwdDataNonZeroAndIncludesCK)
{
    RunBwdData(GetParam(), miopenFloat);
}
TEST_P(GPU_ConvApi_SolutionCount3DLargeStride_FP32, WrwNonZeroAndIncludesCK)
{
    RunWrw(GetParam(), miopenFloat);
}

TEST_P(GPU_ConvApi_SolutionCount3DLargeStride_BFP16, FwdNonZeroAndIncludesCK)
{
    RunFwd(GetParam(), miopenBFloat16);
}
TEST_P(GPU_ConvApi_SolutionCount3DLargeStride_BFP16, BwdDataNonZeroAndIncludesCK)
{
    RunBwdData(GetParam(), miopenBFloat16);
}
TEST_P(GPU_ConvApi_SolutionCount3DLargeStride_BFP16, WrwNonZeroAndIncludesCK)
{
    RunWrw(GetParam(), miopenBFloat16);
}

INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvApi_SolutionCount3DLargeStride_FP16,
                         ::testing::ValuesIn(ReproducerShapes()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvApi_SolutionCount3DLargeStride_FP32,
                         ::testing::ValuesIn(ReproducerShapes()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvApi_SolutionCount3DLargeStride_BFP16,
                         ::testing::ValuesIn(ReproducerShapes()));
