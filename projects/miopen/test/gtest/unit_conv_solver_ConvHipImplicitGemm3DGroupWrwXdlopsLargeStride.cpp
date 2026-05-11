// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "unit_conv_solver_3d_large_stride_shapes.hpp"

namespace {

using TestCase     = miopen::unit_tests::GroupXdlopsNumericData;
using TestDataType = miopen::unit_tests::TestDataType;

template <TestDataType type>
miopen::unit_tests::UnitTestConvSolverParams GetTestParams()
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    Gpu supportedDevices;
    if constexpr(type == TestDataType::FP32)
    {
        supportedDevices = Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950;
    }
    else if constexpr(type == TestDataType::TF32 || type == TestDataType::BF16)
    {
        supportedDevices = Gpu::gfx94X | Gpu::gfx950;
    }
    else
    {
        supportedDevices = Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950 | Gpu::gfx110X |
                           Gpu::gfx115X | Gpu::gfx120X;
    }
#else
    Gpu supportedDevices = Gpu::None;
#endif
    miopen::unit_tests::UnitTestConvSolverParams p(supportedDevices);
    p.ExcludeDevice("gfx1103");
    p.Tunable(5);
    p.UsesCKDynamicLib();

    p.SetTolerance(supportedDevices, miopenFloat, 30.0f);
    p.SetTolerance(Gpu::gfx110X | Gpu::gfx115X, miopenHalf, 4.0f);

    return p;
}

} // namespace

using GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsLargeStride_FP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenHalf>;
using GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsLargeStride_BFP16 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenBFloat16>;
using GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsLargeStride_FP32 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenFloat>;
using GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsLargeStride_TF32 =
    miopen::unit_tests::UnitTestConvSolverGroupXDlops<miopen::conv::Direction::BackwardWeights,
                                                      miopenFloat>;

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsLargeStride_FP16,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsLargeStride_BFP16,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsLargeStride_FP32,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
};

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsLargeStride_TF32,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
};

// Standard tests (PR validation): 1 canary shape near the int32 boundary.
INSTANTIATE_TEST_SUITE_P(
    Standard,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsLargeStride_FP16,
    testing::Combine(
        testing::Values(GetTestParams<TestDataType::FP16>()),
        testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
        testing::ValuesIn(miopen::unit_tests::GetLargeStride3DStandardShapes(false))));

INSTANTIATE_TEST_SUITE_P(
    Standard,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsLargeStride_BFP16,
    testing::Combine(
        testing::Values(GetTestParams<TestDataType::BF16>()),
        testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
        testing::ValuesIn(miopen::unit_tests::GetLargeStride3DStandardShapes(false))));

INSTANTIATE_TEST_SUITE_P(
    Standard,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsLargeStride_FP32,
    testing::Combine(
        testing::Values(GetTestParams<TestDataType::FP32>()),
        testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
        testing::ValuesIn(miopen::unit_tests::GetLargeStride3DStandardShapes(false))));

INSTANTIATE_TEST_SUITE_P(
    Standard,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsLargeStride_TF32,
    testing::Combine(
        testing::Values(GetTestParams<TestDataType::TF32>()),
        testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
        testing::ValuesIn(miopen::unit_tests::GetLargeStride3DStandardShapes(true))));

// Full tests (comprehensive/full categories): all 50 shapes.
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsLargeStride_FP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(miopen::unit_tests::GetLargeStride3DFullShapes(false))));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsLargeStride_BFP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::BF16>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(miopen::unit_tests::GetLargeStride3DFullShapes(false))));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsLargeStride_FP32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP32>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(miopen::unit_tests::GetLargeStride3DFullShapes(false))));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlopsLargeStride_TF32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::TF32>()),
                     testing::Values(miopenTensorNDHWC, miopenTensorNCDHW),
                     testing::ValuesIn(miopen::unit_tests::GetLargeStride3DFullShapes(true))));
