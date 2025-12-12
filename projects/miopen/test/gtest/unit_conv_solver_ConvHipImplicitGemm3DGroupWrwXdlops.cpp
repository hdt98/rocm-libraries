// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "unit_conv_solver.hpp"

using TestDataType = miopen::unit_tests::TestDataType;

namespace {

// Template version of GetConvSmokeTestCases
template <TestDataType type>
auto GetConvSmokeTestCases()
{
    using TestCase                      = miopen::unit_tests::ConvTestCase;
    constexpr miopenDataType_t datatype = miopen::unit_tests::GetDataType(type);
    const bool tf32_compute             = type == TestDataType::TF32;

    return std::vector{
        // clang-format off
        TestCase{{datatype, miopenTensorNDHWC, {1, 64, 8, 8, 8}},
                 {datatype, miopenTensorNDHWC, {96, 64, 1, 1, 1}},
                 datatype, {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}, 1, false, tf32_compute}},
        // clang-format on
    };
}

// Template version of GetConvFullTestCases
template <TestDataType type>
auto GetConvFullTestCases()
{
    using TestCase                      = miopen::unit_tests::ConvTestCase;
    constexpr miopenDataType_t datatype = miopen::unit_tests::GetDataType(type);
    const bool tf32_compute             = type == TestDataType::TF32;

    return std::vector{
        // clang-format off
        TestCase{{datatype, miopenTensorNDHWC, {1, 64, 8, 8, 8}},
                    {datatype, miopenTensorNDHWC, {96, 64, 1, 1, 1}},
                    datatype, {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1, false, tf32_compute}}, // non-zero padding
        TestCase{{datatype, miopenTensorNDHWC, {1, 64, 8, 8, 8}},
                    {datatype, miopenTensorNDHWC, {96, 64, 1, 1, 1}},
                    datatype, {{0, 0, 0}, {2, 2, 2}, {1, 1, 1}, 1, false, tf32_compute}}, // stride > 1
        TestCase{{datatype, miopenTensorNDHWC, {1, 64, 8, 8, 8}},
                    {datatype, miopenTensorNDHWC, {96, 64, 1, 1, 1}},
                    datatype, {{0, 0, 0}, {1, 1, 1}, {2, 2, 2}, 1, false, tf32_compute}}, // dilation > 1
        TestCase{{datatype, miopenTensorNDHWC, {1, 64, 12, 24, 48}},
                    {datatype, miopenTensorNDHWC, {384, 64, 1, 1, 1}},
                    datatype, {{0, 0, 0}, {1, 1, 1}, {1, 1, 1}, 1, false, tf32_compute}}, // some different NCHW and k parameters
        // clang-format on
    };
}

// Template version of GetTestParams
template <TestDataType type>
const auto& GetTestParams()
{
    static const auto params = [&] {
// If MIOpen is built without CK these tests will fail, skip them to avoid failing
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
        Gpu supportedDevices;
        if constexpr(type == TestDataType::TF32 || type == TestDataType::BF16)
        {
            supportedDevices = Gpu::gfx94X | Gpu::gfx950;
        }
        else
        {
            supportedDevices = Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950;
        }
#else
        Gpu supportedDevices = Gpu::None;
#endif
        auto p = miopen::unit_tests::UnitTestConvSolverParams(supportedDevices);
        p.Tunable(5);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverHipImplicitGemm3DGroupWrwXdlops_FP16  = GPU_UnitTestConvSolverWrw_FP16;
using GPU_UnitTestConvSolverHipImplicitGemm3DGroupWrwXdlops_BFP16 = GPU_UnitTestConvSolverWrw_BFP16;
using GPU_UnitTestConvSolverHipImplicitGemm3DGroupWrwXdlops_FP32  = GPU_UnitTestConvSolverWrw_FP32;
using GPU_UnitTestConvSolverHipImplicitGemm3DGroupWrwXdlops_TF32  = GPU_UnitTestConvSolverWrw_TF32;
using CPU_UnitTestConvSolverHipImplicitGemm3DGroupWrwXdlopsDevApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;

TEST_P(GPU_UnitTestConvSolverHipImplicitGemm3DGroupWrwXdlops_FP16,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
};

TEST_P(GPU_UnitTestConvSolverHipImplicitGemm3DGroupWrwXdlops_BFP16,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
};

TEST_P(GPU_UnitTestConvSolverHipImplicitGemm3DGroupWrwXdlops_FP32,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
};
TEST_P(GPU_UnitTestConvSolverHipImplicitGemm3DGroupWrwXdlops_TF32,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
};
TEST_P(CPU_UnitTestConvSolverHipImplicitGemm3DGroupWrwXdlopsDevApplicability_NONE,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverHipImplicitGemm3DGroupWrwXdlops_FP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::FP16>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverHipImplicitGemm3DGroupWrwXdlops_BFP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::BF16>()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::BF16>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverHipImplicitGemm3DGroupWrwXdlops_FP32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP32>()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::FP32>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverHipImplicitGemm3DGroupWrwXdlops_TF32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::TF32>()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::TF32>())));

// Full tests
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverHipImplicitGemm3DGroupWrwXdlops_FP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::FP16>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverHipImplicitGemm3DGroupWrwXdlops_BFP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::BF16>()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::BF16>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverHipImplicitGemm3DGroupWrwXdlops_FP32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP32>()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::FP32>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverHipImplicitGemm3DGroupWrwXdlops_TF32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::TF32>()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::TF32>())));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverHipImplicitGemm3DGroupWrwXdlopsDevApplicability_NONE,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(GetConvSmokeTestCases<TestDataType::FP16>()[0])));
