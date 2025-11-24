/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "unit_conv_solver.hpp"

using TestDataType = miopen::unit_tests::TestDataType;

namespace {

// Template version of GetConvSmokeTestCases
template <TestDataType type>
auto GetConvSmokeTestCases()
{
    using TestCase                      = miopen::unit_tests::ConvTestCase;
    constexpr miopenDataType_t datatype = miopen::unit_tests::GetDataType(type);

    return std::vector{
        // clang-format off
        TestCase{{datatype, miopenTensorNHWC, {1, 64, 8, 8}},
                 {datatype, miopenTensorNHWC, {96, 64, 1, 1}},
                 datatype, {{0, 0}, {1, 1}, {1, 1}}},
        // clang-format on
    };
}

// Template version of GetConvFullTestCases
template <TestDataType type>
auto GetConvFullTestCases()
{
    using TestCase                      = miopen::unit_tests::ConvTestCase;
    constexpr miopenDataType_t datatype = miopen::unit_tests::GetDataType(type);

    return std::vector{
        // clang-format off
        TestCase{{datatype, miopenTensorNHWC, {1, 64, 8, 8}},
                    {datatype, miopenTensorNHWC, {96, 64, 1, 1}},
                    datatype, {{1, 1}, {1, 1}, {1, 1}}}, // non-zero padding
        TestCase{{datatype, miopenTensorNHWC, {1, 64, 8, 8}},
                    {datatype, miopenTensorNHWC, {96, 64, 1, 1}},
                    datatype, {{0, 0}, {2, 2}, {1, 1}}}, // stride > 1
        TestCase{{datatype, miopenTensorNHWC, {1, 64, 8, 8}},
                    {datatype, miopenTensorNHWC, {96, 64, 1, 1}},
                    datatype, {{0, 0}, {1, 1}, {2, 2}}}, // dilation > 1
        TestCase{{datatype, miopenTensorNHWC, {1, 64, 12, 24}},
                    {datatype, miopenTensorNHWC, {384, 64, 1, 1}},
                    datatype, {{0, 0}, {1, 1}, {1, 1}}}, // some different NCHW and k parameters
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
        if constexpr(type == TestDataType::TF32)
        {
            supportedDevices = Gpu::gfx94X;
        }
        else if constexpr(type == TestDataType::BF16)
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

using GPU_UnitTestConvSolverHipImplicitGemmGroupFwdXdlops_FP16  = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverHipImplicitGemmGroupFwdXdlops_BFP16 = GPU_UnitTestConvSolverFwd_BFP16;
using GPU_UnitTestConvSolverHipImplicitGemmGroupFwdXdlops_FP32  = GPU_UnitTestConvSolverFwd_FP32;
using GPU_UnitTestConvSolverHipImplicitGemmGroupFwdXdlops_TF32  = GPU_UnitTestConvSolverFwd_TF32;
using CPU_UnitTestConvSolverHipImplicitGemmGroupFwdXdlopsDevApplicability_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmGroupFwdXdlops_FP16, ConvHipImplicitGemmGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmGroupFwdXdlops_BFP16, ConvHipImplicitGemmGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverHipImplicitGemmGroupFwdXdlops_FP32, ConvHipImplicitGemmGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{});
};
TEST_P(GPU_UnitTestConvSolverHipImplicitGemmGroupFwdXdlops_TF32, ConvHipImplicitGemmGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{});
};
TEST_P(CPU_UnitTestConvSolverHipImplicitGemmGroupFwdXdlopsDevApplicability_NONE,
       ConvHipImplicitGemmGroupFwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverHipImplicitGemmGroupFwdXdlops_FP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::FP16>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverHipImplicitGemmGroupFwdXdlops_BFP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::BF16>()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::BF16>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverHipImplicitGemmGroupFwdXdlops_FP32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP32>()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::FP32>())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverHipImplicitGemmGroupFwdXdlops_TF32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::TF32>()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvSmokeTestCases<TestDataType::TF32>())));

// Full tests
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverHipImplicitGemmGroupFwdXdlops_FP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::FP16>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverHipImplicitGemmGroupFwdXdlops_BFP16,
    testing::Combine(testing::Values(GetTestParams<TestDataType::BF16>()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::BF16>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverHipImplicitGemmGroupFwdXdlops_FP32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP32>()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::FP32>())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverHipImplicitGemmGroupFwdXdlops_TF32,
    testing::Combine(testing::Values(GetTestParams<TestDataType::TF32>()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvFullTestCases<TestDataType::TF32>())));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverHipImplicitGemmGroupFwdXdlopsDevApplicability_NONE,
    testing::Combine(testing::Values(GetTestParams<TestDataType::FP16>()),
                     testing::Values(GetConvSmokeTestCases<TestDataType::FP16>()[0])));
