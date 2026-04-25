// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GpuConvWgradRefShapeCatalog.hpp"

using namespace gpu_conv_wgrad_ref_test;

// ============================================================================
// Slow (integration) weight-gradient (wgrad) shape tests.
// These are medium/large shapes previously DISABLED_ in the fast binary.
// Run via ninja integration-check or ctest -L slow.
// ============================================================================

// TEST_P definitions (must be in each binary that instantiates suites)

TEST_P(TestGpuConvWrwRefShapesFp32, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRefShapesFp16, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRefShapesBfp16, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}

// ============================================================================
// Default layout (NCW/NCHW/NCDHW) — medium/large shapes
// ============================================================================

// fp32
INSTANTIATE_TEST_SUITE_P(Medium1d,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(getMedium1dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Medium2d,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(getMedium2dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Medium3d,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(getMedium3dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large1d,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(getLarge1dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large2d,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(getLarge2dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large3d,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(getLarge3dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });

// fp16
INSTANTIATE_TEST_SUITE_P(Medium1d,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(getMedium1dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Medium2d,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(getMedium2dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Medium3d,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(getMedium3dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large1d,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(getLarge1dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large2d,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(getLarge2dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large3d,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(getLarge3dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });

// bfp16
INSTANTIATE_TEST_SUITE_P(Medium1d,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(getMedium1dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Medium2d,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(getMedium2dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Medium3d,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(getMedium3dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large1d,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(getLarge1dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large2d,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(getLarge2dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large3d,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(getLarge3dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });

// ============================================================================
// Channel-last (NLC/NHWC/NDHWC) — medium/large shapes
// ============================================================================

// fp32
INSTANTIATE_TEST_SUITE_P(Nlc1dMedium,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nhwc2dMedium,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dMedium,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nlc1dLarge,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge1dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nhwc2dLarge,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge2dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dLarge,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge3dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });

// fp16
INSTANTIATE_TEST_SUITE_P(Nlc1dMedium,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nhwc2dMedium,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dMedium,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nlc1dLarge,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge1dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nhwc2dLarge,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge2dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dLarge,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge3dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });

// bfp16
INSTANTIATE_TEST_SUITE_P(Nlc1dMedium,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nhwc2dMedium,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dMedium,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nlc1dLarge,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge1dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nhwc2dLarge,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge2dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dLarge,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge3dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
