// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GpuConvBwdRefShapeCatalog.hpp"

using namespace gpu_conv_bwd_ref_test;

// ============================================================================
// Slow (integration) backward-data (dgrad) shape tests.
// These are medium/large shapes previously DISABLED_ in the fast binary.
// Run via ninja integration-check or ctest -L slow.
// ============================================================================

// TEST_P definitions (must be in each binary that instantiates suites)

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
// Default layout (NCW/NCHW/NCDHW) — medium/large shapes
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
                         ::testing::ValuesIn(getMedium2dDgradCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Medium3d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(getMedium3dDgradCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large1d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(getLarge1dDgradCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large2d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(getLarge2dDgradCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large3d,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(getLarge3dDgradCases()),
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
                         ::testing::ValuesIn(getMedium2dDgradCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Medium3d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(getMedium3dDgradCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large1d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(getLarge1dDgradCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large2d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(getLarge2dDgradCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large3d,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(getLarge3dDgradCases()),
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
                         ::testing::ValuesIn(getMedium2dDgradCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Medium3d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(getMedium3dDgradCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large1d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(getLarge1dDgradCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large2d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(getLarge2dDgradCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large3d,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(getLarge3dDgradCases()),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });

// ============================================================================
// Channel-last (NLC/NHWC/NDHWC) — medium/large shapes
// ============================================================================

// fp32
INSTANTIATE_TEST_SUITE_P(Nlc1dMedium,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nhwc2dMedium,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dMedium,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nlc1dLarge,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge1dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nhwc2dLarge,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge2dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dLarge,
                         TestGpuConvBwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge3dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });

// fp16
INSTANTIATE_TEST_SUITE_P(Nlc1dMedium,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nhwc2dMedium,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dMedium,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nlc1dLarge,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge1dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nhwc2dLarge,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge2dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dLarge,
                         TestGpuConvBwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge3dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });

// bfp16
INSTANTIATE_TEST_SUITE_P(Nlc1dMedium,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nhwc2dMedium,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dMedium,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nlc1dLarge,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge1dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nhwc2dLarge,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge2dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dLarge,
                         TestGpuConvBwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge3dDgradCases())),
                         [](const ::testing::TestParamInfo<ConvBwdShapeCase>& info) {
                             return info.param.tag;
                         });
