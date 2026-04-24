// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GpuConvFwdRefShapeCatalog.hpp"

using namespace gpu_conv_fwd_ref_test;

// ============================================================================
// Slow (integration) forward convolution shape tests.
// These are medium/large shapes previously DISABLED_ in the fast binary.
// Run via ninja integration-check or ctest -L slow.
// ============================================================================

// TEST_P definitions (must be in each binary that instantiates suites)

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
// Default layout (NCHW) — medium/large shapes
// ============================================================================

// fp32
INSTANTIATE_TEST_SUITE_P(Medium2d,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(getMedium2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Large2d,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(getLarge2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });

// fp16
INSTANTIATE_TEST_SUITE_P(Medium2d,
                         TestGpuConvFwdRefShapesFp16,
                         ::testing::ValuesIn(getMedium2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });

// bfp16
INSTANTIATE_TEST_SUITE_P(Medium2d,
                         TestGpuConvFwdRefShapesBfp16,
                         ::testing::ValuesIn(getMedium2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });

// ============================================================================
// Channel-last (NHWC) — medium/large shapes
// ============================================================================

// fp32
INSTANTIATE_TEST_SUITE_P(Nhwc2dMedium,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nhwc2dLarge,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });

// fp16
INSTANTIATE_TEST_SUITE_P(Nhwc2dMedium,
                         TestGpuConvFwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });

// bfp16
INSTANTIATE_TEST_SUITE_P(Nhwc2dMedium,
                         TestGpuConvFwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
