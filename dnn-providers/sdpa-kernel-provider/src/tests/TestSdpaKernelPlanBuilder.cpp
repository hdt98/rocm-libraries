// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "SdpaKernelHandle.hpp"
#include "SdpaKernelPlanBuilder.hpp"

namespace sdpa_kernel_provider
{
namespace
{

class TestSdpaKernelPlanBuilder : public ::testing::Test
{
protected:
    SdpaKernelPlanBuilder _planBuilder;
    SdpaKernelHandle _handle;
};

TEST_F(TestSdpaKernelPlanBuilder, IsApplicableReturnsFalseForNonSdpaGraph)
{
    // Create a batchnorm inference graph - this does not use SDPA attributes
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaKernelPlanBuilder, IsApplicableReturnsTrueForSdpaGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFpropGraph();

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaKernelPlanBuilder, GetMaxWorkspaceSizeCalculatesCorrectly)
{
    // Create an SDPA graph with known dimensions
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFpropGraph();

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    // Get the workspace size from the plan builder
    SdpaKernelSettings settings;
    size_t workspaceSize = _planBuilder.getMaxWorkspaceSize(_handle, graphWrapper, settings);

    // Workspace = LSE buffer: [B, H_q, S_q] in float32
    // The test graph has standard dimensions, verify workspace is non-zero and reasonable
    // For typical SDPA graphs: B=1, H_q=8, S_q=128 → workspace = 1*8*128*4 = 4096 bytes
    EXPECT_GT(workspaceSize, 0u);
    EXPECT_TRUE(workspaceSize % 4 == 0);  // Should be multiple of sizeof(float)
}

} // namespace
} // namespace sdpa_kernel_provider
