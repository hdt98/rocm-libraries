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
    using namespace hipdnn_data_sdk::data_objects;
    using namespace hipdnn_data_sdk::utilities;

    // Create a batchnorm inference graph - this does not use SDPA attributes
    std::vector<int64_t> dims = {4, 8, 256, 128};
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph(
        hipdnn_data_sdk::utilities::generateStrides(dims), dims, DataType::BFLOAT16);

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaKernelPlanBuilder, IsApplicableReturnsTrueForSdpaGraph)
{
    using namespace hipdnn_data_sdk::data_objects;
    using namespace hipdnn_data_sdk::utilities;

    std::vector<int64_t> dims = {4, 8, 256, 128};
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFpropGraph(dims,
                                                                         generateStrides(dims),
                                                                         dims,
                                                                         generateStrides(dims),
                                                                         dims,
                                                                         generateStrides(dims),
                                                                         dims,
                                                                         generateStrides(dims),
                                                                         DataType::BFLOAT16);

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaKernelPlanBuilder, GetMaxWorkspaceSizeCalculatesCorrectly)
{
    // Create an SDPA graph with known dimensions (withStats = false by default)
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFpropGraph();

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    // Get the workspace size from the plan builder
    SdpaKernelSettings settings;
    size_t workspaceSize = _planBuilder.getMaxWorkspaceSize(_handle, graphWrapper, settings);

    // Forward-only kernel uses LDS internally, no external workspace needed
    // LSE (when present) is an optional output tensor (stats_tensor_uid), not workspace
    // The default test graph has withStats = false, so workspace should be 0
    EXPECT_EQ(workspaceSize, 0u);
}

} // namespace
} // namespace sdpa_kernel_provider
