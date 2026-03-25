// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffer_builder.h>
#include <gtest/gtest.h>

#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "SdpaKernelGraphCreation.hpp"
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

struct GraphTest
{
    std::shared_ptr<flatbuffers::DetachedBuffer> buffer;
    std::string message;

    GraphTest(flatbuffers::FlatBufferBuilder&& builder, std::string inMessage)
        : buffer(std::make_shared<flatbuffers::DetachedBuffer>(builder.Release()))
        , message(std::move(inMessage))
    {
    }

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graphWrapper() const
    {
        return hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper(buffer->data(), buffer->size());
    }
};

TEST_F(TestSdpaKernelPlanBuilder, IsApplicableSdpaVariations)
{
    using namespace hipdnn_data_sdk::data_objects;

    std::vector<std::pair<GraphTest, bool>> applicabilityTests = {
        {GraphTest{createValidSdpaFpropGraph(), "Valid test"}, true},
        {GraphTest{createValidSdpaFpropGraph({4, 8, 256, 100}), "Final dimension not 128"}, false},
        {GraphTest{createValidSdpaFpropGraph({4, 8, 256, 128}, DataType::HALF),
                   "Half precision tensor data type"},
         false},
        // TODO: Determine compute data type for this kernel and add corresponding test and check
        // {GraphTest{
        //      createValidSdpaFpropGraph({4, 8, 256, 128}, DataType::BFLOAT16, DataType::BFLOAT16),
        //      "Compute data type bfloat16"},
        //  false},
        {GraphTest{
             createValidSdpaFpropGraph({4, 8, 256, 128}, DataType::BFLOAT16, DataType::FLOAT, true),
             "attn_mask = true"},
         false},
        {GraphTest{createValidSdpaFpropGraph(
                       {4, 8, 256, 128}, DataType::BFLOAT16, DataType::FLOAT, false, true),
                   "scale = true"},
         true},
        {GraphTest{
             createValidSdpaFpropGraph(
                 {4, 8, 256, 128}, DataType::BFLOAT16, DataType::FLOAT, false, true, false, true),
             "alibi_mask = true"},
         false},
        {GraphTest{createValidSdpaFpropGraph({4, 8, 256, 128},
                                             DataType::BFLOAT16,
                                             DataType::FLOAT,
                                             false,
                                             true,
                                             false,
                                             false,
                                             true),
                   "padding_mask = true"},
         false},
        {GraphTest{createValidSdpaFpropGraph({4, 8, 256, 128},
                                             DataType::BFLOAT16,
                                             DataType::FLOAT,
                                             false,
                                             true,
                                             false,
                                             false,
                                             false,
                                             true),
                   "causal_mask = true"},
         false}};

    for(const auto& [test, applicability] : applicabilityTests)
    {
        EXPECT_EQ(_planBuilder.isApplicable(_handle, test.graphWrapper()), applicability)
            << test.message;
    }
}

TEST_F(TestSdpaKernelPlanBuilder, GetMaxWorkspaceSizeCalculatesCorrectly)
{
    // Create an SDPA graph with known dimensions (withStats = false by default)
    auto builder = createValidSdpaFpropGraph();

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
