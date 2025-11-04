/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */
/* SPDX-License-Identifier:  MIT */

#include <gtest/gtest.h>
#include <numeric>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/test_utils/MockGraph.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "engines/plans/MiopenBatchnormPlanBuilder.hpp"

#include "mocks/MockHipdnnEnginePluginExecutionContext.hpp"

using namespace miopen_legacy_plugin;
using namespace hipdnn_plugin;

class TestMiopenBatchnormPlanBuilder : public ::testing::Test
{
protected:
    MiopenBatchnormPlanBuilder _planBuilder;
    HipdnnEnginePluginHandle _dummyHandle;
};

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForMultiNodeGraph)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));

    bool applicable = _planBuilder.isApplicable(_dummyHandle, mockGraph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForUnsupportedAttributes)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillOnce(::testing::Return(1));
    EXPECT_CALL(mockGraph, hasOnlySupportedAttributes(::testing::_))
        .WillOnce(::testing::Return(false));

    bool applicable = _planBuilder.isApplicable(_dummyHandle, mockGraph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsTrueForSupportedSingleNodeGraph)
{
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormInferenceGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);

    EXPECT_TRUE(applicable);
}

TEST_F(TestMiopenBatchnormPlanBuilder, GetWorkspaceSizeReturnsExpectedValue)
{
    MockGraph mockGraph;

    size_t workspaceSize = _planBuilder.getWorkspaceSize(_dummyHandle, mockGraph);

    EXPECT_EQ(workspaceSize, 0u);
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanSetsPlanForSupportedNode)
{
    // Use a real flatbuffer graph with a valid batchnorm node
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormInferenceGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    // Should not throw
    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanThrowsForUnsupportedNodeType)
{
    // Create a graph with a node of unsupported type
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;

    // Node with NONE attributes type
    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder, "unsupported", hipdnn_sdk::data_objects::NodeAttributes::NONE, 0);
    nodes.push_back(node);

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::HALF,
                                                      hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graphOffset);

    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, graph, ctx),
                 hipdnn_plugin::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForBatchnormWithRunningStatistics)
{
    // Create a batchnorm training graph with all running statistics tensors
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;

    std::vector<int64_t> strides = {1, 3, 14, 14};
    std::vector<int64_t> dims = {1, 3, 14, 14};
    std::vector<int64_t> derivedStrides = {1, 3, 1, 1};
    std::vector<int64_t> derivedDims = {1, 3, 1, 1};

    // Required tensors
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    // Epsilon (pass-by-value)
    hipdnn_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributes(
        builder,
        5,
        builder.CreateString("epsilon"),
        hipdnn_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    // Running statistics tensors
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        8,
        "prev_running_mean",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        9,
        "prev_running_variance",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        10,
        "next_running_mean",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        11,
        "next_running_variance",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    // Momentum (pass-by-value)
    hipdnn_sdk::data_objects::Float32Value momentumVal(0.1f);
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributes(
        builder,
        12,
        builder.CreateString("momentum"),
        hipdnn_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(momentumVal).Union()));

    auto bnormAttributes = hipdnn_sdk::data_objects::CreateBatchnormAttributes(
        builder,
        1, // x_tensor_uid
        3, // scale_tensor_uid
        4, // bias_tensor_uid
        5, // epsilon_tensor_uid
        0, // peer_stats_tensor_uid (no peer statistics)
        flatbuffers::Optional<int64_t>(8), // prev_running_mean_tensor_uid
        flatbuffers::Optional<int64_t>(9), // prev_running_variance_tensor_uid
        flatbuffers::Optional<int64_t>(12), // momentum_tensor_uid
        2, // y_tensor_uid
        flatbuffers::nullopt, // mean_tensor_uid
        flatbuffers::nullopt, // inv_variance_tensor_uid
        flatbuffers::Optional<int64_t>(10), // next_running_mean_tensor_uid
        flatbuffers::Optional<int64_t>(11) // next_running_variance_tensor_uid
    );

    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_training_with_running_stats",
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::HALF,
                                                      hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graphOffset);

    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);

    EXPECT_FALSE(applicable);
}
