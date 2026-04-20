// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <unordered_map>
#include <vector>

#include "harness/CpuReferenceGraphExecutorAdapter.hpp"
#include "harness/IReferenceGraphExecutor.hpp"
#include "harness/ReferenceGraphExecutorFactory.hpp"
#include "harness/gpu_graph_executor/GpuReferenceGraphExecutor.hpp"

namespace
{

using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_integration_tests::ReferenceExecutorType;
using hipdnn_integration_tests::ReferenceGraphExecutorFactory;

// Creates a minimal pointwise graph with two FLOAT tensors (input + output).
flatbuffers::FlatBufferBuilder createSimplePointwiseGraph(int64_t inputUid,
                                                          int64_t outputUid,
                                                          const std::vector<int64_t>& dims,
                                                          const std::vector<int64_t>& strides)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, inputUid, "in_0", DataType::FLOAT, &strides, &dims));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, outputUid, "out_0", DataType::FLOAT, &strides, &dims));

    auto pointwiseAttrs
        = CreatePointwiseAttributes(builder,
                                    PointwiseMode::RELU_FWD,
                                    flatbuffers::nullopt, // relu_lower_clip
                                    flatbuffers::nullopt, // relu_upper_clip
                                    flatbuffers::nullopt, // relu_lower_clip_slope
                                    flatbuffers::nullopt, // axis_tensor_uid
                                    inputUid,
                                    flatbuffers::nullopt, // in_1_tensor_uid
                                    flatbuffers::nullopt, // in_2_tensor_uid
                                    outputUid);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "pointwise_node",
                                     DataType::FLOAT,
                                     NodeAttributes::PointwiseAttributes,
                                     pointwiseAttrs.Union()));

    auto graph = CreateGraphDirect(
        builder, "TestGraph", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);

    builder.Finish(graph);
    return builder;
}

} // namespace

// NOLINTBEGIN(readability-identifier-naming) -- gtest macro-generated names

TEST(TestReferenceGraphExecutorFactory, CreateCpuExecutor)
{
    auto executor = ReferenceGraphExecutorFactory::create(ReferenceExecutorType::CPU);
    ASSERT_NE(executor, nullptr);
    EXPECT_FALSE(executor->requiresDeviceMemory());
}

TEST(TestReferenceGraphExecutorFactory, CreateGpuExecutor)
{
    auto executor = ReferenceGraphExecutorFactory::create(ReferenceExecutorType::GPU);
    ASSERT_NE(executor, nullptr);
    EXPECT_TRUE(executor->requiresDeviceMemory());
}

TEST(TestReferenceGraphExecutorFactory, CpuExecutorExecutesGraph)
{
    auto executor = ReferenceGraphExecutorFactory::create(ReferenceExecutorType::CPU);

    constexpr int64_t INPUT_UID = 1;
    constexpr int64_t OUTPUT_UID = 2;
    const std::vector<int64_t> dims = {4};
    const std::vector<int64_t> strides = {1};

    auto builder = createSimplePointwiseGraph(INPUT_UID, OUTPUT_UID, dims, strides);

    std::array<float, 4> input = {2.0f, 3.0f, 5.0f, 7.0f};
    std::array<float, 4> output = {};

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[INPUT_UID] = input.data();
    variantPack[OUTPUT_UID] = output.data();

    executor->execute(builder.GetBufferPointer(), builder.GetSize(), variantPack);

    // CPU reference pointwise RELU_FWD: output = max(0, input), all inputs are positive
    for(size_t i = 0; i < input.size(); ++i)
    {
        EXPECT_FLOAT_EQ(output[i], input[i]) << "Mismatch at index " << i;
    }
}

TEST(TestReferenceGraphExecutorFactory, GpuExecutorRequiresDeviceMemory)
{
    auto executor = ReferenceGraphExecutorFactory::create(ReferenceExecutorType::GPU);
    EXPECT_TRUE(executor->requiresDeviceMemory());
}

TEST(TestReferenceGraphExecutorFactory, DefaultConfigReturnsCpu)
{
    // TestConfig is initialized by TestConfigInitialized (TestTestConfig.cpp) without a
    // reference executor type. When run with --gtest_filter that excludes TestConfigInitialized,
    // the singleton may not be initialized — skip in that case.
    try
    {
        static_cast<void>(hipdnn_integration_tests::TestConfig::get().getReferenceExecutorType());
    }
    catch(const std::runtime_error&)
    {
        GTEST_SKIP() << "TestConfig not initialized (requires TestConfigInitialized suite)";
    }

    auto executor = ReferenceGraphExecutorFactory::createFromConfig();
    ASSERT_NE(executor, nullptr);
    EXPECT_FALSE(executor->requiresDeviceMemory());
}

// NOLINTEND(readability-identifier-naming)
