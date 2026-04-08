// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
<<<<<<< HEAD
#include <string>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/node/ConvolutionWgradNode.hpp>
#include <hipdnn_test_sdk/constants/ConvWgradConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LiftingTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::liftGraph;
using hipdnn_tests::liftGraphWithoutFinalization;
using hipdnn_tests::TestableGraphLifting;
=======
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/ConvolutionWgradNode.hpp>
#include <hipdnn_test_sdk/constants/ConvWgradConstants.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants::conv_wgrad;
>>>>>>> d9e199e220 (merge b-shi branch)
using hipdnn_tests::toVec;

namespace
{
<<<<<<< HEAD
// Lifts a frontend graph via build_operation_graph(handle), then
// reconstructs it with fromBackendDescriptor() for verification.
class IntegrationConvolutionWgradDescriptorLifting : public IntegrationTestFixture
{
protected:
    /// Builds a standard ConvolutionWrw graph for round-trip testing.
    static std::shared_ptr<TestableGraphLifting> buildGraph()
    {
        auto graph = std::make_shared<TestableGraphLifting>();
=======

// Exposes protected Graph methods for lifting integration tests
class TestableGraph : public Graph
{
public:
    using Graph::build_operation_graph;
    using Graph::deserialize_via_backend;
    using Graph::fromBackendDescriptor;
    using Graph::get_raw_graph_descriptor;

    const std::vector<std::shared_ptr<INode>>& getSubNodes() const
    {
        return _sub_nodes;
    }
};

// Lifts a frontend graph via build_operation_graph(handle), then
// reconstructs it with fromBackendDescriptor() for verification.
class IntegrationConvolutionWgradDescriptorLifting : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);

        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            hipdnnDestroy(_handle);
        }
    }

    /// Builds a standard ConvolutionWrw graph for round-trip testing.
    static std::shared_ptr<TestableGraph> buildGraph()
    {
        auto graph = std::make_shared<TestableGraph>();
>>>>>>> d9e199e220 (merge b-shi branch)
        graph->set_name("ConvolutionWrwLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto dy = std::make_shared<TensorAttributes>();
<<<<<<< HEAD
        dy->set_uid(K_WGRAD_TENSOR_DY_UID).set_name("dy").set_data_type(DataType::FLOAT);
        dy->set_dim(toVec(K_WGRAD_TENSOR_DY_DIMS)).set_stride(toVec(K_WGRAD_TENSOR_DY_STRIDES));

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_WGRAD_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_WGRAD_TENSOR_X_DIMS)).set_stride(toVec(K_WGRAD_TENSOR_X_STRIDES));
=======
        dy->set_uid(K_TENSOR_DY_UID).set_name("dy").set_data_type(DataType::FLOAT);
        dy->set_dim(toVec(K_TENSOR_DY_DIMS)).set_stride(toVec(K_TENSOR_DY_STRIDES));

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_TENSOR_X_DIMS)).set_stride(toVec(K_TENSOR_X_STRIDES));
>>>>>>> d9e199e220 (merge b-shi branch)

        ConvWgradAttributes attrs;
        attrs.set_name("test_op");
        attrs.set_convolution_mode(ConvolutionMode::CONVOLUTION);
<<<<<<< HEAD
        attrs.set_pre_padding(toVec(K_WGRAD_CONV_PADDING));
        attrs.set_post_padding(toVec(K_WGRAD_CONV_PADDING));
        attrs.set_stride(toVec(K_WGRAD_CONV_STRIDE));
        attrs.set_dilation(toVec(K_WGRAD_CONV_DILATION));

        auto dw = graph->conv_wgrad(dy, x, attrs);
        dw->set_uid(K_WGRAD_TENSOR_DW_UID).set_output(true).set_name("dw");

        return graph;
    }
=======
        attrs.set_pre_padding(toVec(K_CONV_PADDING));
        attrs.set_post_padding(toVec(K_CONV_PADDING));
        attrs.set_stride(toVec(K_CONV_STRIDE));
        attrs.set_dilation(toVec(K_CONV_DILATION));

        auto dw = graph->conv_wgrad(dy, x, attrs);
        dw->set_uid(K_TENSOR_DW_UID).set_output(true).set_name("dw");

        return graph;
    }

    hipdnnHandle_t _handle = nullptr;
>>>>>>> d9e199e220 (merge b-shi branch)
};

// Builds a standard ConvolutionWrw graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationConvolutionWgradDescriptorLifting, BasicConvolutionWrwRoundTrip)
{
    auto originalGraph = buildGraph();

<<<<<<< HEAD
    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);
=======
    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
>>>>>>> d9e199e220 (merge b-shi branch)

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);

    // Verify x tensor
<<<<<<< HEAD
    ASSERT_NE(tensorMap.count(K_WGRAD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_X_UID]->get_uid(), K_WGRAD_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_X_UID]->get_dim(), toVec(K_WGRAD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_X_UID]->get_stride(), toVec(K_WGRAD_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_X_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_X_UID]->get_name(), "x");

    // Verify dy tensor
    ASSERT_NE(tensorMap.count(K_WGRAD_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_DY_UID]->get_uid(), K_WGRAD_TENSOR_DY_UID);
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_DY_UID]->get_dim(), toVec(K_WGRAD_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_DY_UID]->get_stride(), toVec(K_WGRAD_TENSOR_DY_STRIDES));
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_DY_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_DY_UID]->get_name(), "dy");

    // Verify dw tensor (dims/strides are inferred by infer_properties)
    ASSERT_NE(tensorMap.count(K_WGRAD_TENSOR_DW_UID), 0u);
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_DW_UID]->get_uid(), K_WGRAD_TENSOR_DW_UID);
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_DW_UID]->get_dim(), toVec(K_WGRAD_TENSOR_DW_DIMS));
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_DW_UID]->get_stride(), toVec(K_WGRAD_TENSOR_DW_STRIDES));
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_DW_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_DW_UID]->get_name(), "dw");

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    auto* opNode = dynamic_cast<ConvolutionWgradNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a ConvolutionWgradNode"; // NOLINT(readability-implicit-bool-conversion)
=======
    ASSERT_NE(tensorMap.count(K_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_uid(), K_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_dim(), toVec(K_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_stride(), toVec(K_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_name(), "x");

    // Verify dy tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_uid(), K_TENSOR_DY_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_dim(), toVec(K_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_stride(), toVec(K_TENSOR_DY_STRIDES));
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_name(), "dy");

    // Verify dw tensor (dims/strides are inferred by infer_properties)
    ASSERT_NE(tensorMap.count(K_TENSOR_DW_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_DW_UID]->get_uid(), K_TENSOR_DW_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_DW_UID]->get_dim(), toVec(K_TENSOR_DW_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_DW_UID]->get_stride(), toVec(K_TENSOR_DW_STRIDES));
    EXPECT_EQ(tensorMap[K_TENSOR_DW_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_TENSOR_DW_UID]->get_name(), "dw");

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* opNode = dynamic_cast<ConvolutionWgradNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr) << "Expected a ConvolutionWgradNode";
>>>>>>> d9e199e220 (merge b-shi branch)

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_convolution_mode(), ConvolutionMode::CONVOLUTION);

    // Verify pre_padding
<<<<<<< HEAD
    EXPECT_EQ(opNode->attributes.get_pre_padding(), toVec(K_WGRAD_CONV_PADDING));
    // Verify post_padding
    EXPECT_EQ(opNode->attributes.get_post_padding(), toVec(K_WGRAD_CONV_PADDING));
    // Verify stride
    EXPECT_EQ(opNode->attributes.get_stride(), toVec(K_WGRAD_CONV_STRIDE));
    // Verify dilation
    EXPECT_EQ(opNode->attributes.get_dilation(), toVec(K_WGRAD_CONV_DILATION));
=======
    EXPECT_EQ(opNode->attributes.get_pre_padding(), toVec(K_CONV_PADDING));
    // Verify post_padding
    EXPECT_EQ(opNode->attributes.get_post_padding(), toVec(K_CONV_PADDING));
    // Verify stride
    EXPECT_EQ(opNode->attributes.get_stride(), toVec(K_CONV_STRIDE));
    // Verify dilation
    EXPECT_EQ(opNode->attributes.get_dilation(), toVec(K_CONV_DILATION));
>>>>>>> d9e199e220 (merge b-shi branch)

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationConvolutionWgradDescriptorLifting, ConvolutionWrwTensorSharingPreserved)
{
    auto originalGraph = buildGraph();

<<<<<<< HEAD
    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);
=======
    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
>>>>>>> d9e199e220 (merge b-shi branch)

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<ConvolutionWgradNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify x tensor sharing
<<<<<<< HEAD
    EXPECT_EQ(opNode->attributes.get_x()->get_uid(), K_WGRAD_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_X_UID].get(), opNode->attributes.get_x().get());
    // Verify dy tensor sharing
    EXPECT_EQ(opNode->attributes.get_dy()->get_uid(), K_WGRAD_TENSOR_DY_UID);
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_DY_UID].get(), opNode->attributes.get_dy().get());
    // Verify dw tensor sharing
    EXPECT_EQ(opNode->attributes.get_dw()->get_uid(), K_WGRAD_TENSOR_DW_UID);
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_DW_UID].get(), opNode->attributes.get_dw().get());
=======
    EXPECT_EQ(opNode->attributes.get_x()->get_uid(), K_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID].get(), opNode->attributes.get_x().get());
    // Verify dy tensor sharing
    EXPECT_EQ(opNode->attributes.get_dy()->get_uid(), K_TENSOR_DY_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID].get(), opNode->attributes.get_dy().get());
    // Verify dw tensor sharing
    EXPECT_EQ(opNode->attributes.get_dw()->get_uid(), K_TENSOR_DW_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_DW_UID].get(), opNode->attributes.get_dw().get());
>>>>>>> d9e199e220 (merge b-shi branch)

    // Verify tensor names
    EXPECT_EQ(opNode->attributes.get_x()->get_name(), "x");
    EXPECT_EQ(opNode->attributes.get_dy()->get_name(), "dy");
    EXPECT_EQ(opNode->attributes.get_dw()->get_name(), "dw");
}

// Builds a ConvolutionWrw graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
<<<<<<< HEAD
// all fields survive the backend C API serialization path.
=======
// all fields survive the FlatBuffer-direct path.
>>>>>>> d9e199e220 (merge b-shi branch)
TEST_F(IntegrationConvolutionWgradDescriptorLifting, ConvolutionWrwLiftWithoutFinalization)
{
    auto originalGraph = buildGraph();

<<<<<<< HEAD
    auto liftedGraph = liftGraphWithoutFinalization(*originalGraph);
    ASSERT_NE(liftedGraph, nullptr);
=======
    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Serialize to binary via the frontend
    auto data = originalGraph->toBinary();
    ASSERT_FALSE(data.empty());

    // Create a backend graph descriptor from serialized bytes (no handle, no finalize)
    const detail::ScopedHipdnnBackendDescriptor graphDesc(data.data(), data.size());
    ASSERT_TRUE(graphDesc.valid()) << "Failed to create backend graph descriptor";

    // Lift into a new graph via fromBackendDescriptor
    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(graphDesc.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
>>>>>>> d9e199e220 (merge b-shi branch)

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify the lifted graph has 1 operation node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<ConvolutionWgradNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_convolution_mode(), ConvolutionMode::CONVOLUTION);

    // Verify pre_padding
<<<<<<< HEAD
    EXPECT_EQ(opNode->attributes.get_pre_padding(), toVec(K_WGRAD_CONV_PADDING));
    // Verify post_padding
    EXPECT_EQ(opNode->attributes.get_post_padding(), toVec(K_WGRAD_CONV_PADDING));
    // Verify stride
    EXPECT_EQ(opNode->attributes.get_stride(), toVec(K_WGRAD_CONV_STRIDE));
    // Verify dilation
    EXPECT_EQ(opNode->attributes.get_dilation(), toVec(K_WGRAD_CONV_DILATION));
=======
    EXPECT_EQ(opNode->attributes.get_pre_padding(), toVec(K_CONV_PADDING));
    // Verify post_padding
    EXPECT_EQ(opNode->attributes.get_post_padding(), toVec(K_CONV_PADDING));
    // Verify stride
    EXPECT_EQ(opNode->attributes.get_stride(), toVec(K_CONV_STRIDE));
    // Verify dilation
    EXPECT_EQ(opNode->attributes.get_dilation(), toVec(K_CONV_DILATION));
>>>>>>> d9e199e220 (merge b-shi branch)

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);

<<<<<<< HEAD
    ASSERT_NE(tensorMap.count(K_WGRAD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_X_UID]->get_dim(), toVec(K_WGRAD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_X_UID]->get_stride(), toVec(K_WGRAD_TENSOR_X_STRIDES));
    ASSERT_NE(tensorMap.count(K_WGRAD_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_DY_UID]->get_dim(), toVec(K_WGRAD_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_DY_UID]->get_stride(), toVec(K_WGRAD_TENSOR_DY_STRIDES));
    ASSERT_NE(tensorMap.count(K_WGRAD_TENSOR_DW_UID), 0u);
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_DW_UID]->get_dim(), toVec(K_WGRAD_TENSOR_DW_DIMS));
    EXPECT_EQ(tensorMap[K_WGRAD_TENSOR_DW_UID]->get_stride(), toVec(K_WGRAD_TENSOR_DW_STRIDES));
=======
    ASSERT_NE(tensorMap.count(K_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_dim(), toVec(K_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_stride(), toVec(K_TENSOR_X_STRIDES));
    ASSERT_NE(tensorMap.count(K_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_dim(), toVec(K_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_stride(), toVec(K_TENSOR_DY_STRIDES));
    ASSERT_NE(tensorMap.count(K_TENSOR_DW_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_DW_UID]->get_dim(), toVec(K_TENSOR_DW_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_DW_UID]->get_stride(), toVec(K_TENSOR_DW_STRIDES));
>>>>>>> d9e199e220 (merge b-shi branch)
}

// Creates tensors without explicit set_uid(), verifies that auto-assigned UIDs
// survive the lifting round trip and are all distinct.
TEST_F(IntegrationConvolutionWgradDescriptorLifting, AutoAssignedUidsPreservedInLiftingRoundTrip)
{
    constexpr std::array<int64_t, 4> K_AUTO_X_DIMS = {1, 4, 8, 8};
    constexpr std::array<int64_t, 4> K_AUTO_X_STRIDES = {256, 64, 8, 1};
    constexpr std::array<int64_t, 4> K_AUTO_DY_DIMS = {1, 16, 6, 6};
    constexpr std::array<int64_t, 4> K_AUTO_DY_STRIDES = {576, 36, 6, 1};

<<<<<<< HEAD
    auto graph = std::make_shared<TestableGraphLifting>();
=======
    auto graph = std::make_shared<TestableGraph>();
>>>>>>> d9e199e220 (merge b-shi branch)
    graph->set_name("AutoUidWgradLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_AUTO_X_DIMS)).set_stride(toVec(K_AUTO_X_STRIDES));

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_name("dy").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_AUTO_DY_DIMS)).set_stride(toVec(K_AUTO_DY_STRIDES));

    ConvWgradAttributes convAttrs;
    convAttrs.set_name("auto_uid_wgrad_op");
    convAttrs.set_pre_padding({1, 1});
    convAttrs.set_post_padding({1, 1});
    convAttrs.set_stride({1, 1});
    convAttrs.set_dilation({1, 1});
    convAttrs.set_convolution_mode(ConvolutionMode::CONVOLUTION);

    auto dw = graph->conv_wgrad(dy, x, convAttrs);
    dw->set_output(true).set_name("dw");

<<<<<<< HEAD
    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u)
        << "Expected 3 tensors in lifted graph"; // NOLINT(readability-implicit-bool-conversion)
=======
    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u) << "Expected 3 tensors in lifted graph";
>>>>>>> d9e199e220 (merge b-shi branch)

    // Collect all UIDs and verify they are distinct
    std::vector<int64_t> uids;
    uids.reserve(tensorMap.size());
    for(const auto& [uid, tensor] : tensorMap)
    {
        uids.push_back(uid);
    }
    std::sort(uids.begin(), uids.end());
    EXPECT_EQ(std::adjacent_find(uids.begin(), uids.end()), uids.end())
<<<<<<< HEAD
        << "All auto-assigned UIDs must be distinct"; // NOLINT(readability-implicit-bool-conversion)
=======
        << "All auto-assigned UIDs must be distinct";
>>>>>>> d9e199e220 (merge b-shi branch)

    // Verify the node references tensors with auto-assigned UIDs
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<ConvolutionWgradNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify tensor dims survived the round trip
    auto xUid = opNode->attributes.get_x()->get_uid();
    auto dyUid = opNode->attributes.get_dy()->get_uid();
    auto dwUid = opNode->attributes.get_dw()->get_uid();

    EXPECT_NE(xUid, dyUid);
    EXPECT_NE(xUid, dwUid);
    EXPECT_NE(dyUid, dwUid);

    EXPECT_EQ(tensorMap[xUid]->get_dim(), toVec(K_AUTO_X_DIMS));
    EXPECT_EQ(tensorMap[dyUid]->get_dim(), toVec(K_AUTO_DY_DIMS));
}

// Builds a conv wgrad graph with asymmetric padding (pre_padding={1,1},
// post_padding={2,2}), lowers, lifts, and verifies asymmetric padding
// values survive the round trip.
TEST_F(IntegrationConvolutionWgradDescriptorLifting, AsymmetricPaddingPreservedInLiftingRoundTrip)
{
    constexpr std::array<int64_t, 2> K_ASYM_PRE_PADDING = {1, 1};
    constexpr std::array<int64_t, 2> K_ASYM_POST_PADDING = {2, 2};

<<<<<<< HEAD
    auto graph = std::make_shared<TestableGraphLifting>();
=======
    auto graph = std::make_shared<TestableGraph>();
>>>>>>> d9e199e220 (merge b-shi branch)
    graph->set_name("AsymPaddingWgradLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
<<<<<<< HEAD
    dy->set_uid(K_WGRAD_TENSOR_DY_UID).set_name("dy").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_WGRAD_TENSOR_DY_DIMS)).set_stride(toVec(K_WGRAD_TENSOR_DY_STRIDES));

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_WGRAD_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_WGRAD_TENSOR_X_DIMS)).set_stride(toVec(K_WGRAD_TENSOR_X_STRIDES));
=======
    dy->set_uid(K_TENSOR_DY_UID).set_name("dy").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_TENSOR_DY_DIMS)).set_stride(toVec(K_TENSOR_DY_STRIDES));

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_TENSOR_X_DIMS)).set_stride(toVec(K_TENSOR_X_STRIDES));
>>>>>>> d9e199e220 (merge b-shi branch)

    ConvWgradAttributes convAttrs;
    convAttrs.set_name("asym_wgrad_op");
    convAttrs.set_pre_padding(toVec(K_ASYM_PRE_PADDING));
    convAttrs.set_post_padding(toVec(K_ASYM_POST_PADDING));
    convAttrs.set_stride({1, 1});
    convAttrs.set_dilation({1, 1});
    convAttrs.set_convolution_mode(ConvolutionMode::CONVOLUTION);

    auto dw = graph->conv_wgrad(dy, x, convAttrs);
<<<<<<< HEAD
    dw->set_uid(K_WGRAD_TENSOR_DW_UID).set_output(true).set_name("dw");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);
=======
    dw->set_uid(K_TENSOR_DW_UID).set_output(true).set_name("dw");

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
>>>>>>> d9e199e220 (merge b-shi branch)

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<ConvolutionWgradNode*>(subNodes[0].get());
<<<<<<< HEAD
    ASSERT_NE(opNode, nullptr)
        << "Expected a ConvolutionWgradNode"; // NOLINT(readability-implicit-bool-conversion)
=======
    ASSERT_NE(opNode, nullptr) << "Expected a ConvolutionWgradNode";
>>>>>>> d9e199e220 (merge b-shi branch)

    EXPECT_EQ(opNode->attributes.get_pre_padding(), toVec(K_ASYM_PRE_PADDING));
    EXPECT_EQ(opNode->attributes.get_post_padding(), toVec(K_ASYM_POST_PADDING));
    EXPECT_EQ(opNode->attributes.get_stride(), std::vector<int64_t>({1, 1}));
    EXPECT_EQ(opNode->attributes.get_dilation(), std::vector<int64_t>({1, 1}));
}

<<<<<<< HEAD
// Exercises the JSON serialize/deserialize path with a handle (full finalization)
// for a conv wgrad graph.
TEST_F(IntegrationConvolutionWgradDescriptorLifting, JsonRoundTripWithHandle)
{
    auto originalGraph = buildGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Serialize to JSON (auto-lowers internally)
    std::string jsonData;
    result = originalGraph->serialize(jsonData);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_FALSE(jsonData.empty());

    // Deserialize from JSON with handle
    auto liftedGraph = std::make_shared<TestableGraphLifting>();
    result = liftedGraph->deserialize(_handle, jsonData);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify graph-level attributes
    EXPECT_EQ(liftedGraph->get_name(), "ConvolutionWrwLiftingTestGraph");
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u) << "Expected 3 tensors (dy, x, dw)";

    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_WGRAD_TENSOR_DY_UID,
                                      "dy",
                                      toVec(K_WGRAD_TENSOR_DY_DIMS),
                                      toVec(K_WGRAD_TENSOR_DY_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_WGRAD_TENSOR_X_UID,
                                      "x",
                                      toVec(K_WGRAD_TENSOR_X_DIMS),
                                      toVec(K_WGRAD_TENSOR_X_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_WGRAD_TENSOR_DW_UID,
                                      "dw",
                                      toVec(K_WGRAD_TENSOR_DW_DIMS),
                                      toVec(K_WGRAD_TENSOR_DW_STRIDES),
                                      DataType::FLOAT);

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* opNode = dynamic_cast<ConvolutionWgradNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr) << "Expected a ConvolutionWgradNode";

    // Verify convolution parameters
    EXPECT_EQ(opNode->attributes.get_convolution_mode(), ConvolutionMode::CONVOLUTION);
    EXPECT_EQ(opNode->attributes.get_pre_padding(), toVec(K_WGRAD_CONV_PADDING));
    EXPECT_EQ(opNode->attributes.get_post_padding(), toVec(K_WGRAD_CONV_PADDING));
    EXPECT_EQ(opNode->attributes.get_stride(), toVec(K_WGRAD_CONV_STRIDE));
    EXPECT_EQ(opNode->attributes.get_dilation(), toVec(K_WGRAD_CONV_DILATION));
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

=======
>>>>>>> d9e199e220 (merge b-shi branch)
} // namespace
