// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_set>
#include <vector>

#include <hipdnn_frontend.hpp>
<<<<<<< HEAD
#include <hipdnn_frontend/node/RMSNormNode.hpp>
#include <hipdnn_test_sdk/constants/RMSNormConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LiftingTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::liftGraph;
using hipdnn_tests::liftGraphWithoutFinalization;
using hipdnn_tests::TestableGraphLifting;
=======
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/RMSNormNode.hpp>
#include <hipdnn_test_sdk/constants/RMSNormConstants.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
>>>>>>> d9e199e220 (merge b-shi branch)
using hipdnn_tests::toVec;
namespace rms_constants = hipdnn_tests::constants;

namespace
{
<<<<<<< HEAD
class IntegrationRMSNormDescriptorLifting : public IntegrationTestFixture
{
protected:
    // Builds a standard training rmsnorm graph with all tensors set
    static std::shared_ptr<TestableGraphLifting> buildTrainingGraph()
    {
        auto graph = std::make_shared<TestableGraphLifting>();
=======

// Exposes protected Graph methods for testing
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

class IntegrationRMSNormDescriptorLifting : public ::testing::Test
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

    // Builds a standard training rmsnorm graph with all tensors set
    static std::shared_ptr<TestableGraph> buildTrainingGraph()
    {
        auto graph = std::make_shared<TestableGraph>();
>>>>>>> d9e199e220 (merge b-shi branch)
        graph->set_name("RMSNormLiftingTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(rms_constants::K_RMSNORM_TENSOR_X_UID)
            .set_name("X")
            .set_data_type(DataType::FLOAT)
            .set_dim(toVec(rms_constants::K_RMSNORM_TENSOR_X_DIMS))
            .set_stride(toVec(rms_constants::K_RMSNORM_TENSOR_X_STRIDES));

        auto scale = std::make_shared<TensorAttributes>();
        scale->set_uid(rms_constants::K_RMSNORM_TENSOR_SCALE_UID)
            .set_name("SCALE")
            .set_data_type(DataType::FLOAT)
            .set_dim(toVec(rms_constants::K_RMSNORM_TENSOR_SCALE_DIMS))
            .set_stride(toVec(rms_constants::K_RMSNORM_TENSOR_SCALE_STRIDES));

        auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
        epsilon->set_uid(rms_constants::K_RMSNORM_TENSOR_EPSILON_UID)
            .set_name("EPSILON")
            .set_data_type(DataType::FLOAT)
            .set_dim(toVec(rms_constants::K_RMSNORM_TENSOR_EPSILON_DIMS))
            .set_stride(toVec(rms_constants::K_RMSNORM_TENSOR_EPSILON_STRIDES));

        RMSNormAttributes rmsnormAttrs;
        rmsnormAttrs.set_name("rmsnorm_op");
        rmsnormAttrs.set_forward_phase(NormFwdPhase::TRAINING);
        rmsnormAttrs.set_epsilon(epsilon);

        auto [y, invRms] = graph->rmsnorm(x, scale, std::move(rmsnormAttrs));
        y->set_uid(rms_constants::K_RMSNORM_TENSOR_Y_UID).set_output(true).set_name("Y");
        invRms->set_uid(rms_constants::K_RMSNORM_TENSOR_INV_RMS_UID)
            .set_output(true)
            .set_name("INV_RMS");

        return graph;
    }
<<<<<<< HEAD
=======

    hipdnnHandle_t _handle = nullptr;
>>>>>>> d9e199e220 (merge b-shi branch)
};

// Builds an rmsnorm graph in TRAINING mode (with inv_rms), lowers
// via build_operation_graph(handle), lifts back with fromBackendDescriptor(),
// and verifies all operation attributes and tensors.
TEST_F(IntegrationRMSNormDescriptorLifting, RMSNormTrainingRoundTripViaCApi)
{
    auto graph = buildTrainingGraph();

<<<<<<< HEAD
    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);
=======
    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    // Lift back into a new graph
    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
>>>>>>> d9e199e220 (merge b-shi branch)

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID (5 tensors: x, scale, epsilon, y, inv_rms)
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 5u);

    ASSERT_NE(tensorMap.count(rms_constants::K_RMSNORM_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_X_UID]->get_name(), "X");
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_X_UID]->get_dim(),
              toVec(rms_constants::K_RMSNORM_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_X_UID]->get_stride(),
              toVec(rms_constants::K_RMSNORM_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_X_UID]->get_data_type(), DataType::FLOAT);

    ASSERT_NE(tensorMap.count(rms_constants::K_RMSNORM_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_SCALE_UID]->get_name(), "SCALE");
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_SCALE_UID]->get_dim(),
              toVec(rms_constants::K_RMSNORM_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_SCALE_UID]->get_stride(),
              toVec(rms_constants::K_RMSNORM_TENSOR_SCALE_STRIDES));

    ASSERT_NE(tensorMap.count(rms_constants::K_RMSNORM_TENSOR_EPSILON_UID), 0u);
    auto liftedEpsilon = tensorMap[rms_constants::K_RMSNORM_TENSOR_EPSILON_UID];
    EXPECT_EQ(liftedEpsilon->get_name(), "EPSILON");
    EXPECT_EQ(liftedEpsilon->get_dim(), toVec(rms_constants::K_RMSNORM_TENSOR_EPSILON_DIMS));
    EXPECT_EQ(liftedEpsilon->get_stride(), toVec(rms_constants::K_RMSNORM_TENSOR_EPSILON_STRIDES));
    EXPECT_EQ(liftedEpsilon->get_data_type(), DataType::FLOAT);
    EXPECT_TRUE(liftedEpsilon->get_pass_by_value());
    ASSERT_TRUE(liftedEpsilon->get_pass_by_value<float>().has_value());
    EXPECT_FLOAT_EQ(liftedEpsilon->get_pass_by_value<float>().value(), 1e-5f);

    ASSERT_NE(tensorMap.count(rms_constants::K_RMSNORM_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_Y_UID]->get_name(), "Y");
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_Y_UID]->get_dim(),
              toVec(rms_constants::K_RMSNORM_TENSOR_Y_DIMS));
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_Y_UID]->get_stride(),
              toVec(rms_constants::K_RMSNORM_TENSOR_Y_STRIDES));
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_Y_UID]->get_data_type(), DataType::FLOAT);

    ASSERT_NE(tensorMap.count(rms_constants::K_RMSNORM_TENSOR_INV_RMS_UID), 0u);
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_INV_RMS_UID]->get_name(), "INV_RMS");
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_INV_RMS_UID]->get_dim(),
              toVec(rms_constants::K_RMSNORM_TENSOR_INV_RMS_DIMS));

    // Verify the lifted graph has 1 rmsnorm sub-node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* rmsNode = dynamic_cast<RMSNormNode*>(subNodes[0].get());
    ASSERT_NE(rmsNode, nullptr);

    EXPECT_EQ(rmsNode->attributes.get_forward_phase(), NormFwdPhase::TRAINING);
    EXPECT_EQ(rmsNode->attributes.get_name(), "rmsnorm_op");

    // Verify tensor references on the node
    EXPECT_EQ(rmsNode->attributes.get_x()->get_uid(), rms_constants::K_RMSNORM_TENSOR_X_UID);
    EXPECT_EQ(rmsNode->attributes.get_scale()->get_uid(),
              rms_constants::K_RMSNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(rmsNode->attributes.get_epsilon()->get_uid(),
              rms_constants::K_RMSNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(rmsNode->attributes.get_y()->get_uid(), rms_constants::K_RMSNORM_TENSOR_Y_UID);
    ASSERT_NE(rmsNode->attributes.get_inv_rms(), nullptr);
    EXPECT_EQ(rmsNode->attributes.get_inv_rms()->get_uid(),
              rms_constants::K_RMSNORM_TENSOR_INV_RMS_UID);
}

// Builds an rmsnorm graph in INFERENCE mode (no inv_rms), lowers
// via build_operation_graph(handle), lifts back with fromBackendDescriptor(),
// and verifies all operation attributes and that optional tensors are absent.
TEST_F(IntegrationRMSNormDescriptorLifting, RMSNormInferenceRoundTripViaCApi)
{
<<<<<<< HEAD
    auto graph = std::make_shared<TestableGraphLifting>();
=======
    auto graph = std::make_shared<TestableGraph>();
>>>>>>> d9e199e220 (merge b-shi branch)
    graph->set_name("RMSNormInferenceLiftingTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(rms_constants::K_RMSNORM_TENSOR_X_UID)
        .set_name("X")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(rms_constants::K_RMSNORM_TENSOR_X_DIMS))
        .set_stride(toVec(rms_constants::K_RMSNORM_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(rms_constants::K_RMSNORM_TENSOR_SCALE_UID)
        .set_name("SCALE")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(rms_constants::K_RMSNORM_TENSOR_SCALE_DIMS))
        .set_stride(toVec(rms_constants::K_RMSNORM_TENSOR_SCALE_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_uid(rms_constants::K_RMSNORM_TENSOR_EPSILON_UID)
        .set_name("EPSILON")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(rms_constants::K_RMSNORM_TENSOR_EPSILON_DIMS))
        .set_stride(toVec(rms_constants::K_RMSNORM_TENSOR_EPSILON_STRIDES));

    RMSNormAttributes rmsnormAttrs;
    rmsnormAttrs.set_name("rmsnorm_inference_op");
    rmsnormAttrs.set_forward_phase(NormFwdPhase::INFERENCE);
    rmsnormAttrs.set_epsilon(epsilon);

    auto [y, invRms] = graph->rmsnorm(x, scale, std::move(rmsnormAttrs));
    y->set_uid(rms_constants::K_RMSNORM_TENSOR_Y_UID).set_output(true).set_name("Y");
    // invRms should be nullptr in INFERENCE mode
    EXPECT_EQ(invRms, nullptr);

<<<<<<< HEAD
    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);
=======
    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    // Lift back into a new graph
    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
>>>>>>> d9e199e220 (merge b-shi branch)

    // Verify tensors by UID (4 tensors: x, scale, epsilon, y -- no inv_rms)
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 4u);

    ASSERT_NE(tensorMap.count(rms_constants::K_RMSNORM_TENSOR_X_UID), 0u);
    ASSERT_NE(tensorMap.count(rms_constants::K_RMSNORM_TENSOR_SCALE_UID), 0u);
    ASSERT_NE(tensorMap.count(rms_constants::K_RMSNORM_TENSOR_EPSILON_UID), 0u);
    ASSERT_NE(tensorMap.count(rms_constants::K_RMSNORM_TENSOR_Y_UID), 0u);

    // inv_rms should NOT be in the tensor map
    EXPECT_EQ(tensorMap.count(rms_constants::K_RMSNORM_TENSOR_INV_RMS_UID), 0u);

    // Verify the lifted graph has 1 rmsnorm sub-node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* rmsNode = dynamic_cast<RMSNormNode*>(subNodes[0].get());
    ASSERT_NE(rmsNode, nullptr);

    EXPECT_EQ(rmsNode->attributes.get_forward_phase(), NormFwdPhase::INFERENCE);
    EXPECT_EQ(rmsNode->attributes.get_name(), "rmsnorm_inference_op");

    // Optional tensors should be null on the lifted node
    EXPECT_EQ(rmsNode->attributes.get_inv_rms(), nullptr);
}

// Verifies that tensor objects are shared between the tensor map and the
// rmsnorm node attributes after lifting.
TEST_F(IntegrationRMSNormDescriptorLifting, RMSNormTensorSharingPreserved)
{
    auto graph = buildTrainingGraph();
    graph->set_name("RMSNormTensorSharingTest");

<<<<<<< HEAD
    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);
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
>>>>>>> d9e199e220 (merge b-shi branch)

    auto tensorMap = liftedGraph->getTensorsByUid();
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* rmsNode = dynamic_cast<RMSNormNode*>(subNodes[0].get());
    ASSERT_NE(rmsNode, nullptr);

    // Verify tensor objects are shared (same pointer) between map and node
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_X_UID].get(),
              rmsNode->attributes.get_x().get());
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_SCALE_UID].get(),
              rmsNode->attributes.get_scale().get());
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_EPSILON_UID].get(),
              rmsNode->attributes.get_epsilon().get());
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_Y_UID].get(),
              rmsNode->attributes.get_y().get());
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_INV_RMS_UID].get(),
              rmsNode->attributes.get_inv_rms().get());
}

// Builds an rmsnorm graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// the rmsnorm operation survives the FlatBuffer-direct deserialization path.
TEST_F(IntegrationRMSNormDescriptorLifting, RMSNormLiftWithoutFinalization)
{
    auto graph = buildTrainingGraph();
    graph->set_name("RMSNormFlatBufferLiftTest");

<<<<<<< HEAD
    auto liftedGraph = liftGraphWithoutFinalization(*graph);
    ASSERT_NE(liftedGraph, nullptr);
=======
    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Serialize to binary
    auto data = graph->toBinary();
    ASSERT_FALSE(data.empty());

    // Create backend descriptor from bytes (no handle, no finalize)
    const detail::ScopedHipdnnBackendDescriptor graphDesc(data.data(), data.size());
    ASSERT_TRUE(graphDesc.valid()) << "Failed to create backend graph descriptor";

    // Lift into a new graph
    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(graphDesc.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
>>>>>>> d9e199e220 (merge b-shi branch)

    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* rmsNode = dynamic_cast<RMSNormNode*>(subNodes[0].get());
    ASSERT_NE(rmsNode, nullptr);

    EXPECT_EQ(rmsNode->attributes.get_forward_phase(), NormFwdPhase::TRAINING);
    EXPECT_EQ(rmsNode->attributes.get_name(), "rmsnorm_op");
    EXPECT_EQ(rmsNode->attributes.get_x()->get_uid(), rms_constants::K_RMSNORM_TENSOR_X_UID);
    EXPECT_EQ(rmsNode->attributes.get_scale()->get_uid(),
              rms_constants::K_RMSNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(rmsNode->attributes.get_epsilon()->get_uid(),
              rms_constants::K_RMSNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(rmsNode->attributes.get_y()->get_uid(), rms_constants::K_RMSNORM_TENSOR_Y_UID);
    ASSERT_NE(rmsNode->attributes.get_inv_rms(), nullptr);
    EXPECT_EQ(rmsNode->attributes.get_inv_rms()->get_uid(),
              rms_constants::K_RMSNORM_TENSOR_INV_RMS_UID);

    // Verify tensor dims
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 5u);
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_X_UID]->get_dim(),
              toVec(rms_constants::K_RMSNORM_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_X_UID]->get_stride(),
              toVec(rms_constants::K_RMSNORM_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_Y_UID]->get_dim(),
              toVec(rms_constants::K_RMSNORM_TENSOR_Y_DIMS));
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_Y_UID]->get_stride(),
              toVec(rms_constants::K_RMSNORM_TENSOR_Y_STRIDES));

    // Verify epsilon pass-by-value preserved through FlatBuffer path
    auto liftedEpsilon = tensorMap[rms_constants::K_RMSNORM_TENSOR_EPSILON_UID];
    EXPECT_EQ(liftedEpsilon->get_dim(), toVec(rms_constants::K_RMSNORM_TENSOR_EPSILON_DIMS));
    EXPECT_EQ(liftedEpsilon->get_stride(), toVec(rms_constants::K_RMSNORM_TENSOR_EPSILON_STRIDES));
    EXPECT_TRUE(liftedEpsilon->get_pass_by_value());
    ASSERT_TRUE(liftedEpsilon->get_pass_by_value<float>().has_value());
    EXPECT_FLOAT_EQ(liftedEpsilon->get_pass_by_value<float>().value(), 1e-5f);
}

<<<<<<< HEAD
// Exercises the deserialize() path with a handle for an rmsnorm graph.
=======
// Exercises the deserialize_via_backend() path with a handle for an rmsnorm graph.
>>>>>>> d9e199e220 (merge b-shi branch)
TEST_F(IntegrationRMSNormDescriptorLifting, RMSNormDeserializeViaBackendWithHandle)
{
    auto graph = buildTrainingGraph();
    graph->set_name("RMSNormDeserializeTest");

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

<<<<<<< HEAD
    auto [data, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    // Create a new graph and use deserialize with handle
    auto liftedGraph = std::make_shared<TestableGraphLifting>();
    result = liftedGraph->deserialize(_handle, data);
=======
    auto data = graph->toBinary();
    ASSERT_FALSE(data.empty());

    // Create a new graph and use deserialize_via_backend with handle
    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->deserialize_via_backend(_handle, data);
>>>>>>> d9e199e220 (merge b-shi branch)
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify the lifted graph has 1 operation node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* rmsNode = dynamic_cast<RMSNormNode*>(subNodes[0].get());
    ASSERT_NE(rmsNode, nullptr);

    EXPECT_EQ(rmsNode->attributes.get_forward_phase(), NormFwdPhase::TRAINING);
    EXPECT_EQ(rmsNode->attributes.get_name(), "rmsnorm_op");

    // Verify tensor dims
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 5u);
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_X_UID]->get_dim(),
              toVec(rms_constants::K_RMSNORM_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_SCALE_UID]->get_dim(),
              toVec(rms_constants::K_RMSNORM_TENSOR_SCALE_DIMS));
}

// Roundtrip with optional bias tensor set.
TEST_F(IntegrationRMSNormDescriptorLifting, RMSNormWithBiasRoundTrip)
{
<<<<<<< HEAD
    auto graph = std::make_shared<TestableGraphLifting>();
=======
    auto graph = std::make_shared<TestableGraph>();
>>>>>>> d9e199e220 (merge b-shi branch)
    graph->set_name("BiasRMSNormLiftingTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(rms_constants::K_RMSNORM_TENSOR_X_UID)
        .set_name("X")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(rms_constants::K_RMSNORM_TENSOR_X_DIMS))
        .set_stride(toVec(rms_constants::K_RMSNORM_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(rms_constants::K_RMSNORM_TENSOR_SCALE_UID)
        .set_name("SCALE")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(rms_constants::K_RMSNORM_TENSOR_SCALE_DIMS))
        .set_stride(toVec(rms_constants::K_RMSNORM_TENSOR_SCALE_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_uid(rms_constants::K_RMSNORM_TENSOR_EPSILON_UID)
        .set_name("EPSILON")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(rms_constants::K_RMSNORM_TENSOR_EPSILON_DIMS))
        .set_stride(toVec(rms_constants::K_RMSNORM_TENSOR_EPSILON_STRIDES));

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(rms_constants::K_RMSNORM_TENSOR_BIAS_UID)
        .set_name("BIAS")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(rms_constants::K_RMSNORM_TENSOR_BIAS_DIMS))
        .set_stride(toVec(rms_constants::K_RMSNORM_TENSOR_BIAS_STRIDES));

    RMSNormAttributes rmsnormAttrs;
    rmsnormAttrs.set_name("rmsnorm_bias_op");
    rmsnormAttrs.set_forward_phase(NormFwdPhase::TRAINING);
    rmsnormAttrs.set_epsilon(epsilon);
    rmsnormAttrs.set_bias(bias);

    auto [y, invRms] = graph->rmsnorm(x, scale, std::move(rmsnormAttrs));
    y->set_uid(rms_constants::K_RMSNORM_TENSOR_Y_UID).set_output(true).set_name("Y");
    ASSERT_NE(invRms, nullptr);
    invRms->set_uid(rms_constants::K_RMSNORM_TENSOR_INV_RMS_UID)
        .set_output(true)
        .set_name("INV_RMS");

<<<<<<< HEAD
    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);
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
>>>>>>> d9e199e220 (merge b-shi branch)

    // 6 tensors: x, scale, epsilon, bias, y, inv_rms
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 6u);

    ASSERT_NE(tensorMap.count(rms_constants::K_RMSNORM_TENSOR_BIAS_UID), 0u);
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_BIAS_UID]->get_name(), "BIAS");
    EXPECT_EQ(tensorMap[rms_constants::K_RMSNORM_TENSOR_BIAS_UID]->get_dim(),
              toVec(rms_constants::K_RMSNORM_TENSOR_BIAS_DIMS));

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* rmsNode = dynamic_cast<RMSNormNode*>(subNodes[0].get());
    ASSERT_NE(rmsNode, nullptr);

    ASSERT_NE(rmsNode->attributes.get_bias(), nullptr);
    EXPECT_EQ(rmsNode->attributes.get_bias()->get_uid(), rms_constants::K_RMSNORM_TENSOR_BIAS_UID);
    ASSERT_NE(rmsNode->attributes.get_inv_rms(), nullptr);
    EXPECT_EQ(rmsNode->attributes.get_inv_rms()->get_uid(),
              rms_constants::K_RMSNORM_TENSOR_INV_RMS_UID);
}

// Builds a training rmsnorm graph WITHOUT explicit UIDs, lowers,
// lifts back, and verifies that auto-assigned UIDs are preserved and unique.
TEST_F(IntegrationRMSNormDescriptorLifting, AutoAssignedUidsPreservedInRoundTrip)
{
<<<<<<< HEAD
    auto graph = std::make_shared<TestableGraphLifting>();
=======
    auto graph = std::make_shared<TestableGraph>();
>>>>>>> d9e199e220 (merge b-shi branch)
    graph->set_name("AutoUidRMSNormLiftingTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    // Create tensors WITHOUT setting UIDs
    auto x = std::make_shared<TensorAttributes>();
    x->set_name("X")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(rms_constants::K_RMSNORM_TENSOR_X_DIMS))
        .set_stride(toVec(rms_constants::K_RMSNORM_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_name("SCALE")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(rms_constants::K_RMSNORM_TENSOR_SCALE_DIMS))
        .set_stride(toVec(rms_constants::K_RMSNORM_TENSOR_SCALE_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_name("EPSILON")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(rms_constants::K_RMSNORM_TENSOR_EPSILON_DIMS))
        .set_stride(toVec(rms_constants::K_RMSNORM_TENSOR_EPSILON_STRIDES));

    RMSNormAttributes rmsnormAttrs;
    rmsnormAttrs.set_name("rmsnorm_auto_uid_op");
    rmsnormAttrs.set_forward_phase(NormFwdPhase::TRAINING);
    rmsnormAttrs.set_epsilon(epsilon);

    auto [y, invRms] = graph->rmsnorm(x, scale, std::move(rmsnormAttrs));
    y->set_output(true).set_name("Y");
    ASSERT_NE(invRms, nullptr);
    invRms->set_output(true).set_name("INV_RMS");

<<<<<<< HEAD
    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);
=======
    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    // Lift back into a new graph
    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
>>>>>>> d9e199e220 (merge b-shi branch)

    // Verify all auto-assigned UIDs are unique
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 5u); // x, scale, epsilon, y, inv_rms

    std::unordered_set<int64_t> uids;
    for(const auto& [uid, tensor] : tensorMap)
    {
        uids.insert(uid);
    }
<<<<<<< HEAD
    EXPECT_EQ(uids.size(), 5u)
        << "Tensor UIDs are not unique"; // NOLINT(readability-implicit-bool-conversion)
=======
    EXPECT_EQ(uids.size(), 5u) << "Tensor UIDs are not unique";
>>>>>>> d9e199e220 (merge b-shi branch)

    // Verify the lifted node references UIDs that exist in the tensor map
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* rmsNode = dynamic_cast<RMSNormNode*>(subNodes[0].get());
    ASSERT_NE(rmsNode, nullptr);

    EXPECT_EQ(rmsNode->attributes.get_name(), "rmsnorm_auto_uid_op");
    EXPECT_EQ(rmsNode->attributes.get_forward_phase(), NormFwdPhase::TRAINING);

    EXPECT_TRUE(uids.count(rmsNode->attributes.get_x()->get_uid()) > 0);
    EXPECT_TRUE(uids.count(rmsNode->attributes.get_scale()->get_uid()) > 0);
    EXPECT_TRUE(uids.count(rmsNode->attributes.get_epsilon()->get_uid()) > 0);
    EXPECT_TRUE(uids.count(rmsNode->attributes.get_y()->get_uid()) > 0);
    ASSERT_NE(rmsNode->attributes.get_inv_rms(), nullptr);
    EXPECT_TRUE(uids.count(rmsNode->attributes.get_inv_rms()->get_uid()) > 0);

    // All node tensor UIDs should be distinct
    const std::unordered_set<int64_t> nodeUids = {rmsNode->attributes.get_x()->get_uid(),
                                                  rmsNode->attributes.get_scale()->get_uid(),
                                                  rmsNode->attributes.get_epsilon()->get_uid(),
                                                  rmsNode->attributes.get_y()->get_uid(),
                                                  rmsNode->attributes.get_inv_rms()->get_uid()};
<<<<<<< HEAD
    EXPECT_EQ(nodeUids.size(), 5u)
        << "RMSNorm node tensor UIDs are not distinct"; // NOLINT(readability-implicit-bool-conversion)
=======
    EXPECT_EQ(nodeUids.size(), 5u) << "RMSNorm node tensor UIDs are not distinct";
>>>>>>> d9e199e220 (merge b-shi branch)
}

} // namespace
