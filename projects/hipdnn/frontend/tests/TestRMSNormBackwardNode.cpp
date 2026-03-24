// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormBackwardAttributes.hpp>
#include <hipdnn_frontend/node/RMSNormBackwardNode.hpp>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/data_objects/rmsnorm_backward_attributes_generated.h>

#include <memory>
#include <unordered_set>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// --- Helper: create fully configured attributes for a valid node ---
namespace
{

RMSNormBackwardAttributes createValidAttributes()
{
    RMSNormBackwardAttributes attrs;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    scaleTensor->set_stride({64, 1, 1, 1});
    attrs.set_scale(scaleTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);

    return attrs;
}

} // namespace

// --- GetNodeType ---

TEST(TestRMSNormBackwardNode, GetNodeTypeReturns)
{
    GraphAttributes graphAttrs;
    RMSNormBackwardNode node(RMSNormBackwardAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::RMS_NORM_BACKWARD);
}

// --- PreValidateNode (success case) ---

TEST(TestRMSNormBackwardNode, PreValidateNode)
{
    auto attrs = createValidAttributes();

    GraphAttributes graphAttributes;
    RMSNormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode: missing required tensors ---

TEST(TestRMSNormBackwardNode, PreValidateNodeMissingDyTensor)
{
    RMSNormBackwardAttributes attrs;

    // Set all required tensors except dy
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    scaleTensor->set_stride({64, 1, 1, 1});
    attrs.set_scale(scaleTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);

    // dy tensor is missing
    GraphAttributes graphAttributes;
    RMSNormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestRMSNormBackwardNode, PreValidateNodeMissingXTensor)
{
    RMSNormBackwardAttributes attrs;

    // Set all required tensors except x
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    scaleTensor->set_stride({64, 1, 1, 1});
    attrs.set_scale(scaleTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);

    // x tensor is missing
    GraphAttributes graphAttributes;
    RMSNormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestRMSNormBackwardNode, PreValidateNodeMissingScaleTensor)
{
    RMSNormBackwardAttributes attrs;

    // Set all required tensors except scale
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);

    // scale tensor is missing
    GraphAttributes graphAttributes;
    RMSNormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestRMSNormBackwardNode, PreValidateNodeMissingDxTensor)
{
    RMSNormBackwardAttributes attrs;

    // Set all required tensors except dx
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    scaleTensor->set_stride({64, 1, 1, 1});
    attrs.set_scale(scaleTensor);

    // dx tensor is missing
    GraphAttributes graphAttributes;
    RMSNormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestRMSNormBackwardNode, PreValidateNodeAllValuesSet)
{
    auto attrs = createValidAttributes();

    GraphAttributes graphAttributes;
    RMSNormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- InferPropertiesNode ---

TEST(TestRMSNormBackwardNode, InferPropertiesNode)
{
    auto attrs = createValidAttributes();

    GraphAttributes graphAttributes;
    RMSNormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    // Stub implementation: verify the method can be called without error
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PackNode ---

TEST(TestRMSNormBackwardNode, PackNode)
{
    RMSNormBackwardAttributes attrs;
    attrs.set_name("RMSNormBackwardNode");

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(90).set_name("DyTensor").set_data_type(DataType_t::FLOAT);
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(91).set_name("XTensor").set_data_type(DataType_t::FLOAT);
    xTensor->set_dim({1, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(92).set_name("ScaleTensor").set_data_type(DataType_t::FLOAT);
    scaleTensor->set_dim({1, 64, 1, 1});
    scaleTensor->set_stride({64, 1, 1, 1});
    attrs.set_scale(scaleTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(94).set_name("DxTensor").set_data_type(DataType_t::FLOAT);
    dxTensor->set_dim({1, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);

    GraphAttributes graphAttributes;
    RMSNormBackwardNode node(std::move(attrs), graphAttributes);

    flatbuffers::FlatBufferBuilder builder;
    auto offset = node.pack_node(builder);
    EXPECT_NE(offset.o, 0);

    builder.Finish(offset);
    auto bufferPointer = builder.GetBufferPointer();
    auto nodeFlatbuffer = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Node>(bufferPointer);

    EXPECT_STREQ(nodeFlatbuffer->name()->c_str(), "RMSNormBackwardNode");
    EXPECT_EQ(nodeFlatbuffer->attributes_type(),
              hipdnn_data_sdk::data_objects::NodeAttributes::RMSNormBackwardAttributes);

    auto packedAttributes = nodeFlatbuffer->attributes_as_RMSNormBackwardAttributes();
    ASSERT_NE(packedAttributes, nullptr);

    // Verify tensor UIDs
    EXPECT_EQ(packedAttributes->dy_tensor_uid(), dyTensor->get_uid());
    EXPECT_EQ(packedAttributes->x_tensor_uid(), xTensor->get_uid());
    EXPECT_EQ(packedAttributes->scale_tensor_uid(), scaleTensor->get_uid());
    EXPECT_EQ(packedAttributes->dx_tensor_uid(), dxTensor->get_uid());

    // Verify data fields
}

// --- GatherHipdnnTensors ---

TEST(TestRMSNormBackwardNode, GatherHipdnnTensor)
{
    RMSNormBackwardAttributes attrs;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(90).set_name("DyTensor");
    attrs.set_dy(dyTensor);
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(91).set_name("XTensor");
    attrs.set_x(xTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(92).set_name("ScaleTensor");
    attrs.set_scale(scaleTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(94).set_name("DxTensor");
    attrs.set_dx(dxTensor);

    GraphAttributes graphAttributes;
    RMSNormBackwardNode node(std::move(attrs), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;

    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(dyTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(scaleTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(dxTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 4u);
}
