// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TestMacros.hpp"
#include "descriptors/ConvolutionFwdOperationDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

#include <array>
#include <map>
#include <memory>
#include <set>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;

class TestGraphDescriptorOps : public ::testing::Test
{
public:
    // Create a finalized tensor descriptor with the given UID
    static std::unique_ptr<HipdnnBackendDescriptor>
        createFinalizedTensor(int64_t uid,
                              std::vector<int64_t> dims = {1, 3, 32, 32},
                              std::vector<int64_t> strides = {3072, 1024, 32, 1},
                              DataType dataType = DataType::FLOAT)
    {
        auto wrapper = createDescriptor<TensorDescriptor>();
        auto desc = wrapper->asDescriptor<TensorDescriptor>();

        desc->setAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &uid);
        desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS,
                           HIPDNN_TYPE_INT64,
                           static_cast<int64_t>(dims.size()),
                           dims.data());
        desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES,
                           HIPDNN_TYPE_INT64,
                           static_cast<int64_t>(strides.size()),
                           strides.data());
        desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
        desc->finalize();

        return wrapper;
    }

    // Create a finalized conv fwd operation with given tensor UIDs
    static std::unique_ptr<HipdnnBackendDescriptor>
        createFinalizedConvOp(HipdnnBackendDescriptor* xDesc,
                              HipdnnBackendDescriptor* wDesc,
                              HipdnnBackendDescriptor* yDesc,
                              DataType computeType = DataType::FLOAT)
    {
        auto wrapper = createDescriptor<ConvolutionFwdOperationDescriptor>();
        auto desc = wrapper->asDescriptor<ConvolutionFwdOperationDescriptor>();

        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xDesc);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &wDesc);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yDesc);

        std::vector<int64_t> padding = {1, 1};
        std::vector<int64_t> stride = {1, 1};
        std::vector<int64_t> dilation = {1, 1};

        desc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, padding.data());
        desc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, padding.data());
        desc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());
        desc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, dilation.data());
        desc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

        desc->finalize();
        return wrapper;
    }

    // Find a tensor in a GraphT by UID, returns nullptr if not found
    static const TensorAttributesT* findTensorByUid(const GraphT& graphT, int64_t uid)
    {
        for(const auto& tensor : graphT.tensors)
        {
            if(tensor->uid == uid)
            {
                return tensor.get();
            }
        }
        return nullptr;
    }

    // Validate a tensor's fields against expected values
    static void verifyTensor(const TensorAttributesT* tensor,
                             int64_t expectedUid,
                             const std::vector<int64_t>& expectedDims,
                             const std::vector<int64_t>& expectedStrides,
                             DataType expectedDataType,
                             bool expectedVirtual = false)
    {
        ASSERT_NE(tensor, nullptr) << "Tensor with UID " << expectedUid << " not found";
        EXPECT_EQ(tensor->uid, expectedUid);
        EXPECT_EQ(tensor->dims, expectedDims);
        EXPECT_EQ(tensor->strides, expectedStrides);
        EXPECT_EQ(tensor->data_type, expectedDataType);
        EXPECT_EQ(tensor->virtual_, expectedVirtual);
    }

    // Validate a node's convolution forward attributes
    static void verifyConvFwdNode(const NodeT& node,
                                  DataType expectedComputeType,
                                  int64_t expectedXUid,
                                  int64_t expectedWUid,
                                  int64_t expectedYUid,
                                  const std::vector<int64_t>& expectedPrePadding,
                                  const std::vector<int64_t>& expectedPostPadding,
                                  const std::vector<int64_t>& expectedStride,
                                  const std::vector<int64_t>& expectedDilation)
    {
        EXPECT_EQ(node.compute_data_type, expectedComputeType);
        ASSERT_EQ(node.attributes.type, NodeAttributes::ConvolutionFwdAttributes);

        auto* convAttrs = node.attributes.AsConvolutionFwdAttributes();
        ASSERT_NE(convAttrs, nullptr);

        EXPECT_EQ(convAttrs->x_tensor_uid, expectedXUid);
        EXPECT_EQ(convAttrs->w_tensor_uid, expectedWUid);
        EXPECT_EQ(convAttrs->y_tensor_uid, expectedYUid);
        EXPECT_EQ(convAttrs->pre_padding, expectedPrePadding);
        EXPECT_EQ(convAttrs->post_padding, expectedPostPadding);
        EXPECT_EQ(convAttrs->stride, expectedStride);
        EXPECT_EQ(convAttrs->dilation, expectedDilation);
    }

    std::shared_ptr<GraphDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<GraphDescriptor>();
    }

    void setHandle() const
    {
        auto desc = getDescriptor();
        hipdnnHandle_t handle = &_mockHandle;
        desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle);
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    mutable MockHandle _mockHandle;

    void SetUp() override
    {
        _wrapper = createDescriptor<GraphDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
    }
};

// =============================================================================
// Build From Operations Tests
// =============================================================================

TEST_F(TestGraphDescriptorOps, BuildFromSingleOperation)
{
    auto xDesc = createFinalizedTensor(1);
    auto wDesc = createFinalizedTensor(2, {64, 3, 3, 3}, {27, 9, 3, 1});
    auto yDesc = createFinalizedTensor(3, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto convOp = createFinalizedConvOp(xDesc.get(), wDesc.get(), yDesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {convOp.get()};
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data()));
    ASSERT_NO_THROW(desc->finalize());

    // Verify the built graph
    auto serialized = desc->getSerializedGraph();
    ASSERT_NE(serialized.ptr, nullptr);
    ASSERT_GT(serialized.size, 0UL);

    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(serialized.ptr), serialized.size);
    ASSERT_TRUE(verifier.VerifyBuffer<Graph>());

    auto graph = GetGraph(serialized.ptr);
    auto graphT = graph->UnPack();

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 3);

    // Verify each tensor has correct fields
    verifyTensor(
        findTensorByUid(*graphT, 1), 1, {1, 3, 32, 32}, {3072, 1024, 32, 1}, DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, 2), 2, {64, 3, 3, 3}, {27, 9, 3, 1}, DataType::FLOAT);
    verifyTensor(
        findTensorByUid(*graphT, 3), 3, {1, 64, 32, 32}, {65536, 1024, 32, 1}, DataType::FLOAT);

    // Verify the node's convolution attributes and tensor UID references
    verifyConvFwdNode(*graphT->nodes[0], DataType::FLOAT, 1, 2, 3, {1, 1}, {1, 1}, {1, 1}, {1, 1});
}

TEST_F(TestGraphDescriptorOps, BuildFromMultipleOperations)
{
    // First conv: tensors 1, 2, 3
    auto xDesc1 = createFinalizedTensor(1);
    auto wDesc1 = createFinalizedTensor(2, {64, 3, 3, 3}, {27, 9, 3, 1});
    auto yDesc1 = createFinalizedTensor(3, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto convOp1 = createFinalizedConvOp(xDesc1.get(), wDesc1.get(), yDesc1.get());

    // Second conv: tensors 4, 5, 6
    auto xDesc2 = createFinalizedTensor(4, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto wDesc2 = createFinalizedTensor(5, {128, 64, 3, 3}, {576, 9, 3, 1});
    auto yDesc2 = createFinalizedTensor(6, {1, 128, 32, 32}, {131072, 1024, 32, 1});
    auto convOp2 = createFinalizedConvOp(xDesc2.get(), wDesc2.get(), yDesc2.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 2> ops = {convOp1.get(), convOp2.get()};
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, ops.data()));
    ASSERT_NO_THROW(desc->finalize());

    auto serialized = desc->getSerializedGraph();
    auto graph = GetGraph(serialized.ptr);
    auto graphT = graph->UnPack();

    ASSERT_EQ(graphT->nodes.size(), 2);
    ASSERT_EQ(graphT->tensors.size(), 6);

    // Verify all tensors from first conv op
    verifyTensor(
        findTensorByUid(*graphT, 1), 1, {1, 3, 32, 32}, {3072, 1024, 32, 1}, DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, 2), 2, {64, 3, 3, 3}, {27, 9, 3, 1}, DataType::FLOAT);
    verifyTensor(
        findTensorByUid(*graphT, 3), 3, {1, 64, 32, 32}, {65536, 1024, 32, 1}, DataType::FLOAT);

    // Verify all tensors from second conv op
    verifyTensor(
        findTensorByUid(*graphT, 4), 4, {1, 64, 32, 32}, {65536, 1024, 32, 1}, DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, 5), 5, {128, 64, 3, 3}, {576, 9, 3, 1}, DataType::FLOAT);
    verifyTensor(
        findTensorByUid(*graphT, 6), 6, {1, 128, 32, 32}, {131072, 1024, 32, 1}, DataType::FLOAT);

    // Verify first node references tensors 1, 2, 3
    verifyConvFwdNode(*graphT->nodes[0], DataType::FLOAT, 1, 2, 3, {1, 1}, {1, 1}, {1, 1}, {1, 1});

    // Verify second node references tensors 4, 5, 6
    verifyConvFwdNode(*graphT->nodes[1], DataType::FLOAT, 4, 5, 6, {1, 1}, {1, 1}, {1, 1}, {1, 1});
}

TEST_F(TestGraphDescriptorOps, TensorDeduplication)
{
    // Two operations sharing the same output tensor (tensor 3)
    auto xDesc1 = createFinalizedTensor(1);
    auto wDesc1 = createFinalizedTensor(2, {64, 3, 3, 3}, {27, 9, 3, 1});
    auto yDesc = createFinalizedTensor(3, {1, 64, 32, 32}, {65536, 1024, 32, 1}); // Shared

    auto xDesc2 = createFinalizedTensor(4, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto wDesc2 = createFinalizedTensor(5, {128, 64, 3, 3}, {576, 9, 3, 1});

    auto convOp1 = createFinalizedConvOp(xDesc1.get(), wDesc1.get(), yDesc.get());
    auto convOp2 = createFinalizedConvOp(xDesc2.get(), wDesc2.get(), yDesc.get()); // Reuse yDesc

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 2> ops = {convOp1.get(), convOp2.get()};
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, ops.data()));
    ASSERT_NO_THROW(desc->finalize());

    auto serialized = desc->getSerializedGraph();
    auto graph = GetGraph(serialized.ptr);
    auto graphT = graph->UnPack();

    ASSERT_EQ(graphT->nodes.size(), 2);
    // Should have 5 unique tensors, not 6 (tensor 3 deduplicated)
    ASSERT_EQ(graphT->tensors.size(), 5);

    // Verify tensor UIDs are unique
    std::set<int64_t> tensorUids;
    for(const auto& tensor : graphT->tensors)
    {
        tensorUids.insert(tensor->uid);
    }
    EXPECT_EQ(tensorUids.size(), 5);

    // Verify each tensor's fields
    verifyTensor(
        findTensorByUid(*graphT, 1), 1, {1, 3, 32, 32}, {3072, 1024, 32, 1}, DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, 2), 2, {64, 3, 3, 3}, {27, 9, 3, 1}, DataType::FLOAT);
    verifyTensor(
        findTensorByUid(*graphT, 3), 3, {1, 64, 32, 32}, {65536, 1024, 32, 1}, DataType::FLOAT);
    verifyTensor(
        findTensorByUid(*graphT, 4), 4, {1, 64, 32, 32}, {65536, 1024, 32, 1}, DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, 5), 5, {128, 64, 3, 3}, {576, 9, 3, 1}, DataType::FLOAT);

    // Verify first node: x=1, w=2, y=3
    verifyConvFwdNode(*graphT->nodes[0], DataType::FLOAT, 1, 2, 3, {1, 1}, {1, 1}, {1, 1}, {1, 1});

    // Verify second node also references y=3 (the shared tensor)
    verifyConvFwdNode(*graphT->nodes[1], DataType::FLOAT, 4, 5, 3, {1, 1}, {1, 1}, {1, 1}, {1, 1});
}

TEST_F(TestGraphDescriptorOps, ComputeDataTypePreserved)
{
    auto xDesc = createFinalizedTensor(1);
    auto wDesc = createFinalizedTensor(2, {64, 3, 3, 3}, {27, 9, 3, 1});
    auto yDesc = createFinalizedTensor(3, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto convOp = createFinalizedConvOp(xDesc.get(), wDesc.get(), yDesc.get(), DataType::HALF);

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {convOp.get()};
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data());
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graph = GetGraph(serialized.ptr);
    auto graphT = graph->UnPack();

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 3);

    // Verify tensors retain FLOAT data type (tensor data type is independent of compute type)
    verifyTensor(
        findTensorByUid(*graphT, 1), 1, {1, 3, 32, 32}, {3072, 1024, 32, 1}, DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, 2), 2, {64, 3, 3, 3}, {27, 9, 3, 1}, DataType::FLOAT);
    verifyTensor(
        findTensorByUid(*graphT, 3), 3, {1, 64, 32, 32}, {65536, 1024, 32, 1}, DataType::FLOAT);

    // Verify node compute data type is HALF and all conv attributes are correct
    verifyConvFwdNode(*graphT->nodes[0], DataType::HALF, 1, 2, 3, {1, 1}, {1, 1}, {1, 1}, {1, 1});
}

TEST_F(TestGraphDescriptorOps, ConvolutionAttributesPreserved)
{
    auto xDesc = createFinalizedTensor(1);
    auto wDesc = createFinalizedTensor(2, {64, 3, 3, 3}, {27, 9, 3, 1});
    auto yDesc = createFinalizedTensor(3, {1, 64, 32, 32}, {65536, 1024, 32, 1});

    // Create conv op with specific parameters
    auto wrapper = createDescriptor<ConvolutionFwdOperationDescriptor>();
    auto convDesc = wrapper->asDescriptor<ConvolutionFwdOperationDescriptor>();

    HipdnnBackendDescriptor* x = xDesc.get();
    HipdnnBackendDescriptor* w = wDesc.get();
    HipdnnBackendDescriptor* y = yDesc.get();

    convDesc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x);
    convDesc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w);
    convDesc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y);

    std::vector<int64_t> prePadding = {2, 3};
    std::vector<int64_t> postPadding = {4, 5};
    std::vector<int64_t> stride = {2, 2};
    std::vector<int64_t> dilation = {1, 1};

    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());
    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, dilation.data());
    convDesc->finalize();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {wrapper.get()};
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data());
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graph = GetGraph(serialized.ptr);
    auto graphT = graph->UnPack();

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 3);

    // Verify tensors
    verifyTensor(
        findTensorByUid(*graphT, 1), 1, {1, 3, 32, 32}, {3072, 1024, 32, 1}, DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, 2), 2, {64, 3, 3, 3}, {27, 9, 3, 1}, DataType::FLOAT);
    verifyTensor(
        findTensorByUid(*graphT, 3), 3, {1, 64, 32, 32}, {65536, 1024, 32, 1}, DataType::FLOAT);

    // Verify node with asymmetric padding and non-unit stride (compute type not explicitly set)
    verifyConvFwdNode(*graphT->nodes[0], DataType::UNSET, 1, 2, 3, {2, 3}, {4, 5}, {2, 2}, {1, 1});
}

// =============================================================================
// SetAttribute Order Tests
// =============================================================================

TEST_F(TestGraphDescriptorOps, SetOperationsAndHandleAnyOrder)
{
    // Test: operations first, then handle
    {
        auto wrapper = createDescriptor<GraphDescriptor>();
        auto desc = wrapper->asDescriptor<GraphDescriptor>();

        auto xDesc = createFinalizedTensor(1);
        auto wDesc = createFinalizedTensor(2, {64, 3, 3, 3}, {27, 9, 3, 1});
        auto yDesc = createFinalizedTensor(3, {1, 64, 32, 32}, {65536, 1024, 32, 1});
        auto convOp = createFinalizedConvOp(xDesc.get(), wDesc.get(), yDesc.get());

        std::array<HipdnnBackendDescriptor*, 1> ops = {convOp.get()};
        ASSERT_NO_THROW(desc->setAttribute(
            HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data()));

        hipdnnHandle_t handle = &_mockHandle;
        ASSERT_NO_THROW(
            desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle));

        ASSERT_NO_THROW(desc->finalize());

        auto serialized = desc->getSerializedGraph();
        auto graphT = GetGraph(serialized.ptr)->UnPack();

        ASSERT_EQ(graphT->tensors.size(), 3);
        ASSERT_EQ(graphT->nodes.size(), 1);
        verifyTensor(
            findTensorByUid(*graphT, 1), 1, {1, 3, 32, 32}, {3072, 1024, 32, 1}, DataType::FLOAT);
        verifyTensor(findTensorByUid(*graphT, 2), 2, {64, 3, 3, 3}, {27, 9, 3, 1}, DataType::FLOAT);
        verifyTensor(
            findTensorByUid(*graphT, 3), 3, {1, 64, 32, 32}, {65536, 1024, 32, 1}, DataType::FLOAT);
        verifyConvFwdNode(
            *graphT->nodes[0], DataType::FLOAT, 1, 2, 3, {1, 1}, {1, 1}, {1, 1}, {1, 1});
    }

    // Test: handle first, then operations
    {
        auto wrapper = createDescriptor<GraphDescriptor>();
        auto desc = wrapper->asDescriptor<GraphDescriptor>();

        auto xDesc = createFinalizedTensor(11);
        auto wDesc = createFinalizedTensor(12, {64, 3, 3, 3}, {27, 9, 3, 1});
        auto yDesc = createFinalizedTensor(13, {1, 64, 32, 32}, {65536, 1024, 32, 1});
        auto convOp = createFinalizedConvOp(xDesc.get(), wDesc.get(), yDesc.get());

        hipdnnHandle_t handle = &_mockHandle;
        ASSERT_NO_THROW(
            desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle));

        std::array<HipdnnBackendDescriptor*, 1> ops = {convOp.get()};
        ASSERT_NO_THROW(desc->setAttribute(
            HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data()));

        ASSERT_NO_THROW(desc->finalize());

        auto serialized = desc->getSerializedGraph();
        auto graphT = GetGraph(serialized.ptr)->UnPack();

        ASSERT_EQ(graphT->tensors.size(), 3);
        ASSERT_EQ(graphT->nodes.size(), 1);
        verifyTensor(
            findTensorByUid(*graphT, 11), 11, {1, 3, 32, 32}, {3072, 1024, 32, 1}, DataType::FLOAT);
        verifyTensor(
            findTensorByUid(*graphT, 12), 12, {64, 3, 3, 3}, {27, 9, 3, 1}, DataType::FLOAT);
        verifyTensor(findTensorByUid(*graphT, 13),
                     13,
                     {1, 64, 32, 32},
                     {65536, 1024, 32, 1},
                     DataType::FLOAT);
        verifyConvFwdNode(
            *graphT->nodes[0], DataType::FLOAT, 11, 12, 13, {1, 1}, {1, 1}, {1, 1}, {1, 1});
    }
}

TEST_F(TestGraphDescriptorOps, SetOperationsMultipleBatches)
{
    auto xDesc1 = createFinalizedTensor(1);
    auto wDesc1 = createFinalizedTensor(2, {64, 3, 3, 3}, {27, 9, 3, 1});
    auto yDesc1 = createFinalizedTensor(3, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto convOp1 = createFinalizedConvOp(xDesc1.get(), wDesc1.get(), yDesc1.get());

    auto xDesc2 = createFinalizedTensor(4, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto wDesc2 = createFinalizedTensor(5, {128, 64, 3, 3}, {576, 9, 3, 1});
    auto yDesc2 = createFinalizedTensor(6, {1, 128, 32, 32}, {131072, 1024, 32, 1});
    auto convOp2 = createFinalizedConvOp(xDesc2.get(), wDesc2.get(), yDesc2.get());

    auto desc = getDescriptor();
    setHandle();

    // Set multiple operations in a single setAttribute call
    std::array<HipdnnBackendDescriptor*, 2> ops = {convOp1.get(), convOp2.get()};
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, ops.data()));

    ASSERT_NO_THROW(desc->finalize());

    auto serialized = desc->getSerializedGraph();
    auto graph = GetGraph(serialized.ptr);
    auto graphT = graph->UnPack();

    // Both operations should be present in the graph
    ASSERT_EQ(graphT->nodes.size(), 2);
    ASSERT_EQ(graphT->tensors.size(), 6);

    // Verify tensors from first operation
    verifyTensor(
        findTensorByUid(*graphT, 1), 1, {1, 3, 32, 32}, {3072, 1024, 32, 1}, DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, 2), 2, {64, 3, 3, 3}, {27, 9, 3, 1}, DataType::FLOAT);
    verifyTensor(
        findTensorByUid(*graphT, 3), 3, {1, 64, 32, 32}, {65536, 1024, 32, 1}, DataType::FLOAT);

    // Verify tensors from second operation
    verifyTensor(
        findTensorByUid(*graphT, 4), 4, {1, 64, 32, 32}, {65536, 1024, 32, 1}, DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, 5), 5, {128, 64, 3, 3}, {576, 9, 3, 1}, DataType::FLOAT);
    verifyTensor(
        findTensorByUid(*graphT, 6), 6, {1, 128, 32, 32}, {131072, 1024, 32, 1}, DataType::FLOAT);

    // Verify both nodes reference correct tensor UIDs
    verifyConvFwdNode(*graphT->nodes[0], DataType::FLOAT, 1, 2, 3, {1, 1}, {1, 1}, {1, 1}, {1, 1});
    verifyConvFwdNode(*graphT->nodes[1], DataType::FLOAT, 4, 5, 6, {1, 1}, {1, 1}, {1, 1}, {1, 1});
}

// =============================================================================
// Error Cases
// =============================================================================

TEST_F(TestGraphDescriptorOps, SetOperationsFailsUnfinalized)
{
    auto desc = getDescriptor();

    // Create unfinalized operation
    auto unfinalizedOp = createDescriptor<ConvolutionFwdOperationDescriptor>();

    std::array<HipdnnBackendDescriptor*, 1> ops = {unfinalizedOp.get()};
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data()),
        HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestGraphDescriptorOps, SetOperationsFailsNullDescriptor)
{
    auto desc = getDescriptor();

    std::array<HipdnnBackendDescriptor*, 1> ops = {nullptr};
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data()),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestGraphDescriptorOps, SetOperationsFailsWrongType)
{
    auto desc = getDescriptor();

    // Use a TensorDescriptor instead of an operation descriptor
    auto tensorDesc = createFinalizedTensor(1);

    std::array<HipdnnBackendDescriptor*, 1> ops = {tensorDesc.get()};
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data()),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestGraphDescriptorOps, SetOperationsFailsAfterFinalize)
{
    auto xDesc = createFinalizedTensor(1);
    auto wDesc = createFinalizedTensor(2, {64, 3, 3, 3}, {27, 9, 3, 1});
    auto yDesc = createFinalizedTensor(3, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto convOp = createFinalizedConvOp(xDesc.get(), wDesc.get(), yDesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {convOp.get()};
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data());
    desc->finalize();

    // Try to set operations after finalize
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data()),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestGraphDescriptorOps, FinalizeFailsWithoutHandle)
{
    auto xDesc = createFinalizedTensor(1);
    auto wDesc = createFinalizedTensor(2, {64, 3, 3, 3}, {27, 9, 3, 1});
    auto yDesc = createFinalizedTensor(3, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto convOp = createFinalizedConvOp(xDesc.get(), wDesc.get(), yDesc.get());

    auto desc = getDescriptor();
    // Don't set handle

    std::array<HipdnnBackendDescriptor*, 1> ops = {convOp.get()};
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGraphDescriptorOps, FinalizeFailsWithoutOperationsOrGraph)
{
    auto desc = getDescriptor();
    setHandle();
    // Don't set operations or deserialize graph

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// Serialization Tests
// =============================================================================

TEST_F(TestGraphDescriptorOps, SerializedGraphVerifiable)
{
    auto xDesc = createFinalizedTensor(1);
    auto wDesc = createFinalizedTensor(2, {64, 3, 3, 3}, {27, 9, 3, 1});
    auto yDesc = createFinalizedTensor(3, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto convOp = createFinalizedConvOp(xDesc.get(), wDesc.get(), yDesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {convOp.get()};
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data());
    desc->finalize();

    auto serialized = desc->getSerializedGraph();

    // Verify FlatBuffer is valid
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(serialized.ptr), serialized.size);
    ASSERT_TRUE(verifier.VerifyBuffer<Graph>());

    // Verify the verified buffer contains the expected graph structure
    auto graphT = GetGraph(serialized.ptr)->UnPack();
    ASSERT_EQ(graphT->tensors.size(), 3);
    ASSERT_EQ(graphT->nodes.size(), 1);
    verifyConvFwdNode(*graphT->nodes[0], DataType::FLOAT, 1, 2, 3, {1, 1}, {1, 1}, {1, 1}, {1, 1});
}

TEST_F(TestGraphDescriptorOps, SerializedGraphUnpackable)
{
    auto xDesc = createFinalizedTensor(1);
    auto wDesc = createFinalizedTensor(2, {64, 3, 3, 3}, {27, 9, 3, 1});
    auto yDesc = createFinalizedTensor(3, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto convOp = createFinalizedConvOp(xDesc.get(), wDesc.get(), yDesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {convOp.get()};
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data());
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graph = GetGraph(serialized.ptr);
    ASSERT_NE(graph, nullptr);

    // Unpack should work and produce correct values
    auto graphT = graph->UnPack();
    ASSERT_NE(graphT, nullptr);
    ASSERT_EQ(graphT->tensors.size(), 3);
    ASSERT_EQ(graphT->nodes.size(), 1);

    // Verify unpacked tensor values match input
    verifyTensor(
        findTensorByUid(*graphT, 1), 1, {1, 3, 32, 32}, {3072, 1024, 32, 1}, DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, 2), 2, {64, 3, 3, 3}, {27, 9, 3, 1}, DataType::FLOAT);
    verifyTensor(
        findTensorByUid(*graphT, 3), 3, {1, 64, 32, 32}, {65536, 1024, 32, 1}, DataType::FLOAT);

    // Verify unpacked node values match input
    verifyConvFwdNode(*graphT->nodes[0], DataType::FLOAT, 1, 2, 3, {1, 1}, {1, 1}, {1, 1}, {1, 1});
}

TEST_F(TestGraphDescriptorOps, GetSerializedGraphMultipleCalls)
{
    auto xDesc = createFinalizedTensor(1);
    auto wDesc = createFinalizedTensor(2, {64, 3, 3, 3}, {27, 9, 3, 1});
    auto yDesc = createFinalizedTensor(3, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto convOp = createFinalizedConvOp(xDesc.get(), wDesc.get(), yDesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {convOp.get()};
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data());
    desc->finalize();

    // Call getSerializedGraph multiple times
    auto serialized1 = desc->getSerializedGraph();
    auto serialized2 = desc->getSerializedGraph();

    // Should return same data
    EXPECT_EQ(serialized1.ptr, serialized2.ptr);
    EXPECT_EQ(serialized1.size, serialized2.size);

    // Both should unpack to identical graph values
    auto graphT1 = GetGraph(serialized1.ptr)->UnPack();
    auto graphT2 = GetGraph(serialized2.ptr)->UnPack();

    ASSERT_EQ(graphT1->tensors.size(), 3);
    ASSERT_EQ(graphT2->tensors.size(), 3);
    ASSERT_EQ(graphT1->nodes.size(), 1);
    ASSERT_EQ(graphT2->nodes.size(), 1);

    // Verify both unpacked graphs contain identical tensor data
    for(size_t i = 0; i < graphT1->tensors.size(); ++i)
    {
        EXPECT_EQ(graphT1->tensors[i]->uid, graphT2->tensors[i]->uid);
        EXPECT_EQ(graphT1->tensors[i]->dims, graphT2->tensors[i]->dims);
        EXPECT_EQ(graphT1->tensors[i]->strides, graphT2->tensors[i]->strides);
        EXPECT_EQ(graphT1->tensors[i]->data_type, graphT2->tensors[i]->data_type);
    }

    // Verify both unpacked graphs contain identical node data
    EXPECT_EQ(graphT1->nodes[0]->compute_data_type, graphT2->nodes[0]->compute_data_type);
    EXPECT_EQ(graphT1->nodes[0]->attributes.type, graphT2->nodes[0]->attributes.type);
}

// =============================================================================
// Graph Structure Verification Tests
// =============================================================================

TEST_F(TestGraphDescriptorOps, GraphHasCorrectNodeCount)
{
    auto xDesc = createFinalizedTensor(1);
    auto wDesc = createFinalizedTensor(2, {64, 3, 3, 3}, {27, 9, 3, 1});
    auto yDesc = createFinalizedTensor(3, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto convOp = createFinalizedConvOp(xDesc.get(), wDesc.get(), yDesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {convOp.get()};
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data());
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graph = GetGraph(serialized.ptr);
    auto graphT = graph->UnPack();

    ASSERT_EQ(graphT->nodes.size(), 1);

    // Verify node has ConvolutionFwdAttributes and correct tensor UID references
    verifyConvFwdNode(*graphT->nodes[0], DataType::FLOAT, 1, 2, 3, {1, 1}, {1, 1}, {1, 1}, {1, 1});
}

TEST_F(TestGraphDescriptorOps, GraphHasCorrectTensorCount)
{
    auto xDesc = createFinalizedTensor(1);
    auto wDesc = createFinalizedTensor(2, {64, 3, 3, 3}, {27, 9, 3, 1});
    auto yDesc = createFinalizedTensor(3, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto convOp = createFinalizedConvOp(xDesc.get(), wDesc.get(), yDesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {convOp.get()};
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data());
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graph = GetGraph(serialized.ptr);
    auto graphT = graph->UnPack();

    ASSERT_EQ(graphT->tensors.size(), 3);

    // Verify each tensor's full field values (not just UIDs)
    verifyTensor(
        findTensorByUid(*graphT, 1), 1, {1, 3, 32, 32}, {3072, 1024, 32, 1}, DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, 2), 2, {64, 3, 3, 3}, {27, 9, 3, 1}, DataType::FLOAT);
    verifyTensor(
        findTensorByUid(*graphT, 3), 3, {1, 64, 32, 32}, {65536, 1024, 32, 1}, DataType::FLOAT);
}

// =============================================================================
// Equivalence Tests - Compare FlatBuffer path vs Descriptor path
// =============================================================================

class TestGraphDescriptorEquivalence : public TestGraphDescriptorOps
{
public:
    // Build a graph via FlatBuffer serialization (the "old" path)
    static flatbuffers::DetachedBuffer
        buildGraphViaFlatBuffer(const TensorAttributesT& xTensor,
                                const TensorAttributesT& wTensor,
                                const TensorAttributesT& yTensor,
                                const ConvolutionFwdAttributesT& convAttrs,
                                DataType computeDataType)
    {
        flatbuffers::FlatBufferBuilder builder;

        // Build tensors
        std::vector<flatbuffers::Offset<TensorAttributes>> tensorOffsets;
        tensorOffsets.push_back(TensorAttributes::Pack(builder, &xTensor));
        tensorOffsets.push_back(TensorAttributes::Pack(builder, &wTensor));
        tensorOffsets.push_back(TensorAttributes::Pack(builder, &yTensor));

        // Build node with conv attributes
        NodeT nodeT;
        nodeT.compute_data_type = computeDataType;
        nodeT.attributes.Set(ConvolutionFwdAttributesT(convAttrs));

        std::vector<flatbuffers::Offset<Node>> nodeOffsets;
        nodeOffsets.push_back(Node::Pack(builder, &nodeT));

        // Build graph
        auto graphOffset = CreateGraphDirect(builder,
                                             nullptr, // name
                                             DataType::UNSET,
                                             DataType::UNSET,
                                             DataType::UNSET,
                                             &tensorOffsets,
                                             &nodeOffsets);
        builder.Finish(graphOffset);
        return builder.Release();
    }

    // Build a graph via descriptors (the "new" path)
    std::unique_ptr<GraphT> buildGraphViaDescriptors(int64_t xUid,
                                                     int64_t wUid,
                                                     int64_t yUid,
                                                     const std::vector<int64_t>& xDims,
                                                     const std::vector<int64_t>& xStrides,
                                                     const std::vector<int64_t>& wDims,
                                                     const std::vector<int64_t>& wStrides,
                                                     const std::vector<int64_t>& yDims,
                                                     const std::vector<int64_t>& yStrides,
                                                     const std::vector<int64_t>& prePadding,
                                                     const std::vector<int64_t>& postPadding,
                                                     const std::vector<int64_t>& stride,
                                                     const std::vector<int64_t>& dilation,
                                                     DataType tensorDataType,
                                                     DataType computeDataType)
    {
        // Create tensor descriptors
        auto xDesc = createFinalizedTensor(xUid, xDims, xStrides, tensorDataType);
        auto wDesc = createFinalizedTensor(wUid, wDims, wStrides, tensorDataType);
        auto yDesc = createFinalizedTensor(yUid, yDims, yStrides, tensorDataType);

        // Create conv op descriptor
        auto convWrapper = createDescriptor<ConvolutionFwdOperationDescriptor>();
        auto convDesc = convWrapper->asDescriptor<ConvolutionFwdOperationDescriptor>();

        HipdnnBackendDescriptor* x = xDesc.get();
        HipdnnBackendDescriptor* w = wDesc.get();
        HipdnnBackendDescriptor* y = yDesc.get();

        convDesc->setAttribute(
            HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x);
        convDesc->setAttribute(
            HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w);
        convDesc->setAttribute(
            HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y);
        convDesc->setAttribute(HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                               HIPDNN_TYPE_INT64,
                               static_cast<int64_t>(prePadding.size()),
                               prePadding.data());
        convDesc->setAttribute(HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                               HIPDNN_TYPE_INT64,
                               static_cast<int64_t>(postPadding.size()),
                               postPadding.data());
        convDesc->setAttribute(HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                               HIPDNN_TYPE_INT64,
                               static_cast<int64_t>(stride.size()),
                               stride.data());
        convDesc->setAttribute(HIPDNN_ATTR_CONVOLUTION_DILATIONS,
                               HIPDNN_TYPE_INT64,
                               static_cast<int64_t>(dilation.size()),
                               dilation.data());
        convDesc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeDataType);
        convDesc->finalize();

        // Build graph via GraphDescriptor
        auto graphWrapper = createDescriptor<GraphDescriptor>();
        auto graphDesc = graphWrapper->asDescriptor<GraphDescriptor>();

        hipdnnHandle_t handle = &_mockHandle;
        graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle);

        std::array<HipdnnBackendDescriptor*, 1> ops = {convWrapper.get()};
        graphDesc->setAttribute(
            HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data());
        graphDesc->finalize();

        auto serialized = graphDesc->getSerializedGraph();
        auto graph = GetGraph(serialized.ptr);
        return std::unique_ptr<GraphT>(graph->UnPack());
    }
};

TEST_F(TestGraphDescriptorEquivalence, SingleConvOpEquivalence)
{
    // Test parameters
    const int64_t xUid = 1;
    const int64_t wUid = 2;
    const int64_t yUid = 3;
    const std::vector<int64_t> xDims = {1, 3, 32, 32};
    const std::vector<int64_t> xStrides = {3072, 1024, 32, 1};
    const std::vector<int64_t> wDims = {64, 3, 3, 3};
    const std::vector<int64_t> wStrides = {27, 9, 3, 1};
    const std::vector<int64_t> yDims = {1, 64, 32, 32};
    const std::vector<int64_t> yStrides = {65536, 1024, 32, 1};
    const std::vector<int64_t> prePadding = {1, 1};
    const std::vector<int64_t> postPadding = {1, 1};
    const std::vector<int64_t> stride = {1, 1};
    const std::vector<int64_t> dilation = {1, 1};
    const auto tensorDataType = DataType::FLOAT;
    const auto computeDataType = DataType::FLOAT;

    // Build via FlatBuffer path
    TensorAttributesT xTensor;
    xTensor.uid = xUid;
    xTensor.dims = xDims;
    xTensor.strides = xStrides;
    xTensor.data_type = tensorDataType;

    TensorAttributesT wTensor;
    wTensor.uid = wUid;
    wTensor.dims = wDims;
    wTensor.strides = wStrides;
    wTensor.data_type = tensorDataType;

    TensorAttributesT yTensor;
    yTensor.uid = yUid;
    yTensor.dims = yDims;
    yTensor.strides = yStrides;
    yTensor.data_type = tensorDataType;

    ConvolutionFwdAttributesT convAttrs;
    convAttrs.x_tensor_uid = xUid;
    convAttrs.w_tensor_uid = wUid;
    convAttrs.y_tensor_uid = yUid;
    convAttrs.pre_padding = prePadding;
    convAttrs.post_padding = postPadding;
    convAttrs.stride = stride;
    convAttrs.dilation = dilation;

    auto flatbufferBuffer
        = buildGraphViaFlatBuffer(xTensor, wTensor, yTensor, convAttrs, computeDataType);
    auto flatbufferGraph = GetGraph(flatbufferBuffer.data());
    auto flatbufferGraphT = flatbufferGraph->UnPack();

    // Build via descriptor path
    auto descriptorGraphT = buildGraphViaDescriptors(xUid,
                                                     wUid,
                                                     yUid,
                                                     xDims,
                                                     xStrides,
                                                     wDims,
                                                     wStrides,
                                                     yDims,
                                                     yStrides,
                                                     prePadding,
                                                     postPadding,
                                                     stride,
                                                     dilation,
                                                     tensorDataType,
                                                     computeDataType);

    // Verify structural equivalence
    ASSERT_EQ(flatbufferGraphT->tensors.size(), descriptorGraphT->tensors.size());
    ASSERT_EQ(flatbufferGraphT->nodes.size(), descriptorGraphT->nodes.size());

    // Compare tensors (order may differ, so compare by UID)
    std::map<int64_t, const TensorAttributesT*> fbTensors;
    for(const auto& t : flatbufferGraphT->tensors)
    {
        fbTensors[t->uid] = t.get();
    }

    for(const auto& descTensor : descriptorGraphT->tensors)
    {
        auto it = fbTensors.find(descTensor->uid);
        ASSERT_NE(it, fbTensors.end())
            << "Tensor UID " << descTensor->uid << " not found in FlatBuffer graph";

        const auto* fbTensor = it->second;
        EXPECT_EQ(fbTensor->uid, descTensor->uid);
        EXPECT_EQ(fbTensor->dims, descTensor->dims);
        EXPECT_EQ(fbTensor->strides, descTensor->strides);
        EXPECT_EQ(fbTensor->data_type, descTensor->data_type);
    }

    // Compare nodes
    ASSERT_EQ(flatbufferGraphT->nodes.size(), 1);
    ASSERT_EQ(descriptorGraphT->nodes.size(), 1);

    const auto& fbNode = flatbufferGraphT->nodes[0];
    const auto& descNode = descriptorGraphT->nodes[0];

    EXPECT_EQ(fbNode->compute_data_type, descNode->compute_data_type);
    EXPECT_EQ(fbNode->attributes.type, descNode->attributes.type);
    EXPECT_EQ(fbNode->attributes.type, NodeAttributes::ConvolutionFwdAttributes);

    // Compare conv attributes
    const auto* fbConv = fbNode->attributes.AsConvolutionFwdAttributes();
    const auto* descConv = descNode->attributes.AsConvolutionFwdAttributes();

    ASSERT_NE(fbConv, nullptr);
    ASSERT_NE(descConv, nullptr);

    EXPECT_EQ(fbConv->x_tensor_uid, descConv->x_tensor_uid);
    EXPECT_EQ(fbConv->w_tensor_uid, descConv->w_tensor_uid);
    EXPECT_EQ(fbConv->y_tensor_uid, descConv->y_tensor_uid);
    EXPECT_EQ(fbConv->pre_padding, descConv->pre_padding);
    EXPECT_EQ(fbConv->post_padding, descConv->post_padding);
    EXPECT_EQ(fbConv->stride, descConv->stride);
    EXPECT_EQ(fbConv->dilation, descConv->dilation);
}

TEST_F(TestGraphDescriptorEquivalence, ConvOpWithHalfPrecision)
{
    // Test with HALF data type
    const int64_t xUid = 10;
    const int64_t wUid = 20;
    const int64_t yUid = 30;
    const std::vector<int64_t> xDims = {2, 64, 56, 56};
    const std::vector<int64_t> xStrides = {200704, 3136, 56, 1};
    const std::vector<int64_t> wDims = {128, 64, 3, 3};
    const std::vector<int64_t> wStrides = {576, 9, 3, 1};
    const std::vector<int64_t> yDims = {2, 128, 56, 56};
    const std::vector<int64_t> yStrides = {401408, 3136, 56, 1};
    const std::vector<int64_t> prePadding = {1, 1};
    const std::vector<int64_t> postPadding = {1, 1};
    const std::vector<int64_t> stride = {1, 1};
    const std::vector<int64_t> dilation = {1, 1};
    const auto tensorDataType = DataType::HALF;
    const auto computeDataType = DataType::HALF;

    // Build via FlatBuffer path
    TensorAttributesT xTensor;
    xTensor.uid = xUid;
    xTensor.dims = xDims;
    xTensor.strides = xStrides;
    xTensor.data_type = tensorDataType;

    TensorAttributesT wTensor;
    wTensor.uid = wUid;
    wTensor.dims = wDims;
    wTensor.strides = wStrides;
    wTensor.data_type = tensorDataType;

    TensorAttributesT yTensor;
    yTensor.uid = yUid;
    yTensor.dims = yDims;
    yTensor.strides = yStrides;
    yTensor.data_type = tensorDataType;

    ConvolutionFwdAttributesT convAttrs;
    convAttrs.x_tensor_uid = xUid;
    convAttrs.w_tensor_uid = wUid;
    convAttrs.y_tensor_uid = yUid;
    convAttrs.pre_padding = prePadding;
    convAttrs.post_padding = postPadding;
    convAttrs.stride = stride;
    convAttrs.dilation = dilation;

    auto flatbufferBuffer
        = buildGraphViaFlatBuffer(xTensor, wTensor, yTensor, convAttrs, computeDataType);
    auto flatbufferGraph = GetGraph(flatbufferBuffer.data());
    auto flatbufferGraphT = flatbufferGraph->UnPack();

    // Build via descriptor path
    auto descriptorGraphT = buildGraphViaDescriptors(xUid,
                                                     wUid,
                                                     yUid,
                                                     xDims,
                                                     xStrides,
                                                     wDims,
                                                     wStrides,
                                                     yDims,
                                                     yStrides,
                                                     prePadding,
                                                     postPadding,
                                                     stride,
                                                     dilation,
                                                     tensorDataType,
                                                     computeDataType);

    // Verify structural equivalence
    ASSERT_EQ(flatbufferGraphT->tensors.size(), descriptorGraphT->tensors.size());
    ASSERT_EQ(flatbufferGraphT->nodes.size(), descriptorGraphT->nodes.size());

    // Compare tensors by UID with full field verification
    std::map<int64_t, const TensorAttributesT*> fbTensors;
    for(const auto& t : flatbufferGraphT->tensors)
    {
        fbTensors[t->uid] = t.get();
    }

    for(const auto& descTensor : descriptorGraphT->tensors)
    {
        auto it = fbTensors.find(descTensor->uid);
        ASSERT_NE(it, fbTensors.end()) << "Tensor UID " << descTensor->uid << " not found";

        const auto* fbTensor = it->second;
        EXPECT_EQ(fbTensor->dims, descTensor->dims);
        EXPECT_EQ(fbTensor->strides, descTensor->strides);
        EXPECT_EQ(fbTensor->data_type, descTensor->data_type);
        EXPECT_EQ(fbTensor->data_type, DataType::HALF);
    }

    // Compare nodes
    const auto& fbNode = flatbufferGraphT->nodes[0];
    const auto& descNode = descriptorGraphT->nodes[0];

    EXPECT_EQ(fbNode->compute_data_type, DataType::HALF);
    EXPECT_EQ(descNode->compute_data_type, DataType::HALF);

    // Compare conv attributes with full field verification
    const auto* fbConv = fbNode->attributes.AsConvolutionFwdAttributes();
    const auto* descConv = descNode->attributes.AsConvolutionFwdAttributes();

    ASSERT_NE(fbConv, nullptr);
    ASSERT_NE(descConv, nullptr);

    EXPECT_EQ(fbConv->x_tensor_uid, descConv->x_tensor_uid);
    EXPECT_EQ(fbConv->w_tensor_uid, descConv->w_tensor_uid);
    EXPECT_EQ(fbConv->y_tensor_uid, descConv->y_tensor_uid);
    EXPECT_EQ(fbConv->pre_padding, descConv->pre_padding);
    EXPECT_EQ(fbConv->post_padding, descConv->post_padding);
    EXPECT_EQ(fbConv->stride, descConv->stride);
    EXPECT_EQ(fbConv->dilation, descConv->dilation);
}

TEST_F(TestGraphDescriptorEquivalence, ConvOpWithNonUnitStrideAndDilation)
{
    // Test with non-unit stride and dilation
    const int64_t xUid = 100;
    const int64_t wUid = 200;
    const int64_t yUid = 300;
    const std::vector<int64_t> xDims = {1, 256, 28, 28};
    const std::vector<int64_t> xStrides = {200704, 784, 28, 1};
    const std::vector<int64_t> wDims = {512, 256, 3, 3};
    const std::vector<int64_t> wStrides = {2304, 9, 3, 1};
    const std::vector<int64_t> yDims = {1, 512, 14, 14};
    const std::vector<int64_t> yStrides = {100352, 196, 14, 1};
    const std::vector<int64_t> prePadding = {2, 2};
    const std::vector<int64_t> postPadding = {2, 2};
    const std::vector<int64_t> stride = {2, 2}; // Non-unit stride
    const std::vector<int64_t> dilation = {2, 2}; // Non-unit dilation
    const auto tensorDataType = DataType::FLOAT;
    const auto computeDataType = DataType::FLOAT;

    // Build via FlatBuffer path
    TensorAttributesT xTensor;
    xTensor.uid = xUid;
    xTensor.dims = xDims;
    xTensor.strides = xStrides;
    xTensor.data_type = tensorDataType;

    TensorAttributesT wTensor;
    wTensor.uid = wUid;
    wTensor.dims = wDims;
    wTensor.strides = wStrides;
    wTensor.data_type = tensorDataType;

    TensorAttributesT yTensor;
    yTensor.uid = yUid;
    yTensor.dims = yDims;
    yTensor.strides = yStrides;
    yTensor.data_type = tensorDataType;

    ConvolutionFwdAttributesT convAttrs;
    convAttrs.x_tensor_uid = xUid;
    convAttrs.w_tensor_uid = wUid;
    convAttrs.y_tensor_uid = yUid;
    convAttrs.pre_padding = prePadding;
    convAttrs.post_padding = postPadding;
    convAttrs.stride = stride;
    convAttrs.dilation = dilation;

    auto flatbufferBuffer
        = buildGraphViaFlatBuffer(xTensor, wTensor, yTensor, convAttrs, computeDataType);
    auto flatbufferGraph = GetGraph(flatbufferBuffer.data());
    auto flatbufferGraphT = flatbufferGraph->UnPack();

    // Build via descriptor path
    auto descriptorGraphT = buildGraphViaDescriptors(xUid,
                                                     wUid,
                                                     yUid,
                                                     xDims,
                                                     xStrides,
                                                     wDims,
                                                     wStrides,
                                                     yDims,
                                                     yStrides,
                                                     prePadding,
                                                     postPadding,
                                                     stride,
                                                     dilation,
                                                     tensorDataType,
                                                     computeDataType);

    // Compare conv attributes - especially stride and dilation
    const auto* fbConv = flatbufferGraphT->nodes[0]->attributes.AsConvolutionFwdAttributes();
    const auto* descConv = descriptorGraphT->nodes[0]->attributes.AsConvolutionFwdAttributes();

    EXPECT_EQ(fbConv->pre_padding, descConv->pre_padding);
    EXPECT_EQ(fbConv->post_padding, descConv->post_padding);
    EXPECT_EQ(fbConv->stride, descConv->stride);
    EXPECT_EQ(fbConv->dilation, descConv->dilation);

    // Verify the actual values
    EXPECT_EQ(descConv->stride[0], 2);
    EXPECT_EQ(descConv->stride[1], 2);
    EXPECT_EQ(descConv->dilation[0], 2);
    EXPECT_EQ(descConv->dilation[1], 2);
}

// =============================================================================
// Graph-Level Attribute Tests
// =============================================================================

TEST_F(TestGraphDescriptorOps, GraphLevelDataTypesPreserved)
{
    auto xDesc = createFinalizedTensor(1);
    auto wDesc = createFinalizedTensor(2, {64, 3, 3, 3}, {27, 9, 3, 1});
    auto yDesc = createFinalizedTensor(3, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto convOp = createFinalizedConvOp(xDesc.get(), wDesc.get(), yDesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {convOp.get()};
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data());

    // Set graph-level data types before finalize
    auto computeDt = DataType::HALF;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_COMPUTE_DATA_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeDt);

    auto intermediateDt = DataType::BFLOAT16;
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_INTERMEDIATE_DATA_TYPE_EXT,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &intermediateDt);

    auto ioDt = DataType::FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_IO_DATA_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &ioDt);

    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = GetGraph(serialized.ptr)->UnPack();

    EXPECT_EQ(graphT->compute_data_type, DataType::HALF);
    EXPECT_EQ(graphT->intermediate_data_type, DataType::BFLOAT16);
    EXPECT_EQ(graphT->io_data_type, DataType::FLOAT);
}

TEST_F(TestGraphDescriptorOps, PreferredEngineIdPreserved)
{
    auto xDesc = createFinalizedTensor(1);
    auto wDesc = createFinalizedTensor(2, {64, 3, 3, 3}, {27, 9, 3, 1});
    auto yDesc = createFinalizedTensor(3, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto convOp = createFinalizedConvOp(xDesc.get(), wDesc.get(), yDesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {convOp.get()};
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data());

    int64_t engineId = 42;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_PREFERRED_ENGINE_ID_EXT, HIPDNN_TYPE_INT64, 1, &engineId);

    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = GetGraph(serialized.ptr)->UnPack();

    EXPECT_EQ(graphT->preferred_engine_id, 42);
}

TEST_F(TestGraphDescriptorOps, GraphLevelDataTypesDefaultToUnset)
{
    auto xDesc = createFinalizedTensor(1);
    auto wDesc = createFinalizedTensor(2, {64, 3, 3, 3}, {27, 9, 3, 1});
    auto yDesc = createFinalizedTensor(3, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto convOp = createFinalizedConvOp(xDesc.get(), wDesc.get(), yDesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {convOp.get()};
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data());

    // Finalize without setting any graph-level data types
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = GetGraph(serialized.ptr)->UnPack();

    EXPECT_EQ(graphT->compute_data_type, DataType::UNSET);
    EXPECT_EQ(graphT->intermediate_data_type, DataType::UNSET);
    EXPECT_EQ(graphT->io_data_type, DataType::UNSET);
}

TEST_F(TestGraphDescriptorOps, SharedTensorDifferentPositions)
{
    // Same tensor used as Y in first op and X in second op
    auto graphWrapper = createDescriptor<GraphDescriptor>();
    auto graphDesc = graphWrapper->asDescriptor<GraphDescriptor>();

    auto handle = reinterpret_cast<hipdnnHandle_t>(&_mockHandle);
    graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle);

    // Create tensors (uid 3 shared between ops)
    auto xDesc1 = createFinalizedTensor(1);
    auto wDesc1 = createFinalizedTensor(2, {64, 3, 3, 3}, {27, 9, 3, 1});
    auto sharedTensor = createFinalizedTensor(3, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto wDesc2 = createFinalizedTensor(4, {64, 64, 3, 3}, {576, 9, 3, 1});
    auto yDesc2 = createFinalizedTensor(5, {1, 64, 32, 32}, {65536, 1024, 32, 1});

    // Op1: x=1, w=2, y=3
    auto op1 = createFinalizedConvOp(xDesc1.get(), wDesc1.get(), sharedTensor.get());
    // Op2: x=3, w=4, y=5 (shares tensor 3 with op1 in a different position)
    auto op2 = createFinalizedConvOp(sharedTensor.get(), wDesc2.get(), yDesc2.get());

    std::array<HipdnnBackendDescriptor*, 2> opDescs = {op1.get(), op2.get()};
    graphDesc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, opDescs.data());

    graphDesc->finalize();

    auto serialized = graphDesc->getSerializedGraph();
    auto graph = GetGraph(serialized.ptr);
    ASSERT_NE(graph, nullptr);
    auto graphT = graph->UnPack();

    // 5 unique tensors (1,2,3,4,5), tensor 3 is deduplicated
    ASSERT_EQ(graphT->tensors.size(), 5);
    ASSERT_EQ(graphT->nodes.size(), 2);

    // Verify each tensor's fields
    verifyTensor(
        findTensorByUid(*graphT, 1), 1, {1, 3, 32, 32}, {3072, 1024, 32, 1}, DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, 2), 2, {64, 3, 3, 3}, {27, 9, 3, 1}, DataType::FLOAT);
    verifyTensor(
        findTensorByUid(*graphT, 3), 3, {1, 64, 32, 32}, {65536, 1024, 32, 1}, DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, 4), 4, {64, 64, 3, 3}, {576, 9, 3, 1}, DataType::FLOAT);
    verifyTensor(
        findTensorByUid(*graphT, 5), 5, {1, 64, 32, 32}, {65536, 1024, 32, 1}, DataType::FLOAT);

    // Verify first node: x=1, w=2, y=3 (tensor 3 is the shared tensor used as Y here)
    verifyConvFwdNode(*graphT->nodes[0], DataType::FLOAT, 1, 2, 3, {1, 1}, {1, 1}, {1, 1}, {1, 1});

    // Verify second node: x=3, w=4, y=5 (tensor 3 is reused as X here)
    verifyConvFwdNode(*graphT->nodes[1], DataType::FLOAT, 3, 4, 5, {1, 1}, {1, 1}, {1, 1}, {1, 1});
}

TEST_F(TestGraphDescriptorOps, SetOperationsRejectsNonOperationDescriptor)
{
    auto graphWrapper = createDescriptor<GraphDescriptor>();
    auto graphDesc = graphWrapper->asDescriptor<GraphDescriptor>();

    // Set handle first
    auto handle = reinterpret_cast<hipdnnHandle_t>(&_mockHandle);
    graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle);

    // Create a tensor descriptor (does NOT implement IGraphOperation)
    auto tensorWrapper = TestGraphDescriptorOps::createFinalizedTensor(99);
    HipdnnBackendDescriptor* tensorDescPtr = tensorWrapper.get();

    // Attempting to set a non-operation descriptor as an operation should throw
    ASSERT_THROW_HIPDNN_STATUS(
        graphDesc->setAttribute(
            HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tensorDescPtr),
        HIPDNN_STATUS_NOT_SUPPORTED);
}
