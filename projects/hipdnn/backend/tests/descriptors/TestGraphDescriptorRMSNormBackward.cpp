// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/RMSNormBackwardOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/rmsnorm_backward_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <memory>
#include <set>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;
using hipdnn_tests::toVec;

namespace
{

// Helper: create a finalized RMSNormBackwardOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedRMSNormBackwardOp(HipdnnBackendDescriptor* dyDesc,
                                     HipdnnBackendDescriptor* xDesc,
                                     HipdnnBackendDescriptor* scaleDesc,
                                     HipdnnBackendDescriptor* invRmsDesc,
                                     HipdnnBackendDescriptor* dxDesc,
                                     HipdnnBackendDescriptor* dscaleDesc,
                                     HipdnnBackendDescriptor* dbiasDesc,
                                     hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
{
    auto wrapper = createDescriptor<RMSNormBackwardOperationDescriptor>();
    auto desc = wrapper->asDescriptor<RMSNormBackwardOperationDescriptor>();

    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dyDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &scaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_INV_RMS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &invRmsDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DX_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dxDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DSCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dscaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DBIAS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dbiasDesc);
    desc->setAttribute(
        HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorRMSNormBackward : public ::testing::Test
{
public:
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

TEST_F(TestGraphDescriptorRMSNormBackward, BuildFromSingleOperation)
{
    auto dyDesc = createFinalizedTensor(90, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto xDesc = createFinalizedTensor(91, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto scaleDesc = createFinalizedTensor(92, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto invRmsDesc = createFinalizedTensor(93, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto dxDesc = createFinalizedTensor(94, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto dscaleDesc = createFinalizedTensor(95, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto dbiasDesc = createFinalizedTensor(96, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto opDesc = createFinalizedRMSNormBackwardOp(dyDesc.get(),
                                                   xDesc.get(),
                                                   scaleDesc.get(),
                                                   invRmsDesc.get(),
                                                   dxDesc.get(),
                                                   dscaleDesc.get(),
                                                   dbiasDesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data()));
    ASSERT_NO_THROW(desc->finalize());

    // Verify the built graph
    auto serialized = desc->getSerializedGraph();
    ASSERT_NE(serialized.ptr, nullptr);
    ASSERT_GT(serialized.size, 0UL);

    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(serialized.ptr), serialized.size);
    ASSERT_TRUE(verifier.VerifyBuffer<Graph>());

    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 7);

    // Verify tensor attributes
    verifyTensor(
        findTensorByUid(*graphT, 90), 90, {1, 64, 32, 32}, {65536, 1024, 32, 1}, DataType::FLOAT);
    verifyTensor(
        findTensorByUid(*graphT, 91), 91, {1, 64, 32, 32}, {65536, 1024, 32, 1}, DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, 92), 92, {1, 64, 1, 1}, {64, 1, 1, 1}, DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, 93), 93, {1, 64, 1, 1}, {64, 1, 1, 1}, DataType::FLOAT);
    verifyTensor(
        findTensorByUid(*graphT, 94), 94, {1, 64, 32, 32}, {65536, 1024, 32, 1}, DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, 95), 95, {1, 64, 1, 1}, {64, 1, 1, 1}, DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, 96), 96, {1, 64, 1, 1}, {64, 1, 1, 1}, DataType::FLOAT);

    // Verify node attributes
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::RMSNormBackwardAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsRMSNormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::FLOAT);

    // Verify tensor UID references
    EXPECT_EQ(attrs->dy_tensor_uid, 90);
    EXPECT_EQ(attrs->x_tensor_uid, 91);
    EXPECT_EQ(attrs->scale_tensor_uid, 92);
    EXPECT_EQ(attrs->inv_rms_tensor_uid, 93);
    EXPECT_EQ(attrs->dx_tensor_uid, 94);
    EXPECT_EQ(attrs->dscale_tensor_uid, 95);
    EXPECT_EQ(attrs->dbias_tensor_uid, 96);

    // Verify default node name is empty
    EXPECT_TRUE(graphT->nodes[0]->name.empty());
}

TEST_F(TestGraphDescriptorRMSNormBackward, ComputeDataTypePreserved)
{
    auto dyDesc = createFinalizedTensor(90, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto xDesc = createFinalizedTensor(91, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto scaleDesc = createFinalizedTensor(92, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto invRmsDesc = createFinalizedTensor(93, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto dxDesc = createFinalizedTensor(94, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto dscaleDesc = createFinalizedTensor(95, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto dbiasDesc = createFinalizedTensor(96, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto opDesc = createFinalizedRMSNormBackwardOp(dyDesc.get(),
                                                   xDesc.get(),
                                                   scaleDesc.get(),
                                                   invRmsDesc.get(),
                                                   dxDesc.get(),
                                                   dscaleDesc.get(),
                                                   dbiasDesc.get(),
                                                   HIPDNN_DATA_HALF);

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data());
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::HALF);
}

} // namespace
