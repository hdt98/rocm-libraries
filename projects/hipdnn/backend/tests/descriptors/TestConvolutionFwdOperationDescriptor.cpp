// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TestMacros.hpp"
#include "descriptors/ConvolutionFwdOperationDescriptor.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;

class TestConvolutionFwdOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<ConvolutionFwdOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<ConvolutionFwdOperationDescriptor>();
    }

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

    void setTensors() const
    {
        auto desc = getDescriptor();
        desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_xDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_wDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_yDesc);
    }

    void setConvParams() const
    {
        auto desc = getDescriptor();
        std::vector<int64_t> prePadding = {1, 1};
        std::vector<int64_t> postPadding = {1, 1};
        std::vector<int64_t> stride = {1, 1};
        std::vector<int64_t> dilation = {1, 1};

        desc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
        desc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
        desc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());
        desc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, dilation.data());
    }

    void setRequiredAttributes() const
    {
        setTensors();
        setConvParams();
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _xDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _wDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _yDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<ConvolutionFwdOperationDescriptor>();
        _xDesc = createFinalizedTensor(1, {1, 3, 32, 32}, {3072, 1024, 32, 1});
        _wDesc = createFinalizedTensor(2, {64, 3, 3, 3}, {27, 9, 3, 1});
        _yDesc = createFinalizedTensor(3, {1, 64, 32, 32}, {65536, 1024, 32, 1});
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _xDesc.reset();
        _wDesc.reset();
        _yDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestConvolutionFwdOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
}

TEST_F(TestConvolutionFwdOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestConvolutionFwdOperationDescriptor, FinalizeFailsWithoutXTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_wDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc);
    setConvParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionFwdOperationDescriptor, FinalizeFailsWithoutWTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc);
    setConvParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionFwdOperationDescriptor, FinalizeFailsWithoutYTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_wDesc);
    setConvParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionFwdOperationDescriptor, FinalizeFailsWithoutPrePadding)
{
    auto desc = getDescriptor();
    setTensors();
    std::vector<int64_t> postPadding = {1, 1};
    std::vector<int64_t> stride = {1, 1};
    std::vector<int64_t> dilation = {1, 1};

    desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, dilation.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionFwdOperationDescriptor, FinalizeFailsWithoutPostPadding)
{
    auto desc = getDescriptor();
    setTensors();
    std::vector<int64_t> prePadding = {1, 1};
    std::vector<int64_t> stride = {1, 1};
    std::vector<int64_t> dilation = {1, 1};

    desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, dilation.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionFwdOperationDescriptor, FinalizeFailsWithoutStride)
{
    auto desc = getDescriptor();
    setTensors();
    std::vector<int64_t> prePadding = {1, 1};
    std::vector<int64_t> postPadding = {1, 1};
    std::vector<int64_t> dilation = {1, 1};

    desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
    desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, dilation.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionFwdOperationDescriptor, FinalizeFailsWithoutDilation)
{
    auto desc = getDescriptor();
    setTensors();
    std::vector<int64_t> prePadding = {1, 1};
    std::vector<int64_t> postPadding = {1, 1};
    std::vector<int64_t> stride = {1, 1};

    desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
    desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestConvolutionFwdOperationDescriptor, SetTensorDescriptorX)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().x_tensor_uid, 1);
    ASSERT_NE(desc->getXDesc(), nullptr);
}

TEST_F(TestConvolutionFwdOperationDescriptor, SetTensorDescriptorW)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_wDesc));

    ASSERT_EQ(desc->getData().w_tensor_uid, 2);
    ASSERT_NE(desc->getWDesc(), nullptr);
}

TEST_F(TestConvolutionFwdOperationDescriptor, SetTensorDescriptorY)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc));

    ASSERT_EQ(desc->getData().y_tensor_uid, 3);
    ASSERT_NE(desc->getYDesc(), nullptr);
}

TEST_F(TestConvolutionFwdOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestConvolutionFwdOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, HIPDNN_TYPE_INT64, 1, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionFwdOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  2,
                                                  &_xDesc),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionFwdOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Convolution Parameters
// =============================================================================

TEST_F(TestConvolutionFwdOperationDescriptor, SetConvolutionPrePadding)
{
    auto desc = getDescriptor();
    std::vector<int64_t> prePadding = {2, 3};

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data()));

    auto& data = desc->getData();
    ASSERT_EQ(data.pre_padding.size(), 2);
    ASSERT_EQ(data.pre_padding[0], 2);
    ASSERT_EQ(data.pre_padding[1], 3);
}

TEST_F(TestConvolutionFwdOperationDescriptor, SetConvolutionPostPadding)
{
    auto desc = getDescriptor();
    std::vector<int64_t> postPadding = {4, 5};

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data()));

    auto& data = desc->getData();
    ASSERT_EQ(data.post_padding.size(), 2);
    ASSERT_EQ(data.post_padding[0], 4);
    ASSERT_EQ(data.post_padding[1], 5);
}

TEST_F(TestConvolutionFwdOperationDescriptor, SetConvolutionStride)
{
    auto desc = getDescriptor();
    std::vector<int64_t> stride = {2, 2};

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data()));

    auto& data = desc->getData();
    ASSERT_EQ(data.stride.size(), 2);
    ASSERT_EQ(data.stride[0], 2);
    ASSERT_EQ(data.stride[1], 2);
}

TEST_F(TestConvolutionFwdOperationDescriptor, SetConvolutionDilation)
{
    auto desc = getDescriptor();
    std::vector<int64_t> dilation = {3, 3};

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, dilation.data()));

    auto& data = desc->getData();
    ASSERT_EQ(data.dilation.size(), 2);
    ASSERT_EQ(data.dilation[0], 3);
    ASSERT_EQ(data.dilation[1], 3);
}

TEST_F(TestConvolutionFwdOperationDescriptor, SetConvMode)
{
    auto desc = getDescriptor();
    auto convMode = static_cast<int64_t>(ConvMode::CROSS_CORRELATION);

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_INT64, 1, &convMode));

    ASSERT_EQ(desc->getData().conv_mode, ConvMode::CROSS_CORRELATION);
}

TEST_F(TestConvolutionFwdOperationDescriptor, SetConvModeWrongElementCount)
{
    auto desc = getDescriptor();
    int64_t convMode = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_INT64, 2, &convMode),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionFwdOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = DataType::FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestConvolutionFwdOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = DataType::FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionFwdOperationDescriptor, SetConvParamsWrongType)
{
    auto desc = getDescriptor();
    std::vector<int64_t> padding = {1, 1};

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_CHAR, 2, padding.data()),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestConvolutionFwdOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_xDesc),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestConvolutionFwdOperationDescriptor, SetAttributeUnsupported)
{
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// GetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestConvolutionFwdOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedX = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       &retrievedX));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedX, nullptr);
}

// =============================================================================
// GetAttribute Tests - Convolution Parameters
// =============================================================================

TEST_F(TestConvolutionFwdOperationDescriptor, GetAttributeConvParams)
{
    makeFinalized();
    auto desc = getDescriptor();

    std::vector<int64_t> prePadding(2);
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                       HIPDNN_TYPE_INT64,
                                       2,
                                       &elementCount,
                                       prePadding.data()));

    ASSERT_EQ(elementCount, 2);
    ASSERT_EQ(prePadding[0], 1);
    ASSERT_EQ(prePadding[1], 1);
}

TEST_F(TestConvolutionFwdOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    auto computeType = DataType::HALF;
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    setRequiredAttributes();
    desc->finalize();

    DataType retrieved = DataType::UNSET;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &elementCount, &retrieved));

    ASSERT_EQ(retrieved, DataType::HALF);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestConvolutionFwdOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestConvolutionFwdOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestConvolutionFwdOperationDescriptor, GetAttributeUnsupported)
{
    makeFinalized();
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestConvolutionFwdOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getXDesc(), nullptr);
    ASSERT_NE(desc->getWDesc(), nullptr);
    ASSERT_NE(desc->getYDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getXDesc()->getData().uid, 1);
    ASSERT_EQ(desc->getWDesc()->getData().uid, 2);
    ASSERT_EQ(desc->getYDesc()->getData().uid, 3);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestConvolutionFwdOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    std::string str = desc->toString();
    ASSERT_NE(str.find("ConvolutionFwdOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("x_uid=1"), std::string::npos);
    ASSERT_NE(str.find("w_uid=2"), std::string::npos);
    ASSERT_NE(str.find("y_uid=3"), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestConvolutionFwdOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, 1);
    ASSERT_EQ(tensors[1]->getData().uid, 2);
    ASSERT_EQ(tensors[2]->getData().uid, 3);
}

TEST_F(TestConvolutionFwdOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = DataType::FLOAT;
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::ConvolutionFwdAttributes);

    auto* convAttrs = node->attributes.AsConvolutionFwdAttributes();
    ASSERT_NE(convAttrs, nullptr);
    ASSERT_EQ(convAttrs->x_tensor_uid, 1);
    ASSERT_EQ(convAttrs->w_tensor_uid, 2);
    ASSERT_EQ(convAttrs->y_tensor_uid, 3);
    ASSERT_EQ(convAttrs->pre_padding.size(), 2);
    ASSERT_EQ(convAttrs->stride.size(), 2);
    ASSERT_EQ(convAttrs->dilation.size(), 2);
}

TEST_F(TestConvolutionFwdOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = DataType::HALF;
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestConvolutionFwdOperationDescriptor, GetTensorDescriptorsOrderIsXWY)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    // Verify ordering: [X, W, Y] matches UIDs [1, 2, 3]
    EXPECT_EQ(tensors[0], desc->getXDesc());
    EXPECT_EQ(tensors[1], desc->getWDesc());
    EXPECT_EQ(tensors[2], desc->getYDesc());
}

TEST_F(TestConvolutionFwdOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsInterface<IGraphOperation>();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, 1);
}

TEST_F(TestConvolutionFwdOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _xDesc->tryAsInterface<IGraphOperation>();
    EXPECT_EQ(graphOp, nullptr);
}
