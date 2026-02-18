// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ConvolutionFwdOperationDescriptor.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void ConvolutionFwdOperationDescriptor::finalize()
{
    THROW_IF_NULL(_xDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionFwdOperationDescriptor::finalize() failed: X tensor not set");
    THROW_IF_NULL(_wDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionFwdOperationDescriptor::finalize() failed: W tensor not set");
    THROW_IF_NULL(_yDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionFwdOperationDescriptor::finalize() failed: Y tensor not set");
    THROW_IF_TRUE(_data.pre_padding.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionFwdOperationDescriptor::finalize() failed: pre_padding not set");
    THROW_IF_TRUE(_data.post_padding.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionFwdOperationDescriptor::finalize() failed: post_padding not set");
    THROW_IF_TRUE(_data.stride.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionFwdOperationDescriptor::finalize() failed: stride not set");
    THROW_IF_TRUE(_data.dilation.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ConvolutionFwdOperationDescriptor::finalize() failed: dilation not set");

    HipdnnBackendDescriptorImpl<ConvolutionFwdOperationDescriptor>::finalize();
}

void ConvolutionFwdOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                     hipdnnBackendAttributeType_t attributeType,
                                                     int64_t requestedElementCount,
                                                     int64_t* elementCount,
                                                     void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "ConvolutionFwdOperationDescriptor::getAttribute() failed: Not finalized.");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "ConvolutionFwdOperationDescriptor::getAttribute(): arrayOfElements is null");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X:
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W:
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ConvolutionFwdOperationDescriptor::getAttribute(): attributeType is not "
                       "BACKEND_DESCRIPTOR");
        THROW_IF_FALSE(
            requestedElementCount >= 1,
            HIPDNN_STATUS_BAD_PARAM,
            "ConvolutionFwdOperationDescriptor::getAttribute(): requestedElementCount < 1");
        if(elementCount != nullptr)
        {
            *elementCount = 1;
        }
        {
            std::shared_ptr<TensorDescriptor> desc;
            if(attributeName == HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X)
            {
                desc = _xDesc;
            }
            else if(attributeName == HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W)
            {
                desc = _wDesc;
            }
            else
            {
                desc = _yDesc;
            }
            HipdnnBackendDescriptor::packDescriptor(desc, arrayOfElements);
        }
        break;

    case HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ConvolutionFwdOperationDescriptor::getAttribute(): attributeType mismatch");
        {
            auto copyCount = std::min<size_t>(static_cast<size_t>(requestedElementCount),
                                              _data.pre_padding.size());
            if(elementCount != nullptr)
            {
                *elementCount = static_cast<int64_t>(copyCount);
            }
            std::memcpy(arrayOfElements, _data.pre_padding.data(), copyCount * sizeof(int64_t));
        }
        break;

    case HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ConvolutionFwdOperationDescriptor::getAttribute(): attributeType mismatch");
        {
            auto copyCount = std::min<size_t>(static_cast<size_t>(requestedElementCount),
                                              _data.post_padding.size());
            if(elementCount != nullptr)
            {
                *elementCount = static_cast<int64_t>(copyCount);
            }
            std::memcpy(arrayOfElements, _data.post_padding.data(), copyCount * sizeof(int64_t));
        }
        break;

    case HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ConvolutionFwdOperationDescriptor::getAttribute(): attributeType mismatch");
        {
            auto copyCount
                = std::min<size_t>(static_cast<size_t>(requestedElementCount), _data.stride.size());
            if(elementCount != nullptr)
            {
                *elementCount = static_cast<int64_t>(copyCount);
            }
            std::memcpy(arrayOfElements, _data.stride.data(), copyCount * sizeof(int64_t));
        }
        break;

    case HIPDNN_ATTR_CONVOLUTION_DILATIONS:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ConvolutionFwdOperationDescriptor::getAttribute(): attributeType mismatch");
        {
            auto copyCount = std::min<size_t>(static_cast<size_t>(requestedElementCount),
                                              _data.dilation.size());
            if(elementCount != nullptr)
            {
                *elementCount = static_cast<int64_t>(copyCount);
            }
            std::memcpy(arrayOfElements, _data.dilation.data(), copyCount * sizeof(int64_t));
        }
        break;

    case HIPDNN_ATTR_CONVOLUTION_CONV_MODE:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ConvolutionFwdOperationDescriptor::getAttribute(): attributeType mismatch");
        THROW_IF_FALSE(
            requestedElementCount >= 1,
            HIPDNN_STATUS_BAD_PARAM,
            "ConvolutionFwdOperationDescriptor::getAttribute(): requestedElementCount < 1");
        if(elementCount != nullptr)
        {
            *elementCount = 1;
        }
        *static_cast<int64_t*>(arrayOfElements) = static_cast<int64_t>(_data.conv_mode);
        break;

    case HIPDNN_ATTR_CONVOLUTION_COMP_TYPE:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_DATA_TYPE,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ConvolutionFwdOperationDescriptor::getAttribute(): attributeType mismatch");
        THROW_IF_FALSE(
            requestedElementCount >= 1,
            HIPDNN_STATUS_BAD_PARAM,
            "ConvolutionFwdOperationDescriptor::getAttribute(): requestedElementCount < 1");
        if(elementCount != nullptr)
        {
            *elementCount = 1;
        }
        *static_cast<hipdnn_data_sdk::data_objects::DataType*>(arrayOfElements) = _computeDataType;
        break;

    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "ConvolutionFwdOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

void ConvolutionFwdOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                     hipdnnBackendAttributeType_t attributeType,
                                                     int64_t elementCount,
                                                     const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "ConvolutionFwdOperationDescriptor::setAttribute() failed: Already finalized.");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "ConvolutionFwdOperationDescriptor::setAttribute(): arrayOfElements is null");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X:
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W:
    case HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ConvolutionFwdOperationDescriptor::setAttribute(): attributeType is not "
                       "BACKEND_DESCRIPTOR");
        THROW_IF_FALSE(elementCount == 1,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ConvolutionFwdOperationDescriptor::setAttribute(): elementCount is not 1");
        {
            auto tensorDesc = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
                arrayOfElements,
                HIPDNN_STATUS_BAD_PARAM,
                "ConvolutionFwdOperationDescriptor::setAttribute(): Failed to unpack tensor "
                "descriptor");
            THROW_IF_FALSE(tensorDesc->isFinalized(),
                           HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                           "ConvolutionFwdOperationDescriptor::setAttribute(): Tensor descriptor "
                           "not finalized");

            if(attributeName == HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X)
            {
                _xDesc = tensorDesc;
                _data.x_tensor_uid = tensorDesc->getData().uid;
            }
            else if(attributeName == HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W)
            {
                _wDesc = tensorDesc;
                _data.w_tensor_uid = tensorDesc->getData().uid;
            }
            else
            {
                _yDesc = tensorDesc;
                _data.y_tensor_uid = tensorDesc->getData().uid;
            }
        }
        break;

    case HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ConvolutionFwdOperationDescriptor::setAttribute(): attributeType mismatch");
        _data.pre_padding.assign(static_cast<const int64_t*>(arrayOfElements),
                                 static_cast<const int64_t*>(arrayOfElements) + elementCount);
        break;

    case HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ConvolutionFwdOperationDescriptor::setAttribute(): attributeType mismatch");
        _data.post_padding.assign(static_cast<const int64_t*>(arrayOfElements),
                                  static_cast<const int64_t*>(arrayOfElements) + elementCount);
        break;

    case HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ConvolutionFwdOperationDescriptor::setAttribute(): attributeType mismatch");
        _data.stride.assign(static_cast<const int64_t*>(arrayOfElements),
                            static_cast<const int64_t*>(arrayOfElements) + elementCount);
        break;

    case HIPDNN_ATTR_CONVOLUTION_DILATIONS:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ConvolutionFwdOperationDescriptor::setAttribute(): attributeType mismatch");
        _data.dilation.assign(static_cast<const int64_t*>(arrayOfElements),
                              static_cast<const int64_t*>(arrayOfElements) + elementCount);
        break;

    case HIPDNN_ATTR_CONVOLUTION_CONV_MODE:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ConvolutionFwdOperationDescriptor::setAttribute(): attributeType mismatch");
        THROW_IF_FALSE(elementCount == 1,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ConvolutionFwdOperationDescriptor::setAttribute(): elementCount is not 1");
        _data.conv_mode = static_cast<hipdnn_data_sdk::data_objects::ConvMode>(
            *static_cast<const int64_t*>(arrayOfElements));
        break;

    case HIPDNN_ATTR_CONVOLUTION_COMP_TYPE:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_DATA_TYPE,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ConvolutionFwdOperationDescriptor::setAttribute(): attributeType mismatch");
        THROW_IF_FALSE(elementCount == 1,
                       HIPDNN_STATUS_BAD_PARAM,
                       "ConvolutionFwdOperationDescriptor::setAttribute(): elementCount is not 1");
        _computeDataType
            = *static_cast<const hipdnn_data_sdk::data_objects::DataType*>(arrayOfElements);
        break;

    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "ConvolutionFwdOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

hipdnnBackendDescriptorType_t ConvolutionFwdOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
}

std::string ConvolutionFwdOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "ConvolutionFwdOperationDescriptor: {";
    str += "x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", w_uid=" + std::to_string(_data.w_tensor_uid);
    str += ", y_uid=" + std::to_string(_data.y_tensor_uid);
    str += ", pre_padding=" + vecToString(_data.pre_padding);
    str += ", post_padding=" + vecToString(_data.post_padding);
    str += ", stride=" + vecToString(_data.stride);
    str += ", dilation=" + vecToString(_data.dilation);
    str += ", conv_mode=" + std::to_string(static_cast<int>(_data.conv_mode));
    str += "}";
    return str;
}

} // namespace hipdnn_backend
