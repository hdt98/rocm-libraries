// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TensorDescriptor.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void TensorDescriptor::finalize()
{
    THROW_IF_TRUE(_data.dims.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "TensorDescriptor::finalize() failed: dimensions not set");
    THROW_IF_TRUE(_data.strides.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "TensorDescriptor::finalize() failed: strides not set");
    THROW_IF_NE(_data.dims.size(),
                _data.strides.size(),
                HIPDNN_STATUS_BAD_PARAM,
                "TensorDescriptor::finalize() failed: dims and strides size mismatch");
    THROW_IF_TRUE(_data.data_type == hipdnn_data_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "TensorDescriptor::finalize() failed: data type not set");

    HipdnnBackendDescriptorImpl<TensorDescriptor>::finalize();
}

void TensorDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                    hipdnnBackendAttributeType_t attributeType,
                                    int64_t requestedElementCount,
                                    int64_t* elementCount,
                                    void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "TensorDescriptor::getAttribute() failed: Not finalized.");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "TensorDescriptor::getAttribute(): arrayOfElements is null");

    switch(attributeName)
    {
    case HIPDNN_ATTR_TENSOR_UNIQUE_ID:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                       HIPDNN_STATUS_BAD_PARAM,
                       "TensorDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_INT64");
        THROW_IF_FALSE(requestedElementCount >= 1,
                       HIPDNN_STATUS_BAD_PARAM,
                       "TensorDescriptor::getAttribute(): requestedElementCount < 1");
        if(elementCount != nullptr)
        {
            *elementCount = 1;
        }
        *static_cast<int64_t*>(arrayOfElements) = _data.uid;
        break;

    case HIPDNN_ATTR_TENSOR_NAME:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_CHAR,
                       HIPDNN_STATUS_BAD_PARAM,
                       "TensorDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_CHAR");
        {
            auto copyLen = std::min<size_t>(static_cast<size_t>(requestedElementCount),
                                            _data.name.size() + 1);
            if(elementCount != nullptr)
            {
                *elementCount = static_cast<int64_t>(copyLen);
            }
            std::memcpy(arrayOfElements, _data.name.c_str(), copyLen);
        }
        break;

    case HIPDNN_ATTR_TENSOR_DATA_TYPE:
        THROW_IF_FALSE(
            attributeType == HIPDNN_TYPE_DATA_TYPE,
            HIPDNN_STATUS_BAD_PARAM,
            "TensorDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_DATA_TYPE");
        THROW_IF_FALSE(requestedElementCount >= 1,
                       HIPDNN_STATUS_BAD_PARAM,
                       "TensorDescriptor::getAttribute(): requestedElementCount < 1");
        if(elementCount != nullptr)
        {
            *elementCount = 1;
        }
        *static_cast<hipdnn_data_sdk::data_objects::DataType*>(arrayOfElements) = _data.data_type;
        break;

    case HIPDNN_ATTR_TENSOR_DIMENSIONS:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                       HIPDNN_STATUS_BAD_PARAM,
                       "TensorDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_INT64");
        {
            auto copyCount
                = std::min<size_t>(static_cast<size_t>(requestedElementCount), _data.dims.size());
            if(elementCount != nullptr)
            {
                *elementCount = static_cast<int64_t>(copyCount);
            }
            std::memcpy(arrayOfElements, _data.dims.data(), copyCount * sizeof(int64_t));
        }
        break;

    case HIPDNN_ATTR_TENSOR_STRIDES:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                       HIPDNN_STATUS_BAD_PARAM,
                       "TensorDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_INT64");
        {
            auto copyCount = std::min<size_t>(static_cast<size_t>(requestedElementCount),
                                              _data.strides.size());
            if(elementCount != nullptr)
            {
                *elementCount = static_cast<int64_t>(copyCount);
            }
            std::memcpy(arrayOfElements, _data.strides.data(), copyCount * sizeof(int64_t));
        }
        break;

    case HIPDNN_ATTR_TENSOR_IS_VIRTUAL:
        THROW_IF_FALSE(
            attributeType == HIPDNN_TYPE_BOOLEAN,
            HIPDNN_STATUS_BAD_PARAM,
            "TensorDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_BOOLEAN");
        THROW_IF_FALSE(requestedElementCount >= 1,
                       HIPDNN_STATUS_BAD_PARAM,
                       "TensorDescriptor::getAttribute(): requestedElementCount < 1");
        if(elementCount != nullptr)
        {
            *elementCount = 1;
        }
        *static_cast<bool*>(arrayOfElements) = _data.virtual_;
        break;

    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "TensorDescriptor::getAttribute: attributeName not supported");
    }
}

void TensorDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                    hipdnnBackendAttributeType_t attributeType,
                                    int64_t elementCount,
                                    const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "TensorDescriptor::setAttribute() failed: Already finalized.");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "TensorDescriptor::setAttribute(): arrayOfElements is null");

    switch(attributeName)
    {
    case HIPDNN_ATTR_TENSOR_UNIQUE_ID:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                       HIPDNN_STATUS_BAD_PARAM,
                       "TensorDescriptor::setAttribute(): attributeType is not HIPDNN_TYPE_INT64");
        THROW_IF_FALSE(elementCount == 1,
                       HIPDNN_STATUS_BAD_PARAM,
                       "TensorDescriptor::setAttribute(): elementCount is not 1");
        _data.uid = *static_cast<const int64_t*>(arrayOfElements);
        break;

    case HIPDNN_ATTR_TENSOR_NAME:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_CHAR,
                       HIPDNN_STATUS_BAD_PARAM,
                       "TensorDescriptor::setAttribute(): attributeType is not HIPDNN_TYPE_CHAR");
        _data.name = std::string(static_cast<const char*>(arrayOfElements),
                                 static_cast<size_t>(elementCount));
        break;

    case HIPDNN_ATTR_TENSOR_DATA_TYPE:
        THROW_IF_FALSE(
            attributeType == HIPDNN_TYPE_DATA_TYPE,
            HIPDNN_STATUS_BAD_PARAM,
            "TensorDescriptor::setAttribute(): attributeType is not HIPDNN_TYPE_DATA_TYPE");
        THROW_IF_FALSE(elementCount == 1,
                       HIPDNN_STATUS_BAD_PARAM,
                       "TensorDescriptor::setAttribute(): elementCount is not 1");
        _data.data_type
            = *static_cast<const hipdnn_data_sdk::data_objects::DataType*>(arrayOfElements);
        break;

    case HIPDNN_ATTR_TENSOR_DIMENSIONS:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                       HIPDNN_STATUS_BAD_PARAM,
                       "TensorDescriptor::setAttribute(): attributeType is not HIPDNN_TYPE_INT64");
        _data.dims.assign(static_cast<const int64_t*>(arrayOfElements),
                          static_cast<const int64_t*>(arrayOfElements) + elementCount);
        break;

    case HIPDNN_ATTR_TENSOR_STRIDES:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                       HIPDNN_STATUS_BAD_PARAM,
                       "TensorDescriptor::setAttribute(): attributeType is not HIPDNN_TYPE_INT64");
        _data.strides.assign(static_cast<const int64_t*>(arrayOfElements),
                             static_cast<const int64_t*>(arrayOfElements) + elementCount);
        break;

    case HIPDNN_ATTR_TENSOR_IS_VIRTUAL:
        THROW_IF_FALSE(
            attributeType == HIPDNN_TYPE_BOOLEAN,
            HIPDNN_STATUS_BAD_PARAM,
            "TensorDescriptor::setAttribute(): attributeType is not HIPDNN_TYPE_BOOLEAN");
        THROW_IF_FALSE(elementCount == 1,
                       HIPDNN_STATUS_BAD_PARAM,
                       "TensorDescriptor::setAttribute(): elementCount is not 1");
        _data.virtual_ = *static_cast<const bool*>(arrayOfElements);
        break;

    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "TensorDescriptor::setAttribute: attributeName not supported");
    }
}

hipdnnBackendDescriptorType_t TensorDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_TENSOR_DESCRIPTOR;
}

std::string TensorDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "TensorDescriptor: {uid=" + std::to_string(_data.uid);
    str += ", name=" + _data.name;
    str += ", dataType=" + std::to_string(static_cast<int>(_data.data_type));
    str += ", dims=" + vecToString(_data.dims);
    str += ", strides=" + vecToString(_data.strides);
    str += ", virtual=" + std::string(_data.virtual_ ? "true" : "false");
    str += "}";
    return str;
}

} // namespace hipdnn_backend
