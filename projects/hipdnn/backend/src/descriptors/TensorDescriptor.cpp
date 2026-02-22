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

    switch(attributeName)
    {
    case HIPDNN_ATTR_TENSOR_UNIQUE_ID:
        getUniqueId(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_TENSOR_NAME_EXT:
        getName(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_TENSOR_DATA_TYPE:
        getDataType(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_TENSOR_DIMENSIONS:
        getDimensions(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_TENSOR_STRIDES:
        getStrides(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_TENSOR_IS_VIRTUAL:
        getIsVirtual(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_TENSOR_VALUE_EXT:
        getTensorValue(attributeType, requestedElementCount, elementCount, arrayOfElements);
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

    switch(attributeName)
    {
    case HIPDNN_ATTR_TENSOR_UNIQUE_ID:
        setUniqueId(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_TENSOR_NAME_EXT:
        setName(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_TENSOR_DATA_TYPE:
        setDataType(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_TENSOR_DIMENSIONS:
        setDimensions(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_TENSOR_STRIDES:
        setStrides(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_TENSOR_IS_VIRTUAL:
        setIsVirtual(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_TENSOR_VALUE_EXT:
        setTensorValue(attributeType, elementCount, arrayOfElements);
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "TensorDescriptor::setAttribute: attributeName not supported");
    }
}

void TensorDescriptor::setUniqueId(hipdnnBackendAttributeType_t attributeType,
                                   int64_t elementCount,
                                   const void* arrayOfElements)
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::setAttribute(): attributeType is not HIPDNN_TYPE_INT64");
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::setAttribute(): elementCount is not 1");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "TensorDescriptor::setAttribute(): arrayOfElements is null");

    _data.uid = *static_cast<const int64_t*>(arrayOfElements);
}

void TensorDescriptor::getUniqueId(hipdnnBackendAttributeType_t attributeType,
                                   int64_t requestedElementCount,
                                   int64_t* elementCount,
                                   void* arrayOfElements) const
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_INT64");
    THROW_IF_FALSE(requestedElementCount >= 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::getAttribute(): requestedElementCount < 1");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "TensorDescriptor::getAttribute(): arrayOfElements is null");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
    *static_cast<int64_t*>(arrayOfElements) = _data.uid;
}

void TensorDescriptor::setName(hipdnnBackendAttributeType_t attributeType,
                               int64_t elementCount,
                               const void* arrayOfElements)
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_CHAR,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::setAttribute(): attributeType is not HIPDNN_TYPE_CHAR");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "TensorDescriptor::setAttribute(): arrayOfElements is null");

    _data.name
        = std::string(static_cast<const char*>(arrayOfElements), static_cast<size_t>(elementCount));
}

void TensorDescriptor::getName(hipdnnBackendAttributeType_t attributeType,
                               int64_t requestedElementCount,
                               int64_t* elementCount,
                               void* arrayOfElements) const
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_CHAR,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_CHAR");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "TensorDescriptor::getAttribute(): arrayOfElements is null");

    auto copyLen
        = std::min<size_t>(static_cast<size_t>(requestedElementCount), _data.name.size() + 1);
    if(elementCount != nullptr)
    {
        *elementCount = static_cast<int64_t>(copyLen);
    }
    std::memcpy(arrayOfElements, _data.name.c_str(), copyLen);
}

void TensorDescriptor::setDataType(hipdnnBackendAttributeType_t attributeType,
                                   int64_t elementCount,
                                   const void* arrayOfElements)
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_DATA_TYPE,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::setAttribute(): attributeType is not HIPDNN_TYPE_DATA_TYPE");
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::setAttribute(): elementCount is not 1");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "TensorDescriptor::setAttribute(): arrayOfElements is null");

    _data.data_type = *static_cast<const hipdnn_data_sdk::data_objects::DataType*>(arrayOfElements);
}

void TensorDescriptor::getDataType(hipdnnBackendAttributeType_t attributeType,
                                   int64_t requestedElementCount,
                                   int64_t* elementCount,
                                   void* arrayOfElements) const
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_DATA_TYPE,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_DATA_TYPE");
    THROW_IF_FALSE(requestedElementCount >= 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::getAttribute(): requestedElementCount < 1");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "TensorDescriptor::getAttribute(): arrayOfElements is null");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
    *static_cast<hipdnn_data_sdk::data_objects::DataType*>(arrayOfElements) = _data.data_type;
}

void TensorDescriptor::setDimensions(hipdnnBackendAttributeType_t attributeType,
                                     int64_t elementCount,
                                     const void* arrayOfElements)
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::setAttribute(): attributeType is not HIPDNN_TYPE_INT64");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "TensorDescriptor::setAttribute(): arrayOfElements is null");

    _data.dims.assign(static_cast<const int64_t*>(arrayOfElements),
                      static_cast<const int64_t*>(arrayOfElements) + elementCount);
}

void TensorDescriptor::getDimensions(hipdnnBackendAttributeType_t attributeType,
                                     int64_t requestedElementCount,
                                     int64_t* elementCount,
                                     void* arrayOfElements) const
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_INT64");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "TensorDescriptor::getAttribute(): arrayOfElements is null");

    auto copyCount
        = std::min<size_t>(static_cast<size_t>(requestedElementCount), _data.dims.size());
    if(elementCount != nullptr)
    {
        *elementCount = static_cast<int64_t>(copyCount);
    }
    std::memcpy(arrayOfElements, _data.dims.data(), copyCount * sizeof(int64_t));
}

void TensorDescriptor::setStrides(hipdnnBackendAttributeType_t attributeType,
                                  int64_t elementCount,
                                  const void* arrayOfElements)
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::setAttribute(): attributeType is not HIPDNN_TYPE_INT64");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "TensorDescriptor::setAttribute(): arrayOfElements is null");

    _data.strides.assign(static_cast<const int64_t*>(arrayOfElements),
                         static_cast<const int64_t*>(arrayOfElements) + elementCount);
}

void TensorDescriptor::getStrides(hipdnnBackendAttributeType_t attributeType,
                                  int64_t requestedElementCount,
                                  int64_t* elementCount,
                                  void* arrayOfElements) const
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_INT64,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_INT64");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "TensorDescriptor::getAttribute(): arrayOfElements is null");

    auto copyCount
        = std::min<size_t>(static_cast<size_t>(requestedElementCount), _data.strides.size());
    if(elementCount != nullptr)
    {
        *elementCount = static_cast<int64_t>(copyCount);
    }
    std::memcpy(arrayOfElements, _data.strides.data(), copyCount * sizeof(int64_t));
}

void TensorDescriptor::setIsVirtual(hipdnnBackendAttributeType_t attributeType,
                                    int64_t elementCount,
                                    const void* arrayOfElements)
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_BOOLEAN,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::setAttribute(): attributeType is not HIPDNN_TYPE_BOOLEAN");
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::setAttribute(): elementCount is not 1");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "TensorDescriptor::setAttribute(): arrayOfElements is null");

    _data.virtual_ = *static_cast<const bool*>(arrayOfElements);
}

void TensorDescriptor::getIsVirtual(hipdnnBackendAttributeType_t attributeType,
                                    int64_t requestedElementCount,
                                    int64_t* elementCount,
                                    void* arrayOfElements) const
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_BOOLEAN,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::getAttribute(): attributeType is not HIPDNN_TYPE_BOOLEAN");
    THROW_IF_FALSE(requestedElementCount >= 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::getAttribute(): requestedElementCount < 1");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "TensorDescriptor::getAttribute(): arrayOfElements is null");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
    *static_cast<bool*>(arrayOfElements) = _data.virtual_;
}

void TensorDescriptor::setTensorValue(hipdnnBackendAttributeType_t attributeType,
                                      int64_t elementCount,
                                      const void* arrayOfElements)
{
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::setAttribute(): elementCount is not 1");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "TensorDescriptor::setAttribute(): arrayOfElements is null");
    THROW_IF_TRUE(_data.data_type == hipdnn_data_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "TensorDescriptor::setAttribute(): data type must be set before tensor value");

    using namespace hipdnn_data_sdk::data_objects;

    switch(attributeType)
    {
    case HIPDNN_TYPE_FLOAT:
    {
        auto val = *static_cast<const float*>(arrayOfElements);
        switch(_data.data_type)
        {
        case DataType::HALF:
            _data.value.Set(Float16Value(val));
            break;
        case DataType::BFLOAT16:
            _data.value.Set(BFloat16Value(val));
            break;
        default:
            _data.value.Set(Float32Value(val));
            break;
        }
        break;
    }
    case HIPDNN_TYPE_DOUBLE:
        _data.value.Set(Float64Value(*static_cast<const double*>(arrayOfElements)));
        break;
    case HIPDNN_TYPE_INT32:
        _data.value.Set(Int32Value(*static_cast<const int32_t*>(arrayOfElements)));
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_BAD_PARAM,
            "TensorDescriptor::setAttribute(): unsupported attributeType for TENSOR_VALUE");
    }
}

void TensorDescriptor::getTensorValue(hipdnnBackendAttributeType_t attributeType,
                                      int64_t requestedElementCount,
                                      int64_t* elementCount,
                                      void* arrayOfElements) const
{
    THROW_IF_FALSE(requestedElementCount >= 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   "TensorDescriptor::getAttribute(): requestedElementCount < 1");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "TensorDescriptor::getAttribute(): arrayOfElements is null");

    using namespace hipdnn_data_sdk::data_objects;

    THROW_IF_TRUE(_data.value.type == TensorValue::NONE,
                  HIPDNN_STATUS_BAD_PARAM,
                  "TensorDescriptor::getAttribute(): tensor value is not set");

    switch(attributeType)
    {
    case HIPDNN_TYPE_FLOAT:
    {
        const auto* val = _data.value.AsFloat32Value();
        if(val == nullptr)
        {
            // Try Float16 or BFloat16 (both store as float internally)
            const auto* f16 = _data.value.AsFloat16Value();
            const auto* bf16 = _data.value.AsBFloat16Value();
            THROW_IF_TRUE(f16 == nullptr && bf16 == nullptr,
                          HIPDNN_STATUS_BAD_PARAM,
                          "TensorDescriptor::getAttribute(): value type mismatch for "
                          "HIPDNN_TYPE_FLOAT");
            *static_cast<float*>(arrayOfElements) = f16 != nullptr ? f16->value() : bf16->value();
        }
        else
        {
            *static_cast<float*>(arrayOfElements) = val->value();
        }
        break;
    }
    case HIPDNN_TYPE_DOUBLE:
    {
        const auto* val = _data.value.AsFloat64Value();
        THROW_IF_TRUE(val == nullptr,
                      HIPDNN_STATUS_BAD_PARAM,
                      "TensorDescriptor::getAttribute(): value type mismatch for "
                      "HIPDNN_TYPE_DOUBLE");
        *static_cast<double*>(arrayOfElements) = val->value();
        break;
    }
    case HIPDNN_TYPE_INT32:
    {
        const auto* val = _data.value.AsInt32Value();
        THROW_IF_TRUE(val == nullptr,
                      HIPDNN_STATUS_BAD_PARAM,
                      "TensorDescriptor::getAttribute(): value type mismatch for "
                      "HIPDNN_TYPE_INT32");
        *static_cast<int32_t*>(arrayOfElements) = val->value();
        break;
    }
    default:
        throw HipdnnException(
            HIPDNN_STATUS_BAD_PARAM,
            "TensorDescriptor::getAttribute(): unsupported attributeType for TENSOR_VALUE");
    }

    if(elementCount != nullptr)
    {
        *elementCount = 1;
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
