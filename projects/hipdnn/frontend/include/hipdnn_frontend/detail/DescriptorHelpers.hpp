// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend::detail
{

// Sets a string attribute (char array) on a backend descriptor.
inline Error setDescriptorAttrString(hipdnnBackendDescriptor_t desc,
                                     hipdnnBackendAttributeName_t attrName,
                                     const std::string& value,
                                     const std::string& errorContext)
{
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(
            desc, attrName, HIPDNN_TYPE_CHAR, static_cast<int64_t>(value.size()), value.c_str()),
        "Failed to set " + errorContext);
    return {};
}

// Sets a vector-valued attribute on a backend descriptor.
inline Error setDescriptorAttrVec(hipdnnBackendDescriptor_t desc,
                                  hipdnnBackendAttributeName_t attrName,
                                  hipdnnBackendAttributeType_t attrType,
                                  const std::vector<int64_t>& values,
                                  const std::string& errorContext)
{
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(
            desc, attrName, attrType, static_cast<int64_t>(values.size()), values.data()),
        "Failed to set " + errorContext);
    return {};
}

// Sets a scalar attribute on a backend descriptor.
template <typename T>
inline Error setDescriptorAttrScalar(hipdnnBackendDescriptor_t desc,
                                     hipdnnBackendAttributeName_t attrName,
                                     hipdnnBackendAttributeType_t attrType,
                                     const T& value,
                                     const std::string& errorContext)
{
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(desc, attrName, attrType, 1, &value),
        "Failed to set " + errorContext);
    return {};
}

// Sets a tensor reference attribute on an operation descriptor by looking up
// the tensor UID in the tensorDescs map.
inline Error setDescriptorAttrTensorRef(
    hipdnnBackendDescriptor_t desc,
    hipdnnBackendAttributeName_t attrName,
    int64_t tensorUid,
    const std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    const std::string& errorContext)
{
    auto it = tensorDescs.find(tensorUid);
    if(it == tensorDescs.end())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Tensor UID " + std::to_string(tensorUid) + " not found when setting "
                    + errorContext};
    }
    auto descPtr = it->second.get();
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(
            desc, attrName, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &descPtr),
        "Failed to set " + errorContext);
    return {};
}

// Finalizes a backend descriptor.
inline Error finalizeDescriptor(hipdnnBackendDescriptor_t desc, const std::string& errorContext)
{
    HIPDNN_RETURN_ON_BACKEND_FAILURE(hipdnnBackend()->backendFinalize(desc),
                                     "Failed to finalize " + errorContext);
    return {};
}

// Creates a backend tensor descriptor for the given TensorAttributes if one
// does not already exist in the map (keyed by UID). Returns the tensor UID.
inline std::pair<Error, int64_t>
    createOrFindTensorDesc(std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
                           const std::shared_ptr<graph::TensorAttributes>& tensor)
{
    auto uid = tensor->get_uid();
    if(tensorDescs.find(uid) != tensorDescs.end())
    {
        return {{}, uid};
    }

    ScopedHipdnnBackendDescriptor desc(HIPDNN_BACKEND_TENSOR_DESCRIPTOR);
    if(!desc.valid())
    {
        return {Error(ErrorCode::HIPDNN_BACKEND_ERROR,
                      "Failed to create tensor descriptor for uid " + std::to_string(uid)),
                uid};
    }

    auto err = setDescriptorAttrScalar(desc.get(),
                                       HIPDNN_ATTR_TENSOR_UNIQUE_ID,
                                       HIPDNN_TYPE_INT64,
                                       uid,
                                       "tensor UID " + std::to_string(uid));
    if(err.is_bad())
    {
        return {std::move(err), uid};
    }

    auto& name = tensor->get_name();
    if(!name.empty())
    {
        err = setDescriptorAttrString(desc.get(), HIPDNN_ATTR_TENSOR_NAME_EXT, name, "tensor name");
        if(err.is_bad())
        {
            return {std::move(err), uid};
        }
    }

    auto sdkDataType = hipdnn_frontend::toSdkType(tensor->get_data_type());
    err = setDescriptorAttrScalar(desc.get(),
                                  HIPDNN_ATTR_TENSOR_DATA_TYPE,
                                  HIPDNN_TYPE_DATA_TYPE,
                                  sdkDataType,
                                  "tensor data type");
    if(err.is_bad())
    {
        return {std::move(err), uid};
    }

    err = setDescriptorAttrVec(desc.get(),
                               HIPDNN_ATTR_TENSOR_DIMENSIONS,
                               HIPDNN_TYPE_INT64,
                               tensor->get_dim(),
                               "tensor dimensions");
    if(err.is_bad())
    {
        return {std::move(err), uid};
    }

    err = setDescriptorAttrVec(desc.get(),
                               HIPDNN_ATTR_TENSOR_STRIDES,
                               HIPDNN_TYPE_INT64,
                               tensor->get_stride(),
                               "tensor strides");
    if(err.is_bad())
    {
        return {std::move(err), uid};
    }

    bool isVirtual = tensor->get_is_virtual();
    err = setDescriptorAttrScalar(desc.get(),
                                  HIPDNN_ATTR_TENSOR_IS_VIRTUAL,
                                  HIPDNN_TYPE_BOOLEAN,
                                  isVirtual,
                                  "tensor is_virtual");
    if(err.is_bad())
    {
        return {std::move(err), uid};
    }

    err = finalizeDescriptor(desc.get(), "tensor descriptor");
    if(err.is_bad())
    {
        return {std::move(err), uid};
    }

    tensorDescs.emplace(uid, std::move(desc));
    return {{}, uid};
}

} // namespace hipdnn_frontend::detail
