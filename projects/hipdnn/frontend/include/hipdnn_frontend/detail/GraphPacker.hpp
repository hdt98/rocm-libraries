// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/Node.hpp>

#include <optional>
#include <vector>

namespace hipdnn_frontend::detail
{

// Assembles a GraphDescriptor from pre-built operation descriptors.
// Sets the handle, operations, graph-level data types, preferred engine ID,
// then finalizes and returns the scoped descriptor.
inline Error assembleGraphDescriptor(std::vector<ScopedHipdnnBackendDescriptor>& operations,
                                     hipdnnHandle_t handle,
                                     hipdnn_data_sdk::data_objects::DataType computeDataType,
                                     hipdnn_data_sdk::data_objects::DataType intermediateDataType,
                                     hipdnn_data_sdk::data_objects::DataType ioDataType,
                                     const std::optional<int64_t>& preferredEngineId,
                                     std::unique_ptr<ScopedHipdnnBackendDescriptor>& outGraphDesc)
{
    ScopedHipdnnBackendDescriptor graphDesc(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    if(!graphDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create GraphDescriptor"};
    }

    // Set handle on graph
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(
            graphDesc.get(), HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle),
        "Failed to set handle on GraphDescriptor");

    // Set operations on graph
    std::vector<hipdnnBackendDescriptor_t> opDescPtrs;
    opDescPtrs.reserve(operations.size());
    for(auto& desc : operations)
    {
        opDescPtrs.push_back(desc.get());
    }
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(graphDesc.get(),
                                             HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                             static_cast<int64_t>(opDescPtrs.size()),
                                             opDescPtrs.data()),
        "Failed to set operations on GraphDescriptor");

    // Set graph-level data types
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(graphDesc.get(),
                                             HIPDNN_ATTR_OPERATIONGRAPH_COMPUTE_DATA_TYPE_EXT,
                                             HIPDNN_TYPE_DATA_TYPE,
                                             1,
                                             &computeDataType),
        "Failed to set compute data type on GraphDescriptor");

    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(graphDesc.get(),
                                             HIPDNN_ATTR_OPERATIONGRAPH_INTERMEDIATE_DATA_TYPE_EXT,
                                             HIPDNN_TYPE_DATA_TYPE,
                                             1,
                                             &intermediateDataType),
        "Failed to set intermediate data type on GraphDescriptor");

    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(graphDesc.get(),
                                             HIPDNN_ATTR_OPERATIONGRAPH_IO_DATA_TYPE_EXT,
                                             HIPDNN_TYPE_DATA_TYPE,
                                             1,
                                             &ioDataType),
        "Failed to set io data type on GraphDescriptor");

    // Set preferred engine ID if specified
    if(preferredEngineId.has_value())
    {
        auto engineId = preferredEngineId.value();
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendSetAttribute(graphDesc.get(),
                                                 HIPDNN_ATTR_OPERATIONGRAPH_PREFERRED_ENGINE_ID_EXT,
                                                 HIPDNN_TYPE_INT64,
                                                 1,
                                                 &engineId),
            "Failed to set preferred engine ID on GraphDescriptor");
    }

    // Finalize the graph
    HIPDNN_RETURN_ON_BACKEND_FAILURE(hipdnnBackend()->backendFinalize(graphDesc.get()),
                                     "Failed to finalize GraphDescriptor");

    outGraphDesc = std::make_unique<ScopedHipdnnBackendDescriptor>(std::move(graphDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
