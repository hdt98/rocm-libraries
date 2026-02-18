// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GraphDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "FlatbufferUtilities.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"

namespace hipdnn_backend
{

void GraphDescriptor::finalize()
{
    THROW_IF_NULL(_handle, HIPDNN_STATUS_BAD_PARAM, "GraphDescriptor::finalize: handle is null");

    // If operations were set, build graph from them
    if(!_operations.empty())
    {
        buildGraphFromOperations();
    }

    THROW_IF_NULL(_graph, HIPDNN_STATUS_BAD_PARAM, "GraphDescriptor::finalize: graph is null");
    HipdnnBackendDescriptorImpl<GraphDescriptor>::finalize();
}

void GraphDescriptor::buildGraphFromOperations()
{
    _graph = std::make_unique<hipdnn_data_sdk::data_objects::GraphT>();

    // Collect unique tensors from operations
    for(const auto& convOp : _operations)
    {
        // Get tensor descriptors from operation
        auto xDesc = convOp->getXDesc();
        auto wDesc = convOp->getWDesc();
        auto yDesc = convOp->getYDesc();

        // Add tensors (deduplicated by UID)
        for(const auto& tensorDesc : {xDesc, wDesc, yDesc})
        {
            auto uid = tensorDesc->getData().uid;
            if(_tensorUids.find(uid) == _tensorUids.end())
            {
                _tensorUids.insert(uid);
                _graph->tensors.push_back(
                    std::make_unique<hipdnn_data_sdk::data_objects::TensorAttributesT>(
                        tensorDesc->getData()));
            }
        }

        // Build node from operation
        auto node = std::make_unique<hipdnn_data_sdk::data_objects::NodeT>();
        node->compute_data_type = convOp->getComputeDataType();
        node->attributes.Set(
            hipdnn_data_sdk::data_objects::ConvolutionFwdAttributesT(convOp->getData()));
        _graph->nodes.push_back(std::move(node));
    }

    // Clear the serialized buffer since we have a new graph
    _graphSerializedBuffer = flatbuffers::DetachedBuffer();
}

void GraphDescriptor::setHandle(hipdnnBackendAttributeType_t attributeType,
                                int64_t elementCount,
                                const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_HANDLE,
                HIPDNN_STATUS_BAD_PARAM,
                "GraphDescriptor failed to set handle: Invalid attribute type.");
    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "GraphDescriptor failed to set handle: Invalid element count.");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "GraphDescriptor failed to set handle: Null pointer.");

    hipdnnHandle_t handle = *static_cast<const hipdnnHandle_t*>(arrayOfElements);

    THROW_IF_NULL(handle,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "GraphDescriptor failed to set handle: Handle is null.");

    _handle = handle;
}

void GraphDescriptor::getAttribute([[maybe_unused]] hipdnnBackendAttributeName_t attributeName,
                                   [[maybe_unused]] hipdnnBackendAttributeType_t attributeType,
                                   [[maybe_unused]] int64_t requestedElementCount,
                                   [[maybe_unused]] int64_t* elementCount,
                                   [[maybe_unused]] void* arrayOfElements) const
{
    throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                          "GraphDescriptor::getAttribute: not supported");
}

void GraphDescriptor::setOperations(hipdnnBackendAttributeType_t attributeType,
                                    int64_t elementCount,
                                    const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "GraphDescriptor::setOperations: attributeType mismatch");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "GraphDescriptor::setOperations: arrayOfElements is null");

    auto descriptors = static_cast<HipdnnBackendDescriptor* const*>(arrayOfElements);
    for(int64_t i = 0; i < elementCount; ++i)
    {
        THROW_IF_NULL(descriptors[i],
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "GraphDescriptor::setOperations: descriptor is null");
        THROW_IF_FALSE(descriptors[i]->isFinalized(),
                       HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                       "GraphDescriptor::setOperations: Operation descriptor not finalized");

        // For now, only support convolution forward
        auto descType = descriptors[i]->getType();
        if(descType == HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
        {
            auto opDesc
                = HipdnnBackendDescriptor::unpackDescriptor<ConvolutionFwdOperationDescriptor>(
                    descriptors[i],
                    HIPDNN_STATUS_BAD_PARAM,
                    "GraphDescriptor::setOperations: Failed to unpack conv fwd descriptor");
            _operations.push_back(opDesc);
        }
        else
        {
            throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                                  "GraphDescriptor::setOperations: Unsupported operation type");
        }
    }

    // Clear the serialized graph when operations are set
    _graphSerializedBuffer = flatbuffers::DetachedBuffer();
}

void GraphDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                   hipdnnBackendAttributeType_t attributeType,
                                   int64_t elementCount,
                                   const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "GraphDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATIONGRAPH_HANDLE:
        setHandle(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_OPERATIONGRAPH_OPS:
        setOperations(attributeType, elementCount, arrayOfElements);
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("GraphDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void GraphDescriptor::deserializeGraph(const uint8_t* serializedGraph, size_t graphByteSize)
{
    THROW_IF_NULL(serializedGraph,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "GraphDescriptor::deserializeGraph: serializedGraph is null");
    THROW_IF_TRUE(graphByteSize == 0,
                  HIPDNN_STATUS_BAD_PARAM,
                  "GraphDescriptor::deserializeGraph: graphByteSize is 0");

    // TODO: Consider skipping validation entirely, or maybe add an API option to skip it for schema extension cases.
    flatbuffer_utilities::convertSerializedGraphToGraph(serializedGraph, graphByteSize, _graph);
}

hipdnnPluginConstData_t GraphDescriptor::getSerializedGraph() const
{
    if(_graphSerializedBuffer.size() == 0)
    {
        THROW_IF_NULL(_graph,
                      HIPDNN_STATUS_INTERNAL_ERROR,
                      "GraphDescriptor::getSerializedGraph: graph is null");

        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(hipdnn_data_sdk::data_objects::Graph::Pack(builder, _graph.get()));

        _graphSerializedBuffer = builder.Release();
    }

    return {_graphSerializedBuffer.data(), _graphSerializedBuffer.size()};
}

hipdnnBackendDescriptorType_t GraphDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR;
}

hipdnnHandle_t GraphDescriptor::getHandle() const
{
    THROW_IF_NULL(_handle, HIPDNN_STATUS_BAD_PARAM, "GraphDescriptor::getHandle: handle is null");
    return _handle;
}

std::string GraphDescriptor::toString() const
{
    std::string str = "GraphDescriptor: {handle=";
    str += _handle != nullptr ? fmt::format("{:p}", static_cast<const void*>(_handle)) : "null";
    str += ", serializedGraphSize="
           + std::to_string(_graphSerializedBuffer.size() > 0 ? _graphSerializedBuffer.size() : 0);
    str += "}";
    return str;
}

} // namespace hipdnn_backend
