// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "OperationGraphBuilder.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include <flatbuffers/flatbuffers.h>

namespace hipdnn_backend
{

void OperationGraphBuilder::finalize()
{
    THROW_IF_TRUE(_operations.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "OperationGraphBuilder::finalize() failed: no operations added");
    THROW_IF_TRUE(_tensors.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "OperationGraphBuilder::finalize() failed: no tensors added");
    THROW_IF_NULL(_handle,
                  HIPDNN_STATUS_BAD_PARAM,
                  "OperationGraphBuilder::finalize() failed: handle not set");

    auto status = buildGraph();
    THROW_IF_NE(status,
                HIPDNN_STATUS_SUCCESS,
                status,
                "OperationGraphBuilder::finalize() failed: buildGraph failed");

    HipdnnBackendDescriptorImpl<OperationGraphBuilder>::finalize();
}

hipdnnStatus_t OperationGraphBuilder::buildGraph()
{
    // 1. Build GraphT by collecting data from descriptors
    auto graphData = std::make_unique<hipdnn_data_sdk::data_objects::GraphT>();

    // 2. Collect tensors (already deduplicated during setAttribute)
    for(const auto& tensorDesc : _tensors)
    {
        auto tensorT = std::make_unique<hipdnn_data_sdk::data_objects::TensorAttributesT>(
            tensorDesc->getData());
        graphData->tensors.push_back(std::move(tensorT));
    }

    // 3. Build nodes from operation descriptors
    for(const auto& convOp : _operations)
    {
        auto node = std::make_unique<hipdnn_data_sdk::data_objects::NodeT>();
        node->compute_data_type = convOp->getComputeDataType();
        node->attributes.Set(
            hipdnn_data_sdk::data_objects::ConvolutionFwdAttributesT(convOp->getData()));
        graphData->nodes.push_back(std::move(node));
    }

    // 4. Serialize using Pack()
    flatbuffers::FlatBufferBuilder builder;
    auto graphOffset = hipdnn_data_sdk::data_objects::Graph::Pack(builder, graphData.get());
    builder.Finish(graphOffset);

    // 5. Create GraphDescriptor from serialized buffer
    _graphDesc = std::make_shared<GraphDescriptor>();
    _graphDesc->deserializeGraph(builder.GetBufferPointer(), builder.GetSize());

    // 6. Set handle on graph descriptor via setAttribute
    _graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_handle);

    // 7. Finalize the graph descriptor
    _graphDesc->finalize();

    return HIPDNN_STATUS_SUCCESS;
}

void OperationGraphBuilder::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                         hipdnnBackendAttributeType_t attributeType,
                                         int64_t requestedElementCount,
                                         int64_t* elementCount,
                                         void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "OperationGraphBuilder::getAttribute() failed: Not finalized.");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "OperationGraphBuilder::getAttribute(): arrayOfElements is null");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATIONGRAPH_BUILDER_HANDLE:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_HANDLE,
                       HIPDNN_STATUS_BAD_PARAM,
                       "OperationGraphBuilder::getAttribute(): attributeType mismatch");
        THROW_IF_FALSE(requestedElementCount >= 1,
                       HIPDNN_STATUS_BAD_PARAM,
                       "OperationGraphBuilder::getAttribute(): requestedElementCount < 1");
        if(elementCount != nullptr)
        {
            *elementCount = 1;
        }
        *static_cast<hipdnnHandle_t*>(arrayOfElements) = _handle;
        break;

    case HIPDNN_ATTR_OPERATIONGRAPH_BUILDER_GRAPH:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       HIPDNN_STATUS_BAD_PARAM,
                       "OperationGraphBuilder::getAttribute(): attributeType mismatch");
        THROW_IF_FALSE(requestedElementCount >= 1,
                       HIPDNN_STATUS_BAD_PARAM,
                       "OperationGraphBuilder::getAttribute(): requestedElementCount < 1");
        THROW_IF_NULL(_graphDesc,
                      HIPDNN_STATUS_NOT_INITIALIZED,
                      "OperationGraphBuilder::getAttribute(): graph not built yet");
        if(elementCount != nullptr)
        {
            *elementCount = 1;
        }
        HipdnnBackendDescriptor::packDescriptor(_graphDesc, arrayOfElements);
        break;

    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "OperationGraphBuilder::getAttribute: attributeName not supported");
    }
}

void OperationGraphBuilder::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                         hipdnnBackendAttributeType_t attributeType,
                                         int64_t elementCount,
                                         const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "OperationGraphBuilder::setAttribute() failed: Already finalized.");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "OperationGraphBuilder::setAttribute(): arrayOfElements is null");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATIONGRAPH_BUILDER_HANDLE:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_HANDLE,
                       HIPDNN_STATUS_BAD_PARAM,
                       "OperationGraphBuilder::setAttribute(): attributeType mismatch");
        THROW_IF_FALSE(elementCount == 1,
                       HIPDNN_STATUS_BAD_PARAM,
                       "OperationGraphBuilder::setAttribute(): elementCount is not 1");
        _handle = *static_cast<const hipdnnHandle_t*>(arrayOfElements);
        break;

    case HIPDNN_ATTR_OPERATIONGRAPH_BUILDER_TENSORS:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       HIPDNN_STATUS_BAD_PARAM,
                       "OperationGraphBuilder::setAttribute(): attributeType mismatch");
        {
            auto descriptors = static_cast<HipdnnBackendDescriptor* const*>(arrayOfElements);
            for(int64_t i = 0; i < elementCount; ++i)
            {
                auto tensorDesc = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
                    descriptors[i],
                    HIPDNN_STATUS_BAD_PARAM,
                    "OperationGraphBuilder::setAttribute(): Failed to unpack tensor descriptor");
                THROW_IF_FALSE(tensorDesc->isFinalized(),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                               "OperationGraphBuilder::setAttribute(): Tensor descriptor not "
                               "finalized");

                // Deduplicate by UID
                auto uid = tensorDesc->getData().uid;
                if(_tensorUids.find(uid) == _tensorUids.end())
                {
                    _tensorUids.insert(uid);
                    _tensors.push_back(tensorDesc);
                }
            }
        }
        break;

    case HIPDNN_ATTR_OPERATIONGRAPH_BUILDER_OPERATIONS:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       HIPDNN_STATUS_BAD_PARAM,
                       "OperationGraphBuilder::setAttribute(): attributeType mismatch");
        {
            auto descriptors = static_cast<HipdnnBackendDescriptor* const*>(arrayOfElements);
            for(int64_t i = 0; i < elementCount; ++i)
            {
                THROW_IF_NULL(descriptors[i],
                              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                              "OperationGraphBuilder::setAttribute(): descriptor is null");
                THROW_IF_FALSE(descriptors[i]->isFinalized(),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                               "OperationGraphBuilder::setAttribute(): Operation descriptor not "
                               "finalized");

                // For now, only support convolution forward
                auto descType = descriptors[i]->getType();
                if(descType == HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                {
                    auto opDesc = HipdnnBackendDescriptor::unpackDescriptor<
                        ConvolutionFwdOperationDescriptor>(
                        descriptors[i],
                        HIPDNN_STATUS_BAD_PARAM,
                        "OperationGraphBuilder::setAttribute(): Failed to unpack conv fwd "
                        "descriptor");
                    _operations.push_back(opDesc);
                }
                else
                {
                    throw HipdnnException(
                        HIPDNN_STATUS_NOT_SUPPORTED,
                        "OperationGraphBuilder::setAttribute(): Unsupported operation type");
                }
            }
        }
        break;

    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "OperationGraphBuilder::setAttribute: attributeName not supported");
    }
}

hipdnnBackendDescriptorType_t OperationGraphBuilder::getStaticType()
{
    return HIPDNN_BACKEND_OPERATIONGRAPH_BUILDER_DESCRIPTOR;
}

std::string OperationGraphBuilder::toString() const
{
    std::string str = "OperationGraphBuilder: {";
    str += "numTensors=" + std::to_string(_tensors.size());
    str += ", numOperations=" + std::to_string(_operations.size());
    str += ", hasHandle=" + std::string(_handle != nullptr ? "true" : "false");
    str += "}";
    return str;
}

} // namespace hipdnn_backend
