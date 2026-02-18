// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "BackendDescriptor.hpp"
#include "ConvolutionFwdOperationDescriptor.hpp"
#include "TensorDescriptor.hpp"
#include <flatbuffers/detached_buffer.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <memory>
#include <unordered_set>
#include <vector>

namespace hipdnn_backend
{

class GraphDescriptor : public HipdnnBackendDescriptorImpl<GraphDescriptor>
{
private:
    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> _graph;
    hipdnnHandle_t _handle = nullptr;
    mutable flatbuffers::DetachedBuffer _graphSerializedBuffer;

    // For building graph from operation descriptors
    std::vector<std::shared_ptr<ConvolutionFwdOperationDescriptor>> _operations;
    std::unordered_set<int64_t> _tensorUids; // For deduplication

    void setHandle(hipdnnBackendAttributeType_t attributeType,
                   int64_t elementCount,
                   const void* arrayOfElements);

    void setOperations(hipdnnBackendAttributeType_t attributeType,
                       int64_t elementCount,
                       const void* arrayOfElements);

    // Build GraphT from operation descriptors
    void buildGraphFromOperations();

public:
    void finalize() override;

    void getAttribute(hipdnnBackendAttributeName_t attributeName,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements) const override;

    void setAttribute(hipdnnBackendAttributeName_t attributeName,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      const void* arrayOfElements) override;

    void deserializeGraph(const uint8_t* serializedGraph, size_t graphByteSize);

    virtual hipdnnPluginConstData_t getSerializedGraph() const;
    virtual hipdnnHandle_t getHandle() const;

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;
};
}
