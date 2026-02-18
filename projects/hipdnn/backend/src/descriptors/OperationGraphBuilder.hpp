// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BackendDescriptor.hpp"
#include "ConvolutionFwdOperationDescriptor.hpp"
#include "GraphDescriptor.hpp"
#include "TensorDescriptor.hpp"
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <unordered_set>

namespace hipdnn_backend
{

class OperationGraphBuilder : public HipdnnBackendDescriptorImpl<OperationGraphBuilder>
{
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

    // After finalize, get the resulting graph descriptor
    std::shared_ptr<GraphDescriptor> getGraphDescriptor() const
    {
        return _graphDesc;
    }

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;

private:
    hipdnnStatus_t buildGraph();

    std::vector<std::shared_ptr<TensorDescriptor>> _tensors;
    std::vector<std::shared_ptr<ConvolutionFwdOperationDescriptor>> _operations;
    std::unordered_set<int64_t> _tensorUids; // For deduplication
    hipdnnHandle_t _handle = nullptr;
    std::shared_ptr<GraphDescriptor> _graphDesc;
};

} // namespace hipdnn_backend
