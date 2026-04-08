// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BackendDescriptor.hpp"
#include "IGraphOperation.hpp"
#include "TensorDescriptor.hpp"
<<<<<<< HEAD
#include <hipdnn_flatbuffers_sdk/data_objects/block_scale_dequantize_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
=======
#include <hipdnn_data_sdk/data_objects/block_scale_dequantize_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
>>>>>>> d9e199e220 (merge b-shi branch)
#include <unordered_map>

namespace hipdnn_backend
{

class BlockScaleDequantizeOperationDescriptor
    : public HipdnnBackendDescriptorImpl<BlockScaleDequantizeOperationDescriptor>,
      public IGraphOperation
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

    // Direct access to the underlying T struct for OperationGraphBuilder
<<<<<<< HEAD
    const hipdnn_flatbuffers_sdk::data_objects::BlockScaleDequantizeAttributesT& getData() const
=======
    const hipdnn_data_sdk::data_objects::BlockScaleDequantizeAttributesT& getData() const
>>>>>>> d9e199e220 (merge b-shi branch)
    {
        return _data;
    }

    // Access to tensor descriptor references for graph building
    std::shared_ptr<TensorDescriptor> getXDesc() const
    {
        return _xDesc;
    }
    std::shared_ptr<TensorDescriptor> getScaleDesc() const
    {
        return _scaleDesc;
    }
    std::shared_ptr<TensorDescriptor> getYDesc() const
    {
        return _yDesc;
    }

    // Get compute data type for the operation (used when building graph nodes)
<<<<<<< HEAD
    hipdnn_flatbuffers_sdk::data_objects::DataType getComputeDataType() const
=======
    hipdnn_data_sdk::data_objects::DataType getComputeDataType() const
>>>>>>> d9e199e220 (merge b-shi branch)
    {
        return _computeDataType;
    }

    // IGraphOperation interface
    std::vector<std::shared_ptr<TensorDescriptor>> getTensorDescriptors() const override;
<<<<<<< HEAD
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT> buildNode() const override;
=======
    std::unique_ptr<hipdnn_data_sdk::data_objects::NodeT> buildNode() const override;
>>>>>>> d9e199e220 (merge b-shi branch)

    // Creates a finalized BlockScaleDequantizeOperationDescriptor directly from a FlatBuffer NodeT.
    // Casts nodeT.attributes to BlockScaleDequantizeAttributesT internally, then directly assigns
    // the data struct, looks up tensor descriptors from the tensor map, and calls finalize().
    static std::shared_ptr<BlockScaleDequantizeOperationDescriptor>
<<<<<<< HEAD
        fromNode(const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
=======
        fromNode(const hipdnn_data_sdk::data_objects::NodeT& nodeT,
>>>>>>> d9e199e220 (merge b-shi branch)
                 const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap);

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;

private:
<<<<<<< HEAD
    hipdnn_flatbuffers_sdk::data_objects::BlockScaleDequantizeAttributesT _data;
=======
    hipdnn_data_sdk::data_objects::BlockScaleDequantizeAttributesT _data;
>>>>>>> d9e199e220 (merge b-shi branch)

    // Store tensor descriptor references for validation and graph building
    std::shared_ptr<TensorDescriptor> _xDesc;
    std::shared_ptr<TensorDescriptor> _scaleDesc;
    std::shared_ptr<TensorDescriptor> _yDesc;

    // Compute data type for this operation (stored at node level in graph)
<<<<<<< HEAD
    hipdnn_flatbuffers_sdk::data_objects::DataType _computeDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
=======
    hipdnn_data_sdk::data_objects::DataType _computeDataType
        = hipdnn_data_sdk::data_objects::DataType::UNSET;
>>>>>>> d9e199e220 (merge b-shi branch)

    std::string _name;
};

} // namespace hipdnn_backend
