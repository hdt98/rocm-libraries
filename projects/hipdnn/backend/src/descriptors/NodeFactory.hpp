// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "BatchnormBackwardOperationDescriptor.hpp"
#include "BatchnormInferenceOperationDescriptor.hpp"
#include "BatchnormInferenceVarianceExtOperationDescriptor.hpp"
#include "BatchnormOperationDescriptor.hpp"
#include "BlockScaleDequantizeOperationDescriptor.hpp"
#include "BlockScaleQuantizeOperationDescriptor.hpp"
#include "ConvolutionBwdOperationDescriptor.hpp"
#include "ConvolutionFwdOperationDescriptor.hpp"
#include "ConvolutionWrwOperationDescriptor.hpp"
#include "CustomOpOperationDescriptor.hpp"
#include "IGraphOperation.hpp"
#include "LayernormOperationDescriptor.hpp"
#include "MatmulOperationDescriptor.hpp"
#include "PointwiseOperationDescriptor.hpp"
#include "RMSNormOperationDescriptor.hpp"
<<<<<<< HEAD
#include "ReductionOperationDescriptor.hpp"
#include "SdpaBwdOperationDescriptor.hpp"
#include "SdpaFwdOperationDescriptor.hpp"
#include "TensorDescriptor.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
=======
#include "SdpaBpropOperationDescriptor.hpp"
#include "SdpaFpropOperationDescriptor.hpp"
#include "TensorDescriptor.hpp"
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
>>>>>>> d9e199e220 (merge b-shi branch)
#include <memory>

namespace hipdnn_backend
{

// Factory that converts FlatBuffer NodeT objects into backend operation descriptors.
// Each operation descriptor handles its own attribute casting from the FlatBuffer union.
class NodeFactory
{
public:
    // Creates an operation descriptor from a FlatBuffer NodeT.
    // Dispatches to the appropriate operation descriptor's fromNode() method
    // based on nodeT.attributes.type. The operation descriptor handles its own
    // attribute casting internally.
    // Throws HIPDNN_STATUS_NOT_SUPPORTED for unsupported node types.
    static std::shared_ptr<IBackendDescriptor> createOperationFromNode(
<<<<<<< HEAD
        const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
=======
        const hipdnn_data_sdk::data_objects::NodeT& nodeT,
>>>>>>> d9e199e220 (merge b-shi branch)
        const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap);

    // Builds a tensor map from a vector of FlatBuffer TensorAttributesT.
    // Each tensor is created via TensorDescriptor::fromFlatBuffer() and indexed by UID.
    // Throws on null tensors or duplicate UIDs.
    static std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> buildTensorMap(
<<<<<<< HEAD
        const std::vector<std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT>>&
=======
        const std::vector<std::unique_ptr<hipdnn_data_sdk::data_objects::TensorAttributesT>>&
>>>>>>> d9e199e220 (merge b-shi branch)
            tensors);
};

} // namespace hipdnn_backend
