// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormBackwardAttributes.hpp>
#include <hipdnn_frontend/detail/RMSNormBackwardPacker.hpp>
#include <hipdnn_frontend/detail/RMSNormBackwardUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

namespace hipdnn_frontend::graph
{
class RMSNormBackwardNode : public BaseNode<RMSNormBackwardNode, NodeType::RMS_NORM_BACKWARD>
{
public:
    RMSNormBackwardAttributes attributes;

    RMSNormBackwardNode(RMSNormBackwardAttributes&& attrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(attrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        RMSNormBackwardAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackRMSNormBackwardOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    Error pre_validate_node() const override
    {
        // Validate required tensor pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_dy(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormBackwardNode missing dy (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormBackwardNode missing x (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_scale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormBackwardNode missing scale (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dx(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormBackwardNode missing dx (output) for pre-validation");

        return {};
    }

    Error infer_properties_node() override
    {
        // Validate required tensor pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_dy(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormBackwardNode missing dy for setting properties");

        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormBackwardNode missing x for setting properties");

        HIPDNN_RETURN_IF_FALSE(attributes.get_scale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormBackwardNode missing scale for setting properties");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dx(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormBackwardNode missing dx for setting properties");

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        // TODO: Implement output dimension inference

        auto dxTensor = attributes.get_dx();
        auto dyTensor = attributes.get_dy();

        // Infer output strides if not set
        if(dxTensor->get_stride().empty())
        {
            auto& dyStrides = dyTensor->get_stride();
            auto& dxDims = dxTensor->get_dim();

            HIPDNN_RETURN_IF_TRUE(
                dyStrides.empty(),
                ErrorCode::ATTRIBUTE_NOT_SET,
                "RMSNormBackwardNode: Cannot infer output strides - missing input strides");

            HIPDNN_RETURN_IF_TRUE(
                dxDims.empty(),
                ErrorCode::ATTRIBUTE_NOT_SET,
                "RMSNormBackwardNode: Cannot infer output strides - missing output dimensions");

            HIPDNN_RETURN_IF_NE(
                dyStrides.size(),
                dxDims.size(),
                ErrorCode::ATTRIBUTE_NOT_SET,
                "RMSNormBackwardNode: Stride dimension mismatch between input and output tensors");

            auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(dyStrides);
            auto dxStrides = hipdnn_data_sdk::utilities::generateStrides(dxDims, strideOrder);
            dxTensor->set_stride(dxStrides);
        }

        return {};
    }

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>
        pack_node(flatbuffers::FlatBufferBuilder& builder) const override
    {
        return hipdnn_data_sdk::data_objects::CreateNodeDirect(
            builder,
            attributes.get_name().c_str(),
            toSdkType(attributes.compute_data_type),
            hipdnn_data_sdk::data_objects::NodeAttributes::RMSNormBackwardAttributes,
            attributes.pack_attributes(builder).Union());
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createRMSNormBackwardOperation(attributes, tensorDescs, operations);
    }
};
}
