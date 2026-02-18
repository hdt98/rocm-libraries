// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

namespace hipdnn_frontend::graph
{
class ConvolutionFpropNode : public BaseNode<ConvolutionFpropNode>
{
public:
    ConvFpropAttributes attributes;

    ConvolutionFpropNode(ConvFpropAttributes&& convAttrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(convAttrs))
    {
    }

    Error pre_validate_node() const override
    {
        // Validate tensor pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ConvolutionFpropNode missing x (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_w(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ConvolutionFpropNode missing w (weights) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_y(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ConvolutionFpropNode missing y (output) for pre-validation");

        // Validate convolution parameters
        HIPDNN_RETURN_IF_TRUE(attributes.get_pre_padding().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ConvolutionFpropNode missing pre_padding for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_post_padding().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ConvolutionFpropNode missing post_padding for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_stride().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ConvolutionFpropNode missing stride for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_dilation().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ConvolutionFpropNode missing dilation for pre-validation");

        // Get tensor references
        auto x = attributes.get_x();
        auto w = attributes.get_w();
        auto y = attributes.get_y();

        // Validate input tensor dimensions and strides
        auto& xDims = x->get_dim();

        HIPDNN_RETURN_IF_LT(
            xDims.size(),
            3,
            ErrorCode::INVALID_VALUE,
            "ConvolutionFpropNode: Input tensor must have at least 3 dimensions (N, C, spatial)");

        // Validate weight tensor dimensions and strides
        auto& wDims = w->get_dim();

        HIPDNN_RETURN_IF_NE(
            wDims.size(),
            xDims.size(),
            ErrorCode::INVALID_VALUE,
            "ConvolutionFpropNode: Weight tensor dimension count must match input tensor "
            "dimension count");

        // Validate input channels match between input and weight tensors
        // For regular convolution: x_dims[1] == w_dims[1]
        // For grouped convolution: x_dims[1] % w_dims[1] == 0
        HIPDNN_RETURN_IF_NE(
            xDims[1] % wDims[1],
            0,
            ErrorCode::INVALID_VALUE,
            "ConvolutionFpropNode: Input tensor channels must match weight tensor input "
            "channels or be divisible by them for grouped convolution");

        // For grouped convolution: x_dims[1] / w_dims[1] is group count.
        // Output channels must be divisible by group count.
        auto groupCount = xDims[1] / wDims[1];
        HIPDNN_RETURN_IF_NE(
            wDims[0] % groupCount,
            0,
            ErrorCode::INVALID_VALUE,
            "ConvolutionFpropNode: Weight tensor output channels must be divisible by "
            "the number of groups");

        // Validate output tensor dimensions and strides if they are set
        auto& yDims = y->get_dim();

        if(!yDims.empty())
        {
            HIPDNN_RETURN_IF_NE(
                yDims.size(),
                xDims.size(),
                ErrorCode::INVALID_VALUE,
                "ConvolutionFpropNode: Output tensor dimension count must match input tensor "
                "dimension count");

            // Validate batch size matches
            HIPDNN_RETURN_IF_NE(yDims[0],
                                xDims[0],
                                ErrorCode::INVALID_VALUE,
                                "ConvolutionFpropNode: Output tensor batch size must match input "
                                "tensor batch size");

            // Validate output channels match weight output channels
            HIPDNN_RETURN_IF_NE(yDims[1],
                                wDims[0],
                                ErrorCode::INVALID_VALUE,
                                "ConvolutionFpropNode: Output tensor channels must match weight "
                                "tensor output channels");
        }

        // Validate spatial parameter counts match spatial dimensions
        auto spatialDims = xDims.size() - 2; // Skip N and C dimensions
        auto& prePadding = attributes.get_pre_padding();
        auto& postPadding = attributes.get_post_padding();
        auto& stride = attributes.get_stride();
        auto& dilation = attributes.get_dilation();

        HIPDNN_RETURN_IF_NE(
            prePadding.size(),
            spatialDims,
            ErrorCode::INVALID_VALUE,
            "ConvolutionFpropNode: pre_padding parameter count must match spatial dimension count");

        HIPDNN_RETURN_IF_NE(postPadding.size(),
                            spatialDims,
                            ErrorCode::INVALID_VALUE,
                            "ConvolutionFpropNode: post_padding parameter count must match spatial "
                            "dimension count");

        HIPDNN_RETURN_IF_NE(
            stride.size(),
            spatialDims,
            ErrorCode::INVALID_VALUE,
            "ConvolutionFpropNode: stride parameter count must match spatial dimension count");

        HIPDNN_RETURN_IF_NE(
            dilation.size(),
            spatialDims,
            ErrorCode::INVALID_VALUE,
            "ConvolutionFpropNode: dilation parameter count must match spatial dimension count");

        // Check spatial parameters for each dimension
        for(size_t i = 0; i < spatialDims; ++i)
        {
            auto prePad = prePadding[i];
            auto postPad = postPadding[i];
            auto strideVal = stride[i];
            auto dilationVal = dilation[i];

            // Validate parameters
            HIPDNN_RETURN_IF_LT(
                strideVal, 1, ErrorCode::INVALID_VALUE, "ConvolutionFpropNode: Stride must be > 0");

            HIPDNN_RETURN_IF_LT(dilationVal,
                                1,
                                ErrorCode::INVALID_VALUE,
                                "ConvolutionFpropNode: Dilation must > 0");

            HIPDNN_RETURN_IF_LT(prePad,
                                0,
                                ErrorCode::INVALID_VALUE,
                                "ConvolutionFpropNode: Pre-padding must be non-negative");

            HIPDNN_RETURN_IF_LT(postPad,
                                0,
                                ErrorCode::INVALID_VALUE,
                                "ConvolutionFpropNode: Post-padding must be non-negative");

            if(!yDims.empty())
            {
                auto inputSize = xDims[i + 2];
                auto kernelSize = wDims[i + 2];
                auto outputSize = yDims[i + 2];

                auto dilatedKernelSize = (dilationVal * (kernelSize - 1)) + 1;
                auto numerator = inputSize + prePad + postPad - dilatedKernelSize;

                HIPDNN_RETURN_IF_LT(numerator,
                                    0,
                                    ErrorCode::INVALID_VALUE,
                                    "ConvolutionFpropNode: Input spatial dimension at index "
                                        + std::to_string(i) + " (" + std::to_string(inputSize)
                                        + ") is too small for the kernel size ("
                                        + std::to_string(kernelSize) + ") and dilation ("
                                        + std::to_string(dilationVal) + ")");

                int64_t expectedOutputSize = (numerator / strideVal) + 1;

                HIPDNN_RETURN_IF_NE(
                    outputSize,
                    expectedOutputSize,
                    ErrorCode::INVALID_VALUE,
                    "ConvolutionFpropNode: Output tensor spatial dimension at index "
                        + std::to_string(i) + " (" + std::to_string(outputSize)
                        + ") does not match expected dimension ("
                        + std::to_string(expectedOutputSize)
                        + ") given input dimensions, kernel size, padding, stride, and dilation");
            }
        }

        return {};
    }

    Error infer_properties_node() override
    {
        auto x = attributes.get_x();
        auto w = attributes.get_w();
        auto y = attributes.get_y();

        HIPDNN_RETURN_IF_FALSE(x,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ConvolutionFpropNode missing x for setting properties");

        HIPDNN_RETURN_IF_FALSE(w,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ConvolutionFpropNode missing w for setting properties");

        HIPDNN_RETURN_IF_FALSE(y,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ConvolutionFpropNode missing y for setting properties");

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        auto yDims = y->get_dim();

        // Infer output dimensions if not set
        if(yDims.empty())
        {
            auto& xDims = x->get_dim();
            auto& wDims = w->get_dim();

            yDims.resize(xDims.size());

            auto& prePadding = attributes.get_pre_padding();
            auto& postPadding = attributes.get_post_padding();
            auto& stride = attributes.get_stride();
            auto& dilation = attributes.get_dilation();

            yDims[0] = xDims[0]; // N (batch) matches input
            yDims[1] = wDims[0]; // C (output channels)

            // Calculate spatial dimensions (Optional D, H, W)
            // Starting from dim 2 (skip N and C)
            for(size_t i = 2; i < xDims.size(); ++i)
            {
                auto spatialIdx = i - 2; // Index into spatial dimension arrays

                HIPDNN_RETURN_IF_TRUE(
                    spatialIdx >= prePadding.size() || spatialIdx >= postPadding.size()
                        || spatialIdx >= stride.size() || spatialIdx >= dilation.size(),
                    ErrorCode::INVALID_VALUE,
                    "ConvolutionFpropNode: Insufficient padding/stride/dilation parameters for "
                    "spatial "
                    "dimensions");

                // Standard convolution output size formula:
                // output_size = floor((input_size + pre_padding + post_padding - dilated_kernel_size) / stride) + 1
                // where dilated_kernel_size = dilation * (kernel_size - 1) + 1

                auto inputSize = xDims[i];
                auto kernelSize = wDims[i];
                auto prePad = prePadding[spatialIdx];
                auto postPad = postPadding[spatialIdx];
                auto strideVal = stride[spatialIdx];
                auto dilationVal = dilation[spatialIdx];

                // Validate parameters
                HIPDNN_RETURN_IF_LT(strideVal,
                                    1,
                                    ErrorCode::INVALID_VALUE,
                                    "ConvolutionFpropNode: Stride must be positive");

                HIPDNN_RETURN_IF_LT(dilationVal,
                                    1,
                                    ErrorCode::INVALID_VALUE,
                                    "ConvolutionFpropNode: Dilation must be positive");

                HIPDNN_RETURN_IF_LT(prePad,
                                    0,
                                    ErrorCode::INVALID_VALUE,
                                    "ConvolutionFpropNode: Pre-padding must be non-negative");

                HIPDNN_RETURN_IF_LT(postPad,
                                    0,
                                    ErrorCode::INVALID_VALUE,
                                    "ConvolutionFpropNode: Post-padding must be non-negative");

                // Calculate dilated kernel size
                auto dilatedKernelSize = (dilationVal * (kernelSize - 1)) + 1;

                // Calculate output dimension
                auto numerator = inputSize + prePad + postPad - dilatedKernelSize;
                HIPDNN_RETURN_IF_LT(
                    numerator,
                    0,
                    ErrorCode::INVALID_VALUE,
                    "ConvolutionFpropNode: Invalid convolution parameters result in negative "
                    "output size");

                yDims[i] = (numerator / strideVal) + 1;
            }

            // Set the inferred dimensions
            y->set_dim(yDims);
        }

        // Infer output strides if not set
        if(y->get_stride().empty())
        {
            auto& xStrides = x->get_stride();
            auto& yDimsFinal = y->get_dim();

            HIPDNN_RETURN_IF_TRUE(
                xStrides.empty(),
                ErrorCode::ATTRIBUTE_NOT_SET,
                "ConvolutionFpropNode: Cannot infer output strides - missing input strides");

            HIPDNN_RETURN_IF_TRUE(
                yDimsFinal.empty(),
                ErrorCode::ATTRIBUTE_NOT_SET,
                "ConvolutionFpropNode: Cannot infer output strides - missing output dimensions");

            HIPDNN_RETURN_IF_NE(
                xStrides.size(),
                yDimsFinal.size(),
                ErrorCode::ATTRIBUTE_NOT_SET,
                "ConvolutionFpropNode: Stride dimension mismatch between input and output tensors");

            // Extract stride order from input tensor and apply to output tensor
            auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(xStrides);

            // Generate Y strides using the extracted stride order and Y dimensions
            auto yStrides = hipdnn_data_sdk::utilities::generateStrides(yDimsFinal, strideOrder);

            y->set_stride(yStrides);
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
            hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes,
            attributes.pack_attributes(builder).Union());
    }

    // Create operation descriptor using backend API (no FlatBuffers required)
    Error createOperation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const
    {
        // Helper to create tensor descriptor if not already in map
        auto ensureTensorDesc
            = [&tensorDescs](
                  const std::shared_ptr<TensorAttributes>& tensor) -> std::pair<Error, int64_t> {
            auto uid = tensor->get_uid();
            if(tensorDescs.find(uid) != tensorDescs.end())
            {
                return {{}, uid};
            }

            detail::ScopedHipdnnBackendDescriptor desc(HIPDNN_BACKEND_TENSOR_DESCRIPTOR);
            if(!desc.valid())
            {
                return {Error(ErrorCode::HIPDNN_BACKEND_ERROR,
                              "Failed to create tensor descriptor for uid " + std::to_string(uid)),
                        uid};
            }

            // Set UID
            auto status = detail::hipdnnBackend()->backendSetAttribute(
                desc.get(), HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &uid);
            if(status != HIPDNN_STATUS_SUCCESS)
            {
                return {Error(ErrorCode::HIPDNN_BACKEND_ERROR,
                              "Failed to set tensor UID for " + std::to_string(uid)),
                        uid};
            }

            // Set name
            auto& name = tensor->get_name();
            if(!name.empty())
            {
                status = detail::hipdnnBackend()->backendSetAttribute(
                    desc.get(),
                    HIPDNN_ATTR_TENSOR_NAME,
                    HIPDNN_TYPE_CHAR,
                    static_cast<int64_t>(name.size()),
                    name.c_str());
                if(status != HIPDNN_STATUS_SUCCESS)
                {
                    return {Error(ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to set tensor name"),
                            uid};
                }
            }

            // Set data type
            auto sdkDataType = toSdkType(tensor->get_data_type());
            status = detail::hipdnnBackend()->backendSetAttribute(
                desc.get(), HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &sdkDataType);
            if(status != HIPDNN_STATUS_SUCCESS)
            {
                return {Error(ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to set tensor data type"),
                        uid};
            }

            // Set dimensions
            auto& dims = tensor->get_dim();
            status = detail::hipdnnBackend()->backendSetAttribute(desc.get(),
                                                                  HIPDNN_ATTR_TENSOR_DIMENSIONS,
                                                                  HIPDNN_TYPE_INT64,
                                                                  static_cast<int64_t>(dims.size()),
                                                                  dims.data());
            if(status != HIPDNN_STATUS_SUCCESS)
            {
                return {Error(ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to set tensor dimensions"),
                        uid};
            }

            // Set strides
            auto& strides = tensor->get_stride();
            status
                = detail::hipdnnBackend()->backendSetAttribute(desc.get(),
                                                               HIPDNN_ATTR_TENSOR_STRIDES,
                                                               HIPDNN_TYPE_INT64,
                                                               static_cast<int64_t>(strides.size()),
                                                               strides.data());
            if(status != HIPDNN_STATUS_SUCCESS)
            {
                return {Error(ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to set tensor strides"),
                        uid};
            }

            // Set is_virtual
            bool isVirtual = tensor->get_is_virtual();
            status = detail::hipdnnBackend()->backendSetAttribute(
                desc.get(), HIPDNN_ATTR_TENSOR_IS_VIRTUAL, HIPDNN_TYPE_BOOLEAN, 1, &isVirtual);
            if(status != HIPDNN_STATUS_SUCCESS)
            {
                return {Error(ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to set tensor is_virtual"),
                        uid};
            }

            // Finalize
            status = detail::hipdnnBackend()->backendFinalize(desc.get());
            if(status != HIPDNN_STATUS_SUCCESS)
            {
                return {
                    Error(ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to finalize tensor descriptor"),
                    uid};
            }

            tensorDescs.emplace(uid, std::move(desc));
            return {{}, uid};
        };

        // Ensure tensor descriptors exist for X, W, Y
        auto [errX, xUid] = ensureTensorDesc(attributes.get_x());
        HIPDNN_CHECK_ERROR(errX);
        auto [errW, wUid] = ensureTensorDesc(attributes.get_w());
        HIPDNN_CHECK_ERROR(errW);
        auto [errY, yUid] = ensureTensorDesc(attributes.get_y());
        HIPDNN_CHECK_ERROR(errY);

        // Create operation descriptor
        detail::ScopedHipdnnBackendDescriptor opDesc(
            HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
        if(!opDesc.valid())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Failed to create convolution forward operation descriptor"};
        }

        // Set tensor references
        auto xDescPtr = tensorDescs.at(xUid).get();
        auto status = detail::hipdnnBackend()->backendSetAttribute(
            opDesc.get(),
            HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &xDescPtr);
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to set conv X tensor"};
        }

        auto wDescPtr = tensorDescs.at(wUid).get();
        status = detail::hipdnnBackend()->backendSetAttribute(
            opDesc.get(),
            HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &wDescPtr);
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to set conv W tensor"};
        }

        auto yDescPtr = tensorDescs.at(yUid).get();
        status = detail::hipdnnBackend()->backendSetAttribute(
            opDesc.get(),
            HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &yDescPtr);
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to set conv Y tensor"};
        }

        // Set convolution parameters
        auto& prePadding = attributes.get_pre_padding();
        status = detail::hipdnnBackend()->backendSetAttribute(
            opDesc.get(),
            HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_PRE_PADDINGS,
            HIPDNN_TYPE_INT64,
            static_cast<int64_t>(prePadding.size()),
            prePadding.data());
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to set conv pre_padding"};
        }

        auto& postPadding = attributes.get_post_padding();
        status = detail::hipdnnBackend()->backendSetAttribute(
            opDesc.get(),
            HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_POST_PADDINGS,
            HIPDNN_TYPE_INT64,
            static_cast<int64_t>(postPadding.size()),
            postPadding.data());
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to set conv post_padding"};
        }

        auto& stride = attributes.get_stride();
        status = detail::hipdnnBackend()->backendSetAttribute(
            opDesc.get(),
            HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_STRIDES,
            HIPDNN_TYPE_INT64,
            static_cast<int64_t>(stride.size()),
            stride.data());
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to set conv stride"};
        }

        auto& dilation = attributes.get_dilation();
        status = detail::hipdnnBackend()->backendSetAttribute(
            opDesc.get(),
            HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_DILATIONS,
            HIPDNN_TYPE_INT64,
            static_cast<int64_t>(dilation.size()),
            dilation.data());
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to set conv dilation"};
        }

        auto convMode = static_cast<int64_t>(toSdkType(attributes.get_convolution_mode()));
        status = detail::hipdnnBackend()->backendSetAttribute(
            opDesc.get(),
            HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_MODE,
            HIPDNN_TYPE_INT64,
            1,
            &convMode);
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to set conv mode"};
        }

        // Set compute data type (inherited from graph attributes)
        auto computeDataType = toSdkType(attributes.compute_data_type);
        status = detail::hipdnnBackend()->backendSetAttribute(
            opDesc.get(),
            HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_COMPUTE_DATA_TYPE,
            HIPDNN_TYPE_DATA_TYPE,
            1,
            &computeDataType);
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to set conv compute data type"};
        }

        // Finalize operation descriptor
        status = detail::hipdnnBackend()->backendFinalize(opDesc.get());
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Failed to finalize convolution operation descriptor"};
        }

        operations.push_back(std::move(opDesc));
        return {};
    }
};

typedef ConvolutionFpropNode ConvolutionNode;
}
