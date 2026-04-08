// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/BlockScaleDequantizeAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a block scale dequantize operation descriptor from BlockScaleDequantizeAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createBlockScaleDequantizeOperation(
    const graph::BlockScaleDequantizeAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(
<<<<<<< HEAD
        HIPDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR);
=======
        HIPDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR_EXT);
>>>>>>> d9e199e220 (merge b-shi branch)
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create block scale dequantize operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
<<<<<<< HEAD
                                             HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "block_scale_dequantize X"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetTensorRef(opDesc.get(),
                              HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC,
                              attributes.get_scale(),
                              tensorDescs,
                              "block_scale_dequantize SCALE"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC,
=======
                                             HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_X_EXT,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "block_scale_dequantize X"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_EXT,
                                             attributes.get_scale(),
                                             tensorDescs,
                                             "block_scale_dequantize SCALE"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_Y_EXT,
>>>>>>> d9e199e220 (merge b-shi branch)
                                             attributes.get_y(),
                                             tensorDescs,
                                             "block_scale_dequantize Y"));

    // Set block_size as int32 (matches backend and FBS schema type)
<<<<<<< HEAD
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE,
                                            HIPDNN_TYPE_INT32,
                                            attributes.get_block_size(),
                                            "block_scale_dequantize block_size"));
=======
    HIPDNN_CHECK_ERROR(
        setDescriptorAttrVec(opDesc.get(),
                             HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE_EXT,
                             HIPDNN_TYPE_INT32,
                             attributes.get_block_size(),
                             "block_scale_dequantize block_size"));
>>>>>>> d9e199e220 (merge b-shi branch)

    // Set is_negative_scale
    HIPDNN_CHECK_ERROR(
        setDescriptorAttrScalar(opDesc.get(),
<<<<<<< HEAD
                                HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE,
=======
                                HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_IS_NEGATIVE_SCALE_EXT,
>>>>>>> d9e199e220 (merge b-shi branch)
                                HIPDNN_TYPE_BOOLEAN,
                                attributes.get_is_negative_scale(),
                                "block_scale_dequantize is_negative_scale"));

    // Set compute data type
<<<<<<< HEAD
    HIPDNN_CHECK_ERROR(
        setDescriptorAttrDataType(opDesc.get(),
                                  HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC,
                                  attributes.compute_data_type,
                                  "block_scale_dequantize compute data type"));
=======
    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_BLOCK_SCALE_DEQUANTIZE_MATH_PREC_EXT,
                                                 attributes.compute_data_type,
                                                 "block_scale_dequantize compute data type"));
>>>>>>> d9e199e220 (merge b-shi branch)

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(opDesc.get(),
                                                   HIPDNN_ATTR_OPERATION_NAME_EXT,
                                                   opName,
                                                   "block scale dequantize operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(
        finalizeDescriptor(opDesc.get(), "block_scale_dequantize operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
