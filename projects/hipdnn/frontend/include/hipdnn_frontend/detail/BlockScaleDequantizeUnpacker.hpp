// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/BlockScaleDequantizeAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

/// Unpacks a BLOCK_SCALE_DEQUANTIZE operation descriptor into BlockScaleDequantizeAttributes.
/// Reads X, Scale, and Y tensors; block_size; is_negative_scale; compute data type; and name.
[[nodiscard]] inline Error unpackBlockScaleDequantizeOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::BlockScaleDequantizeAttributes& attributes)
{
    // Unpack x tensor
    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
<<<<<<< HEAD
                                               HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC,
=======
                                               HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_X_EXT,
>>>>>>> d9e199e220 (merge b-shi branch)
                                               tensorMap,
                                               xTensor,
                                               "block_scale_dequantize X tensor"));
    attributes.set_x(xTensor);

    // Unpack scale tensor
    std::shared_ptr<graph::TensorAttributes> scaleTensor;
    HIPDNN_CHECK_ERROR(
        unpackAndRegisterTensor(opDesc,
<<<<<<< HEAD
                                HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC,
=======
                                HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_EXT,
>>>>>>> d9e199e220 (merge b-shi branch)
                                tensorMap,
                                scaleTensor,
                                "block_scale_dequantize SCALE tensor"));
    attributes.set_scale(scaleTensor);

    // Unpack y tensor
    std::shared_ptr<graph::TensorAttributes> yTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
<<<<<<< HEAD
                                               HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC,
=======
                                               HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_Y_EXT,
>>>>>>> d9e199e220 (merge b-shi branch)
                                               tensorMap,
                                               yTensor,
                                               "block_scale_dequantize Y tensor"));
    attributes.set_y(yTensor);

    // Unpack block_size (int32 vector)
    std::vector<int32_t> blockSize;
<<<<<<< HEAD
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(opDesc,
                                            HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE,
                                            HIPDNN_TYPE_INT32,
                                            blockSize,
                                            "block_scale_dequantize block_size"));
=======
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrVec(opDesc,
                             HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE_EXT,
                             HIPDNN_TYPE_INT32,
                             blockSize,
                             "block_scale_dequantize block_size"));
>>>>>>> d9e199e220 (merge b-shi branch)
    attributes.set_block_size(blockSize);

    // Unpack is_negative_scale
    {
        bool isNegativeScale = false;
<<<<<<< HEAD
        HIPDNN_CHECK_ERROR(
            getDescriptorAttrScalar(opDesc,
                                    HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE,
                                    HIPDNN_TYPE_BOOLEAN,
                                    isNegativeScale,
                                    "block_scale_dequantize is_negative_scale"));
=======
        HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(
            opDesc,
            HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_IS_NEGATIVE_SCALE_EXT,
            HIPDNN_TYPE_BOOLEAN,
            isNegativeScale,
            "block_scale_dequantize is_negative_scale"));
>>>>>>> d9e199e220 (merge b-shi branch)
        attributes.set_is_negative_scale(isNegativeScale);
    }

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(opDesc,
<<<<<<< HEAD
                                           HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC,
=======
                                           HIPDNN_ATTR_BLOCK_SCALE_DEQUANTIZE_MATH_PREC_EXT,
>>>>>>> d9e199e220 (merge b-shi branch)
                                           "block_scale_dequantize compute data type");
    if(dtErr.is_bad())
    {
        return dtErr;
    }
    attributes.set_compute_data_type(dt);

    // Unpack operation name
    std::string opName;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrString(opDesc, HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    attributes.set_name(opName);

    return {};
}

} // namespace hipdnn_frontend::detail
