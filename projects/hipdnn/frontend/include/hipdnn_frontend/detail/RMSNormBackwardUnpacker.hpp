// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/RMSNormBackwardAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackRMSNormBackwardOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::RMSNormBackwardAttributes& attributes)
{
    // Unpack dy tensor
    std::shared_ptr<graph::TensorAttributes> dyTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT,
                                               tensorMap,
                                               dyTensor,
                                               "rmsnormbackward DY_EXT tensor"));
    attributes.set_dy(dyTensor);

    // Unpack x tensor
    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_X_EXT,
                                               tensorMap,
                                               xTensor,
                                               "rmsnormbackward X_EXT tensor"));
    attributes.set_x(xTensor);

    // Unpack scale tensor
    std::shared_ptr<graph::TensorAttributes> scaleTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_SCALE_EXT,
                                               tensorMap,
                                               scaleTensor,
                                               "rmsnormbackward SCALE_EXT tensor"));
    attributes.set_scale(scaleTensor);

    // Unpack dx tensor
    std::shared_ptr<graph::TensorAttributes> dxTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DX_EXT,
                                               tensorMap,
                                               dxTensor,
                                               "rmsnormbackward DX_EXT tensor"));
    attributes.set_dx(dxTensor);

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(
        opDesc, HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT, "rmsnormbackward compute data type");
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
