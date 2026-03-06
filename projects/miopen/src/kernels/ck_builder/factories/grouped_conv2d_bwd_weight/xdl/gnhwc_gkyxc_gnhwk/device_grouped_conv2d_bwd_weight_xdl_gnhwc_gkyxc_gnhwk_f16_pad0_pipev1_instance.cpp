// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TODO: This file uses the V3 XDL template (device_grouped_conv_bwd_weight_v3_xdl_instance.hpp)
// with pipeline scheduling parameters (BlockGemmPipelineScheduler::Intrawave,
// BlockGemmPipelineVersion::v1). The corresponding CK Builder factory function for V3 XDL
// instances is not yet available. Once the V3 factory header is created, convert this file.
//
// Original CK tuple: device_grouped_conv_bwd_weight_v3_xdl_c_shuffle_f16_instances<
//     2, GNHWC, GKYXC, GNHWK, ConvBwdWeightFilter1x1Stride1Pad0,
//     BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1>

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/factories/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_xdl_instance.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp"

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvBwdWeight<
    2,
    ck::tensor_layout::convolution::GNHWC,
    ck::tensor_layout::convolution::GKYXC,
    ck::tensor_layout::convolution::GNHWK,
    ck::half_t,
    ck::half_t,
    ck::half_t,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

void add_device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f16_pad0_pipev1_instances(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    // TODO: Needs V3 XDL factory function support
    // device_grouped_conv_bwd_weight_v3_xdl_c_shuffle_f16_instances<
    //     2, GNHWC, GKYXC, GNHWK, ConvBwdWeightFilter1x1Stride1Pad0,
    //     BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1>
    (void)instances;
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
