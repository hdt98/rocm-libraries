// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TODO: Requires v3 (pipeline) bwd weight factory header to be ported.
// Original CK source uses device_grouped_conv_bwd_weight_v3_xdl_c_shuffle_bf16_instances
// with BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v2
// and ConvBwdWeightDefault specialization.
// Layouts: NDHWGC / GKZYXC / NDHWGK, NDimSpatial=3, BF16/BF16/BF16

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp"

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvBwdWeight<
    3,
    ck::tensor_layout::convolution::NDHWGC,
    ck::tensor_layout::convolution::GKZYXC,
    ck::tensor_layout::convolution::NDHWGK,
    ck::bhalf_t,
    ck::bhalf_t,
    ck::bhalf_t,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

void add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_default_pipev2_instances(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    // TODO: Port v3 pipeline factory
    // Original: device_grouped_conv_bwd_weight_v3_xdl_c_shuffle_bf16_instances<3, NDHWGC, GKZYXC,
    // NDHWGK,
    //     ConvBwdWeightDefault, BlockGemmPipelineScheduler::Intrawave,
    //     BlockGemmPipelineVersion::v2>
    std::ignore = instances;
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
