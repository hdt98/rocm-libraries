// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/factories/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_two_stage_xdl_instance.hpp>
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

void add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_pipev1_instances(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    namespace ckb = ck_tile::builder;
    using namespace factories::grouped_conv_bwd_weight;
    add_device_operation_instances<
        device_grouped_conv_bwd_weight_two_stage_nhwgc_xdl_c_shuffle_bf16_instances(
            3,
            ckb::TensorLayout::NDHWGC,
            ckb::TensorLayout::GKZYXC,
            ckb::TensorLayout::NDHWGK,
            ConvBwdWeightDefault,
            Intrawave,
            PipeV1)>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
