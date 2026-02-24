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
    ck::tensor_layout::convolution::NGCDHW,
    ck::tensor_layout::convolution::GKZYXC,
    ck::tensor_layout::convolution::NGKDHW,
    ck::half_t,
    ck::half_t,
    ck::half_t,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

void add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ngcdhw_gkzyxc_ngkdhw_f16_pipev1_instances(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    namespace ckb = ck_tile::builder;
    using namespace factories::grouped_conv_bwd_weight;
    add_device_operation_instances<
        device_grouped_conv_bwd_weight_two_stage_ngchw_xdl_c_shuffle_f16_generic_instances(
            3, ckb::TensorLayout::NGCDHW, ckb::TensorLayout::GKZYXC, ckb::TensorLayout::NGKDHW,
            ConvBwdWeightDefault, Intrawave, PipeV1)>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
