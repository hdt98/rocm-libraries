// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/factories/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_xdl_instance.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp"

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvBwdWeight<
    3,
    ck::tensor_layout::convolution::NGCDHW,
    ck::tensor_layout::convolution::GKCZYX,
    ck::tensor_layout::convolution::NGKDHW,
    ck::bhalf_t,
    ck::bhalf_t,
    ck::bhalf_t,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

void add_device_grouped_conv3d_bwd_weight_xdl_ngcdhw_gkczyx_ngkdhw_bf16_instances(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    namespace ckb = ck_tile::builder;
    using namespace factories::grouped_conv_bwd_weight;

    // Layout aliases
    constexpr auto NGCDHW = ckb::TensorLayout::NGCDHW;
    constexpr auto GKCZYX = ckb::TensorLayout::GKCZYX;
    constexpr auto NGKDHW = ckb::TensorLayout::NGKDHW;

    // Specialization aliases
    constexpr auto ConvBwdWeightDefault = ckb::ConvSpecialization::DEFAULT;

    // 1. Default
    add_device_operation_instances<
        device_grouped_conv_bwd_weight_xdl_c_shuffle_bf16_instances(
            3, NGCDHW, GKCZYX, NGKDHW, ConvBwdWeightDefault)>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
