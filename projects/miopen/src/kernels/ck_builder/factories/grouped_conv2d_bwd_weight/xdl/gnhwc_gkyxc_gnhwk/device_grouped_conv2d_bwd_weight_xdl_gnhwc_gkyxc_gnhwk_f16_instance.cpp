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

void add_device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    namespace ckb = ck_tile::builder;
    using namespace factories::grouped_conv_bwd_weight;

    // Layout aliases
    constexpr auto GNHWC = ckb::TensorLayout::GNHWC;
    constexpr auto GKYXC = ckb::TensorLayout::GKYXC;
    constexpr auto GNHWK = ckb::TensorLayout::GNHWK;

    // Specialization aliases
    constexpr auto ConvBwdWeightDefault = ckb::ConvSpecialization::DEFAULT;
    constexpr auto ConvBwdWeight1x1S1P0 = ckb::ConvSpecialization::FILTER_1X1_STRIDE1_PAD0;

    // 1. Default
    add_device_operation_instances<
        device_grouped_conv_bwd_weight_xdl_c_shuffle_f16_generic_instances(
            2, GNHWC, GKYXC, GNHWK, ConvBwdWeightDefault)>(instances);

    // 2. Filter1x1Stride1Pad0
    add_device_operation_instances<
        device_grouped_conv_bwd_weight_xdl_c_shuffle_f16_generic_instances(
            2, GNHWC, GKYXC, GNHWK, ConvBwdWeight1x1S1P0)>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
