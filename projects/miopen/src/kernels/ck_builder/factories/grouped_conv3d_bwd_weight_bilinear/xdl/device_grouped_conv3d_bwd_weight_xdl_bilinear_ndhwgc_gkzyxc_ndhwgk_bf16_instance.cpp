// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/kernel_instantiation.hpp>
// TODO: This header may not exist yet - create
// device_grouped_conv_bwd_weight_xdl_bilinear_instance.hpp when the bilinear factory infrastructure
// is ready.
#include <miopen/ck_builder/factories/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_xdl_bilinear_instance.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp"

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<
    3,
    ck::tensor_layout::convolution::NDHWGC,
    ck::tensor_layout::convolution::GKZYXC,
    ck::tensor_layout::convolution::NDHWGK,
    ck::Tuple<ck::tensor_layout::convolution::GKZYXC>,
    ck::bhalf_t,
    float,
    ck::bhalf_t,
    ck::Tuple<float>,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::Bilinear,
    ck::tensor_operation::element_wise::PassThrough>;

void add_device_grouped_conv3d_bwd_weight_xdl_bilinear_ndhwgc_gkzyxc_ndhwgk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    namespace ckb = ck_tile::builder;
    using namespace factories::grouped_conv_bwd_weight;

    // Layout aliases
    constexpr auto NDHWGC = ckb::TensorLayout::NDHWGC;
    constexpr auto GKZYXC = ckb::TensorLayout::GKZYXC;
    constexpr auto NDHWGK = ckb::TensorLayout::NDHWGK;

    // Specialization aliases
    constexpr auto ConvBwdWeightDefault = ckb::ConvSpecialization::DEFAULT;
    constexpr auto ConvBwdWeight1x1S1P0 = ckb::ConvSpecialization::FILTER_1X1_STRIDE1_PAD0;

    // 1. Default
    add_device_operation_instances<
        device_grouped_conv_bwd_weight_xdl_c_shuffle_bf16_bilinear_instances(
            3, NDHWGC, GKZYXC, NDHWGK, ConvBwdWeightDefault)>(instances);

    // 2. Filter1x1Stride1Pad0
    add_device_operation_instances<
        device_grouped_conv_bwd_weight_xdl_c_shuffle_bf16_bilinear_instances(
            3, NDHWGC, GKZYXC, NDHWGK, ConvBwdWeight1x1S1P0)>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
