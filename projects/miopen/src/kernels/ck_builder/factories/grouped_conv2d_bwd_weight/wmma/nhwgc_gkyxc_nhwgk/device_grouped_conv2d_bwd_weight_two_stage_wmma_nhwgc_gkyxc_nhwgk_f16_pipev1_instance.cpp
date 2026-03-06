// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/kernel_instantiation.hpp>
// TODO: This header may not exist yet - create
// device_grouped_conv_bwd_weight_two_stage_wmma_instance.hpp when the two-stage WMMA factory
// infrastructure is ready.
#include <miopen/ck_builder/factories/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_two_stage_wmma_instance.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp"

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvBwdWeight<
    2,
    ck::tensor_layout::convolution::NHWGC,
    ck::tensor_layout::convolution::GKYXC,
    ck::tensor_layout::convolution::NHWGK,
    ck::half_t,
    ck::half_t,
    ck::half_t,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

void add_device_grouped_conv2d_bwd_weight_two_stage_wmma_nhwgc_gkyxc_nhwgk_f16_pipev1_instances(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    namespace ckb = ck_tile::builder;
    using namespace factories::grouped_conv_bwd_weight;

    // Layout aliases
    constexpr auto NHWGC = ckb::TensorLayout::NHWGC;
    constexpr auto GKYXC = ckb::TensorLayout::GKYXC;
    constexpr auto NHWGK = ckb::TensorLayout::NHWGK;

    // Specialization aliases
    constexpr auto ConvBwdWeightDefault = ckb::ConvSpecialization::DEFAULT;

    // 1. Default
    add_device_operation_instances<
        device_grouped_conv_bwd_weight_two_stage_nhwgc_wmma_c_shuffle_f16_instances(
            2, NHWGC, GKYXC, NHWGK, ConvBwdWeightDefault)>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
