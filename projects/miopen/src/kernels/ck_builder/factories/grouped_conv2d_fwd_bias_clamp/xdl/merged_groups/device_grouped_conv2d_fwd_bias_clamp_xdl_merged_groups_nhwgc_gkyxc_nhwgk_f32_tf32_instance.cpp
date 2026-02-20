// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/device_grouped_conv_fwd_xdl_merged_groups_instance.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bias_clamp.hpp"

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
    2,
    ck::tensor_layout::convolution::NHWGC,
    ck::tensor_layout::convolution::GKYXC,
    ck::Tuple<ck::tensor_layout::convolution::NHWGK>,
    ck::tensor_layout::convolution::NHWGK,
    float,
    float,
    ck::Tuple<float>,
    float,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::AddClamp,
    ck::tf32_t,
    ck::tf32_t>;

// Compilation parameters for in[n, hi, wi, g, c] * wei[g, k, y, x, c] = out[n, ho, wo, g, k]
void add_device_grouped_conv2d_fwd_bias_clamp_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    namespace ckb = ck_tile::builder;
    using namespace factories::grouped_conv_fwd;

    // Layout aliases
    constexpr auto NHWGC = ckb::TensorLayout::NHWGC;
    constexpr auto GKYXC = ckb::TensorLayout::GKYXC;
    constexpr auto NHWGK = ckb::TensorLayout::NHWGK;

    // Specialization aliases
    constexpr auto ConvFwdDefault    = ckb::ConvSpecialization::DEFAULT;
    constexpr auto ConvFwd1x1P0      = ckb::ConvSpecialization::FILTER_1X1_PAD0;
    constexpr auto ConvFwd1x1S1P0    = ckb::ConvSpecialization::FILTER_1X1_STRIDE1_PAD0;

    add_device_operation_instances<device_grouped_conv_fwd_xdl_merged_groups_f32_tf32_instances<1>(
        2, NHWGC, GKYXC, {NHWGK}, NHWGK, ConvFwdDefault, {F32}, AddClamp)>(instances);

    add_device_operation_instances<device_grouped_conv_fwd_xdl_merged_groups_f32_tf32_instances<1>(
        2, NHWGC, GKYXC, {NHWGK}, NHWGK, ConvFwd1x1P0, {F32}, AddClamp)>(instances);

    add_device_operation_instances<device_grouped_conv_fwd_xdl_merged_groups_f32_tf32_instances<1>(
        2, NHWGC, GKYXC, {NHWGK}, NHWGK, ConvFwd1x1S1P0, {F32}, AddClamp)>(instances);

}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
