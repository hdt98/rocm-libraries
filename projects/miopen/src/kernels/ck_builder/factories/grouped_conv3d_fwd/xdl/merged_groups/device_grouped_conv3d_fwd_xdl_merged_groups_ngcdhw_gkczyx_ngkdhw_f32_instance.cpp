// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/device_grouped_conv_fwd_xdl_merged_groups_instance.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
    3,
    ck::tensor_layout::convolution::NGCDHW,
    ck::tensor_layout::convolution::GKCZYX,
    ck::Tuple<>,
    ck::tensor_layout::convolution::NGKDHW,
    float,
    float,
    ck::Tuple<>,
    float,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

// Compilation parameters for in[n, g, c, di, hi, wi] * wei[g, k, c, z, y, x] = out[n, g, k, do, ho,
// wo]
void add_device_grouped_conv3d_fwd_xdl_merged_groups_ngcdhw_gkczyx_ngkdhw_f32_instances(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    namespace ckb = ck_tile::builder;
    using namespace factories::grouped_conv_fwd;

    // Layout aliases
    constexpr auto GKCZYX = ckb::TensorLayout::GKCZYX;
    constexpr auto NGCDHW = ckb::TensorLayout::NGCDHW;
    constexpr auto NGKDHW = ckb::TensorLayout::NGKDHW;

    // Specialization aliases
    constexpr auto ConvFwdDefault = ckb::ConvSpecialization::DEFAULT;
    constexpr auto ConvFwd3x3     = ckb::ConvSpecialization::FILTER_3x3;

    add_device_operation_instances<device_grouped_conv_fwd_xdl_merged_groups_f32_instances<0>(
        3, NGCDHW, GKCZYX, {}, NGKDHW, ConvFwdDefault)>(instances);

    add_device_operation_instances<device_grouped_conv_fwd_xdl_merged_groups_f32_instances<0>(
        3, NGCDHW, GKCZYX, {}, NGKDHW, ConvFwd3x3)>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
