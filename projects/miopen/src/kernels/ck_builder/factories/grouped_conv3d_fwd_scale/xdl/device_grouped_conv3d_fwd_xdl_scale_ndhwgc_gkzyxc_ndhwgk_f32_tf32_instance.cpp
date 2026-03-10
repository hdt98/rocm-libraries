// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scale.hpp"

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
    3,
    ck::tensor_layout::convolution::NDHWGC,
    ck::tensor_layout::convolution::GKZYXC,
    ck::Tuple<>,
    ck::tensor_layout::convolution::NDHWGK,
    float,
    float,
    ck::Tuple<>,
    float,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::Scale,
    ck::tf32_t,
    ck::tf32_t>;

// Compilation parameters for in[n, di, hi, wi, g, c] * wei[g, k, z, y, x, c] = out[n, do, ho, wo,
// g, k]
void add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    // TODO: TF32 compute type is not yet supported in ckb::DataType enum
    // add_device_operation_instances<device_grouped_conv_fwd_xdl_f32_tf32_instances<0>(
    //     3, NDHWGC, GKZYXC, {}, NDHWGK, ConvFwdDefault, {}, Scale)>(instances);
    //
    // add_device_operation_instances<device_grouped_conv_fwd_xdl_f32_tf32_instances<0>(
    //     3, NDHWGC, GKZYXC, {}, NDHWGK, ConvFwd1x1P0, {}, Scale)>(instances);
    //
    // add_device_operation_instances<device_grouped_conv_fwd_xdl_f32_tf32_instances<0>(
    //     3, NDHWGC, GKZYXC, {}, NDHWGK, ConvFwd1x1S1P0, {}, Scale)>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
