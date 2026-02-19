// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
    3,
    ck::tensor_layout::convolution::GNDHWC,
    ck::tensor_layout::convolution::GKZYXC,
    ck::Tuple<>,
    ck::tensor_layout::convolution::GNDHWK,
    ck::bhalf_t,
    ck::bhalf_t,
    ck::Tuple<>,
    ck::bhalf_t,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

// Compilation parameters for in[g, n, di, hi, wi, c] * wei[g, k, z, y, x, c] = out[g, n, do, ho,
// wo, k]
void add_device_grouped_conv3d_fwd_xdl_gndhwc_gkzyxc_gndhwk_bf16_instances(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    namespace ckb = ck_tile::builder;
    using namespace factories::grouped_conv_fwd;

    // Layout aliases
    constexpr auto GKZYXC = ckb::TensorLayout::GKZYXC;
    constexpr auto GNDHWC = ckb::TensorLayout::GNDHWC;
    constexpr auto GNDHWK = ckb::TensorLayout::GNDHWK;

    // Specialization aliases
    constexpr auto ConvFwdDefault = ckb::ConvSpecialization::DEFAULT;
    constexpr auto ConvFwd1x1P0   = ckb::ConvSpecialization::FILTER_1X1_PAD0;
    constexpr auto ConvFwd1x1S1P0 = ckb::ConvSpecialization::FILTER_1X1_STRIDE1_PAD0;

    add_device_operation_instances<device_grouped_conv_fwd_xdl_bf16_instances<0>(
        3, GNDHWC, GKZYXC, {}, GNDHWK, ConvFwdDefault)>(instances);

    add_device_operation_instances<device_grouped_conv_fwd_xdl_bf16_instances<0>(
        3, GNDHWC, GKZYXC, {}, GNDHWK, ConvFwd1x1P0)>(instances);

    add_device_operation_instances<device_grouped_conv_fwd_xdl_bf16_instances<0>(
        3, GNDHWC, GKZYXC, {}, GNDHWK, ConvFwd1x1S1P0)>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
