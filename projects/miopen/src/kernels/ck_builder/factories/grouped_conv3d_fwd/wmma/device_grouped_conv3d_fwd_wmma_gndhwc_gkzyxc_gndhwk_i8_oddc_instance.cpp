// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/device_grouped_conv_fwd_wmma_instance.hpp>
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
    int8_t,
    int8_t,
    ck::Tuple<>,
    int8_t,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

// Compilation parameters for in[g, n, di, hi, wi, c] * wei[g, k, z, y, x, c] = out[g, n, do, ho, wo, k]
void add_device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_i8_oddc_instances(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    namespace ckb = ck_tile::builder;
    using namespace factories::grouped_conv_fwd;

    // Layout aliases
    constexpr auto GKZYXC = ckb::TensorLayout::GKZYXC;
    constexpr auto GNDHWC = ckb::TensorLayout::GNDHWC;
    constexpr auto GNDHWK = ckb::TensorLayout::GNDHWK;

    // Specialization aliases
    constexpr auto ConvFwdOddC = ckb::ConvSpecialization::ODD_C;

    add_device_operation_instances<device_grouped_conv_fwd_wmma_i8_instances<0>(
        3, GNDHWC, GKZYXC, {}, GNDHWK, ConvFwdOddC)>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
