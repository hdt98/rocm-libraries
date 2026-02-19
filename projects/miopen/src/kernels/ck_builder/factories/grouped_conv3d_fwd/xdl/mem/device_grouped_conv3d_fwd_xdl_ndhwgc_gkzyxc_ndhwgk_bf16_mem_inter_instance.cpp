// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/device_grouped_conv_fwd_xdl_mem_instance.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"

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
    ck::bhalf_t,
    ck::bhalf_t,
    ck::Tuple<>,
    ck::bhalf_t,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

// Compilation parameters for in[n, di, hi, wi, g, c] * wei[g, k, z, y, x, c] = out[n, do, ho, wo,
// g, k]
void add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_mem_inter_instances(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    namespace ckb = ck_tile::builder;
    using namespace factories::grouped_conv_fwd;

    // Layout aliases
    constexpr auto GKZYXC = ckb::TensorLayout::GKZYXC;
    constexpr auto NDHWGC = ckb::TensorLayout::NDHWGC;
    constexpr auto NDHWGK = ckb::TensorLayout::NDHWGK;

    // Specialization aliases
    constexpr auto ConvFwdDefault = ckb::ConvSpecialization::DEFAULT;
    constexpr auto ConvFwd1x1P0   = ckb::ConvSpecialization::FILTER_1X1_PAD0;
    constexpr auto ConvFwd1x1S1P0 = ckb::ConvSpecialization::FILTER_1X1_STRIDE1_PAD0;

    constexpr auto Interwave = ckb::PipelineScheduler::INTERWAVE;

    add_device_operation_instances<device_grouped_conv_fwd_xdl_bf16_mem_instances<0>(
        3, NDHWGC, GKZYXC, {}, NDHWGK, ConvFwdDefault, Interwave)>(instances);

    add_device_operation_instances<device_grouped_conv_fwd_xdl_bf16_mem_instances<0>(
        3, NDHWGC, GKZYXC, {}, NDHWGK, ConvFwd1x1P0, Interwave)>(instances);

    add_device_operation_instances<device_grouped_conv_fwd_xdl_bf16_mem_instances<0>(
        3, NDHWGC, GKZYXC, {}, NDHWGK, ConvFwd1x1S1P0, Interwave)>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
