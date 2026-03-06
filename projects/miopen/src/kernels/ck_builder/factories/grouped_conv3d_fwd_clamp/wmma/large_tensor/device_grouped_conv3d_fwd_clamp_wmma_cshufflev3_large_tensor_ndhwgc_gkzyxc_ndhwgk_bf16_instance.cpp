// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_large_tensor_instance.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_clamp.hpp"

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
    ck::tensor_operation::element_wise::Clamp>;

// Compilation parameters for in[n, di, hi, wi, g, c] * wei[g, k, z, y, x, c] = out[n, do, ho, wo,
// g, k]
void add_device_grouped_conv3d_fwd_clamp_wmma_cshufflev3_large_tensor_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    namespace ckb = ck_tile::builder;
    using namespace factories::grouped_conv_fwd;

    // Layout aliases
    constexpr auto NDHWGC = ckb::TensorLayout::NDHWGC;
    constexpr auto GKZYXC = ckb::TensorLayout::GKZYXC;
    constexpr auto NDHWGK = ckb::TensorLayout::NDHWGK;

    // Specialization aliases
    constexpr auto ConvFwdDefault = ckb::ConvSpecialization::DEFAULT;

    add_device_operation_instances<device_grouped_conv_fwd_wmma_large_tensor_bf16_instances<0>(
        3, NDHWGC, GKZYXC, {}, NDHWGK, ConvFwdDefault, {}, Clamp)>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
