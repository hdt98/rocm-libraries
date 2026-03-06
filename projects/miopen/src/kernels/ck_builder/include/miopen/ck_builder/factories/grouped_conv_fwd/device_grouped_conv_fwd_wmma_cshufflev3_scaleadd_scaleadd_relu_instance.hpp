// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include <miopen/ck_builder/instance_data/wmma_v3.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/common_aliases.hpp>
#include <array>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_fwd {

using namespace instance;

// Note: DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3 is UNSUPPORTED by CK Builder.
// All functions return empty arrays as stubs.

template <std::size_t NumDTensor = 2>
constexpr auto device_grouped_conv_fwd_wmma_cshufflev3_scaleadd_scaleadd_relu_bf16_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {BF16, BF16})
{
    return std::array<int, 0>{};
}

template <std::size_t NumDTensor = 2>
constexpr auto device_grouped_conv_fwd_wmma_cshufflev3_scaleadd_scaleadd_relu_f16_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {F16, F16})
{
    return std::array<int, 0>{};
}

} // namespace grouped_conv_fwd
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
