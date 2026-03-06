// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

#ifdef CK_ENABLE_BF16

void add_device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instances_part1(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
                                                                ck::tensor_layout::convolution::NHWGC,
                                                                ck::tensor_layout::convolution::GKYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NHWGK>,
                                                                ck::tensor_layout::convolution::NHWGK,
                                                                ck::bhalf_t,
                                                                ck::bhalf_t,
                                                                ck::Tuple<ck::bhalf_t>,
                                                                ck::bhalf_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instances_part2(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
                                                                ck::tensor_layout::convolution::NHWGC,
                                                                ck::tensor_layout::convolution::GKYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NHWGK>,
                                                                ck::tensor_layout::convolution::NHWGK,
                                                                ck::bhalf_t,
                                                                ck::bhalf_t,
                                                                ck::Tuple<ck::bhalf_t>,
                                                                ck::bhalf_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instances_part3(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
                                                                ck::tensor_layout::convolution::NHWGC,
                                                                ck::tensor_layout::convolution::GKYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NHWGK>,
                                                                ck::tensor_layout::convolution::NHWGK,
                                                                ck::bhalf_t,
                                                                ck::bhalf_t,
                                                                ck::Tuple<ck::bhalf_t>,
                                                                ck::bhalf_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instances_part4(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
                                                                ck::tensor_layout::convolution::NHWGC,
                                                                ck::tensor_layout::convolution::GKYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NHWGK>,
                                                                ck::tensor_layout::convolution::NHWGK,
                                                                ck::bhalf_t,
                                                                ck::bhalf_t,
                                                                ck::Tuple<ck::bhalf_t>,
                                                                ck::bhalf_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
                                                                ck::tensor_layout::convolution::NHWGC,
                                                                ck::tensor_layout::convolution::GKYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NHWGK>,
                                                                ck::tensor_layout::convolution::NHWGK,
                                                                ck::bhalf_t,
                                                                ck::bhalf_t,
                                                                ck::Tuple<ck::bhalf_t>,
                                                                ck::bhalf_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_bf16_generic_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
                                                                ck::tensor_layout::convolution::NHWGC,
                                                                ck::tensor_layout::convolution::GKYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NHWGK>,
                                                                ck::tensor_layout::convolution::NHWGK,
                                                                ck::bhalf_t,
                                                                ck::bhalf_t,
                                                                ck::Tuple<ck::bhalf_t>,
                                                                ck::bhalf_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_wmma_cshufflev3_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part1(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                ck::bhalf_t,
                                                                ck::bhalf_t,
                                                                ck::Tuple<ck::bhalf_t>,
                                                                ck::bhalf_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_wmma_cshufflev3_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part2(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                ck::bhalf_t,
                                                                ck::bhalf_t,
                                                                ck::Tuple<ck::bhalf_t>,
                                                                ck::bhalf_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_wmma_cshufflev3_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part3(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                ck::bhalf_t,
                                                                ck::bhalf_t,
                                                                ck::Tuple<ck::bhalf_t>,
                                                                ck::bhalf_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_wmma_cshufflev3_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part4(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                ck::bhalf_t,
                                                                ck::bhalf_t,
                                                                ck::Tuple<ck::bhalf_t>,
                                                                ck::bhalf_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_wmma_cshufflev3_large_tensor_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                ck::bhalf_t,
                                                                ck::bhalf_t,
                                                                ck::Tuple<ck::bhalf_t>,
                                                                ck::bhalf_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_wmma_cshufflev3_large_tensor_ndhwgc_gkzyxc_ndhwgk_bf16_generic_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                ck::bhalf_t,
                                                                ck::bhalf_t,
                                                                ck::Tuple<ck::bhalf_t>,
                                                                ck::bhalf_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

#endif

#ifdef CK_ENABLE_FP16

void add_device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instances_part1(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
                                                                ck::tensor_layout::convolution::NHWGC,
                                                                ck::tensor_layout::convolution::GKYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NHWGK>,
                                                                ck::tensor_layout::convolution::NHWGK,
                                                                ck::half_t,
                                                                ck::half_t,
                                                                ck::Tuple<ck::half_t>,
                                                                ck::half_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instances_part2(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
                                                                ck::tensor_layout::convolution::NHWGC,
                                                                ck::tensor_layout::convolution::GKYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NHWGK>,
                                                                ck::tensor_layout::convolution::NHWGK,
                                                                ck::half_t,
                                                                ck::half_t,
                                                                ck::Tuple<ck::half_t>,
                                                                ck::half_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instances_part3(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
                                                                ck::tensor_layout::convolution::NHWGC,
                                                                ck::tensor_layout::convolution::GKYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NHWGK>,
                                                                ck::tensor_layout::convolution::NHWGK,
                                                                ck::half_t,
                                                                ck::half_t,
                                                                ck::Tuple<ck::half_t>,
                                                                ck::half_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instances_part4(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
                                                                ck::tensor_layout::convolution::NHWGC,
                                                                ck::tensor_layout::convolution::GKYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NHWGK>,
                                                                ck::tensor_layout::convolution::NHWGK,
                                                                ck::half_t,
                                                                ck::half_t,
                                                                ck::Tuple<ck::half_t>,
                                                                ck::half_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_f16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
                                                                ck::tensor_layout::convolution::NHWGC,
                                                                ck::tensor_layout::convolution::GKYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NHWGK>,
                                                                ck::tensor_layout::convolution::NHWGK,
                                                                ck::half_t,
                                                                ck::half_t,
                                                                ck::Tuple<ck::half_t>,
                                                                ck::half_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_f16_generic_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
                                                                ck::tensor_layout::convolution::NHWGC,
                                                                ck::tensor_layout::convolution::GKYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NHWGK>,
                                                                ck::tensor_layout::convolution::NHWGK,
                                                                ck::half_t,
                                                                ck::half_t,
                                                                ck::Tuple<ck::half_t>,
                                                                ck::half_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_wmma_cshufflev3_ndhwgc_gkzyxc_ndhwgk_f16_instances_part1(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                ck::half_t,
                                                                ck::half_t,
                                                                ck::Tuple<ck::half_t>,
                                                                ck::half_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_wmma_cshufflev3_ndhwgc_gkzyxc_ndhwgk_f16_instances_part2(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                ck::half_t,
                                                                ck::half_t,
                                                                ck::Tuple<ck::half_t>,
                                                                ck::half_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_wmma_cshufflev3_ndhwgc_gkzyxc_ndhwgk_f16_instances_part3(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                ck::half_t,
                                                                ck::half_t,
                                                                ck::Tuple<ck::half_t>,
                                                                ck::half_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_wmma_cshufflev3_ndhwgc_gkzyxc_ndhwgk_f16_instances_part4(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                ck::half_t,
                                                                ck::half_t,
                                                                ck::Tuple<ck::half_t>,
                                                                ck::half_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_wmma_cshufflev3_large_tensor_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                ck::half_t,
                                                                ck::half_t,
                                                                ck::Tuple<ck::half_t>,
                                                                ck::half_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_wmma_cshufflev3_large_tensor_ndhwgc_gkzyxc_ndhwgk_f16_generic_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                ck::half_t,
                                                                ck::half_t,
                                                                ck::Tuple<ck::half_t>,
                                                                ck::half_t,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

#endif

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
