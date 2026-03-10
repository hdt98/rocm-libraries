// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

#ifdef CK_ENABLE_BF16

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_instances(
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

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_16x16_instances(
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

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instances(
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

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_merged_groups_nhwgc_gkyxc_nhwgk_bf16_instances(
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

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_instances(
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

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_2x_instances(
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

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_part2_instances(
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

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_intra_instances(
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

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_inter_instances(
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

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_direct_load_instances(
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

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
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

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_16x16_instances(
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

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_large_tensor_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
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

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_merged_groups_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
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

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_comp_instances(
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

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_mem_intra_instances(
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

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_mem_inter_instances(
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

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_f16_instances(
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

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_f16_16x16_instances(
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

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f16_instances(
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

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f16_instances(
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

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_f16_comp_instances(
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

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_f16_comp_2x_instances(
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

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_f16_comp_part2_instances(
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

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_f16_mem_intra_instances(
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

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_f16_mem_inter_instances(
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

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_f16_direct_load_instances(
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

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_instances(
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

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_16x16_instances(
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

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_large_tensor_ndhwgc_gkzyxc_ndhwgk_f16_instances(
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

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_merged_groups_ndhwgc_gkzyxc_ndhwgk_f16_instances(
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

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_comp_instances(
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

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_mem_intra_instances(
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

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_mem_inter_instances(
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

#ifdef CK_ENABLE_FP32

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
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
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_16x16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
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
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f32_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
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
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f32_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
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
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_comp_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
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
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_mem_intra_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
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
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_mem_inter_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
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
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                float,
                                                                float,
                                                                ck::Tuple<float>,
                                                                float,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_16x16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                float,
                                                                float,
                                                                ck::Tuple<float>,
                                                                float,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_large_tensor_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                float,
                                                                float,
                                                                ck::Tuple<float>,
                                                                float,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_merged_groups_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                float,
                                                                float,
                                                                ck::Tuple<float>,
                                                                float,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_comp_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                float,
                                                                float,
                                                                ck::Tuple<float>,
                                                                float,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_mem_intra_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                float,
                                                                float,
                                                                ck::Tuple<float>,
                                                                float,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_mem_inter_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                float,
                                                                float,
                                                                ck::Tuple<float>,
                                                                float,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp>>>& instances);

#endif

#ifdef CK_ENABLE_TF32

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
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
                                                                ck::tf32_t>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_16x16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
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
                                                                ck::tf32_t>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
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
                                                                ck::tf32_t>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
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
                                                                ck::tf32_t>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_comp_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
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
                                                                ck::tf32_t>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_mem_intra_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
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
                                                                ck::tf32_t>>>& instances);

void add_device_grouped_conv2d_fwd_bias_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_mem_inter_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
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
                                                                ck::tf32_t>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                float,
                                                                float,
                                                                ck::Tuple<float>,
                                                                float,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp,
                                                                ck::tf32_t,
                                                                ck::tf32_t>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_16x16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                float,
                                                                float,
                                                                ck::Tuple<float>,
                                                                float,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp,
                                                                ck::tf32_t,
                                                                ck::tf32_t>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_large_tensor_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                float,
                                                                float,
                                                                ck::Tuple<float>,
                                                                float,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp,
                                                                ck::tf32_t,
                                                                ck::tf32_t>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_merged_groups_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                float,
                                                                float,
                                                                ck::Tuple<float>,
                                                                float,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp,
                                                                ck::tf32_t,
                                                                ck::tf32_t>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_comp_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                float,
                                                                float,
                                                                ck::Tuple<float>,
                                                                float,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp,
                                                                ck::tf32_t,
                                                                ck::tf32_t>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_mem_intra_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                float,
                                                                float,
                                                                ck::Tuple<float>,
                                                                float,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp,
                                                                ck::tf32_t,
                                                                ck::tf32_t>>>& instances);

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_mem_inter_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                ck::tensor_layout::convolution::NDHWGC,
                                                                ck::tensor_layout::convolution::GKZYXC,
                                                                ck::Tuple<ck::tensor_layout::convolution::NDHWGK>,
                                                                ck::tensor_layout::convolution::NDHWGK,
                                                                float,
                                                                float,
                                                                ck::Tuple<float>,
                                                                float,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                ck::tensor_operation::element_wise::AddClamp,
                                                                ck::tf32_t,
                                                                ck::tf32_t>>>& instances);

#endif

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
