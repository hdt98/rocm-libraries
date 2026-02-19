// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

#ifdef CK_ENABLE_BF16
// grouped conv3d forward, NDHWGC/GKZYXC/NDHWGK
void add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                Empty_Tuple,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Scale>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Scale>>>& instances);
#endif

#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Scale>>>& instances);
#endif

#ifdef CK_ENABLE_TF32
void add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Scale,
                                                                TF32,
                                                                TF32>>>& instances);
#endif

#ifdef CK_ENABLE_INT8
void add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_int8_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                Scale>>>& instances);
#endif

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
