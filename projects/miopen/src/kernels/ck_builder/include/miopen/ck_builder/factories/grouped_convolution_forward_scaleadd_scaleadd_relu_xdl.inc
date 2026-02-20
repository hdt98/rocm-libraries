// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

#ifdef CK_ENABLE_BF16
// grouped conv3d forward, NDHWGC/GKZYXC/NDHWGK
void add_device_grouped_conv3d_fwd_xdl_scaleadd_scaleadd_relu_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<NDHWGK, G_K>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                ck::Tuple<BF16, BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                ScaleAddScaleAddRelu>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_fwd_xdl_scaleadd_scaleadd_relu_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<NDHWGK, G_K>,
                                                                NDHWGK,
                                                                F16,
                                                                F16,
                                                                ck::Tuple<F16, F16>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                ScaleAddScaleAddRelu>>>& instances);
#endif

#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_fwd_xdl_scaleadd_scaleadd_relu_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<NDHWGK, G_K>,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                ck::Tuple<F32, F32>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                ScaleAddScaleAddRelu>>>& instances);
#endif

#ifdef CK_ENABLE_INT8
void add_device_grouped_conv3d_fwd_xdl_scaleadd_scaleadd_relu_ndhwgc_gkzyxc_ndhwgk_int8_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<NDHWGK, G_K>,
                                                                NDHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                ck::Tuple<F32, F32>,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                ScaleAddScaleAddRelu>>>& instances);
#endif

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
