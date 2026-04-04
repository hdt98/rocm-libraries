// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv2d_bwd_data_xdl_gnhwk_gkyxc_gnhwc_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances)
{
    // [COMMENTED OUT — CShuffle instances win 0/579 cases, NoShuffle wins 100%]
    // Uncomment for final full-search verification after tuning is complete.
#if 0
    // 1. Default (16x16 XDL, small tiles)
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f32_16_16_instances<2,
                                                             GNHWK,
                                                             GKYXC,
                                                             Empty_Tuple,
                                                             GNHWC,
                                                             ConvBwdDataDefault>{});
    // 2. Filter1x1Stride1Pad0 (16x16 XDL, small tiles)
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f32_16_16_instances<2,
                                                             GNHWK,
                                                             GKYXC,
                                                             Empty_Tuple,
                                                             GNHWC,
                                                             ConvBwdDataFilter1x1Stride1Pad0>{});
    // 3. Default (32x32 XDL, large tiles)
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f32_instances<2,
                                                       GNHWK,
                                                       GKYXC,
                                                       Empty_Tuple,
                                                       GNHWC,
                                                       ConvBwdDataDefault>{});
    // 4. Filter1x1Stride1Pad0 (32x32 XDL, large tiles)
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f32_instances<2,
                                                       GNHWK,
                                                       GKYXC,
                                                       Empty_Tuple,
                                                       GNHWC,
                                                       ConvBwdDataFilter1x1Stride1Pad0>{});
#endif
    // 5. Default — noshuffle epilogue (ScalarPerVector=1)
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f32_noshuffle_instances<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  ConvBwdDataDefault>{});
    // 6. Filter1x1Stride1Pad0 — noshuffle epilogue (ScalarPerVector=1)
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f32_noshuffle_instances<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  ConvBwdDataFilter1x1Stride1Pad0>{});
    // 7. Default — BBlockTransfer matching non-grouped kernel (NoShuffle, SPV=1,1)
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f32_nongrouped_match_instances<2,
                                                                         GNHWK,
                                                                         GKYXC,
                                                                         Empty_Tuple,
                                                                         GNHWC,
                                                                         ConvBwdDataDefault>{});
    // 8. Filter1x1Stride1Pad0 — BBlockTransfer matching non-grouped kernel (NoShuffle, SPV=1,1)
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f32_nongrouped_match_instances<2,
                                                                         GNHWK,
                                                                         GKYXC,
                                                                         Empty_Tuple,
                                                                         GNHWC,
                                                                         ConvBwdDataFilter1x1Stride1Pad0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
