// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv2d_bwd_data_xdl_gnhwk_gkyxc_gnhwc_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances)
{
    // 1. Default
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f16_16_16_instances<2,
                                                             GNHWK,
                                                             GKYXC,
                                                             Empty_Tuple,
                                                             GNHWC,
                                                             ConvBwdDataDefault>{});
    // 2. Filter1x1Stride1Pad0
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f16_16_16_instances<2,
                                                             GNHWK,
                                                             GKYXC,
                                                             Empty_Tuple,
                                                             GNHWC,
                                                             ConvBwdDataFilter1x1Stride1Pad0>{});
    // 3. Default
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f16_instances<2,
                                                       GNHWK,
                                                       GKYXC,
                                                       Empty_Tuple,
                                                       GNHWC,
                                                       ConvBwdDataDefault>{});
    // 4. Filter1x1Stride1Pad0
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f16_instances<2,
                                                       GNHWK,
                                                       GKYXC,
                                                       Empty_Tuple,
                                                       GNHWC,
                                                       ConvBwdDataFilter1x1Stride1Pad0>{});
    // 5. Default — noshuffle epilogue
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f16_noshuffle_instances<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  ConvBwdDataDefault>{});
    // 6. Filter1x1Stride1Pad0 — noshuffle epilogue
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f16_noshuffle_instances<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  ConvBwdDataFilter1x1Stride1Pad0>{});
    // 7. Default — nongrouped_match instances
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f16_nongrouped_match_instances<2,
                                                                         GNHWK,
                                                                         GKYXC,
                                                                         Empty_Tuple,
                                                                         GNHWC,
                                                                         ConvBwdDataDefault>{});
    // 8. Filter1x1Stride1Pad0 — nongrouped_match instances
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f16_nongrouped_match_instances<2,
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
