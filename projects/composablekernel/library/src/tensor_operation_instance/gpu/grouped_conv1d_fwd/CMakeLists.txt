# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# All 1D convolution layouts are not needed by MIOpen
set(GROUPED_CONV1D_FWD)

if(NOT MIOPEN_REQ_LIBS_ONLY)
    list(APPEND GROUPED_CONV1D_FWD
        xdl/device_grouped_conv1d_fwd_xdl_gnwc_gkxc_gnwk_bf16_instance.cpp
        xdl/device_grouped_conv1d_fwd_xdl_gnwc_gkxc_gnwk_f16_instance.cpp
        xdl/device_grouped_conv1d_fwd_xdl_gnwc_gkxc_gnwk_f32_instance.cpp
        xdl/device_grouped_conv1d_fwd_xdl_gnwc_gkxc_gnwk_int8_instance.cpp
    )
endif()

add_instance_library(device_grouped_conv1d_fwd_instance ${GROUPED_CONV1D_FWD})
