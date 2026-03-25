# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# =============================================================================
# Layout-Specific Convolution Libraries
# =============================================================================
# This section defines layout-specific convolution library targets that combine
# all operations for a given layout (e.g., all NHWGC operations).
#
# These libraries enable selective linking - e.g., MIOpen can link only the
# layouts it needs (NHWGC, NDHWGC, OLD) instead of all layouts.
# =============================================================================

# -----------------------------------------------------------------------------
# 1D Convolution Layout Libraries
# -----------------------------------------------------------------------------

# GNWC Layout (1D)
if(TARGET device_grouped_conv1d_fwd_gnwc_instance OR
   TARGET device_grouped_conv1d_bwd_weight_gnwc_instance)
    add_library(device_conv1d_gnwc_operations INTERFACE)
    if(TARGET device_grouped_conv1d_fwd_gnwc_instance)
        target_link_libraries(device_conv1d_gnwc_operations INTERFACE
            device_grouped_conv1d_fwd_gnwc_instance
        )
    endif()
    if(TARGET device_grouped_conv1d_bwd_weight_gnwc_instance)
        target_link_libraries(device_conv1d_gnwc_operations INTERFACE
            device_grouped_conv1d_bwd_weight_gnwc_instance
        )
    endif()
    message(STATUS "Created device_conv1d_gnwc_operations")
endif()

# NWGC Layout (1D)
if(TARGET device_grouped_conv1d_bwd_weight_nwgc_instance)
    add_library(device_conv1d_nwgc_operations INTERFACE)
    target_link_libraries(device_conv1d_nwgc_operations INTERFACE
        device_grouped_conv1d_bwd_weight_nwgc_instance
    )
    message(STATUS "Created device_conv1d_nwgc_operations")
endif()

# -----------------------------------------------------------------------------
# 2D Convolution Layout Libraries
# -----------------------------------------------------------------------------

# GNHWC Layout (2D)
add_library(device_conv2d_gnhwc_operations INTERFACE)
target_link_libraries(device_conv2d_gnhwc_operations INTERFACE
    device_grouped_conv2d_fwd_gnhwc_instance
    device_grouped_conv2d_bwd_data_gnhwc_instance
    device_grouped_conv2d_bwd_weight_gnhwc_instance
)
message(STATUS "Created device_conv2d_gnhwc_operations")

# NHWGC Layout (2D) - ⭐ Required by MIOpen
add_library(device_conv2d_nhwgc_operations INTERFACE)
target_link_libraries(device_conv2d_nhwgc_operations INTERFACE
    device_grouped_conv2d_fwd_nhwgc_instance
    device_grouped_conv2d_bwd_data_nhwgc_instance
    device_grouped_conv2d_bwd_weight_nhwgc_instance
    device_grouped_conv2d_fwd_clamp_nhwgc_instance
    device_grouped_conv2d_fwd_bias_clamp_nhwgc_instance
    device_grouped_conv2d_fwd_bias_bnorm_clamp_nhwgc_instance
    device_grouped_conv2d_fwd_dynamic_op_nhwgc_instance
)
message(STATUS "Created device_conv2d_nhwgc_operations (MIOpen required)")

# NGCHW Layout (2D)
add_library(device_conv2d_ngchw_operations INTERFACE)
target_link_libraries(device_conv2d_ngchw_operations INTERFACE
    device_grouped_conv2d_fwd_ngchw_instance
    device_grouped_conv2d_bwd_data_ngchw_instance
    device_grouped_conv2d_bwd_weight_ngchw_instance
)
message(STATUS "Created device_conv2d_ngchw_operations")

# -----------------------------------------------------------------------------
# 3D Convolution Layout Libraries
# -----------------------------------------------------------------------------

# GNDHWC Layout (3D)
add_library(device_conv3d_gndhwc_operations INTERFACE)
target_link_libraries(device_conv3d_gndhwc_operations INTERFACE
    device_grouped_conv3d_fwd_gndhwc_instance
    device_grouped_conv3d_bwd_data_gndhwc_instance
    device_grouped_conv3d_bwd_weight_gndhwc_instance
)
message(STATUS "Created device_conv3d_gndhwc_operations")

# NDHWGC Layout (3D) - ⭐ Required by MIOpen
add_library(device_conv3d_ndhwgc_operations INTERFACE)
target_link_libraries(device_conv3d_ndhwgc_operations INTERFACE
    device_grouped_conv3d_fwd_ndhwgc_instance
    device_grouped_conv3d_bwd_data_ndhwgc_instance
    device_grouped_conv3d_bwd_weight_ndhwgc_instance
    device_grouped_conv3d_fwd_scale_ndhwgc_instance
    device_grouped_conv3d_fwd_clamp_ndhwgc_instance
    device_grouped_conv3d_fwd_bias_clamp_ndhwgc_instance
    device_grouped_conv3d_fwd_bias_bnorm_clamp_ndhwgc_instance
    device_grouped_conv3d_fwd_bilinear_ndhwgc_instance
    device_grouped_conv3d_fwd_convscale_ndhwgc_instance
    device_grouped_conv3d_fwd_convscale_add_ndhwgc_instance
    device_grouped_conv3d_fwd_convscale_relu_ndhwgc_instance
    device_grouped_conv3d_fwd_convinvscale_ndhwgc_instance
    device_grouped_conv3d_fwd_dynamic_op_ndhwgc_instance
    device_grouped_conv3d_fwd_scaleadd_ab_ndhwgc_instance
    device_grouped_conv3d_fwd_scaleadd_scaleadd_relu_ndhwgc_instance
    device_grouped_conv3d_bwd_data_bilinear_ndhwgc_instance
    device_grouped_conv3d_bwd_data_scale_ndhwgc_instance
    device_grouped_conv3d_bwd_weight_bilinear_ndhwgc_instance
    device_grouped_conv3d_bwd_weight_scale_ndhwgc_instance
)
message(STATUS "Created device_conv3d_ndhwgc_operations (MIOpen required)")

# NGCDHW Layout (3D)
add_library(device_conv3d_ngcdhw_operations INTERFACE)
target_link_libraries(device_conv3d_ngcdhw_operations INTERFACE
    device_grouped_conv3d_fwd_ngcdhw_instance
    device_grouped_conv3d_bwd_data_ngcdhw_instance
    device_grouped_conv3d_bwd_weight_ngcdhw_instance
)
message(STATUS "Created device_conv3d_ngcdhw_operations")

# NHWGC Layout (3D) - Only for bias_bnorm_clamp variant
if(TARGET device_grouped_conv3d_fwd_bias_bnorm_clamp_nhwgc_instance)
    add_library(device_conv3d_nhwgc_operations INTERFACE)
    target_link_libraries(device_conv3d_nhwgc_operations INTERFACE
        device_grouped_conv3d_fwd_bias_bnorm_clamp_nhwgc_instance
    )
    message(STATUS "Created device_conv3d_nhwgc_operations")
endif()

# -----------------------------------------------------------------------------
# Special Libraries (not split by layout)
# -----------------------------------------------------------------------------

# Non-grouped convolutions (old/legacy) - ⭐ Required by MIOpen
# These remain in their original locations as they don't have layout variants
file(GLOB_RECURSE CONV_OLD_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/conv1d_bwd_data/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/conv2d_bwd_data/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/conv2d_fwd/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/conv2d_fwd_bias_relu/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/conv2d_fwd_bias_relu_add/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/conv3d_bwd_data/*.cpp"
)
if(CONV_OLD_SOURCES)
    add_instance_library(device_conv_old_operations ${CONV_OLD_SOURCES})
    message(STATUS "Created device_conv_old_operations (MIOpen required)")
endif()

# N-dimensional generic convolutions (layout-agnostic)
file(GLOB_RECURSE CONVND_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/grouped_convnd_bwd_weight/**/*.cpp"
)
if(CONVND_SOURCES)
    add_instance_library(device_convnd_generic_operations ${CONVND_SOURCES})
    message(STATUS "Created device_convnd_generic_operations")
endif()

# Quantization operations
file(GLOB_RECURSE QUANT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/quantization/**/*.cpp"
)
if(QUANT_SOURCES)
    add_instance_library(device_quantization_operations ${QUANT_SOURCES})
    message(STATUS "Created device_quantization_operations")
endif()

# -----------------------------------------------------------------------------
# Umbrella Library (Backward Compatibility)
# -----------------------------------------------------------------------------
# Combines all layout-specific libraries for projects that want everything

add_library(device_conv_operations INTERFACE)

# 1D layouts
if(TARGET device_conv1d_gnwc_operations)
    target_link_libraries(device_conv_operations INTERFACE device_conv1d_gnwc_operations)
endif()
if(TARGET device_conv1d_nwgc_operations)
    target_link_libraries(device_conv_operations INTERFACE device_conv1d_nwgc_operations)
endif()

# 2D layouts
target_link_libraries(device_conv_operations INTERFACE
    device_conv2d_gnhwc_operations
    device_conv2d_nhwgc_operations
    device_conv2d_ngchw_operations
)

# 3D layouts
target_link_libraries(device_conv_operations INTERFACE
    device_conv3d_gndhwc_operations
    device_conv3d_ndhwgc_operations
    device_conv3d_ngcdhw_operations
)
if(TARGET device_conv3d_nhwgc_operations)
    target_link_libraries(device_conv_operations INTERFACE device_conv3d_nhwgc_operations)
endif()

# Special libraries
if(TARGET device_conv_old_operations)
    target_link_libraries(device_conv_operations INTERFACE device_conv_old_operations)
endif()
if(TARGET device_convnd_generic_operations)
    target_link_libraries(device_conv_operations INTERFACE device_convnd_generic_operations)
endif()
if(TARGET device_quantization_operations)
    target_link_libraries(device_conv_operations INTERFACE device_quantization_operations)
endif()

add_library(composablekernels::device_conv_operations ALIAS device_conv_operations)
message(STATUS "Created umbrella device_conv_operations (includes all layouts)")

# =============================================================================
# End of Layout-Specific Libraries
# =============================================================================
