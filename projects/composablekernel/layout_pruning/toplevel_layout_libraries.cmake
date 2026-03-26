# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# =============================================================================
# Layout-Specific Convolution Libraries (Static Libraries)
# =============================================================================
# This section defines layout-specific convolution library targets that combine
# all operations for a given layout (e.g., all NHWGC operations).
#
# These are STATIC libraries (.a files) that MIOpen and other consumers can link against.
# This enables selective linking - e.g., MIOpen can link only the layouts it needs
# (NHWGC, NDHWGC, OLD) instead of all layouts.
# =============================================================================

# -----------------------------------------------------------------------------
# 1D Convolution Layout Libraries
# -----------------------------------------------------------------------------

# GNWC Layout (1D)
if(TARGET device_grouped_conv1d_fwd_gnwc_instance OR
   TARGET device_grouped_conv1d_bwd_weight_gnwc_instance)
    set(CONV1D_GNWC_OBJECTS)
    if(TARGET device_grouped_conv1d_fwd_gnwc_instance)
        list(APPEND CONV1D_GNWC_OBJECTS $<TARGET_OBJECTS:device_grouped_conv1d_fwd_gnwc_instance>)
    endif()
    if(TARGET device_grouped_conv1d_bwd_weight_gnwc_instance)
        list(APPEND CONV1D_GNWC_OBJECTS $<TARGET_OBJECTS:device_grouped_conv1d_bwd_weight_gnwc_instance>)
    endif()
    add_library(device_conv1d_gnwc_operations ${CONV1D_GNWC_OBJECTS})
    set_target_properties(device_conv1d_gnwc_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(composablekernels::device_conv1d_gnwc_operations ALIAS device_conv1d_gnwc_operations)
    message(STATUS "Created device_conv1d_gnwc_operations")
endif()

# NWGC Layout (1D)
if(TARGET device_grouped_conv1d_bwd_weight_nwgc_instance)
    add_library(device_conv1d_nwgc_operations
        $<TARGET_OBJECTS:device_grouped_conv1d_bwd_weight_nwgc_instance>
    )
    set_target_properties(device_conv1d_nwgc_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(composablekernels::device_conv1d_nwgc_operations ALIAS device_conv1d_nwgc_operations)
    message(STATUS "Created device_conv1d_nwgc_operations")
endif()

# -----------------------------------------------------------------------------
# 2D Convolution Layout Libraries
# -----------------------------------------------------------------------------

# GNHWC Layout (2D)
add_library(device_conv2d_gnhwc_operations
    $<TARGET_OBJECTS:device_grouped_conv2d_fwd_gnhwc_instance>
    $<TARGET_OBJECTS:device_grouped_conv2d_bwd_data_gnhwc_instance>
    $<TARGET_OBJECTS:device_grouped_conv2d_bwd_weight_gnhwc_instance>
)
set_target_properties(device_conv2d_gnhwc_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(composablekernels::device_conv2d_gnhwc_operations ALIAS device_conv2d_gnhwc_operations)
message(STATUS "Created device_conv2d_gnhwc_operations")

# NHWGC Layout (2D) - ⭐ Required by MIOpen
add_library(device_conv2d_nhwgc_operations
    $<TARGET_OBJECTS:device_grouped_conv2d_fwd_nhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv2d_bwd_data_nhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv2d_bwd_weight_nhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv2d_fwd_clamp_nhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv2d_fwd_bias_clamp_nhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv2d_fwd_bias_bnorm_clamp_nhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv2d_fwd_dynamic_op_nhwgc_instance>
)
set_target_properties(device_conv2d_nhwgc_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(composablekernels::device_conv2d_nhwgc_operations ALIAS device_conv2d_nhwgc_operations)
message(STATUS "Created device_conv2d_nhwgc_operations (MIOpen required)")

# NGCHW Layout (2D)
add_library(device_conv2d_ngchw_operations
    $<TARGET_OBJECTS:device_grouped_conv2d_fwd_ngchw_instance>
    $<TARGET_OBJECTS:device_grouped_conv2d_bwd_data_ngchw_instance>
    $<TARGET_OBJECTS:device_grouped_conv2d_bwd_weight_ngchw_instance>
)
set_target_properties(device_conv2d_ngchw_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(composablekernels::device_conv2d_ngchw_operations ALIAS device_conv2d_ngchw_operations)
message(STATUS "Created device_conv2d_ngchw_operations")

# -----------------------------------------------------------------------------
# 3D Convolution Layout Libraries
# -----------------------------------------------------------------------------

# GNDHWC Layout (3D)
add_library(device_conv3d_gndhwc_operations
    $<TARGET_OBJECTS:device_grouped_conv3d_fwd_gndhwc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_bwd_data_gndhwc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_bwd_weight_gndhwc_instance>
)
set_target_properties(device_conv3d_gndhwc_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(composablekernels::device_conv3d_gndhwc_operations ALIAS device_conv3d_gndhwc_operations)
message(STATUS "Created device_conv3d_gndhwc_operations")

# NDHWGC Layout (3D) - ⭐ Required by MIOpen
add_library(device_conv3d_ndhwgc_operations
    $<TARGET_OBJECTS:device_grouped_conv3d_fwd_ndhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_bwd_data_ndhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_bwd_weight_ndhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_fwd_scale_ndhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_fwd_clamp_ndhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_fwd_bias_clamp_ndhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_fwd_bias_bnorm_clamp_ndhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_fwd_bilinear_ndhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_fwd_convscale_ndhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_fwd_convscale_add_ndhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_fwd_convscale_relu_ndhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_fwd_convinvscale_ndhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_fwd_dynamic_op_ndhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_fwd_scaleadd_ab_ndhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_fwd_scaleadd_scaleadd_relu_ndhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_bwd_data_bilinear_ndhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_bwd_data_scale_ndhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_bwd_weight_bilinear_ndhwgc_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_bwd_weight_scale_ndhwgc_instance>
)
set_target_properties(device_conv3d_ndhwgc_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(composablekernels::device_conv3d_ndhwgc_operations ALIAS device_conv3d_ndhwgc_operations)
message(STATUS "Created device_conv3d_ndhwgc_operations (MIOpen required)")

# NGCDHW Layout (3D)
add_library(device_conv3d_ngcdhw_operations
    $<TARGET_OBJECTS:device_grouped_conv3d_fwd_ngcdhw_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_bwd_data_ngcdhw_instance>
    $<TARGET_OBJECTS:device_grouped_conv3d_bwd_weight_ngcdhw_instance>
)
set_target_properties(device_conv3d_ngcdhw_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(composablekernels::device_conv3d_ngcdhw_operations ALIAS device_conv3d_ngcdhw_operations)
message(STATUS "Created device_conv3d_ngcdhw_operations")

# NHWGC Layout (3D) - Only for bias_bnorm_clamp variant
if(TARGET device_grouped_conv3d_fwd_bias_bnorm_clamp_nhwgc_instance)
    add_library(device_conv3d_nhwgc_operations
        $<TARGET_OBJECTS:device_grouped_conv3d_fwd_bias_bnorm_clamp_nhwgc_instance>
    )
    set_target_properties(device_conv3d_nhwgc_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(composablekernels::device_conv3d_nhwgc_operations ALIAS device_conv3d_nhwgc_operations)
    message(STATUS "Created device_conv3d_nhwgc_operations")
endif()

# -----------------------------------------------------------------------------
# Special Libraries (not split by layout) - Already defined above as OBJECT
# -----------------------------------------------------------------------------
# device_conv_old_operations - created via add_instance_library() above
# device_convnd_generic_operations - created via add_instance_library() above
# device_quantization_operations - created via add_instance_library() above

# -----------------------------------------------------------------------------
# Umbrella Library (Backward Compatibility) - Links all layout libraries
# -----------------------------------------------------------------------------

add_library(device_conv_operations
    $<TARGET_OBJECTS:device_conv2d_gnhwc_operations>
    $<TARGET_OBJECTS:device_conv2d_nhwgc_operations>
    $<TARGET_OBJECTS:device_conv2d_ngchw_operations>
    $<TARGET_OBJECTS:device_conv3d_gndhwc_operations>
    $<TARGET_OBJECTS:device_conv3d_ndhwgc_operations>
    $<TARGET_OBJECTS:device_conv3d_ngcdhw_operations>
    $<TARGET_OBJECTS:device_conv_old_operations>
    $<TARGET_OBJECTS:device_convnd_generic_operations>
    $<TARGET_OBJECTS:device_quantization_operations>
)

# Add 1D and 3D NHWGC if they exist
if(TARGET device_conv1d_gnwc_operations)
    target_sources(device_conv_operations PRIVATE $<TARGET_OBJECTS:device_conv1d_gnwc_operations>)
endif()
if(TARGET device_conv1d_nwgc_operations)
    target_sources(device_conv_operations PRIVATE $<TARGET_OBJECTS:device_conv1d_nwgc_operations>)
endif()
if(TARGET device_conv3d_nhwgc_operations)
    target_sources(device_conv_operations PRIVATE $<TARGET_OBJECTS:device_conv3d_nhwgc_operations>)
endif()

set_target_properties(device_conv_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(composablekernels::device_conv_operations ALIAS device_conv_operations)
message(STATUS "Created umbrella device_conv_operations (includes all layouts)")

# =============================================================================
# End of Layout-Specific Libraries
# =============================================================================
