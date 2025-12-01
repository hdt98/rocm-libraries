# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Configure version header
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/hipblas-version.h.in"
    "${PROJECT_BINARY_DIR}/include/hipblas/hipblas-version.h"
    @ONLY
)

# Copy public header
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/hipblas.h"
    "${PROJECT_BINARY_DIR}/include/hipblas/hipblas.h"
    COPYONLY
)

# Add headers to target
target_sources(hipblas
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hipblas.h>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/hipblas/hipblas-version.h>
)
