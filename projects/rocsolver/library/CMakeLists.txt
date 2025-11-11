# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# This is incremented when the ABI to the library changes
set(rocsolver_SOVERSION 0.7)

# Create version header from templated .in file using CMake info
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/rocsolver/rocsolver-version.h.in"
    "${PROJECT_BINARY_DIR}/include/rocsolver/rocsolver-version.h"
)

add_subdirectory(src)
