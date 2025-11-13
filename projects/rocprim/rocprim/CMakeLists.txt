# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Configure version header
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/rocprim/rocprim_version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/rocprim/rocprim_version.hpp"
    @ONLY
)

# Header-only library target
add_library(rocprim INTERFACE)
add_library(roc::rocprim ALIAS rocprim)

target_include_directories(rocprim
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/rocprim>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(rocprim INTERFACE cxx_std_17)

# Library target that includes HIP dependency
add_library(rocprim_hip INTERFACE)
add_library(roc::rocprim_hip ALIAS rocprim_hip)

target_link_libraries(rocprim_hip
    INTERFACE
        roc::rocprim
        hip::device
)

# Installation
if(ROCPRIM_ENABLE_INSTALL)
    rocm_install_targets(
        TARGETS rocprim rocprim_hip
    )

    rocm_install(
        DIRECTORY
            "${CMAKE_CURRENT_SOURCE_DIR}/include/"
            "${CMAKE_CURRENT_BINARY_DIR}/include/"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
    )

    rocm_export_targets(
        TARGETS roc::rocprim roc::rocprim_hip
        DEPENDS PACKAGE hip
        NAMESPACE roc::
    )
endif()
