# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Thin INTERFACE wrapper around `roc::mxDataGenerator`.
#
# This file is added by the project root via `add_subdirectory(clients/common)`
# only when:
#   * HIPBLASLT_ENABLE_MXDATAGENERATOR is ON, and
#   * at least one consumer subtree is enabled (HIPBLASLT_ENABLE_CLIENT,
#     TENSILELITE_ENABLE_CLIENT, or TENSILELITE_BUILD_TESTING).
if(NOT ROCM_LIBS_SUPERBUILD)
    if(HIPBLASLT_ENABLE_THEROCK)
        find_package(mxDataGenerator REQUIRED)
    else()
        # `${PROJECT_SOURCE_DIR}/../../shared/mxdatagenerator` resolves to the
        # sibling `shared/mxdatagenerator` subtree of the project. The binary
        # dir is anchored under the hipblaslt project's build tree to match
        # the upstream subdirectory's existing layout assumptions.
        add_subdirectory(
            "${PROJECT_SOURCE_DIR}/../../shared/mxdatagenerator"
            "${PROJECT_BINARY_DIR}/mxdatagenerator"
        )
    endif()
endif()

add_library(hipblaslt-mxdatagen INTERFACE)
add_library(hipblaslt::mxdatagen ALIAS hipblaslt-mxdatagen)

target_link_libraries(hipblaslt-mxdatagen
    INTERFACE
        roc::mxDataGenerator
)

target_compile_features(hipblaslt-mxdatagen INTERFACE cxx_std_20)
target_compile_definitions(hipblaslt-mxdatagen INTERFACE HIPBLASLT_ENABLE_MXDATAGENERATOR)
