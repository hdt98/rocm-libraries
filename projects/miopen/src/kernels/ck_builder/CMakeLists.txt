# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

###############################################################################
# CK Builder Kernels
# This subproject uses CK Builder to provide kernels to MIOpen
###############################################################################

find_package(composable_kernel 1.0.0 COMPONENTS device_conv_operations)

add_library(ck_builder_kernels STATIC 
    generate_kernels.cpp
)

add_definitions(-DCK_EXPERIMENTAL_BUILDER)

target_link_libraries(ck_builder_kernels composable_kernel::device_conv_operations)

target_include_directories(ck_builder_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/includes)