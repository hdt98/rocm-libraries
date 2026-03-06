# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_library(
    exprtk
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/exprtk.hpp
)

add_library(exprtk::exprtk ALIAS exprtk)

target_include_directories(
    exprtk
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
