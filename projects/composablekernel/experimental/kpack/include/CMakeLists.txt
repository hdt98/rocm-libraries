# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# INTERFACE library for rocm_ck shared headers.
# Consumers write: #include <rocm_ck/hip_check.hpp>

add_library(rocm_ck INTERFACE)
target_include_directories(rocm_ck INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
