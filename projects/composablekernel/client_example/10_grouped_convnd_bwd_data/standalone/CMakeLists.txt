# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Build the MiOpenDriver kernel example alone (no full CK/client_example build).
# Only needs: ROCm (hipcc) and path to CK include (composablekernel root).
#
# From this directory (standalone/):
#   mkdir build && cd build
#   cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
#   make
# CK_SOURCE_DIR defaults to composablekernel root (../../.. from standalone/).
# CK_BUILD_DIR defaults to CK_SOURCE_DIR/build (must have run main CK cmake once for config.h).
# Override if needed: cmake -DCK_SOURCE_DIR=... -DCK_BUILD_DIR=... ..
# ROCM_PATH defaults to /opt/rocm (set ROCM_PATH if ROCm is elsewhere).

cmake_minimum_required(VERSION 3.16)
project(miopen_driver_kernel_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT CK_SOURCE_DIR)
  get_filename_component(CK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
endif()
if(NOT EXISTS "${CK_SOURCE_DIR}/include/ck/ck.hpp")
  message(FATAL_ERROR "CK_SOURCE_DIR must point to composablekernel root (containing include/ck/ck.hpp). Current: ${CK_SOURCE_DIR}")
endif()

if(NOT CK_BUILD_DIR)
  set(CK_BUILD_DIR "${CK_SOURCE_DIR}/build" CACHE PATH "CK build dir (contains include/ck/config.h)")
endif()
if(NOT EXISTS "${CK_BUILD_DIR}/include/ck/config.h")
  message(FATAL_ERROR "CK_BUILD_DIR must point to composablekernel build dir (containing include/ck/config.h). Run main CK cmake once. Current: ${CK_BUILD_DIR}")
endif()

if(NOT ROCM_PATH)
  set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "ROCm installation path")
  if(NOT ROCM_PATH)
    set(ROCM_PATH "/opt/rocm")
  endif()
endif()

add_executable(client_grouped_conv2d_bwd_data_miopen_driver_kernel
  ${CMAKE_CURRENT_SOURCE_DIR}/../grouped_conv2d_bwd_data_miopen_driver_kernel.cpp
)
# Use config.h from main CK build (CK_BUILD_DIR/include/ck/config.h)
target_include_directories(client_grouped_conv2d_bwd_data_miopen_driver_kernel PRIVATE
  ${CK_BUILD_DIR}/include
  ${CK_SOURCE_DIR}/include
  ${ROCM_PATH}/include
)
# hipcc links HIP runtime when building; ensure lib path is available
target_link_directories(client_grouped_conv2d_bwd_data_miopen_driver_kernel PRIVATE ${ROCM_PATH}/lib)
target_link_libraries(client_grouped_conv2d_bwd_data_miopen_driver_kernel PRIVATE amdhip64)
