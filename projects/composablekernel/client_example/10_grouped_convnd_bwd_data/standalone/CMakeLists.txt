# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Build two standalone MiOpenDriver kernel examples (V1 and V3 DirectLoad), no full CK/client_example build.
# Only needs: ROCm (hipcc) and path to CK include (composablekernel root).
# Targets: client_grouped_conv2d_bwd_data_miopen_driver_kernel_v1, client_grouped_conv2d_bwd_data_miopen_driver_kernel_v3.
#
# From this directory (standalone/):
#   mkdir build && cd build
#   cmake ..
#   make
# CMAKE_CXX_COMPILER defaults to /opt/rocm/bin/hipcc (override with -DCMAKE_CXX_COMPILER=...).
# CK_SOURCE_DIR defaults to composablekernel root (../../.. from standalone/).
# CK_BUILD_DIR defaults to CK_SOURCE_DIR/build (must have run main CK cmake once for config.h).
# Override if needed: cmake -DCK_SOURCE_DIR=... -DCK_BUILD_DIR=... ..
# ROCM_PATH defaults to /opt/rocm (set ROCM_PATH if ROCm is elsewhere).

cmake_minimum_required(VERSION 3.16)

# Default to hipcc if no compiler was specified (must be before project()).
if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc" CACHE STRING "C++ compiler" FORCE)
endif()

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

# Standalone V1: DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1
add_executable(client_grouped_conv2d_bwd_data_miopen_driver_kernel_v1
  ${CMAKE_CURRENT_SOURCE_DIR}/grouped_conv2d_bwd_data_miopen_driver_kernel_v1.cpp
)
target_include_directories(client_grouped_conv2d_bwd_data_miopen_driver_kernel_v1 PRIVATE
  ${CK_BUILD_DIR}/include
  ${CK_SOURCE_DIR}/include
  ${ROCM_PATH}/include
)
target_compile_options(client_grouped_conv2d_bwd_data_miopen_driver_kernel_v1 PRIVATE -save-temps)
target_link_directories(client_grouped_conv2d_bwd_data_miopen_driver_kernel_v1 PRIVATE ${ROCM_PATH}/lib)
target_link_libraries(client_grouped_conv2d_bwd_data_miopen_driver_kernel_v1 PRIVATE amdhip64)

# Standalone V3: DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffleV3 with DirectLoad=true
add_executable(client_grouped_conv2d_bwd_data_miopen_driver_kernel_v3
  ${CMAKE_CURRENT_SOURCE_DIR}/grouped_conv2d_bwd_data_miopen_driver_kernel_v3.cpp
)
target_include_directories(client_grouped_conv2d_bwd_data_miopen_driver_kernel_v3 PRIVATE
  ${CK_BUILD_DIR}/include
  ${CK_SOURCE_DIR}/include
  ${ROCM_PATH}/include
)
target_compile_options(client_grouped_conv2d_bwd_data_miopen_driver_kernel_v3 PRIVATE -save-temps)
target_link_directories(client_grouped_conv2d_bwd_data_miopen_driver_kernel_v3 PRIVATE ${ROCM_PATH}/lib)
target_link_libraries(client_grouped_conv2d_bwd_data_miopen_driver_kernel_v3 PRIVATE amdhip64)