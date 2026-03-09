# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Build the MiOpenDriver kernel example alone (no full CK/client_example build).
# Only needs: ROCm (hipcc) and path to CK include (composablekernel root).
#
# From this directory (standalone/):
#   mkdir build && cd build
#   cmake ..
#   make
# Debug build: cmake -DCMAKE_BUILD_TYPE=Debug ..
# CMAKE_CXX_COMPILER defaults to ROCm hipcc; override with -DCMAKE_CXX_COMPILER=...
# CK_SOURCE_DIR defaults to composablekernel root (../../.. from standalone/).
# CK_BUILD_DIR defaults to CK_SOURCE_DIR/build (must have run main CK cmake once for config.h).
# Override if needed: cmake -DCK_SOURCE_DIR=... -DCK_BUILD_DIR=... ..
# ROCM_PATH defaults to /opt/rocm (set ROCM_PATH if ROCm is elsewhere).

cmake_minimum_required(VERSION 3.16)

# Default C++ compiler to ROCm hipcc (override with -DCMAKE_CXX_COMPILER=...)
set(_ROCM_DEFAULT "$ENV{ROCM_PATH}")
if(NOT _ROCM_DEFAULT)
  set(_ROCM_DEFAULT "/opt/rocm")
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER OR CMAKE_CXX_COMPILER STREQUAL "")
  set(CMAKE_CXX_COMPILER "${_ROCM_DEFAULT}/bin/hipcc" CACHE STRING "C++ compiler")
endif()

project(miopen_driver_kernel_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Default to Release if not set (single-config generators)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# Debug: use -Og -g for debuggability without -O0 template blow-up (match main CK)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS_DEBUG "-Og -g" CACHE STRING "Debug compiler flags" FORCE)
  set(CMAKE_HIP_FLAGS_DEBUG "-Og -g" CACHE STRING "Debug HIP flags" FORCE)
endif()

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

# Option: build ck_tile standalone with COMPUTE_ASYNC pipeline (amd_async_buffer_load, gfx950)
option(USE_ASYNC_GROUPED_CONV_BWD_DATA "Use COMPUTE_ASYNC pipeline in grouped conv bwd data standalone" ON)

# Second executable: ck_tile grouped conv bwd data (same problem, for perf comparison)
add_executable(client_grouped_conv2d_bwd_data_ck_tile_standalone
  ${CMAKE_CURRENT_SOURCE_DIR}/../grouped_conv2d_bwd_data_ck_tile_standalone.cpp
)
target_include_directories(client_grouped_conv2d_bwd_data_ck_tile_standalone PRIVATE
  ${CK_BUILD_DIR}/include
  ${CK_SOURCE_DIR}/include
  ${CK_SOURCE_DIR}/example/ck_tile/20_grouped_convolution
  ${ROCM_PATH}/include
)
if(USE_ASYNC_GROUPED_CONV_BWD_DATA)
  target_compile_definitions(client_grouped_conv2d_bwd_data_ck_tile_standalone PRIVATE USE_ASYNC_GROUPED_CONV_BWD_DATA=1)
else()
  target_compile_definitions(client_grouped_conv2d_bwd_data_ck_tile_standalone PRIVATE USE_ASYNC_GROUPED_CONV_BWD_DATA=0)
endif()
target_link_directories(client_grouped_conv2d_bwd_data_ck_tile_standalone PRIVATE ${ROCM_PATH}/lib)
target_link_libraries(client_grouped_conv2d_bwd_data_ck_tile_standalone PRIVATE amdhip64)
