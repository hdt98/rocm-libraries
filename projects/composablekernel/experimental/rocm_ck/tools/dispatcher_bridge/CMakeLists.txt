# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Dispatcher Bridge PoC — load kpack kernels into the CK dispatcher registry.
#
# Usage:
#   cmake -B build -S . -G Ninja \
#       -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
#       -DCMAKE_PREFIX_PATH=/opt/rocm
#   ninja -C build -j16
#
# Run:
#   build/dispatcher_bridge ../../examples/04_gemm/build/gemm.kpack

cmake_minimum_required(VERSION 3.21)
project(dispatcher_bridge LANGUAGES CXX)

enable_language(HIP)
find_package(hip REQUIRED)

# CK Tile headers (four levels up from this tool directory)
set(CK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../include")
get_filename_component(CK_INCLUDE_DIR "${CK_INCLUDE_DIR}" ABSOLUTE)

# Dispatcher (compile its two source files directly — avoids separate build step)
set(DISPATCHER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../dispatcher")
get_filename_component(DISPATCHER_DIR "${DISPATCHER_DIR}" ABSOLUTE)

add_library(ck_tile_dispatcher STATIC
    ${DISPATCHER_DIR}/src/registry.cpp
    ${DISPATCHER_DIR}/src/dispatcher.cpp)
target_compile_features(ck_tile_dispatcher PRIVATE cxx_std_20)
target_include_directories(ck_tile_dispatcher PUBLIC
    ${DISPATCHER_DIR}/include
    ${CK_INCLUDE_DIR})
target_link_libraries(ck_tile_dispatcher PUBLIC hip::host)

# rocm_ck headers and rocm_kpack runtime
if(NOT TARGET rocm_kpack)
    add_subdirectory(../../rocm_kpack rocm_kpack_build)
endif()
if(NOT TARGET rocm_ck)
    add_subdirectory(../../include rocm_ck_build)
endif()

# Bridge demo executable
add_executable(dispatcher_bridge main.cpp)
target_compile_features(dispatcher_bridge PRIVATE cxx_std_20)
target_include_directories(dispatcher_bridge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dispatcher_bridge PRIVATE
    rocm_ck rocm_kpack ck_tile_dispatcher hip::host)
