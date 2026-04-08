# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Minimal GEMM example without kpack. Single HIP compilation — the kernel is
# compiled into the binary and launched with <<<>>> syntax.
#
# Usage:
#   cmake -B build -S . -G Ninja \
#       -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
#       -DCMAKE_PREFIX_PATH=/opt/rocm \
#       -DGPU_TARGETS="gfx90a"
#   ninja -C build -j16
#   build/gemm_no_kpack

cmake_minimum_required(VERSION 3.21)
project(gemm_no_kpack LANGUAGES CXX HIP)

find_package(hip REQUIRED)

# CK Tile headers (device pass needs these for gemm_dev.hpp)
set(CK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../include")
get_filename_component(CK_INCLUDE_DIR "${CK_INCLUDE_DIR}" ABSOLUTE)

# rocm_ck header-only target
if(NOT TARGET rocm_ck)
    add_subdirectory(../../include rocm_ck_build)
endif()

add_executable(gemm_no_kpack gemm_no_kpack.hip)
target_compile_features(gemm_no_kpack PRIVATE cxx_std_20)
target_include_directories(gemm_no_kpack PRIVATE ${CK_INCLUDE_DIR})
target_link_libraries(gemm_no_kpack PRIVATE rocm_ck hip::host)
