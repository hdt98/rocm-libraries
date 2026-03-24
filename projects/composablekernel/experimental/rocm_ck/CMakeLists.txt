# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# rocm_ck — CK Tile kernel APIs with multiarch distribution.
#
# Each example has its own CMakeLists.txt and can be built independently:
#
#   cd examples/01_hello_world
#   cmake -B build -S . -G Ninja \
#       -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
#       -DCMAKE_PREFIX_PATH=/opt/rocm \
#       -DGPU_TARGETS="gfx90a;gfx942"
#   ninja -C build

cmake_minimum_required(VERSION 3.21)
project(rocm_ck)

add_subdirectory(rocm_kpack)
add_subdirectory(include)
add_subdirectory(examples/01_hello_world)
add_subdirectory(examples/02_ck_tile_vector_add)
add_subdirectory(examples/03_rocm_ck_vector_add)
add_subdirectory(examples/04_gemm)
add_subdirectory(examples/06_rocm_ck_fmha_bwd)
