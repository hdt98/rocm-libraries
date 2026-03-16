# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Kpack experiment — see examples/hello_world/ for a standalone build.
#
# The hello_world example has its own CMakeLists.txt and can be built
# independently. Build it directly:
#
#   cd examples/hello_world
#   cmake -B build -S . -G Ninja \
#       -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
#       -DCMAKE_PREFIX_PATH=/opt/rocm \
#       -DGPU_TARGETS="gfx90a;gfx942"
#   ninja -C build

cmake_minimum_required(VERSION 3.21)
project(kpack_experiment)

add_subdirectory(examples/hello_world)
