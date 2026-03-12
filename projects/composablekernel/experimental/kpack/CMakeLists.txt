# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.21)
project(kpack_hello_world LANGUAGES CXX)

enable_language(HIP)
find_package(hip REQUIRED)

set(GPU_TARGETS "gfx90a;gfx942;gfx1101" CACHE STRING "GPU architectures to build .hsaco for")
set(KPACK_ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation (for --rocm-path)")

# Build vendored kpack runtime
add_subdirectory(rocm_kpack)

# Compile .hsaco device code objects per architecture using clang++
# Uses --cuda-device-only to produce raw .hsaco code objects
set(HSACO_FILES "")
set(HSACO_ARGS "")
foreach(arch IN LISTS GPU_TARGETS)
    set(hsaco_output "${CMAKE_BINARY_DIR}/vector_add_${arch}.hsaco")
    add_custom_command(
        OUTPUT ${hsaco_output}
        COMMAND ${CMAKE_HIP_COMPILER}
                --offload-arch=${arch}
                --cuda-device-only
                -x hip
                --rocm-path=${KPACK_ROCM_PATH}
                -o ${hsaco_output}
                ${CMAKE_CURRENT_SOURCE_DIR}/vector_add_kernel.hip
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vector_add_kernel.hip
        COMMENT "Compiling vector_add_kernel.hip for ${arch}"
    )
    list(APPEND HSACO_FILES ${hsaco_output})
    list(APPEND HSACO_ARGS --hsaco ${hsaco_output}:${arch})
endforeach()

# Find Python for archive creation
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Create .kpack archive from .hsaco files
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/kernels.kpack
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/create_kpack_archive.py
            --output ${CMAKE_BINARY_DIR}/kernels.kpack
            --binary-name vector_add_kernel
            ${HSACO_ARGS}
    DEPENDS ${HSACO_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/create_kpack_archive.py
    COMMENT "Creating kernels.kpack archive"
)
add_custom_target(kpack_archive ALL DEPENDS ${CMAKE_BINARY_DIR}/kernels.kpack)

# Build demo loader (host-only -- loads device code at runtime via hipModule API)
add_executable(kpack_hello_world kpack_hello_world.cpp)
target_link_libraries(kpack_hello_world PRIVATE rocm_kpack hip::host)
add_dependencies(kpack_hello_world kpack_archive)
