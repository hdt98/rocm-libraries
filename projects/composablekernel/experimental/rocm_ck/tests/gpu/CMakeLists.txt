# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# GPU integration tests: bare CK Tile vs rocm_ck abstraction.
# Requires HIP compiler and a GPU at runtime.
#
# Standalone build:
#   cmake -B build -S . -G Ninja \
#       -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
#       -DCMAKE_PREFIX_PATH=/opt/rocm \
#       -DGPU_TARGETS="gfx90a"
#   ninja -C build -j4
#   ctest --test-dir build --output-on-failure

cmake_minimum_required(VERSION 3.21)
project(rocm_ck_gpu_tests LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

find_package(hip REQUIRED)

# =============================================================================
# Google Test
# =============================================================================

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/gtest.cmake)

# =============================================================================
# CK Tile headers (device pass needs these)
# =============================================================================

set(CK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../include")
get_filename_component(CK_INCLUDE_DIR "${CK_INCLUDE_DIR}" ABSOLUTE)

# =============================================================================
# rocm_ck headers (INTERFACE library)
# =============================================================================

if(NOT TARGET rocm_ck)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../include rocm_ck_build)
endif()

# =============================================================================
# GPU integration tests
# =============================================================================

set(ROCM_CK_GPU_TESTS
    test_vectoradd.hip
    test_gemm.hip
    test_gemm_epilogue.hip
)

add_custom_target(rocm_ck_gpu_tests)

foreach(test_source ${ROCM_CK_GPU_TESTS})
    get_filename_component(test_name ${test_source} NAME_WE)

    add_executable(${test_name} ${test_source})
    target_compile_features(${test_name} PRIVATE cxx_std_20)
    target_include_directories(${test_name} PRIVATE
        ${CK_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(${test_name} PRIVATE rocm_ck hip::host GTest::gtest_main)

    # CK Tile requires aggressive optimization and inlining for device code.
    # Without these, template-heavy kernels are 100-400x slower.
    target_compile_options(${test_name} PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:
            -O3
            -DNDEBUG
            -fno-unwind-tables
            -mllvm=-amdgpu-function-calls=false
            -mllvm=-amdgpu-early-inline-all=true>
    )

    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES LABELS "rocm_ck_gpu")

    add_dependencies(rocm_ck_gpu_tests ${test_name})
endforeach()
