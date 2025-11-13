# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

include(CheckLanguage)
include(CMakeDependentOption)

check_language(HIP)
cmake_dependent_option(USE_HIPCXX "Use CMake HIP language support" OFF CMAKE_HIP_COMPILER OFF)
if(USE_HIPCXX)
    set(PROJECT_LANG HIP)
else()
    set(PROJECT_LANG CXX)
endif()

project(rocprim_package_install_test ${PROJECT_LANG})

list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake"
)

if(USE_HIPCXX)
    enable_language(HIP)
endif()

include(VerifyCompiler)

find_package(rocprim REQUIRED CONFIG)

if(NOT DEFINED CMAKE_${PROJECT_LANG}_STANDARD)
    set(CMAKE_${PROJECT_LANG}_STANDARD 17)
endif()
set(CMAKE_${PROJECT_LANG}_STANDARD_REQUIRED ON)
set(CMAKE_${PROJECT_LANG}_EXTENSIONS OFF)

if(NOT CMAKE_CXX_STANDARD EQUAL 17)
    message(FATAL_ERROR "Only C++17 is supported")
endif()

enable_testing()

function(add_rocprim_test TEST_NAME TEST_SOURCES)
    list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
    get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET}
        PRIVATE
            roc::rocprim_hip
    )
    target_compile_options(${TEST_TARGET}
        PRIVATE
            -Wall
            -Wextra
            -Werror
    )
    add_test(${TEST_NAME} ${TEST_TARGET})
endfunction()

add_rocprim_test("test_rocprim_package" test_rocprim_package.cpp)
