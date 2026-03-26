# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(GTest QUIET)
if(NOT GTest_FOUND)
    return()
endif()

# Integration tests require GPU -- skip if not available
include(CheckLanguage)
check_language(HIP)
if(NOT CMAKE_HIP_COMPILER)
    message(STATUS "HIP not found, skipping CK FMHA integration tests")
    return()
endif()

add_executable(ck_fmha_integration_tests
    EndToEndTest.cpp)

target_link_libraries(ck_fmha_integration_tests
    PRIVATE ck_fmha_plugin_impl
            hipdnn_data_sdk hipdnn_plugin_sdk
            ck_tile_dispatcher
            hip::host
            GTest::gtest_main)

target_include_directories(ck_fmha_integration_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

include(GoogleTest)
gtest_discover_tests(ck_fmha_integration_tests
    PROPERTIES LABELS "integration;gpu")

# End-to-end TFLOPS benchmark demo (not a gtest, standalone binary)
add_executable(ck_fmha_e2e_demo
    EndToEndSdpaDemo.cpp)

target_link_libraries(ck_fmha_e2e_demo
    PRIVATE hipdnn_frontend hipdnn_backend
            hip::host)

# The demo discovers the CK FMHA plugin at runtime via dlopen,
# so it only needs the hipDNN frontend/backend at link time.
# Make sure the plugin .so is in the plugin search path at runtime.
