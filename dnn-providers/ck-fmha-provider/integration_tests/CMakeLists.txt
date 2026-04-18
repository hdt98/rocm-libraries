# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Integration tests require GPU -- skip if not available
include(CheckLanguage)
check_language(HIP)
if(NOT CMAKE_HIP_COMPILER)
    message(STATUS "HIP not found, skipping CK FMHA integration tests")
    return()
endif()

# GTest-backed integration tests (optional).
find_package(GTest QUIET)
if(GTest_FOUND)
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
endif()

# End-to-end TFLOPS benchmark demo (not a gtest, standalone binary).
# Builds even when GTest is unavailable, since the demo discovers the
# plugin via dlopen at runtime and only needs hipDNN's frontend +
# backend at link time.
add_executable(ck_fmha_e2e_demo
    EndToEndSdpaDemo.cpp)

target_link_libraries(ck_fmha_e2e_demo
    PRIVATE hipdnn_frontend hipdnn_backend
            hip::host)

# A variant that exercises the hipRTC JIT path and performs a
# hipcc-vs-rtc parity comparison (Phase 12 follow-up). Produces a
# side-by-side table of wall-time + output-parity for every shape.
add_executable(ck_fmha_e2e_rtc_demo
    EndToEndSdpaRtcDemo.cpp)

target_link_libraries(ck_fmha_e2e_rtc_demo
    PRIVATE hipdnn_frontend hipdnn_backend
            hip::host)

# The demo discovers the CK FMHA plugin at runtime via dlopen,
# so it only needs the hipDNN frontend/backend at link time.
# Make sure the plugin .so is in the plugin search path at runtime.
