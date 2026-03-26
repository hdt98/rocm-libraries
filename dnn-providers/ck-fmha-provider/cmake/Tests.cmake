# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(GTest QUIET)
if(NOT GTest_FOUND)
    return()
endif()

add_executable(ck_fmha_provider_tests
    tests/CkFmhaPluginTest.cpp)

target_link_libraries(ck_fmha_provider_tests
    PRIVATE ck_fmha_plugin_impl hipdnn_data_sdk hipdnn_plugin_sdk GTest::gtest_main)

target_include_directories(ck_fmha_provider_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

include(GoogleTest)
gtest_discover_tests(ck_fmha_provider_tests)
