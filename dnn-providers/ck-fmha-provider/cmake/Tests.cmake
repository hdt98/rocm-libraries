# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# GTest-backed plugin tests (optional -- only build when a GTest is
# discoverable on the host).
find_package(GTest QUIET)
if(GTest_FOUND)
    add_executable(ck_fmha_provider_tests
        tests/CkFmhaPluginTest.cpp)
    target_link_libraries(ck_fmha_provider_tests
        PRIVATE ck_fmha_plugin_impl hipdnn_data_sdk hipdnn_plugin_sdk GTest::gtest_main)
    target_include_directories(ck_fmha_provider_tests
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    include(GoogleTest)
    gtest_discover_tests(ck_fmha_provider_tests)
endif()

# Phase 15: lightweight unit tests for the hipRTC wiring that do NOT
# depend on GTest. These cover the CPU-side invariants of
# RtcFmhaKernelInstance::supports(), pick_jit_backend(), and the
# shape-hash used by compile_rtc() for registry disambiguation.
add_executable(ck_fmha_rtc_unit_tests
    tests/CkFmhaRtcUnitTest.cpp)
target_link_libraries(ck_fmha_rtc_unit_tests
    PRIVATE ck_fmha_plugin_impl
            ck_tile_dispatcher
            hip::host)
if(CK_FMHA_WITH_RTC AND TARGET ck_host)
    target_link_libraries(ck_fmha_rtc_unit_tests PRIVATE ck_host)
    target_compile_definitions(ck_fmha_rtc_unit_tests PRIVATE CK_FMHA_WITH_RTC=1)
endif()
target_include_directories(ck_fmha_rtc_unit_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME ck_fmha_rtc_unit_tests COMMAND ck_fmha_rtc_unit_tests)

# Phase 14/15: HSACO cache-verifier tests. Stand-alone; the fake ELF
# headers it writes do not touch the live FmhaRegistry or the GPU.
add_executable(ck_fmha_rtc_cache_verify_tests
    tests/CkFmhaRtcCacheVerifyTest.cpp)
target_link_libraries(ck_fmha_rtc_cache_verify_tests PRIVATE hip::host)
add_test(NAME ck_fmha_rtc_cache_verify_tests COMMAND ck_fmha_rtc_cache_verify_tests)
