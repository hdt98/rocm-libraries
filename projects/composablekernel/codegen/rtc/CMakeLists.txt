# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Public ck_rtc library. This is the Phase 1 public promotion of the
# previously test-only helper library in codegen/test/rtc/.
#
# Design notes:
#   - Sources still live under codegen/test/rtc/{src,include} to avoid
#     breaking the existing codegen_test_* build paths. This CMakeLists
#     references them via ${CMAKE_CURRENT_SOURCE_DIR}/../test/rtc.
#   - ck_rtc always requires hipRTC now (the prior opt-in
#     USE_HIPRTC_FOR_CODEGEN_TESTS switch is retained as an alias for
#     backwards compatibility with the MGX test subproject).
#   - The target is promoted to the ck_host_targets install/export set
#     alongside ck_host, ck_headers, ck_tile_headers, ck_codegen_headers.

find_package(hip REQUIRED)
find_package(hiprtc REQUIRED)

set(_RTC_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../test/rtc)
file(GLOB RTC_SOURCES CONFIGURE_DEPENDS ${_RTC_SRC_DIR}/src/*.cpp)

add_library(ck_rtc ${RTC_SOURCES})
target_include_directories(ck_rtc PUBLIC
    $<BUILD_INTERFACE:${_RTC_SRC_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(ck_rtc PUBLIC hip::host hiprtc::hiprtc -lstdc++fs)
# The existing compile_kernel.cpp branches on HIPRTC_FOR_CODEGEN_TESTS;
# we now always enable the hipRTC code path in ck_rtc.
target_compile_definitions(ck_rtc PUBLIC HIPRTC_FOR_CODEGEN_TESTS)

set_target_properties(ck_rtc PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Install headers from the live location.
install(DIRECTORY ${_RTC_SRC_DIR}/include/rtc
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp")

message(STATUS "ck_rtc promoted to public install target (hipRTC enabled)")
