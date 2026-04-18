# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)
project(ck_fmha_provider LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CK_FMHA_PROVIDER_BUILD_TESTS "Build CK FMHA provider tests" ON)

# Phase 9: CK_FMHA_WITH_RTC compiles the hipRTC backend into the
# plugin. Requires composable_kernel::ck_host (built from
# projects/composablekernel/codegen/). When OFF the plugin builds
# without an RTC backend and CkFmhaJit::compile_rtc always returns
# "not available", leaving CK_FMHA_JIT_BACKEND=hipcc as the only live path.
option(CK_FMHA_WITH_RTC "Enable in-process hipRTC JIT backend (Phase 9)" ON)

# Phase 10: when set, RTC becomes the compile-time default (same as
# CK_FMHA_JIT_BACKEND=rtc) and hipcc is only used when explicitly
# requested. Keep OFF until deployment validates RTC on the full
# (arch, dtype, family, shape) matrix. Deployments that want the
# runtime Python path entirely removed should also drop
# CK_DISPATCHER_PYTHON_PATH-related environment variables from their
# orchestration -- see ck-fmha-provider/README.md.
option(CK_FMHA_DEFAULT_BACKEND_RTC
       "Make RTC the default runtime JIT backend (Phase 10)" OFF)

include(cmake/Dependencies.cmake)

add_library(ck_fmha_plugin_impl OBJECT
    src/CkFmhaPluginPublic.cpp
    src/CkFmhaContainer.cpp
    src/CkFmhaHandle.cpp
    src/CkFmhaJit.cpp
    src/CkFmhaRtcKernelInstance.cpp
    src/CkFmhaRtcStats.cpp
    src/engines/CkFmhaEngine.cpp
    src/engines/CkFmhaParamParser.cpp
    src/engines/plans/CkFmhaFwdPlanBuilder.cpp
    src/engines/plans/CkFmhaFwdPlan.cpp
    src/engines/plans/CkFmhaBwdPlanBuilder.cpp
    src/engines/plans/CkFmhaBwdPlan.cpp)

target_include_directories(ck_fmha_plugin_impl
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(ck_fmha_plugin_impl
    PUBLIC hipdnn_data_sdk hipdnn_plugin_sdk
    PRIVATE ck_tile_dispatcher hip::host)

if(CK_FMHA_WITH_RTC)
    if(TARGET ck_host)
        target_link_libraries(ck_fmha_plugin_impl PRIVATE ck_host)
        target_compile_definitions(ck_fmha_plugin_impl PRIVATE CK_FMHA_WITH_RTC=1)
        if(CK_FMHA_DEFAULT_BACKEND_RTC)
            target_compile_definitions(ck_fmha_plugin_impl PRIVATE
                CK_FMHA_DEFAULT_BACKEND_RTC=1)
            message(STATUS "ck_fmha_plugin: hipRTC backend enabled AND default (Phase 10)")
        else()
            message(STATUS "ck_fmha_plugin: hipRTC backend enabled, default remains Auto")
        endif()
    else()
        message(STATUS "ck_fmha_plugin: CK_FMHA_WITH_RTC requested but ck_host "
                       "target not available; building without RTC backend")
    endif()
endif()

set_target_properties(ck_fmha_plugin_impl PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

add_library(ck_fmha_plugin SHARED $<TARGET_OBJECTS:ck_fmha_plugin_impl>)
target_link_libraries(ck_fmha_plugin
    PRIVATE hipdnn_data_sdk hipdnn_plugin_sdk ck_tile_dispatcher hip::host)
if(CK_FMHA_WITH_RTC AND TARGET ck_host)
    # ck_host pulls in ck_rtc + hiprtc transitively.
    target_link_libraries(ck_fmha_plugin PRIVATE ck_host)
endif()
set_target_properties(ck_fmha_plugin PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    OUTPUT_NAME "ck_fmha_provider_plugin")

if(DEFINED HIPDNN_BUILD_PLUGIN_ENGINE_DIR)
    set_target_properties(ck_fmha_plugin PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR})
endif()

install(TARGETS ck_fmha_plugin
    LIBRARY DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
    RUNTIME DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR})

if(CK_FMHA_PROVIDER_BUILD_TESTS)
    enable_testing()
    include(CTest)
    include(cmake/Tests.cmake)
    add_subdirectory(integration_tests)
endif()
