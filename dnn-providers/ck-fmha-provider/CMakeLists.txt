# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)
project(ck_fmha_provider LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CK_FMHA_PROVIDER_BUILD_TESTS "Build CK FMHA provider tests" ON)

include(cmake/Dependencies.cmake)

add_library(ck_fmha_plugin_impl OBJECT
    src/CkFmhaPluginPublic.cpp
    src/CkFmhaContainer.cpp
    src/CkFmhaHandle.cpp
    src/CkFmhaJit.cpp
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

set_target_properties(ck_fmha_plugin_impl PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

add_library(ck_fmha_plugin SHARED $<TARGET_OBJECTS:ck_fmha_plugin_impl>)
target_link_libraries(ck_fmha_plugin
    PRIVATE hipdnn_data_sdk hipdnn_plugin_sdk ck_tile_dispatcher hip::host)
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
    include(cmake/Tests.cmake)
    add_subdirectory(integration_tests)
endif()
