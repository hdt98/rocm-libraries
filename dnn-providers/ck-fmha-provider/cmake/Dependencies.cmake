# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# HIP toolchain (hip::host etc.)
find_package(hip REQUIRED)

# hipDNN SDK + backend + frontend: when `HIPDNN_ROOT` points at a
# hipdnn source tree, include its helper cmake modules and the
# SDK/frontend/backend subdirectories so the plugin's
# `hipdnn_{data,plugin}_sdk`, `hipdnn_frontend`, and `hipdnn_backend`
# target references resolve.
if(NOT TARGET hipdnn_plugin_sdk OR NOT TARGET hipdnn_data_sdk OR NOT TARGET hipdnn_frontend OR NOT TARGET hipdnn_backend)
    if(DEFINED HIPDNN_ROOT)
        # Import prebuilt hipDNN artefacts as INTERFACE / IMPORTED
        # targets. Much simpler than replicating hipdnn's top-level
        # CMakeLists helper-macro setup.
        set(HIPDNN_BUILD_DIR "${HIPDNN_ROOT}/build/release" CACHE PATH
            "Location of pre-built hipdnn artefacts (libhipdnn_backend.so etc.)")

        # flatbuffers (vendored by hipdnn build via FetchContent)
        set(HIPDNN_FB_INCLUDE
            "${HIPDNN_BUILD_DIR}/_deps/flatbuffers-src/include")
        # nlohmann/json (also vendored via FetchContent)
        set(HIPDNN_JSON_INCLUDE
            "${HIPDNN_BUILD_DIR}/_deps/json-src/include")

        # Versioned flatbuffer-generated header dir. hipdnn installs
        # these under data_sdk/include/hipdnn_data_sdk/data_objects/vNN_NN_NN/.
        file(GLOB _HIPDNN_DATA_SDK_VDIRS
            LIST_DIRECTORIES TRUE
            "${HIPDNN_ROOT}/data_sdk/include/hipdnn_data_sdk/data_objects/v*")
        # Pick the newest (last after sort).
        if(_HIPDNN_DATA_SDK_VDIRS)
            list(SORT _HIPDNN_DATA_SDK_VDIRS)
            list(GET _HIPDNN_DATA_SDK_VDIRS -1 HIPDNN_DATA_SDK_GENERATED_DIR)
        endif()

        # hipdnn_data_sdk (header-only + generated headers + flatbuffers)
        add_library(hipdnn_data_sdk INTERFACE IMPORTED)
        target_include_directories(hipdnn_data_sdk INTERFACE
            ${HIPDNN_ROOT}/data_sdk/include
            ${HIPDNN_DATA_SDK_GENERATED_DIR}
            ${HIPDNN_BUILD_DIR}/data_sdk/include
            ${HIPDNN_FB_INCLUDE})
        # hipdnn_plugin_sdk (header-only)
        add_library(hipdnn_plugin_sdk INTERFACE IMPORTED)
        target_include_directories(hipdnn_plugin_sdk INTERFACE
            ${HIPDNN_ROOT}/plugin_sdk/include)
        target_link_libraries(hipdnn_plugin_sdk INTERFACE hipdnn_data_sdk)
        # hipdnn_frontend (header-only; version.h is generated at build
        # time, and engine-override helpers #include <nlohmann/json.hpp>)
        add_library(hipdnn_frontend INTERFACE IMPORTED)
        target_include_directories(hipdnn_frontend INTERFACE
            ${HIPDNN_ROOT}/frontend/include
            ${HIPDNN_BUILD_DIR}/frontend/include
            ${HIPDNN_JSON_INCLUDE})
        target_link_libraries(hipdnn_frontend INTERFACE hipdnn_data_sdk)

        # hipdnn_backend (prebuilt shared library). The generated
        # hipdnn_backend_export.h (produced by
        # GENERATE_EXPORT_HEADER()) lives under the backend build dir;
        # we have to expose both the source include path and the
        # build-time generated include path.
        add_library(hipdnn_backend SHARED IMPORTED)
        set_target_properties(hipdnn_backend PROPERTIES
            IMPORTED_LOCATION ${HIPDNN_BUILD_DIR}/lib/libhipdnn_backend.so)
        target_include_directories(hipdnn_backend INTERFACE
            ${HIPDNN_ROOT}/backend/include
            ${HIPDNN_BUILD_DIR}/backend/src/backend/include)
    endif()
endif()

# CK dispatcher: prefer pre-built package, fall back to subdirectory build
find_package(ck_tile_dispatcher QUIET)

if(NOT ck_tile_dispatcher_FOUND)
    if(DEFINED CK_ROOT)
        add_subdirectory(${CK_ROOT}/dispatcher dispatcher_build EXCLUDE_FROM_ALL)
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../projects/composablekernel/dispatcher)
        add_subdirectory(
            ${CMAKE_CURRENT_SOURCE_DIR}/../../projects/composablekernel/dispatcher
            dispatcher_build EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR
            "ck_tile_dispatcher not found. Set CK_ROOT or install the package.")
    endif()
endif()

# Phase 9: ck_host (codegen library) for the RTC backend. Only required
# when CK_FMHA_WITH_RTC is ON; fall through silently otherwise.
if(CK_FMHA_WITH_RTC)
    find_package(composable_kernel QUIET COMPONENTS ck_host)
    if(NOT TARGET ck_host)
        # ck_host isn't a standalone find_package; try building it as a subproject.
        if(DEFINED CK_ROOT AND EXISTS ${CK_ROOT}/codegen/CMakeLists.txt)
            add_subdirectory(${CK_ROOT}/codegen ck_host_build EXCLUDE_FROM_ALL)
        elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../projects/composablekernel/codegen/CMakeLists.txt)
            add_subdirectory(
                ${CMAKE_CURRENT_SOURCE_DIR}/../../projects/composablekernel/codegen
                ck_host_build EXCLUDE_FROM_ALL)
        else()
            message(WARNING
                "CK_FMHA_WITH_RTC=ON but ck_host source not found at "
                "${CK_ROOT}/codegen; RTC backend will be disabled.")
        endif()
    endif()
endif()
