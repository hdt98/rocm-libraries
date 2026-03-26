# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

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
