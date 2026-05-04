# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Config Heuristic Plugin
# RFC 0007: Well-known policy that honors graph-level preferences

project(hipdnn_heuristic_config VERSION 1.0.0 LANGUAGES CXX)

# Create the plugin as a shared library. Do not set OUTPUT_NAME or PREFIX;
# the target name already matches the desired output name, and relying on the
# platform default avoids the `lib` prefix appearing on Windows DLLs.
add_library(hipdnn_heuristic_config SHARED
    ConfigPlugin.cpp
)

# Include directories
target_include_directories(hipdnn_heuristic_config PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against hipdnn_plugin_sdk (C ABI types), hipdnn_data_sdk (utilities),
# and hipdnn_flatbuffers_sdk (graph FlatBuffer schema + FlatBuffers headers).
target_link_libraries(hipdnn_heuristic_config PRIVATE
    hipdnn_plugin_sdk
    hipdnn_data_sdk
    hipdnn_flatbuffers_sdk
)

# Compiler options
target_compile_options(hipdnn_heuristic_config PRIVATE
    ${HIPDNN_WARNING_COMPILE_OPTIONS}
)

# Export only the C ABI symbols on Linux. --exclude-libs=ALL prevents symbols
# from statically-linked dependencies from being re-exported, matching the
# engine plugin pattern used for TheRock / multi-arch / wheel builds.
if(NOT WIN32)
    target_link_libraries(hipdnn_heuristic_config PRIVATE
        "-Xlinker --exclude-libs=ALL"
    )
    target_link_options(hipdnn_heuristic_config PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
    )
endif()

# Build the plugin directly into the standard build-tree heuristic plugin
# directory so the loader (and tests) can discover it relative to the backend
# library location, and set THEROCK_INSTALL_RPATH_ORIGIN so the installed
# plugin can locate its sibling libraries when packaged for TheRock / wheels.
set_target_properties(hipdnn_heuristic_config PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_HEURISTIC_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_HEURISTIC_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_HEURISTIC_DIR}"
    THEROCK_INSTALL_RPATH_ORIGIN "${HIPDNN_RELATIVE_INSTALL_PLUGIN_HEURISTIC_DIR}"
)

clang_tidy_check(hipdnn_heuristic_config)

# Installation. HIPDNN_RELATIVE_INSTALL_PLUGIN_HEURISTIC_DIR is the standard
# install subfolder where hipDNN heuristic plugins are located (e.g.
# lib/hipdnn_plugins/heuristics on Linux, bin/hipdnn_plugins/heuristics on Windows).
install(TARGETS hipdnn_heuristic_config
    LIBRARY DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_HEURISTIC_DIR}
    RUNTIME DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_HEURISTIC_DIR}
    COMPONENT Runtime
)
