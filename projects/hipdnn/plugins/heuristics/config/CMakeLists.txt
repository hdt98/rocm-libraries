# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Config Heuristic Plugin
# RFC 0007: Well-known policy that honors graph-level preferences
# (Graph.preferred_engine_id) and the HIPDNN_ENGINE_OVERRIDE_FILE JSON config.

project(hipdnn_heuristic_config VERSION 1.0.0 LANGUAGES CXX)

# nlohmann_json is used directly by EngineOverrideConfig.hpp for parsing the
# HIPDNN_ENGINE_OVERRIDE_FILE. Declare it explicitly so the plugin builds
# even when downstream consumers gate flatbuffers_sdk's JSON headers off.
if(NOT TARGET nlohmann_json::nlohmann_json)
    hipdnn_add_dependency(nlohmann_json VERSION ${HIPDNN_NLOHMANN_JSON_VERSION})
endif()

# OBJECT library: contains the plugin implementation. Built once and shared
# between the SHARED plugin and the STATIC library used by unit tests.
add_library(hipdnn_heuristic_config_impl OBJECT
    ConfigPlugin.cpp
)
target_include_directories(hipdnn_heuristic_config_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(hipdnn_heuristic_config_impl PUBLIC
    hipdnn_plugin_sdk
    hipdnn_data_sdk
    hipdnn_flatbuffers_sdk
)
hipdnn_add_dependency_includes(hipdnn_heuristic_config_impl nlohmann_json::nlohmann_json)
target_compile_options(hipdnn_heuristic_config_impl PRIVATE
    ${HIPDNN_WARNING_COMPILE_OPTIONS}
)
set_target_properties(hipdnn_heuristic_config_impl PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)

# STATIC library: exposes plugin internals for unit tests that link directly
# against the implementation rather than dlopen'ing the shared plugin.
add_library(hipdnn_heuristic_config_private STATIC
    $<TARGET_OBJECTS:hipdnn_heuristic_config_impl>
)
target_include_directories(hipdnn_heuristic_config_private PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(hipdnn_heuristic_config_private PUBLIC
    hipdnn_plugin_sdk
    hipdnn_data_sdk
    hipdnn_flatbuffers_sdk
)
hipdnn_add_dependency_includes(hipdnn_heuristic_config_private nlohmann_json::nlohmann_json)

# SHARED library: the actual plugin loaded at runtime. Re-uses the OBJECTs
# from the impl target. CXX_VISIBILITY_PRESET hidden + HIPDNN_HEURISTIC_PLUGIN_EXPORT
# annotations on the C ABI entry points keep all other symbols hidden, and
# --exclude-libs=ALL prevents statically-linked dependencies from re-exporting
# their symbols. This matches the example engine plugin pattern.
add_library(hipdnn_heuristic_config SHARED
    $<TARGET_OBJECTS:hipdnn_heuristic_config_impl>
)
target_link_libraries(hipdnn_heuristic_config PRIVATE
    hipdnn_plugin_sdk
    hipdnn_data_sdk
    hipdnn_flatbuffers_sdk
)
hipdnn_add_dependency_includes(hipdnn_heuristic_config nlohmann_json::nlohmann_json)
if(NOT WIN32)
    target_link_libraries(hipdnn_heuristic_config PRIVATE
        "-Xlinker --exclude-libs=ALL"
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

add_subdirectory(tests)
