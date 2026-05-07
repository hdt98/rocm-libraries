# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIPDNN_SKIP_TESTS)
    message(STATUS "Skipping HIP-DNN Default Heuristics Plugin Unit tests")
    return()
endif()

find_package(Threads REQUIRED)

add_executable(hipdnn_heuristic_default_tests
    TestDefaultHeuristicsPlugin.cpp
    TestConfigPolicy.cpp
    TestStaticOrderingPolicy.cpp
    TestEngineOverrideConfig.cpp
)

# The matcher header is internal to the plugin; expose its parent directory so
# the tests can include it the same way DefaultHeuristicsPlugin.cpp does.
target_include_directories(hipdnn_heuristic_default_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# The tests link the plugin's STATIC archive directly so the C ABI entry points
# can be exercised without dlopen. Define the *_STATIC_DEFINE macros so the
# export-annotated declarations in the public headers expand to nothing on this
# consumer side (matches the producer OBJECT lib behavior when linked statically
# into an executable).
target_compile_definitions(hipdnn_heuristic_default_tests PRIVATE
    HIPDNN_PLUGIN_STATIC_DEFINE
    HIPDNN_HEURISTIC_PLUGIN_STATIC_DEFINE
)

target_compile_options(hipdnn_heuristic_default_tests PRIVATE
    ${HIPDNN_WARNING_COMPILE_OPTIONS}
)

target_link_libraries(hipdnn_heuristic_default_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    hipdnn_data_sdk
    hipdnn_plugin_sdk
    hipdnn_flatbuffers_sdk
    hipdnn_test_sdk
    hipdnn_heuristic_default_private
)

# nlohmann_json is consumed via the matcher header for JSON loading paths.
if(NOT TARGET nlohmann_json::nlohmann_json)
    hipdnn_add_dependency(nlohmann_json VERSION ${HIPDNN_NLOHMANN_JSON_VERSION})
endif()
hipdnn_add_dependency_includes(hipdnn_heuristic_default_tests nlohmann_json::nlohmann_json)

clang_tidy_check(hipdnn_heuristic_default_tests)

add_unit_test_target(hipdnn_heuristic_default_tests ${CMAKE_CURRENT_BINARY_DIR})
