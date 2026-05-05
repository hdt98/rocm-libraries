# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIPDNN_SKIP_TESTS)
    message(STATUS "Skipping HIP-DNN Heuristic Config Plugin Unit tests")
    return()
endif()

find_package(Threads REQUIRED)

add_executable(hipdnn_heuristic_config_tests
    TestEngineOverrideConfig.cpp
)

# The matcher header is internal to the plugin; expose its parent directory so
# the test can include it the same way ConfigPlugin.cpp does.
target_include_directories(hipdnn_heuristic_config_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_options(hipdnn_heuristic_config_tests PRIVATE
    ${HIPDNN_WARNING_COMPILE_OPTIONS}
)

target_link_libraries(hipdnn_heuristic_config_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    hipdnn_data_sdk
)

# nlohmann_json is consumed via the matcher header for JSON loading paths.
if(NOT TARGET nlohmann_json::nlohmann_json)
    hipdnn_add_dependency(nlohmann_json VERSION ${HIPDNN_NLOHMANN_JSON_VERSION})
endif()
hipdnn_add_dependency_includes(hipdnn_heuristic_config_tests nlohmann_json::nlohmann_json)

clang_tidy_check(hipdnn_heuristic_config_tests)

add_unit_test_target(hipdnn_heuristic_config_tests ${CMAKE_CURRENT_BINARY_DIR})
