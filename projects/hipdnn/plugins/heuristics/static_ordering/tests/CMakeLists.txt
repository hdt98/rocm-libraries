# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIPDNN_SKIP_TESTS)
    message(STATUS "Skipping HIP-DNN Heuristic StaticOrdering Plugin Unit tests")
    return()
endif()

find_package(Threads REQUIRED)

# Links the plugin's STATIC archive directly so the C ABI entry points can be
# exercised without dlopen'ing the SHARED plugin. Define the *_STATIC_DEFINE
# macros so the export-annotated declarations in the public headers expand to
# nothing on this consumer side (matches the producer OBJECT lib behavior when
# linked statically into an executable).
add_executable(hipdnn_heuristic_static_ordering_tests
    TestStaticOrderingPlugin.cpp
)

target_compile_definitions(hipdnn_heuristic_static_ordering_tests PRIVATE
    HIPDNN_PLUGIN_STATIC_DEFINE
    HIPDNN_HEURISTIC_PLUGIN_STATIC_DEFINE
)

target_compile_options(hipdnn_heuristic_static_ordering_tests PRIVATE
    ${HIPDNN_WARNING_COMPILE_OPTIONS}
)

target_link_libraries(hipdnn_heuristic_static_ordering_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    hipdnn_heuristic_static_ordering_private
    hipdnn_plugin_sdk
    hipdnn_data_sdk
)

clang_tidy_check(hipdnn_heuristic_static_ordering_tests)

add_unit_test_target(hipdnn_heuristic_static_ordering_tests ${CMAKE_CURRENT_BINARY_DIR})
