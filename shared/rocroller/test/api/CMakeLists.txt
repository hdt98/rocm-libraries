# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Catch2 QUIET)
if(NOT Catch2_FOUND)
    if(ROCROLLER_ENABLE_FETCH)
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG devel
        )
        FetchContent_MakeAvailable(Catch2)
    else()
        message(FATAL_ERROR "Failed to find Catch2")
    endif()
endif()
if(DEFINED Catch2_SOURCE_DIR)
    list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
else()
    list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}")
endif()
include(Catch)

add_executable(rocroller-tests-api)

target_sources(rocroller-tests-api
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/LazySingletonAPITest.cpp"
)

target_compile_features(rocroller-tests-api PRIVATE cxx_std_20)

target_link_libraries(rocroller-tests-api
    PRIVATE
        roc::rocroller-interface
        Catch2::Catch2WithMain
)

catch_discover_tests(
    rocroller-tests-api
    REPORTER JUnit
    OUTPUT_DIR ${TEST_REPORT_DIR}
    OUTPUT_SUFFIX ".xml"
    DISCOVERY_MODE PRE_TEST
    EXTRA_ARGS "--reporter console::out=-::colour-mode=ansi"
    PROPERTIES "LABELS" "CATCH;API"
               "ENVIRONMENT" "${asan_opts}"
    SKIP_REGULAR_EXPRESSION "[1-9][0-9]* skipped" "[1-9][0-9]* SKIPPED"
)    
