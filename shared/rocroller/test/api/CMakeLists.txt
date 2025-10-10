# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(Catch2 QUIET)
if(NOT Catch2_FOUND)
    if(ROCROLLER_ENABLE_FETCH)
        include(FetchContent)
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.6.0
        )
        FetchContent_MakeAvailable(Catch2)
    else()
        message(FATAL_ERROR "Catch2 not found and fetch disabled (set ROCROLLER_ENABLE_FETCH=ON)")
    endif()
endif()

if(DEFINED Catch2_SOURCE_DIR)
    list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
elseif(DEFINED Catch2_DIR)
    list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}")
endif()

include(CTest)
include(Catch)

add_executable(rocroller-tests-api
    LazySingletonAPITest.cpp
)

target_link_libraries(rocroller-tests-api
    PRIVATE
        rocroller
        Catch2::Catch2WithMain
)

target_include_directories(rocroller-tests-api
    PRIVATE
        ${PROJECT_SOURCE_DIR}/lib/include
)

target_compile_features(rocroller-tests-api PRIVATE cxx_std_20)

catch_discover_tests(
    rocroller-tests-api
    REPORTER JUnit
    OUTPUT_DIR ${PROJECT_BINARY_DIR}/test_report
    OUTPUT_SUFFIX ".xml"
    TEST_PREFIX "API:"
    EXTRA_ARGS "--reporter" "console::out=-::colour-mode=ansi"
    PROPERTIES LABELS "API"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DISCOVERY_MODE POST_BUILD
)
