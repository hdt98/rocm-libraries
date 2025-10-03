# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Catch2 QUIET)
if(NOT Catch2_FOUND)
    if(ROCROLLER_ENABLE_FETCH)
        include(FetchContent)
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
    EXTRA_ARGS "--reporter" "console::out=-::colour-mode=ansi"
    TEST_SPEC "[api]"
    PROPERTIES LABELS "API"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} 
    DISCOVERY_MODE POST_BUILD
)