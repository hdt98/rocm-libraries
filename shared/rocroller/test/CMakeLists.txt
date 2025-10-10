# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(GTest QUIET)
if(NOT GTest_FOUND)
    if(ROCROLLER_ENABLE_FETCH)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
        )
        FetchContent_MakeAvailable(googletest)
    else()
        message(FATAL_ERROR "Failed to find googletest")
    endif()
endif()

if(NOT ROCM_LIBS_SUPERBUILD)
    if(ROCROLLER_ENABLE_THEROCK)
        find_package(mxDataGenerator REQUIRED)
    else()
        find_package(mxDataGenerator QUIET)
        if(NOT mxDataGenerator_FOUND)
            if(ROCROLLER_ENABLE_FETCH)
                FetchContent_Declare(
                    mxDataGenerator
                    GIT_REPOSITORY ${MXDATAGENERATOR_GIT_URL}
                    GIT_TAG ${MXDATAGENERATOR_GIT_TAG}
                )
                FetchContent_MakeAvailable(mxDataGenerator)
            else()
                if(EXISTS "${PROJECT_SOURCE_DIR}/shared/mxdatagenerator/CMakeLists.txt")
                    add_subdirectory(
                        "${PROJECT_SOURCE_DIR}/shared/mxdatagenerator"
                        "${PROJECT_BINARY_DIR}/mxdatagenerator"
                    )
                else()
                    message(WARNING "mxDataGenerator not found; skipping inclusion.")
                endif()
            endif()
        endif()
    endif()
endif()

add_executable(rocroller-tests)

if(ROCROLLER_ENABLE_CATCH AND (ROCROLLER_BUILD_TESTING OR BUILD_TESTING))
    add_executable(rocroller-tests-catch)
    # When detecting tests gtest and catch test discovery modules run
    # something like:
    # execute_process(COMMAND ${_TEST_EXECUTOR} "${_TEST_EXECUTABLE}" --gtest_list_tests ${filter})
    # where TEST_EXECUTABLE is rocroller-tests-catch and TEST_EXECUTOR
    # is a command to prepend primarily for the purposes of running in
    # a cross compiled environment. As such, we can set the CROSSCOMPILING_EMULATOR
    # property on the test targets to use the cmake -E -env command providing
    # the LD_PRELOAD and ASAN_OPTIONS variables necessary to run sanitizer builds.
    # Note that the entries in the string are semicolon separated which is necessary
    # to ensure that the command supplied to execute_process is interpreted as a list
    # of strings rather than one white space delimited string.
    if(ROCROLLER_ENABLE_ASAN)
        rocroller_target_configure_sanitizers(rocroller-tests-catch PRIVATE)
        set_target_properties(rocroller-tests-catch
            PROPERTIES
                CROSSCOMPILING_EMULATOR "${CMAKE_COMMAND};-E;env;${asan_opts}"
        )
    endif()

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
            message(FATAL_ERROR "Catch2 not found and fetch disabled. Set ROCROLLER_ENABLE_FETCH=ON.")
        endif()
    endif()

    if(DEFINED Catch2_SOURCE_DIR)
        list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
    elseif(DEFINED Catch2_DIR)
        list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}")
    endif()
endif()

# Component stand-alone test
add_executable(component-test)

if(ROCROLLER_ENABLE_ASAN)
    rocroller_target_configure_sanitizers(component-test PRIVATE)
    set_target_properties(component-test
        PROPERTIES
            CROSSCOMPILING_EMULATOR "${CMAKE_COMMAND};-E;env;${asan_opts}"
    )
    rocroller_target_configure_sanitizers(rocroller-tests PRIVATE)
    set_target_properties(rocroller-tests
        PROPERTIES
            CROSSCOMPILING_EMULATOR "${CMAKE_COMMAND};-E;env;${asan_opts}"
    )
endif()

if(ROCROLLER_ENABLE_GEMM_CLIENT_TESTS)
    if(NOT ROCROLLER_ENABLE_LLD)
        # There are a few tests that require ReadELF functionality
        # which is disabled by default and on a deprecation path.
        # The tests that fail are missing the amdhsa.version key
        # and as such hit a fatal error that emits the following
        #     Key   key = amdhsa.version
        #      not found: {}
        # If observed, the test will be marked as skipped rather
        # than failing.
        if(ROCROLLER_ENABLE_LLVM)
            set(rocroller_skip_regex "FatalError\\(hsa_version.size\\(\\) == 2\\)")
        else()
            set(rocroller_skip_regex "key = amdhsa.version")
        endif()
    endif()
    find_package(Pytest REQUIRED)  # requires pip install -r requirements.txt
    pytest_discover_tests(
        GEMMClientTests
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/client"
        ENVIRONMENT "ROCROLLER_BUILD_DIR=${PROJECT_BINARY_DIR}" "LD_PRELOAD=${ASAN_LIB_PATH}" "ASAN_OPTIONS=detect_leaks=0"
        PROPERTIES
            SKIP_REGULAR_EXPRESSION ${rocroller_skip_regex}
    )
endif()

add_executable(arch-gen-tests)
if(ROCROLLER_ENABLE_ASAN)
    rocroller_target_configure_sanitizers(arch-gen-tests PRIVATE)
    set_target_properties(arch-gen-tests
        PROPERTIES
            CROSSCOMPILING_EMULATOR "${CMAKE_COMMAND};-E;env;${asan_opts}"
    )
endif()

find_package(BLAS REQUIRED)
find_package(cblas)
find_package(OpenMP REQUIRED)

if(ROCROLLER_ENABLE_YAML_CPP)
    target_compile_definitions(rocroller-tests PRIVATE ROCROLLER_TESTS_USE_YAML_CPP)
else()
    target_compile_definitions(rocroller-tests PRIVATE ROCROLLER_USE_LLVM)
endif()


if(ROCROLLER_ENABLE_CATCH)
    include(Catch)
    add_subdirectory(catch)
    add_subdirectory(api)
endif()

add_subdirectory(common)
add_subdirectory(standalone)
add_subdirectory(unit)

if(ROCROLLER_ENABLE_TEST_DISCOVERY)
    set(TEST_REPORT_DIR "${PROJECT_BINARY_DIR}/test_report")
    file(MAKE_DIRECTORY ${TEST_REPORT_DIR})

    include(GoogleTest)
    gtest_discover_tests(
        rocroller-tests
        XML_OUTPUT_DIR ${TEST_REPORT_DIR}
        TEST_FILTER "-*GPU_*"
        DISCOVERY_MODE PRE_TEST
        PROPERTIES ENVIRONMENT "${asan_opts}"
    )
    gtest_discover_tests(
        rocroller-tests
        XML_OUTPUT_DIR ${TEST_REPORT_DIR}
        TEST_FILTER "*GPU_*"
        PROPERTIES LABELS "GPU"
        DISCOVERY_MODE PRE_TEST
        PROPERTIES ENVIRONMENT "${asan_opts}"
    )
    gtest_discover_tests(
        arch-gen-tests
        XML_OUTPUT_DIR ${TEST_REPORT_DIR}
        DISCOVERY_MODE PRE_TEST
        PROPERTIES ENVIRONMENT "${asan_opts}"
    )

    if(ROCROLLER_ENABLE_CATCH)
        catch_discover_tests(
            rocroller-tests-catch
            REPORTER JUnit
            OUTPUT_DIR ${TEST_REPORT_DIR}
            OUTPUT_SUFFIX ".xml"
            DISCOVERY_MODE PRE_TEST
            EXTRA_ARGS "--reporter" "console::out=-::colour-mode=ansi"
            TEST_SPEC "~[gpu]"
            PROPERTIES LABELS "CATCH" ENVIRONMENT "${asan_opts}"
        )
    endif()
endif()

if(ROCROLLER_ENABLE_COVERAGE OR BUILD_CODE_COVERAGE)
    target_compile_options(rocroller PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(rocroller PRIVATE -fprofile-instr-generate)

    target_compile_options(common-test-utilities PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_compile_options(rocroller-tests PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(rocroller-tests PRIVATE -fprofile-instr-generate)

    if(TARGET rocroller-tests-catch)
        target_compile_options(rocroller-tests-catch PRIVATE -fprofile-instr-generate -fcoverage-mapping)
        target_link_options(rocroller-tests-catch PRIVATE -fprofile-instr-generate)
    endif()
endif()
