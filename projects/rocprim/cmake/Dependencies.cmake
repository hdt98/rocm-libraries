# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ROCm CMake dependencies
include(FetchContent)

find_package(ROCmCMakeBuildTools 0.11.0 CONFIG QUIET)
if(NOT ROCmCMakeBuildTools_FOUND)
    message(STATUS "ROCm CMake not found. Fetching...")
    FetchContent_Declare(
        rocm-cmake
        GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
        GIT_TAG        rocm-6.4.4
        SOURCE_SUBDIR "DISABLE ADDING TO BUILD"
    )
    FetchContent_MakeAvailable(rocm-cmake)
    find_package(ROCmCMakeBuildTools CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${rocm-cmake_SOURCE_DIR}")
else()
    find_package(ROCmCMakeBuildTools 0.11.0 CONFIG REQUIRED)
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMCheckTargetIds)
include(ROCMClients)
if(ROCPRIM_ENABLE_DOCS)
    include(ROCMSphinxDoc)
endif()

# Test dependencies
if(ROCPRIM_BUILD_TESTING)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        if(EXISTS /usr/src/googletest AND NOT DEPENDENCIES_FORCE_DOWNLOAD)
            FetchContent_Declare(
                googletest
                SOURCE_DIR /usr/src/googletest
            )
        else()
            message(STATUS "Google Test not found. Fetching...")
            FetchContent_Declare(
                googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG        e2239ee6043f73722e7aa812a459f54a28552929 # release-1.11.0
            )
        endif()
        set(BUILD_GMOCK OFF CACHE BOOL "")
        set(INSTALL_GTEST OFF CACHE BOOL "")
        FetchContent_MakeAvailable(googletest)
        if(NOT TARGET GTest::GTest)
            add_library(GTest::GTest ALIAS gtest)
            add_library(GTest::Main ALIAS gtest_main)
        endif()
    else()
        find_package(GTest REQUIRED)
        if(TARGET GTest::gtest_main AND NOT TARGET GTest::Main)
            add_library(GTest::GTest ALIAS GTest::gtest)
            add_library(GTest::Main ALIAS GTest::gtest_main)
        endif()
    endif()
endif()

# Benchmark dependencies
if(ROCPRIM_ENABLE_BENCHMARK)
    set(BENCHMARK_VERSION 1.8.0)
    find_package(benchmark ${BENCHMARK_VERSION} CONFIG QUIET)
    if(NOT benchmark_FOUND)
        message(STATUS "Google Benchmark not found. Fetching...")
        FetchContent_Declare(
            googlebench
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG        v${BENCHMARK_VERSION}
        )
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
        set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "")
        set(HAVE_STD_REGEX ON)
        set(RUN_HAVE_STD_REGEX 1)
        FetchContent_MakeAvailable(googlebench)
        if(NOT TARGET benchmark::benchmark)
            add_library(benchmark::benchmark ALIAS benchmark)
        endif()
    else()
        find_package(benchmark CONFIG REQUIRED)
    endif()
endif()

# rocRAND dependency
if(ROCPRIM_ENABLE_ROCRAND)
    find_package(rocrand QUIET)
    if(NOT rocrand_FOUND)
        message(STATUS "rocRAND not found. Fetching and building...")
        include(DownloadProject)
        
        set(ROCRAND_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/rocrand CACHE PATH "")
        set(EXTRA_CMAKE_ARGS "-DGPU_TARGETS=${GPU_TARGETS}")
        string(REPLACE ";" "|" EXTRA_CMAKE_ARGS "${EXTRA_CMAKE_ARGS}")
        
        if(CMAKE_CXX_COMPILER_LAUNCHER)
            set(EXTRA_CMAKE_ARGS "${EXTRA_CMAKE_ARGS} -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}")
        endif()
        
        download_project(
            PROJ                  rocrand
            GIT_REPOSITORY        https://github.com/ROCmSoftwarePlatform/rocRAND.git
            GIT_TAG               develop
            GIT_SHALLOW           TRUE
            INSTALL_DIR           ${ROCRAND_ROOT}
            LIST_SEPARATOR        |
            CMAKE_ARGS            -DCMAKE_CXX_COMPILER=hipcc -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_PREFIX_PATH=/opt/rocm ${EXTRA_CMAKE_ARGS}
            LOG_DOWNLOAD          TRUE
            LOG_CONFIGURE         TRUE
            LOG_BUILD             TRUE
            LOG_INSTALL           TRUE
            LOG_OUTPUT_ON_FAILURE TRUE
            BUILD_PROJECT         TRUE
            UPDATE_DISCONNECTED   TRUE
        )
        find_package(rocrand REQUIRED CONFIG PATHS ${ROCRAND_ROOT})
    endif()
endif()
