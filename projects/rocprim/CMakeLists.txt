# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
cmake_policy(VERSION 3.16...3.25)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(rocprim_shim)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path prefix" FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix" FORCE)
    endif()
endif()

project(rocprim LANGUAGES CXX)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${PROJECT_SOURCE_DIR}")
    set(ROCPRIM_IS_SUBPROJECT FALSE)
else()
    set(ROCPRIM_IS_SUBPROJECT TRUE)
endif()

include(CheckLanguage)
include(CMakeDependentOption)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Get dependencies (required here to get rocm-cmake)
include(Dependencies)

set(VERSION_STRING "4.1.0")
rocm_setup_version(VERSION ${VERSION_STRING})

# Build options
option(ROCPRIM_ENABLE_INSTALL "Enable installation of rocPRIM" ON)
cmake_dependent_option(ROCPRIM_BUILD_TESTING "Build tests" ON "NOT ROCPRIM_IS_SUBPROJECT" OFF)
cmake_dependent_option(ROCPRIM_ENABLE_BENCHMARK "Build benchmarks" OFF "NOT ROCPRIM_IS_SUBPROJECT" OFF)
cmake_dependent_option(ROCPRIM_ENABLE_EXAMPLES "Build examples" OFF "NOT ROCPRIM_IS_SUBPROJECT" OFF)
option(ROCPRIM_ENABLE_DOCS "Build documentation" OFF)
option(ROCPRIM_ENABLE_COVERAGE "Build with code coverage enabled" OFF)
option(ROCPRIM_ENABLE_ASSERTS "Enable asserts in release build" OFF)
option(ROCPRIM_ENABLE_OFFLOAD_COMPRESS "Build rocPRIM with offload compression" ON)
option(ROCPRIM_ENABLE_NAIVE_BENCHMARK "Build naive benchmarks" OFF)
option(ROCPRIM_ENABLE_CONFIG_TUNING "Benchmark device-level functions using various configs" OFF)
option(ROCPRIM_ENABLE_AUTOTUNED_TYPES_ONLY "Benchmark autotuned types only" OFF)
cmake_dependent_option(ROCPRIM_ENABLE_ROCRAND "Build tests with device-side data generation (requires rocRAND)" OFF "ROCPRIM_BUILD_TESTING" OFF)
cmake_dependent_option(ROCPRIM_USE_SYSTEM_LIBS "Use installed ROCm libs when building tests" OFF "ROCPRIM_BUILD_TESTING" OFF)

check_language(HIP)
cmake_dependent_option(ROCPRIM_USE_HIPCXX "Use CMake HIP language support" OFF CMAKE_HIP_COMPILER OFF)

include(CheckCXXCompilerFlag)

if(ROCPRIM_ENABLE_OFFLOAD_COMPRESS)
    check_cxx_compiler_flag("--offload-compress -x hip" CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
    if(CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
        add_compile_options(--offload-compress)
    else()
        message(STATUS "Warning: ROCPRIM_ENABLE_OFFLOAD_COMPRESS=ON but flag not supported by compiler. Ignoring option.")
    endif()
endif()

if(ROCPRIM_ENABLE_ASSERTS)
    add_compile_definitions($<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:ROCPRIM_ENABLE_ASSERTS>)
    if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "")
        string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
        set(BUILD_TYPE_CXX_FLAGS "CMAKE_CXX_FLAGS_${BUILD_TYPE}")
        set(BUILD_TYPE_C_FLAGS "CMAKE_C_FLAGS_${BUILD_TYPE}")
        string(REGEX REPLACE "-DNDEBUG( |$)" "" ${BUILD_TYPE_CXX_FLAGS} "${${BUILD_TYPE_CXX_FLAGS}}")
        string(REGEX REPLACE "-DNDEBUG( |$)" "" ${BUILD_TYPE_C_FLAGS} "${${BUILD_TYPE_C_FLAGS}}")
    endif()
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Add paths to linker search and installed rpath")

# Use target ID syntax if supported for GPU_TARGETS
if(ROCPRIM_USE_HIPCXX)
    enable_language(HIP)
else()
    if(NOT DEFINED AMDGPU_TARGETS)
        set(GPU_TARGETS "all" CACHE STRING "GPU architectures to compile for")
    else()
        set(GPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for")
    endif()
    set_property(CACHE GPU_TARGETS PROPERTY STRINGS "all")

    if(GPU_TARGETS STREQUAL "all")
        if(BUILD_ADDRESS_SANITIZER)
            rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
                TARGETS "gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+;gfx950:xnack+"
            )
        else()
            rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
                TARGETS "gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201"
            )
        endif()
        set(GPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for" FORCE)
    endif()
endif()

# Compressed offload binaries are currently not working with the SPIR-V target
if("amdgcnspirv" IN_LIST GPU_TARGETS)
    if(ROCPRIM_ENABLE_OFFLOAD_COMPRESS)
        message(FATAL_ERROR "Cannot combine SPIR-V and ROCPRIM_ENABLE_OFFLOAD_COMPRESS")
    endif()
endif()

find_package(hip REQUIRED CONFIG)

if(NOT WIN32)
    include(VerifyCompiler)
endif()

if(ROCPRIM_ENABLE_COVERAGE)
    add_compile_options(-fprofile-arcs -ftest-coverage)
    add_link_options(--coverage)
endif()

# Print configuration summary
include(Summary)
print_configuration_summary()

# rocPRIM library
add_subdirectory(rocprim)

if(NOT ROCPRIM_IS_SUBPROJECT AND (ROCPRIM_BUILD_TESTING OR ROCPRIM_ENABLE_BENCHMARK))
    rocm_package_setup_component(clients)
endif()

# Tests
if(ROCPRIM_BUILD_TESTING)
    if(ROCPRIM_USE_SYSTEM_LIBS)
        find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/rocprim")
        if(NOT ${rocprim_VERSION} VERSION_EQUAL ${VERSION_STRING})
            message(WARNING "The installed rocprim version, ${rocprim_VERSION}, does not match project version ${VERSION_STRING}. Building tests with ROCPRIM_USE_SYSTEM_LIBS=ON may not work properly.")
        endif()
    endif()

    if(NOT ROCPRIM_IS_SUBPROJECT)
        rocm_package_setup_client_component(tests)
    endif()
    enable_testing()
    add_subdirectory(test)
endif()

# Benchmarks
if(ROCPRIM_ENABLE_BENCHMARK)
    if(NOT ROCPRIM_IS_SUBPROJECT)
        rocm_package_setup_client_component(benchmarks)
    endif()
    add_subdirectory(benchmark)
endif()

# Examples
if(ROCPRIM_ENABLE_EXAMPLES)
    add_subdirectory(example)
endif()

# Docs
if(ROCPRIM_ENABLE_DOCS)
    add_subdirectory(docs)
endif()

# Package
if(NOT ROCPRIM_IS_SUBPROJECT)
    set(HIP_RUNTIME_MINIMUM 4.5.0)
    if(BUILD_ADDRESS_SANITIZER)
        set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan")
    else()
        set(DEPENDS_HIP_RUNTIME "hip-runtime-amd")
    endif()

    rocm_package_add_dependencies(SHARED_DEPENDS "${DEPENDS_HIP_RUNTIME} >= ${HIP_RUNTIME_MINIMUM}")
    rocm_package_add_deb_dependencies(STATIC_DEPENDS "hip-static-dev >= ${HIP_RUNTIME_MINIMUM}")
    rocm_package_add_rpm_dependencies(STATIC_DEPENDS "hip-static-devel >= ${HIP_RUNTIME_MINIMUM}")

    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
    set(CPACK_RPM_PACKAGE_LICENSE "MIT")
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}")

    rocm_create_package(
        NAME rocprim
        DESCRIPTION "rocPRIM is a header-only library that provides HIP parallel primitives."
        MAINTAINER "rocPRIM Maintainer <rocprim-maintainer@amd.com>"
        HEADER_ONLY
    )
endif()

# Additional targets for code coverage
if(ROCPRIM_ENABLE_COVERAGE)
    add_custom_target(coverage
        DEPENDS rocprim
        COMMAND mkdir -p lcoverage
        COMMAND echo "\\#!/bin/bash" > llvm-gcov.sh
        COMMAND echo "\\# THIS FILE HAS BEEN GENERATED" >> llvm-gcov.sh
        COMMAND printf "exec /opt/rocm/llvm/bin/llvm-cov gcov $$\\@" >> llvm-gcov.sh
        COMMAND chmod +x llvm-gcov.sh
    )

    add_custom_command(TARGET coverage
        COMMAND lcov --directory . --base-directory . --gcov-tool ${CMAKE_BINARY_DIR}/llvm-gcov.sh --capture -o lcoverage/raw_main_coverage.info
        COMMAND lcov --remove lcoverage/raw_main_coverage.info "'/opt/*'" "'/usr/*'" -o lcoverage/main_coverage.info
        COMMAND genhtml lcoverage/main_coverage.info --output-directory lcoverage
    )

    add_custom_target(coverage_cleanup
        COMMAND find ${CMAKE_BINARY_DIR} -name *.gcda -delete
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()
