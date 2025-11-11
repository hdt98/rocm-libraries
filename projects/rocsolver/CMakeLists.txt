# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.13)

# This has to be initialized before the project() command appears
# Set the default build type to Release
if(NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
endif()

if(NOT DEFINED CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC})
    set(CMAKE_Fortran_COMPILER "gfortran")
endif()

# ROCM_BUILD_ID is added to the package name by rocm-cmake. Unsetting it prevents that.
unset(ENV{ROCM_BUILD_ID})

# Disable ROCMClang detection to make CMake v3.21 work the same as CMake v3.20 and earlier.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21.0 AND CMAKE_VERSION VERSION_LESS 3.21.3)
    set(__skip_rocmclang ON)
endif()
message(STATUS "Using CMake ${CMAKE_VERSION}")

# Add our CMake helper files to the lookup path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Load backward compatibility shim for legacy option names
include(rocsolver_shim)

# Project command without VERSION - let rocm_setup_version control versioning
project(rocsolver LANGUAGES CXX C)

# Set C++ standard globally (matches original build)
# Note: While modern CMake recommends per-target compile_features,
# we keep global standard to match reference build exactly
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# Enable Fortran on UNIX for LAPACK
if(UNIX)
    enable_language(Fortran)
endif()

# Project-specific options with ROCSOLVER_ prefix
option(ROCSOLVER_EMBED_FMT "Hide libfmt symbols" ON)
option(OPTIMAL "Build specialized kernels for small matrix sizes" ON)
option(ROCSOLVER_FIND_PACKAGE_LAPACK_CONFIG "Skip module mode search for LAPACK" ON)
option(ROCSOLVER_ENABLE_INTERNAL_BLAS "Use internal implementation of GEMM and TRSM for debugging" OFF)
option(ROCSOLVER_ENABLE_REFERENCE_SECULAR_SOLVER "Use LAPACK like implementation of secular equations solver" ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(fmt REQUIRED)

# Get rocm-cmake
include(get-rocm-cmake)

# Include the rocm-cmake components we use
include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMCheckTargetIds)
include(ROCMClients)
include(ROCMHeaderWrapper)
include(GNUInstallDirs)

include(os-detection)
get_os_id(OS_ID)
message(STATUS "OS detected is ${OS_ID}")

# Versioning via rocm-cmake
set(VERSION_STRING "3.32.0")
rocm_setup_version(VERSION ${VERSION_STRING})

# Workaround until llvm and hip CMake modules fix symlink logic in their config files
list(APPEND CMAKE_PREFIX_PATH
    ${ROCM_PATH}
    ${ROCM_PATH}/llvm
    ${ROCM_PATH}/hip
    /opt/rocm
    /opt/rocm/llvm
    /opt/rocm/hip
)

# Armor level configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DEFAULT_ARMOR_LEVEL 1)
else()
    set(DEFAULT_ARMOR_LEVEL 0)
endif()
set(ARMOR_LEVEL "${DEFAULT_ARMOR_LEVEL}" CACHE STRING "Enables increasingly expensive runtime correctness checks")
include(armor-config)

# This option only works for make, nmake and ninja, but no reason it shouldn't be on all the time
# It creates a compile_commands.json file for use with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# BUILD_SHARED_LIBS is a cmake built-in
# Make it an explicit option such that it shows in cmake-gui
option(BUILD_SHARED_LIBS "Build rocSOLVER as a shared library" ON)

# Include helper functions and wrapper functions
include(util)
include(CheckLanguage)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)

# Standard CMake options with ROCSOLVER_ prefix
option(ROCSOLVER_BUILD_TESTING "Build rocSOLVER tests" OFF)
if(ROCSOLVER_BUILD_TESTING OR BUILD_TESTING)
    enable_testing()
endif()

# Build type-specific sparse default
if(BUILD_SHARED_LIBS)
    set(ROCSOLVER_ENABLE_SPARSE_DEFAULT OFF)
else()
    set(ROCSOLVER_ENABLE_SPARSE_DEFAULT ON)
endif()

# Build options with ROCSOLVER_ prefix
option(ROCSOLVER_BUILD_LIBRARY "Build rocSOLVER library" ON)
option_opposite(ROCSOLVER_BUILD_LIBRARY SKIP_LIBRARY)
option(ROCSOLVER_ENABLE_SPARSE "Build with rocsparse available at build time" "${ROCSOLVER_ENABLE_SPARSE_DEFAULT}")
option(ROCSOLVER_ENABLE_TESTS "Build rocSOLVER test client" "${ROCSOLVER_BUILD_TESTING}")
option(ROCSOLVER_ENABLE_BENCHMARKS "Build rocSOLVER benchmark client" OFF)
option(ROCSOLVER_ENABLE_SAMPLES "Build rocSOLVER samples" OFF)
cmake_dependent_option(ROCSOLVER_ENABLE_EXTRA_TESTS "Build extra tests" OFF ROCSOLVER_BUILD_TESTING OFF)
option(ROCSOLVER_ENABLE_ASAN "Build with address sanitizer enabled" OFF)
option(ROCSOLVER_ENABLE_COVERAGE "Build rocSOLVER with code coverage enabled" OFF)
option(ROCSOLVER_ENABLE_WERROR "Treat warnings as errors" OFF)
option(ROCSOLVER_ENABLE_COMPRESSED_DBG "Enable compressed debug symbols" ON)
check_cxx_compiler_flag("--offload-compress" CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
cmake_dependent_option(ROCSOLVER_ENABLE_OFFLOAD_COMPRESS "Build with offload compression" ON CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS OFF)

message(STATUS "Tests: ${ROCSOLVER_ENABLE_TESTS}")
message(STATUS "Benchmarks: ${ROCSOLVER_ENABLE_BENCHMARKS}")
message(STATUS "Samples: ${ROCSOLVER_ENABLE_SAMPLES}")

check_language(HIP)
cmake_dependent_option(ROCSOLVER_ENABLE_HIPCXX "Use CMake HIP language support" OFF CMAKE_HIP_COMPILER OFF)
if(ROCSOLVER_ENABLE_HIPCXX)
    enable_language(HIP)
else()
    if(NOT DEFINED AMDGPU_TARGETS)
        set(XNACK_PLUS_TARGETS
            gfx90a:xnack+
            gfx908:xnack+
            gfx942:xnack+
            gfx950:xnack+)
        set(XNACK_MINUS_TARGETS gfx90a:xnack-)
        set(MISC_TARGETS
            gfx942
            gfx950
            gfx10-1-generic
            gfx10-3-generic
            gfx1100
            gfx1101
            gfx1102
            gfx1151
            gfx12-generic
        )
        if(ROCSOLVER_ENABLE_ASAN)
            set(OPTIONAL_TARGETS_QUERY ${XNACK_PLUS_TARGETS})
        else()
            set(OPTIONAL_TARGETS_QUERY ${XNACK_MINUS_TARGETS} ${MISC_TARGETS})
            set(DEFAULT_TARGETS
                gfx900
                gfx906:xnack-
                gfx908:xnack-
            )
        endif()
        # Query for compiler support of GPU archs
        rocm_check_target_ids(OPTIONAL_AMDGPU_TARGETS TARGETS ${OPTIONAL_TARGETS_QUERY})
        set(AMDGPU_TARGETS_INIT ${OPTIONAL_AMDGPU_TARGETS} ${DEFAULT_TARGETS})
    endif()

    # Set this before finding hip so that hip::device has the required arch flags
    # added as usage requirements on its interface
    set(AMDGPU_TARGETS "${AMDGPU_TARGETS_INIT}"
        CACHE STRING "List of specific machine types for library to target")
endif()

# Find HIP dependencies
find_package(hip REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm)

find_package(rocblas REQUIRED CONFIG PATHS ${ROCM_PATH})
get_imported_target_location(location roc::rocblas)
message(STATUS "Found rocBLAS: ${location}")
set(rocblas_minimum 5.1)
rocm_package_add_dependencies(SHARED_DEPENDS "rocblas >= ${rocblas_minimum}")
rocm_package_add_rpm_dependencies(STATIC_DEPENDS "rocblas-static-devel >= ${rocblas_minimum}")
rocm_package_add_deb_dependencies(STATIC_DEPENDS "rocblas-static-dev >= ${rocblas_minimum}")

if(ROCSOLVER_ENABLE_SPARSE)
    find_package(rocsparse REQUIRED CONFIG PATHS ${ROCM_PATH})
    get_imported_target_location(location roc::rocsparse)
    message(STATUS "Found rocSPARSE: ${location}")
    set(rocsparse_minimum 2.2)
    rocm_package_add_dependencies(SHARED_DEPENDS "rocsparse >= ${rocsparse_minimum}")
    rocm_package_add_rpm_dependencies(STATIC_DEPENDS "rocsparse-static-devel >= ${rocsparse_minimum}")
    rocm_package_add_deb_dependencies(STATIC_DEPENDS "rocsparse-static-dev >= ${rocsparse_minimum}")
else()
    list(APPEND CPACK_DEBIAN_RUNTIME_PACKAGE_RECOMMENDS "rocsparse")
    list(APPEND CPACK_RPM_RUNTIME_PACKAGE_SUGGESTS "rocsparse")
endif()

find_package(rocprim REQUIRED CONFIG PATHS ${ROCM_PATH})
rocm_package_add_rpm_dependencies(STATIC_DEPENDS "rocprim-static-devel")
rocm_package_add_deb_dependencies(STATIC_DEPENDS "rocprim-static-dev")

add_subdirectory(common)

if(ROCSOLVER_BUILD_LIBRARY)
    add_subdirectory(library)
endif()

if(ROCSOLVER_ENABLE_TESTS OR ROCSOLVER_ENABLE_BENCHMARKS OR ROCSOLVER_ENABLE_SAMPLES)
    if(NOT CLIENTS_OS)
        rocm_set_os_id(CLIENTS_OS)
        string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
        rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
    endif()
    set(GFORTRAN_RPM "libgfortran4")
    set(GFORTRAN_DEB "libgfortran4")
    if(CLIENTS_OS STREQUAL "centos" OR CLIENTS_OS STREQUAL "rhel" OR CLIENTS_OS STREQUAL "almalinux")
        if(CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "8")
            set(GFORTRAN_RPM "libgfortran")
        endif()
    elseif(CLIENTS_OS STREQUAL "ubuntu" AND CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "20.04")
        set(GFORTRAN_DEB "libgfortran5")
    elseif(CLIENTS_OS STREQUAL "mariner" OR CLIENTS_OS STREQUAL "azurelinux")
        set(GFORTRAN_RPM "gfortran")
    endif()
    rocm_package_setup_component(clients)
    if(ROCSOLVER_ENABLE_TESTS)
        rocm_package_setup_client_component(tests DEPENDS DEB "${GFORTRAN_DEB}" RPM "${GFORTRAN_RPM}")
    endif()
    if(ROCSOLVER_ENABLE_BENCHMARKS)
        rocm_package_setup_client_component(benchmarks DEPENDS DEB "${GFORTRAN_DEB}" RPM "${GFORTRAN_RPM}")
    endif()
    add_subdirectory(clients)
endif()

if(OS_ID_sles)
    rocm_package_add_rpm_dependencies("libLLVM >= 7.0.1")
endif()

# Installation - moved from library/src/CMakeLists.txt
set(package_targets rocsolver)

rocm_install(
    DIRECTORY
        "${PROJECT_SOURCE_DIR}/library/include/rocsolver/"
        "${PROJECT_BINARY_DIR}/include/rocsolver/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocsolver
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)

rocm_install_targets(
    TARGETS ${package_targets}
    INCLUDE
        ${CMAKE_SOURCE_DIR}/library/include
        ${CMAKE_BINARY_DIR}/include
)

# Prepare static dependencies list for export
set(static_depends)
if(ROCSOLVER_ENABLE_SPARSE)
    list(APPEND static_depends PACKAGE rocsparse)
endif()
if(NOT ROCSOLVER_EMBED_FMT)
    list(APPEND static_depends PACKAGE fmt)
endif()

rocm_export_targets(
    TARGETS roc::rocsolver
    DEPENDS
        PACKAGE hip
        PACKAGE rocblas
    STATIC_DEPENDS
        ${static_depends}
    NAMESPACE roc::
)

# Packaging configuration
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "BSD")

if(WIN32)
    set(CPACK_SOURCE_GENERATOR "ZIP")
    set(CPACK_GENERATOR "ZIP")
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE)
    endif()
    set(INSTALL_PREFIX "C:/hipSDK")
    set(CPACK_SET_DESTDIR OFF)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK")
    set(CPACK_PACKAGING_INSTALL_PREFIX "")
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
else()
    if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
        set(CPACK_PACKAGING_INSTALL_PREFIX "${ROCM_PATH}")
    endif()
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
    "\${CPACK_PACKAGING_INSTALL_PREFIX}"
)

set(ROCSOLVER_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
    CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
    NAME rocsolver
    DESCRIPTION "AMD ROCm SOLVER library"
    MAINTAINER "RocSOLVER maintainer <rocsolver-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${ROCSOLVER_CONFIG_DIR}
)

# Code Coverage Build Commands:
#   make coverage_cleanup (clean coverage related files)
#   make coverage GTEST_FILTER=<>
#   make coverage_analysis GTEST_FILTER=<> (analyze tests)
#   make coverage_output (generate html documentation)
if(ROCSOLVER_ENABLE_COVERAGE)
    set(coverage_test ./clients/staging/rocsolver-test)

    add_custom_target(
        coverage_analysis
        DEPENDS rocsolver-test
        COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
        COMMAND ${CMAKE_COMMAND} -E rm -rf ./coverage-report
        COMMAND ${CMAKE_COMMAND} -E make_directory ./coverage-report/profraw
        COMMAND ${CMAKE_COMMAND} -E env 
            LLVM_PROFILE_FILE="./coverage-report/profraw/rocSOLVER-coverage_%p.profraw" 
            GTEST_LISTENER=NO_PASS_LINE_IN_LOG 
            ${coverage_test} --gtest_filter=\"\${GTEST_FILTER}\"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    find_program(
        LLVM_PROFDATA
        llvm-profdata
        REQUIRED
        HINTS ${ROCM_PATH}/llvm/bin
        PATHS /opt/rocm/llvm/bin
    )

    find_program(
        LLVM_COV
        llvm-cov
        REQUIRED
        HINTS ${ROCM_PATH}/llvm/bin
        PATHS /opt/rocm/llvm/bin
    )

    add_custom_target(
        coverage
        DEPENDS coverage_analysis
        COMMAND ${LLVM_PROFDATA} merge -sparse 
            ./coverage-report/profraw/rocSOLVER-coverage_*.profraw 
            -o ./coverage-report/rocSOLVER.profdata
        COMMAND ${LLVM_COV} report 
            -object ./library/src/librocsolver.so 
            -instr-profile=./coverage-report/rocSOLVER.profdata
        COMMAND ${LLVM_COV} show 
            -object ./library/src/librocsolver.so 
            -instr-profile=./coverage-report/rocSOLVER.profdata 
            -format=html 
            -output-dir=coverage-report
        COMMAND ${LLVM_COV} export 
            -object ./library/src/librocsolver.so 
            -instr-profile=./coverage-report/rocSOLVER.profdata 
            -format=lcov > ./coverage-report/coverage.info
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()
