# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.5)

# Load backward compatibility shim before project() and option() calls
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(hipblas_shim)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix" FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix" FORCE)
    endif()
endif()

if(NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
endif()

# Export compile commands for tooling support
option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compile_commands.json for clang tooling support" ON)

project(hipblas LANGUAGES CXX)

# Load ROCm CMake dependencies
include(dependencies)
include(GNUInstallDirs)
include(CMakeDependentOption)

set(VERSION_STRING "3.2.0")
rocm_setup_version(VERSION ${VERSION_STRING})

find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)

# Build options
option(HIPBLAS_BUILD_SHARED_LIBS "Build hipBLAS as a shared library" ON)
option(HIPBLAS_ENABLE_SOLVER "Add additional functions from rocSOLVER" ON)
option(HIPBLAS_BUILD_COVERAGE "Build with code coverage enabled" OFF)
option(HIPBLAS_ENABLE_ASAN "Build with address sanitizer enabled" OFF)
option(HIPBLAS_BUILD_DOCS "Build documentation" OFF)

# Backend selection options
option(HIPBLAS_ENABLE_HIP "Build hipBLAS with HIP backend" ON)
option(HIPBLAS_ENABLE_CUDA "Build hipBLAS with CUDA backend" OFF)

if(HIPBLAS_ENABLE_HIP AND HIPBLAS_ENABLE_CUDA)
    message(FATAL_ERROR "Cannot enable both HIP and CUDA backends simultaneously. "
            "Set HIPBLAS_ENABLE_HIP=OFF or HIPBLAS_ENABLE_CUDA=OFF.")
endif()

if(NOT HIPBLAS_ENABLE_HIP AND NOT HIPBLAS_ENABLE_CUDA)
    message(FATAL_ERROR "At least one backend must be enabled. "
            "Set HIPBLAS_ENABLE_HIP=ON or HIPBLAS_ENABLE_CUDA=ON.")
endif()

# Client build options
option(HIPBLAS_ENABLE_CLIENT "Build hipBLAS clients" ON)
if(HIPBLAS_ENABLE_CLIENT)
    cmake_dependent_option(HIPBLAS_BUILD_TESTING "Build hipBLAS unit tests" ON "HIPBLAS_ENABLE_CLIENT" OFF)
    cmake_dependent_option(HIPBLAS_ENABLE_BENCHMARKS "Build hipBLAS benchmarks" ON "HIPBLAS_ENABLE_CLIENT" OFF)
    cmake_dependent_option(HIPBLAS_ENABLE_SAMPLES "Build hipBLAS samples" ON "HIPBLAS_ENABLE_CLIENT" OFF)
    cmake_dependent_option(HIPBLAS_ENABLE_FORTRAN "Build Fortran clients" ON "HIPBLAS_ENABLE_CLIENT;NOT WIN32" OFF)
    cmake_dependent_option(HIPBLAS_ENABLE_OPENMP "Enable OpenMP support" ON "HIPBLAS_ENABLE_CLIENT" OFF)
    cmake_dependent_option(HIPBLAS_ENABLE_BLIS "Link AOCL BLIS reference library" ON "HIPBLAS_ENABLE_CLIENT;NOT HIPBLAS_ENABLE_CUDA" OFF)
endif()

# ASAN and Fortran are incompatible (Clang ASAN runtime can't link with gfortran)
if(HIPBLAS_ENABLE_FORTRAN AND HIPBLAS_ENABLE_ASAN)
    message(FATAL_ERROR "Cannot enable Fortran clients and address sanitizers. "
            "Set HIPBLAS_ENABLE_FORTRAN=OFF or HIPBLAS_ENABLE_ASAN=OFF.")
endif()

# Coverage validation checks
if(HIPBLAS_BUILD_COVERAGE)
    if(HIPBLAS_ENABLE_FORTRAN)
        message(FATAL_ERROR "Cannot enable Fortran clients and code coverage. "
                "Set HIPBLAS_ENABLE_FORTRAN=OFF or HIPBLAS_BUILD_COVERAGE=OFF.")
    elseif(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(FATAL_ERROR "Code coverage is only supported for CMAKE_BUILD_TYPE=Debug. "
                "Set CMAKE_BUILD_TYPE=Debug when using HIPBLAS_BUILD_COVERAGE=ON.")
    endif()
endif()

# Enable Fortran language if building Fortran clients
if(HIPBLAS_ENABLE_FORTRAN)
    enable_language(Fortran)
endif()

# Find HIP package
find_package(hip CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm)

# Support for CUDA backend with hip < 6.0
if(NOT hip_FOUND)
    find_package(HIP MODULE REQUIRED)
    list(APPEND HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include")
endif()

# Find CUDA for CUDA backend
if(HIPBLAS_ENABLE_CUDA)
    find_package(CUDA REQUIRED)
endif()

# Build library
add_subdirectory(library)

# Build clients
if(HIPBLAS_ENABLE_CLIENT)
    # Determine OS for package dependencies
    if(NOT CLIENTS_OS)
        rocm_set_os_id(CLIENTS_OS)
        string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
        rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
    endif()
    message(STATUS "OS: ${CLIENTS_OS} ${CLIENTS_OS_VERSION}")

    # Determine Fortran package dependencies
    if(HIPBLAS_ENABLE_FORTRAN)
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
    endif()

    # Setup package components
    rocm_package_setup_component(runtime)
    
    # Setup client components for packaging
    rocm_package_setup_component(clients)
    rocm_package_setup_client_component(clients-common)
    
    if(HIPBLAS_BUILD_TESTING)
        rocm_package_setup_client_component(tests DEPENDS COMPONENT clients-common)
    endif()
    
    if(HIPBLAS_ENABLE_BENCHMARKS)
        rocm_package_setup_client_component(benchmarks DEPENDS COMPONENT clients-common)
    endif()
    
    if(HIPBLAS_ENABLE_SAMPLES)
        rocm_package_setup_client_component(samples DEPENDS COMPONENT clients-common)
    endif()

    # Add Fortran dependencies to packages
    if(HIPBLAS_ENABLE_FORTRAN)
        if(HIPBLAS_BUILD_TESTING)
            rocm_package_add_rpm_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_RPM}")
            rocm_package_add_deb_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_DEB}")
        endif()
        if(HIPBLAS_ENABLE_BENCHMARKS)
            rocm_package_add_rpm_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_RPM}")
            rocm_package_add_deb_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_DEB}")
        endif()
        if(HIPBLAS_ENABLE_SAMPLES)
            rocm_package_add_rpm_dependencies(COMPONENT samples DEPENDS "${GFORTRAN_RPM}")
            rocm_package_add_deb_dependencies(COMPONENT samples DEPENDS "${GFORTRAN_DEB}")
        endif()
    endif()

    add_subdirectory(clients)
endif()

# Build documentation
if(HIPBLAS_BUILD_DOCS)
    add_subdirectory(docs)
endif()

# Packaging configuration
if(WIN32)
    set(CPACK_SOURCE_GENERATOR "ZIP")
    set(CPACK_GENERATOR "ZIP")
endif()

# Package-specific dependencies
if(HIPBLAS_ENABLE_HIP)
    set(rocblas_minimum 5.2.0)
    set(rocsolver_minimum 3.32.0)
    rocm_package_add_dependencies(SHARED_DEPENDS "rocblas >= ${rocblas_minimum}" "rocsolver >= ${rocsolver_minimum}")
    rocm_package_add_rpm_dependencies(STATIC_DEPENDS "rocblas-static-devel >= ${rocblas_minimum}" "rocsolver-static-devel >= ${rocsolver_minimum}")
    rocm_package_add_deb_dependencies(STATIC_DEPENDS "rocblas-static-dev >= ${rocblas_minimum}" "rocsolver-static-dev >= ${rocsolver_minimum}")
endif()

set(hipblas_common_minimum 1.4.0)

if(HIPBLAS_BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
    rocm_package_add_deb_dependencies(COMPONENT devel DEPENDS "hipblas-common-dev >= ${hipblas_common_minimum}")
    rocm_package_add_rpm_dependencies(COMPONENT devel DEPENDS "hipblas-common-devel >= ${hipblas_common_minimum}")
else()
    rocm_package_add_deb_dependencies(COMPONENT devel DEPENDS "hipblas-common-static-dev >= ${hipblas_common_minimum}")
    rocm_package_add_rpm_dependencies(COMPONENT devel DEPENDS "hipblas-common-static-devel >= ${hipblas_common_minimum}")
endif()

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

if(WIN32)
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE)
    endif()
    set(INSTALL_PREFIX "C:/hipSDK")
    set(CPACK_SET_DESTDIR FALSE)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK")
    set(CPACK_PACKAGING_INSTALL_PREFIX "")
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
else()
    if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
        set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
    endif()
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}")

# Give hipblas compiled for CUDA backend a different name
if(HIPBLAS_ENABLE_HIP)
    set(package_name hipblas)
else()
    set(package_name hipblas-alt)
endif()

set(HIPBLAS_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "ROCm BLAS marshalling library"
    MAINTAINER "hipBLAS Maintainer <hipblas-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${HIPBLAS_CONFIG_DIR}
)

# Code coverage targets
if(HIPBLAS_BUILD_COVERAGE)
    # Use generator expression instead of checking CMAKE_BUILD_TYPE directly
    add_custom_target(coverage_analysis
        COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
        COMMAND ${CMAKE_COMMAND} -E make_directory ./coverage-report/profraw
        COMMAND ${CMAKE_COMMAND} -E env LLVM_PROFILE_FILE="./coverage-report/profraw/hipblas-coverage_%m.profraw" $<TARGET_FILE:hipblas-test> --gtest_filter=\"\${GTEST_FILTER}\"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    add_dependencies(coverage_analysis hipblas hipblas-test)

    find_program(LLVM_PROFDATA llvm-profdata REQUIRED HINTS ${ROCM_PATH}/llvm/bin PATHS /opt/rocm/llvm/bin)
    find_program(LLVM_COV llvm-cov REQUIRED HINTS ${ROCM_PATH}/llvm/bin PATHS /opt/rocm/llvm/bin)

    add_custom_target(coverage
        DEPENDS coverage_analysis
        COMMAND ${LLVM_PROFDATA} merge -sparse ./coverage-report/profraw/hipblas-coverage_*.profraw -o ./coverage-report/hipblas.profdata
        COMMAND ${LLVM_COV} report -object $<TARGET_FILE:hipblas> -instr-profile=./coverage-report/hipblas.profdata
        COMMAND ${LLVM_COV} show -object $<TARGET_FILE:hipblas> -instr-profile=./coverage-report/hipblas.profdata -format=html -output-dir=coverage-report
        COMMAND ${LLVM_COV} export -object $<TARGET_FILE:hipblas> -instr-profile=./coverage-report/hipblas.profdata -format=lcov > ./coverage-report/coverage.info
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    add_custom_target(coverage_cleanup
        COMMAND find ${CMAKE_BINARY_DIR} -name *.gcda -delete
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()
