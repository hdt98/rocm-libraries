# ########################################################################
# Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ########################################################################

# ROCm software requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
# cmake_minimum_required( VERSION 3.5 )
cmake_minimum_required(VERSION 3.21)
project(hipblas VERSION 3.1.0 LANGUAGES CXX)

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW) # option() honors normal variables
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")

include(FetchROCmCMake)
include(CMakeDependentOption)

rocm_setup_version(VERSION ${PROJECT_VERSION})

# We use C++14 features, this will add compile option: -std=c++14
# set( CMAKE_CXX_STANDARD 17 )

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  if(WIN32)
    set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path prefix." FORCE)
  else()
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix." FORCE)
  endif()
endif()

# ============================================================================
# Build Options and Validation
# ============================================================================
option(HIPBLAS_ENABLE_CUDA "Build hipBLAS with CUDA backend." OFF)
cmake_dependent_option(
  HIPBLAS_ENABLE_HIP "Build hipSPARSELt with HIP backend." ON "NOT HIPSPARSELT_ENABLE_CUDA" OFF
)

option(HIPBLAS_BUILD_SHARED_LIBS "Build hipBLAS as a shared library." ON)
option(HIPBLAS_ENABLE_CLIENTS "Build hipBLAS clients (tests, benchmarks, samples)." ON)

if(HIPBLAS_ENABLE_CLIENTS)
  option(HIPBLAS_BUILD_TESTING "Build test clients." ON) # TESTING and whenever theres an if u need to put OR BUILD_TESTING
  option(HIPBLAS_ENABLE_BENCHMARKS "Build benchmark clients." ON) 
  option(HIPBLAS_ENABLE_SAMPLES "Build sample clients." ON) 
  option(HIPBLAS_ENABLE_FORTRAN "Build Fortran clients." ON) 
endif()

option(HIPBLAS_ENABLE_ROCSOLVER "Build with rocSOLVER integration." ON)
option(HIPBLAS_BUILD_COVERAGE "Build with code coverage enabled." OFF) 
option(HIPBLAS_ENABLE_ASAN "Build with address sanitizer enabled." OFF) # doesnt work, test in develop
option(HIPBLAS_ENABLE_DOCS "Build documentation." OFF) 
option(HIPBLAS_ENABLE_VERBOSE "Output additional build information." OFF) #e probably not needed

# CXX_STANDARD?

if(HIPBLAS_ENABLE_CUDA AND HIPBLAS_ENABLE_HIP)
  message(STATUS "HIPBLAS_ENABLE_CUDA: ${HIPBLAS_ENABLE_CUDA}")
  message(STATUS "HIPBLAS_ENABLE_HIP: ${HIPBLAS_ENABLE_HIP}")
  message(FATAL_ERROR "Cannot enable CUDA and HIP support simultaneously.")
endif()

if(HIPBLAS_ENABLE_FORTRAN AND HIPBLAS_ENABLE_ASAN)
  # message(FATAL_ERROR "Cannot enable fortran clients and address sanitizers.")
endif() #test this. also need to test ASAN ON, fortran off.
# FORTRAN OFF doesnt work. ASAN

if(HIPBLAS_BUILD_COVERAGE AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(FATAL_ERROR "Code coverage is only supported for CMAKE_BUILD_TYPE=Debug.")
endif()

if(HIPBLAS_ENABLE_FORTRAN)
  enable_language(Fortran)
endif()

# =====================================
# Dependency Setup
# =======================================
find_package(Python3 REQUIRED COMPONENTS Interpreter) # move to client/tests
find_package(hip REQUIRED)

# Should be linked to a target:
if(HIPBLAS_ENABLE_CUDA)
  find_package(CUDA REQUIRED)
  # set( HIPBLAS_HIP_PLATFORM_COMPILER_DEFINES __HIP_PLATFORM_NVCC__ __HIP_PLATFORM_NVIDIA__ )
endif()
# needs to happen before library add_sub (well move it inside):
if(HIPBLAS_ENABLE_ASAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
endif()

include( clients/cmake/build-options.cmake ) # can we put this logic here?

# ============================================================================
# Project Configuration and Build
# ============================================================================
add_subdirectory(library)

if(HIPBLAS_ENABLE_CLIENTS)
  add_subdirectory(clients)
endif()

if(HIPBLAS_ENABLE_DOCS)
  add_subdirectory(docs)
endif()

# ============================================================================
# Packaging and Installation
# ============================================================================
if(HIPBLAS_ENABLE_CLIENTS)
  rocm_set_os_id(CLIENTS_OS)
  string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
  rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)

  message(STATUS "OS: ${CLIENTS_OS} ${CLIENTS_OS_VERSION}")
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
  rocm_package_setup_client_component(clients-common)
  if(HIPBLAS_BUILD_TESTING)
    rocm_package_setup_client_component(
      tests
      DEPENDS
        COMPONENT clients-common
        DEB "${GFORTRAN_DEB}"
        RPM "${GFORTRAN_RPM}")
  endif()
  if(HIPBLAS_ENABLE_BENCHMARKS)
    rocm_package_setup_client_component(
      benchmarks
      DEPENDS
        COMPONENT clients-common
        DEB "${GFORTRAN_DEB}"
        RPM "${GFORTRAN_RPM}")
  endif()
  if(HIPBLAS_ENABLE_SAMPLES)
    rocm_package_setup_client_component(
      samples
      DEPENDS
        COMPONENT clients-common
        DEB "${GFORTRAN_DEB}"
        RPM "${GFORTRAN_RPM}")
  endif()
endif( )

if(HIPBLAS_ENABLE_HIP)
  set(rocblas_minimum 5.1.0)
  set(rocsolver_minimum 3.31.0)
  rocm_package_add_dependencies(SHARED_DEPENDS "rocblas >= ${rocblas_minimum}" "rocsolver >= ${rocsolver_minimum}")
  rocm_package_add_rpm_dependencies(STATIC_DEPENDS "rocblas-static-devel >= ${rocblas_minimum}" "rocsolver-static-devel >= ${rocsolver_minimum}")
  rocm_package_add_deb_dependencies(STATIC_DEPENDS "rocblas-static-dev >= ${rocblas_minimum}" "rocsolver-static-dev >= ${rocsolver_minimum}")
endif( )

set(hipblas_common_minimum 1.3.0)

if(HIPBLAS_BUILD_SHARED_LIBS)
  rocm_package_add_deb_dependencies(COMPONENT devel DEPENDS "hipblas-common-dev >= ${hipblas_common_minimum}")
  rocm_package_add_rpm_dependencies(COMPONENT devel DEPENDS "hipblas-common-devel >= ${hipblas_common_minimum}")
else()
  rocm_package_add_deb_dependencies(COMPONENT devel DEPENDS "hipblas-common-static-dev >= ${hipblas_common_minimum}")
  rocm_package_add_rpm_dependencies(COMPONENT devel DEPENDS "hipblas-common-static-devel >= ${hipblas_common_minimum}")
endif()

set( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" )
# check if can del (cuda in general):
if(HIPBLAS_ENABLE_CUDA)
  set(package_name "hipblas-alt")
else()
  set(package_name "hipblas")
endif()

set( HIPBLAS_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file" )

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "ROCm BLAS marshalling library"
    MAINTAINER "hipBLAS Maintainer <hipblas-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${HIPBLAS_CONFIG_DIR}
)

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
# if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
#   set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
# endif()

# if (NOT WIN32)
#     if ( NOT DEFINED CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC} )
#       set( CMAKE_Fortran_COMPILER  "gfortran" )
#     endif()
#     set( fortran_language "Fortran" )
# endif( )

# project( hipblas LANGUAGES CXX ${fortran_language} )

# if (NOT python)
#     set(python "python3") # default for linux
# endif()

# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
# list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake  ${ROCM_PATH}/lib/cmake/hip /opt/rocm/lib/cmake/hip ${HIP_DIR}/cmake )

# NOTE:  workaround until hip cmake modules fixes symlink logic in their config files; remove when fixed
# this should not be set to default:
# list( APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake/hip /opt/rocm /opt/rocm/llvm /opt/rocm/hip )

# This finds the rocm-cmake project, and installs it if not found
# rocm-cmake contains common cmake code for rocm projects to help setup and install
# include(dependencies)

# set ( VERSION_STRING "3.1.0" )
# rocm_setup_version( VERSION ${VERSION_STRING} )

# option( BUILD_VERBOSE "Output additional build information" OFF )

# option( BUILD_WITH_SOLVER "Add additional functions from rocSOLVER" ON )

# Should be linked to a target:
# if(HIPBLAS_BUILD_SOLVER)
#     add_definitions( -D__HIP_PLATFORM_SOLVER__ )
# endif( )

# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option such that it shows in cmake-gui
# option( BUILD_SHARED_LIBS "Build hipBLAS as a shared library" ON )

# Deprecated USE_CUDA option
# if(DEFINED USE_CUDA)
#   if(USE_CUDA)
#     set(ENV{HIP_PLATFORM} nvidia)
#     message(DEPRECATED "USE_CUDA is deprecated (use environment variable HIP_PLATFORM=nvidia)")
#   else()
#     set(ENV{HIP_PLATFORM} amd)
#     message(DEPRECATION "USE_CUDA is deprecated (use environment variable HIP_PLATFORM=amd)")
#   endif()
# endif()

# # Hip headers required of all clients; clients use hip to allocate device memory
# # find_package( hip CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm )

# # support for cuda backend with hip < 6.0
# if(NOT hip_FOUND)
#     find_package( HIP MODULE REQUIRED )
#     list( APPEND HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include" )
#     set(ENV{HIP_PLATFORM} nvidia)
#     if (NOT HIP_PLATFORM)
#       set(HIP_PLATFORM nvidia)
#     endif()
# else()
#     if (NOT HIP_PLATFORM)
#       set(HIP_PLATFORM amd)
#     endif()
# endif()
# # Should be linked to a target:
# if(HIP_PLATFORM STREQUAL nvidia)
#   find_package( CUDA REQUIRED )
#   set( HIPBLAS_HIP_PLATFORM_COMPILER_DEFINES __HIP_PLATFORM_NVCC__ __HIP_PLATFORM_NVIDIA__ )
# endif()

# option(BUILD_CODE_COVERAGE "Build with code coverage enabled" OFF)
# if(BUILD_CODE_COVERAGE)
#   #add_compile_options(-fprofile-arcs -ftest-coverage)
#   #add_link_options(--coverage)
# endif()

# option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)
# Should be linked to a target:
# if(HIPBLAS_BUILD_ASAN)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
# endif()

# # add_subdirectory( library )

# include( clients/cmake/build-options.cmake )

# # Build clients of the library
# if(HIPBLAS_BUILD_CLIENTS)
#   rocm_set_os_id(CLIENTS_OS)
#   string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
#   rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)

#   message(STATUS "OS: ${CLIENTS_OS} ${CLIENTS_OS_VERSION}")
#   set(GFORTRAN_RPM "libgfortran4")
#   set(GFORTRAN_DEB "libgfortran4")
#   if(CLIENTS_OS STREQUAL "centos" OR CLIENTS_OS STREQUAL "rhel" OR CLIENTS_OS STREQUAL "almalinux")
#     if(CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "8")
#       set(GFORTRAN_RPM "libgfortran")
#     endif()
#   elseif(CLIENTS_OS STREQUAL "ubuntu" AND CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "20.04")
#     set(GFORTRAN_DEB "libgfortran5")
#   elseif(CLIENTS_OS STREQUAL "mariner" OR CLIENTS_OS STREQUAL "azurelinux")
#     set(GFORTRAN_RPM "gfortran")
#   endif()

#   # set( BUILD_CLIENTS ON )
#   rocm_package_setup_component(clients)
#   rocm_package_setup_client_component(clients-common)
#   if(HIPBLAS_BUILD_TESTS)
#     rocm_package_setup_client_component(
#       tests
#       DEPENDS
#         COMPONENT clients-common
#         DEB "${GFORTRAN_DEB}"
#         RPM "${GFORTRAN_RPM}")
#   endif()
#   if(HIPBLAS_BUILD_BENCHMARKS)
#     rocm_package_setup_client_component(
#       benchmarks
#       DEPENDS
#         COMPONENT clients-common
#         DEB "${GFORTRAN_DEB}"
#         RPM "${GFORTRAN_RPM}")
#   endif()
#   if(HIPBLAS_BUILD_SAMPLES)
#     rocm_package_setup_client_component(
#       samples
#       DEPENDS
#         COMPONENT clients-common
#         DEB "${GFORTRAN_DEB}"
#         RPM "${GFORTRAN_RPM}")
#   endif()
#   add_subdirectory(clients)
# endif( )

# Build docs
# if(HIPBLAS_BUILD_DOCS)
#   add_subdirectory(docs)
# endif()

# The following code is setting variables to control the behavior of CPack to generate our
 # DO WE EVEN NEED THIS
# if( WIN32 )
#     set( CPACK_SOURCE_GENERATOR "ZIP" )
#     set( CPACK_GENERATOR "ZIP" )
# endif( )

# Package specific CPACK vars
# if(HIP_PLATFORM STREQUAL amd)
#   set(rocblas_minimum 5.1.0)
#   set(rocsolver_minimum 3.31.0)
#   rocm_package_add_dependencies(SHARED_DEPENDS "rocblas >= ${rocblas_minimum}" "rocsolver >= ${rocsolver_minimum}")
#   rocm_package_add_rpm_dependencies(STATIC_DEPENDS "rocblas-static-devel >= ${rocblas_minimum}" "rocsolver-static-devel >= ${rocsolver_minimum}")
#   rocm_package_add_deb_dependencies(STATIC_DEPENDS "rocblas-static-dev >= ${rocblas_minimum}" "rocsolver-static-dev >= ${rocsolver_minimum}")
# endif( )

# set(hipblas_common_minimum 1.3.0)

# if(HIPBLAS_BUILD_SHARED_LIBS)
#   rocm_package_add_deb_dependencies(COMPONENT devel DEPENDS "hipblas-common-dev >= ${hipblas_common_minimum}")
#   rocm_package_add_rpm_dependencies(COMPONENT devel DEPENDS "hipblas-common-devel >= ${hipblas_common_minimum}")
# else()
#   rocm_package_add_deb_dependencies(COMPONENT devel DEPENDS "hipblas-common-static-dev >= ${hipblas_common_minimum}")
#   rocm_package_add_rpm_dependencies(COMPONENT devel DEPENDS "hipblas-common-static-devel >= ${hipblas_common_minimum}")
# endif()

# set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md" )
# set( CPACK_RPM_PACKAGE_LICENSE "MIT")
# this can be changed to what hipsparselt has
# if (WIN32)
#   if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
#     SET( CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE )
#   endif()
#   SET( INSTALL_PREFIX "C:/hipSDK" )
#   SET( CPACK_SET_DESTDIR FALSE )
#   SET( CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK" )
#   SET( CPACK_PACKAGING_INSTALL_PREFIX "" )
#   set( CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF )
# else()
#   if( NOT CPACK_PACKAGING_INSTALL_PREFIX )
#     set( CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" )
#   endif()
# endif( )
# is this needed?
# set( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" )

# Give hipblas compiled for CUDA backend a different name
# if(HIP_PLATFORM STREQUAL amd)
#     set( package_name hipblas )
# else( )
#     set( package_name hipblas-alt )
# endif( )

# set( HIPBLAS_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file" )

# rocm_create_package(
#     NAME ${package_name}
#     DESCRIPTION "ROCm BLAS marshalling library"
#     MAINTAINER "hipBLAS Maintainer <hipblas-maintainer@amd.com>"
#     LDCONFIG
#     LDCONFIG_DIR ${HIPBLAS_CONFIG_DIR}
# )


# DO THIS LATER:
# ADDITIONAL TARGETS FOR CODE COVERAGE
#
if(HIPBLAS_BUILD_COVERAGE)
  #
  # > make coverage_cleanup (clean coverage related files.)
  # > make coverage GTEST_FILTER=<>
  # will run:
  #  > make coverage_analysis GTEST_FILTER=<> (analyze tests)
  #  > make coverage_output (generate html documentation)
  #

  #
  # Run coverage analysis
  #
  set(coverage_test ./clients/staging/hipblas-test)
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(coverage_test ./clients/staging/hipblas-test-d)
  endif()

  add_custom_target(coverage_analysis
    COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
    COMMAND ${coverage_test} --gtest_filter=\"\${GTEST_FILTER}\"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

  add_dependencies(coverage_analysis hipblas)

  #
  # Prepare coverage output
  # This little script is generated because the option '--gcov-tool <program name>' of lcov cannot take arguments.
  #

  # source path substitute adjustment only for CI
  get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)
  get_filename_component(GRANDPARENT_DIR ${PARENT_DIR} DIRECTORY)
  set(base_path "/var/jenkins_home/library")
  set(hash_path "${CMAKE_SOURCE_DIR}/library")
  message(STATUS "BASE ${base_path} to HASHED ${hash_path}")

  add_custom_target(coverage
    DEPENDS coverage_analysis
    COMMAND mkdir -p lcoverage
    COMMAND echo "\\#!/bin/bash" > llvm-gcov.sh
    COMMAND echo "\\# THIS FILE HAS BEEN GENERATED" >> llvm-gcov.sh
    COMMAND printf "exec /opt/rocm/llvm/bin/llvm-cov gcov $$\\@" >> llvm-gcov.sh
    COMMAND chmod +x llvm-gcov.sh
    COMMAND lcov --ignore-errors unused --directory . --base-directory "${CMAKE_SOURCE_DIR}" --gcov-tool ./llvm-gcov.sh --capture --substitute "'s#${base_path}#${hash_path}#g'" -o ./lcoverage/raw_main_coverage.info
    COMMAND lcov --ignore-errors unused --remove ./lcoverage/raw_main_coverage.info "'/opt/*'" "'/usr/*'" -o ./lcoverage/main_coverage.info
    COMMAND genhtml --ignore-errors source ./lcoverage/main_coverage.info --output-directory ./lcoverage --prefix ${CMAKE_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )


  #
  # Coverage cleanup
  #
  add_custom_target(coverage_cleanup
    COMMAND find ${CMAKE_BINARY_DIR} -name *.gcda -delete
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()
