# MIT License
#
# Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

# --------------------------------------
# Update these variables at release time
#
# Set the library version
set(VERSION_STRING "3.3.0")
# Set the minimum required rocPRIM version
set(MIN_ROCRAND_PACKAGE_VERSION "4.0.0" CACHE STRING "Minimum version of rocRAND to search for when ROCRAND_FETCH_METHOD is set to PACKAGE.")
# Set download branch for dependency rocPRIM
set(ROCM_DEP_RELEASE_BRANCH "release/therock-7.13" CACHE STRING "Download branch for ROCm dependencies")
# --------------------------------------

if(DEFINED ENV{ROCM_PATH})
  set (ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "The path to the ROCm installation.")
endif()
set (ROCM_PATH "/opt/rocm" CACHE PATH "The path to the ROCm installation.")

# Install prefix
if (WIN32)
  SET( CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" )
  SET( INSTALL_PREFIX "C:/hipSDK" )
else()
  set(CMAKE_INSTALL_PREFIX "${ROCM_PATH}" CACHE PATH "Install path prefix, prepended onto install directories")
  list( APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm ${ROCM_PATH} )
endif()

#
# hipRAND project
#
project(hipRAND CXX)

# Build options
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)
option(BUILD_BENCHMARK "Build benchmarks" OFF)
option(BUILD_CODE_COVERAGE "Build with code coverage enabled" OFF)
option(BUILD_FORTRAN_WRAPPER "Build Fortran wrapper" OFF)
option(BUILD_TEST "Build tests (requires googletest)" OFF)
option(EXTERNAL_DEPS_FORCE_DOWNLOAD "Download non-ROCm dependencies and do not search for packages" OFF)
option(DISABLE_WERROR "Disable building with Werror" ON)
option(DOWNLOAD_ROCRAND "Download rocRAND if it is not found in the default install directory or the manually-specified directory" OFF)
option(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
option(BUILD_SHARED_LIBS "Build shared" ON)

set(BUILD_WITH_LIB "ROCM" CACHE STRING "Select backend library to build with (ROCM or CUDA)")
set_property(CACHE BUILD_WITH_LIB PROPERTY STRINGS "ROCM" "CUDA")

# Set up options for obtaining dependency rocPRIM.
# PACKAGE: Search for an install package that contains the dependency.
# MONOREPO: Assume this is a monorepo checkout and search for the dependency in the directory at ../../projects/.
# DOWNLOAD: Download the dependency from the monorepo.
set(FETCH_METHOD_OPTIONS "PACKAGE" "MONOREPO" "DOWNLOAD")

set(ROCRAND_FETCH_METHOD "PACKAGE" CACHE STRING "How to obtain the rocRAND dependency")

# This function checks to see if the fetch method variable it's passed is defined, and contains a valid value.
# If it does not contain a valid value, it issues a fatal failure with an error message.
function(check_fetch_method method)
  if (DEFINED ${method} AND NOT ${${method}} IN_LIST FETCH_METHOD_OPTIONS)
    message(FATAL_ERROR "Unrecognized ${method}: \"${${method}}\". Valid options are: ${FETCH_METHOD_OPTIONS}.")
  endif()
endfunction()

check_fetch_method(ROCRAND_FETCH_METHOD)

if (BUILD_WITH_LIB STREQUAL "CUDA")
  # Not relevant for CUDA builds
  set(ROCRAND_PATH "" CACHE INTERNAL "")
  set(ROCRAND_ROOT "" CACHE INTERNAL "")
else()
  set(ROCRAND_PATH "" CACHE PATH "Path to rocRAND install.")
  set(ROCRAND_ROOT "${CMAKE_CURRENT_BINARY_DIR}/deps/rocrand" CACHE PATH "Path to downloaded rocRAND install.")
endif()

set(HIPRAND_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

if (BUILD_TEST)
  set(GTEST_ROOT "${CMAKE_CURRENT_BINARY_DIR}/deps/gtest" CACHE PATH "Path to place downloaded googletest.")
else()
  set(GTEST_ROOT "" CACHE INTERNAL "")
endif()

mark_as_advanced(CMAKE_INSTALL_RPATH_USE_LINK_PATH)
mark_as_advanced(HIPRAND_CONFIG_DIR)

# CMake modules
list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
    ${HIP_DIR}/cmake ${ROCM_PATH}/lib/cmake/hip # FindHIP.cmake
    ${ROCM_PATH}/llvm
)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include(ROCmCMakeBuildToolsDependency)

# Set version variables
rocm_setup_version( VERSION ${VERSION_STRING} )
set ( hiprand_VERSION ${VERSION_STRING} )
# Old-style version number used within the library's API. rocrand_get_version should be modified.
math(EXPR hiprand_VERSION_NUMBER "${hipRAND_VERSION_MAJOR} * 100000 + ${hipRAND_VERSION_MINOR} * 100 + ${hipRAND_VERSION_PATCH}")
set(hiprand_SOVERSION 1.1)

# Set target IDs for compilation
if (NOT BUILD_WITH_LIB STREQUAL "CUDA")
  if (NOT DEFINED AMDGPU_TARGETS)
    set(GPU_TARGETS "all" CACHE STRING "GPU architectures to compile for")
  else()
    set(GPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for")
  endif()
  set_property(CACHE GPU_TARGETS PROPERTY STRINGS "all")

  if(GPU_TARGETS STREQUAL "all")
    rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
      TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201"
    )
    set(GPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for" FORCE)
  endif()
else()
  set(GPU_TARGETS "all" CACHE INTERNAL "")
endif()

# When -fsanitize=address is present (injected by a super-project such as TheRock's
# host-ASAN preset), rewrite it so ASAN applies only to the host compilation pass.
# gfx942 and other targets without xnack+ cannot support device-side ASAN: clang
# silently drops the flag after ASAN metadata accounting has already begun, producing
# a malformed HIP fat binary whose .hipFatBinSegment size fields are inconsistent
# with the actual embedded code object.
#
# The correct fix is -Xarch_host -fsanitize=address: this confines ASAN to the host
# pass only. Negation approaches (-Xarch_device -fno-sanitize=address, or appending
# -fno-sanitize=address) do NOT work — clang's unified driver bakes in ASAN metadata
# before per-arch splitting occurs.
#
# IMPORTANT: This must run before include(VerifyCompiler) because find_package(HIP)
# inside VerifyCompiler locks in the device-compiler flags from CMAKE_CXX_FLAGS at
# that moment.
# Diagnostics: log CMAKE_CXX_FLAGS as injected by the parent (TheRock) before
# any per-project modification. Verifies whether the TheRock HOST_ASAN fix in
# cmake/therock_sanitizers.cmake is injecting -Xarch_host -fsanitize=address
# (correct) or bare -fsanitize=address (TheRock fix not active).
message(STATUS "HOST_ASAN_DIAG CMAKE_CXX_FLAGS on entry (from TheRock): ${CMAKE_CXX_FLAGS}")

if(CMAKE_CXX_FLAGS MATCHES "-Xarch_host -fsanitize=address")
  message(STATUS "HOST_ASAN_DIAG TheRock fix ACTIVE: -Xarch_host -fsanitize=address confirmed in CMAKE_CXX_FLAGS")
elseif(CMAKE_CXX_FLAGS MATCHES "-fsanitize=address" OR BUILD_ADDRESS_SANITIZER)
  # Bare -fsanitize=address detected without -Xarch_host: TheRock fix is not
  # active. The per-project replacement is currently disabled (see below).
  # If this warning fires during a HOST_ASAN build, re-enable the commented
  # string(REPLACE ...) block below.
  message(WARNING "HOST_ASAN_DIAG bare -fsanitize=address detected without -Xarch_host: TheRock fix is NOT active. Re-enable the string(REPLACE) below.")
  # Per-project fix (disabled: TheRock now owns this via therock_sanitizers.cmake).
  # Re-enable if the TheRock fix is reverted or this project is built outside TheRock.
  # The AND NOT guard prevents double-replacement if both fixes are simultaneously active.
  # if((CMAKE_CXX_FLAGS MATCHES "-fsanitize=address" OR BUILD_ADDRESS_SANITIZER)
  #     AND NOT CMAKE_CXX_FLAGS MATCHES "-Xarch_host -fsanitize=address")
  #   string(REPLACE "-fsanitize=address" "-Xarch_host -fsanitize=address" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  #   message(STATUS "HOST_ASAN: replaced -fsanitize=address with -Xarch_host -fsanitize=address in CMAKE_CXX_FLAGS")
  # endif()
endif()
message(STATUS "HOST_ASAN CMAKE_CXX_FLAGS after fix: ${CMAKE_CXX_FLAGS}")

# Verify that hipcc or HIP-aware clang is used on ROCM software
# TODO: Fix VerifyCompiler for Windows
if (NOT WIN32)
  include(VerifyCompiler)
else()
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip ${ROCM_PATH}/llvm)
  find_package(hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH})
endif()

# Diagnostics: dump hip::device INTERFACE properties after find_package(HIP) has run.
if(TARGET hip::device)
  get_target_property(_dbg_hip_device_compile_opts hip::device INTERFACE_COMPILE_OPTIONS)
  message(STATUS "DIAG hip::device INTERFACE_COMPILE_OPTIONS : ${_dbg_hip_device_compile_opts}")
  message(STATUS "DIAG CMAKE_CXX_FLAGS after VerifyCompiler  : ${CMAKE_CXX_FLAGS}")
else()
  message(STATUS "DIAG hip::device target NOT found after VerifyCompiler")
endif()

# Include cmake scripts
include(Dependencies)

# Build CXX flags
if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(DISABLE_WERROR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
endif()

if (CMAKE_CXX_STANDARD EQUAL 14)
  message(WARNING "C++14 will be deprecated in the next major release")
elseif(NOT CMAKE_CXX_STANDARD EQUAL 17)
  message(FATAL_ERROR "Only C++14 and C++17 are supported")
endif()

# HIP on Windows: xhip is required with clang++ to get __half defined
if (WIN32)
  add_compile_options(-xhip)
endif()

if(BUILD_ADDRESS_SANITIZER AND BUILD_SHARED_LIBS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
  add_link_options(-fuse-ld=lld)
endif()

# Print configuration summary
include(Summary)
print_configuration_summary()

# hipRAND library
# This adds library targets: hiprand,
# also includes Fortran wrapper
add_subdirectory(library)

# Tests
if(BUILD_TEST)
    enable_testing()
    rocm_package_setup_component(clients)
    rocm_package_setup_client_component(tests)
    add_subdirectory(test)
endif()

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
if(WIN32)
    set(CPACK_SOURCE_GENERATOR "ZIP")
    set(CPACK_GENERATOR "ZIP")
    set(CPACK_SET_DESTDIR OFF)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK")
    set(CPACK_PACKAGING_INSTALL_PREFIX "")
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
else()
    if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
        set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
    endif()
endif()

if( HIP_RUNTIME_LOWER STREQUAL "rocclr" )
    if(BUILD_ADDRESS_SANITIZER)
        set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan" )
    else()
        set(DEPENDS_HIP_RUNTIME "hip-runtime-amd" )
    endif()
    rocm_package_add_dependencies("${DEPENDS_HIP_RUNTIME} >= 4.5.0")
elseif( HIP_RUNTIME STREQUAL "cuda" )
    rocm_package_add_dependencies("hip-nvcc >= 3.5.0")
endif( )
rocm_package_add_dependencies("rocrand >= ${VERSION_STRING}")

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include")

if (BUILD_WITH_LIB STREQUAL "CUDA")
    set(package_name hiprand-alt)
else()
    set(package_name hiprand)
endif()

if(BUILD_CODE_COVERAGE)
  target_compile_options(hiprand PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping)
  target_link_options(hiprand PUBLIC -fprofile-instr-generate)
endif()

rocm_create_package(
        NAME ${package_name}
        DESCRIPTION "GPU-based random number generation library, written in HIP"
        MAINTAINER "hipRAND Maintainer <hiprand-maintainer@amd.com>"
        LDCONFIG
        LDCONFIG_DIR ${HIPRAND_CONFIG_DIR}
)
