# MIT License
#
# Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

##################
# CMake Preamble #
##################

cmake_minimum_required(VERSION 3.21...3.25 FATAL_ERROR)

if(WIN32)
  set(ROCM_PATH "$ENV{HIP_PATH}" CACHE PATH "Root directory of the ROCm installation")
else()
  set(ROCM_PATH "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
endif()

# Configure default CMake install behavior.
set(CMAKE_INSTALL_PREFIX ${ROCM_PATH} CACHE PATH "Install path prefix, prepended onto install directories")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Add paths to linker search and installed rpath")

# Declare project.
project(rocprim LANGUAGES CXX HIP)
set(VERSION_STRING "4.2.0")

###########
# Options #
###########

include(cmake/ConfigUtils.cmake)

register_option(ALLOW_DOWNLOADING_DEPS OFF "Allow downloads during configuration.")
register_option(ONLY_INSTALL "Only install package. Skips building dependencies and tests." OFF)
register_option(ROCPRIM_INSTALL "Enable installation of rocPRIM, projects embedding rocPRIM may want to turn this off." ON)

register_dep_option(BUILD_TEST             "Build tests." OFF "NOT ONLY_INSTALL" OFF)
register_dep_option(BUILD_EXAMPLE          "Build examples." OFF "NOT ONLY_INSTALL" OFF)
register_dep_option(BUILD_BENCHMARK        "Build benchmarks." OFF "NOT ONLY_INSTALL" OFF)
register_dep_option(BUILD_NAIVE_BENCHMARK  "Build naive benchmarks." OFF BUILD_BENCHMARK OFF)
register_dep_option(WITH_ROCRAND           "Build tests with device-side data generation." OFF "NOT ONLY_INSTALL" OFF)
register_dep_option(BUILD_DOCS             "Build documentation." OFF "NOT ONLY_INSTALL" OFF)
register_dep_option(USE_SYSTEM_LIB         "Use installed ROCm libs when building tests." OFF "BUILD_TEST" OFF)
register_dep_option(BUILD_CODE_COVERAGE    "Build with code coverage enabled." OFF "BUILD_TEST" OFF)
register_dep_option(BUILD_OFFLOAD_COMPRESS "Build rocPRIM with offload compression." ON "NOT ONLY_INSTALL" OFF)
register_dep_option(ROCPRIM_ENABLE_ASSERTS "Enable asserts in release build." OFF "NOT ONLY_INSTALL" OFF)

register_set(ROCRAND_FETCH_METHOD PACKAGE STRING "Method to fetch rocRAND dependency.")
register_set(ROCRAND_FETCH_BRANCH develop STRING "Branch to fetch rocRAND from.")

###########
# C++/HIP #
###########

# Set C++/HIP default language options.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_EXTENSIONS OFF)

# Verify C++/HIP language standard.
if(NOT CMAKE_CXX_STANDARD EQUAL 17)
  message(FATAL_ERROR "Only C++17 is supported (C++)")
endif()
if(NOT CMAKE_HIP_STANDARD EQUAL 17)
  message(FATAL_ERROR "Only C++17 is supported (HIP)")
endif()

###############
# dependencies #
###############

include(cmake/Dependencies.cmake)

# The reason we don't scope global state meddling using add_subdirectory
# is because CMake < 3.24 lacks CMAKE_FIND_PACKAGE_TARGETS_GLOBAL which
# would promote IMPORTED targets of find_package(CONFIG) to be visible
# by other parts of the build. So we save and restore global state.
override_cache_variable(BUILD_SHARED_LIBS OFF BOOL)

# We disable the ROCMChecks.cmake warning noting that we meddle with
# global state. This is a consequence of abusing the CMake CXX language
# which HIP piggybacks on top of. This kind of HIP support has one chance
# at observing the global flags: at the find_package(HIP) invocation.
# The device compiler won't be able to pick up changes after that, hence
# the warning.
override_cache_variable(ROCM_WARN_TOOLCHAIN_VAR OFF BOOL)

# We ignore warning-as-errors when building dependencies.
remove_warning_flags(CMAKE_CXX_FLAGS CXX_FLAGS_FILTERED)
remove_warning_flags(CMAKE_HIP_FLAGS HIP_FLAGS_FILTERED)
override_cache_variable(CMAKE_CXX_FLAGS CXX_FLAGS_FILTERED STRING)
override_cache_variable(CMAKE_HIP_FLAGS HIP_FLAGS_FILTERED STRING)

fetch_rocmcmake()
if(BUILD_TEST)
  fetch_googletest()
endif()
if(BUILD_BENCHMARK)
  fetch_googlebench()
endif()
if(WITH_ROCRAND)
  fetch_rocrand()
endif()

restore_cache_variable(BUILD_SHARED_LIBS BOOL)
restore_cache_variable(ROCM_WARN_TOOLCHAIN_VAR BOOL)
restore_cache_variable(CMAKE_CXX_FLAGS STRING)
restore_cache_variable(CMAKE_HIP_FLAGS STRING)

find_package(hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH})

#################
# Compile flags #
#################

# By default, target a release build. 
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include(CheckCompilerFlag)

# Enable offload compress.
if(BUILD_OFFLOAD_COMPRESS)
  # Except on SPIR-V!
  if ("amdgcnspirv" IN_LIST HIP_ARCHITECTURES)
    message(FATAL_ERROR "Cannot combine SPIR-V and BUILD_OFFLOAD_COMPRESS")
  endif()

  # Detect capability and add flag.
  cmake_check_compiler_flag(HIP "--offload-compress" HIP_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
  if(HIP_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
    add_compile_options($<$<COMPILE_LANGUAGE:HIP>:--offload-compress>)
  else()
    message(WARNING "BUILD_OFFLOAD_COMPRESS=ON but flag not supported by compiler. Ignoring option.")
  endif()
endif()

# Allow asserts in release mode if 'ROCPRIM_ENABLE_ASSERTS' is enabled.
if(ROCPRIM_ENABLE_ASSERTS)
  # Remove -DNDEBUG from the default release build type flags.
  string(REGEX REPLACE "-DNDEBUG( |$)" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif()

#################
# Architectures #
#################

# Derive the device architecture we need to compile for. We prioritize the
# the following:
# 1. GPU_TARGETS (user defined, used by hip-config)
# 2. HIP_ARCHITECTURES (user defined)
# 3. CMAKE_HIP_ARCHITECTURES (inferred, requires CMake HIP support)
# 4. 'all' (default, no CMake HIP support)
if (NOT DEFINED GPU_TARGETS)
  if (DEFINED HIP_ARCHITECTURES)
    set(GPU_TARGETS ${HIP_ARCHITECTURES})
  elseif(DEFINED CMAKE_HIP_ARCHITECTURES)
    set(GPU_TARGETS ${CMAKE_HIP_ARCHITECTURES})
  else()
    set(GPU_TARGETS "all")
  endif()
endif()

# Handle 'all'-keyword in architectures.
if(GPU_TARGETS STREQUAL "all")
  if(BUILD_ADDRESS_SANITIZER)
    # ASAN builds require xnack
    rocm_check_target_ids(VERIFIED_GPU_TARGETS
      TARGETS "gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+;gfx950:xnack+"
    )
  else()
    rocm_check_target_ids(VERIFIED_GPU_TARGETS
      TARGETS "gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201"
    )
  endif()
else()
  rocm_check_target_ids(VERIFIED_GPU_TARGETS TARGETS "${GPU_TARGETS}")
endif()

# Overwrite the targets with the subset of supported archs.
# HIP_ARCHITECTURES and CMAKE_HIP_ARCHITECTURES are new CMake
# cache variables. Unfortunately most of the ROCm ecosystem
# still uses GPU_TARGETS. We'll set both, just in case.
set(HIP_ARCHITECTURES "${VERIFIED_GPU_TARGETS}" CACHE INTERNAL "" FORCE)
set(GPU_TARGETS "${VERIFIED_GPU_TARGETS}" CACHE STRING "" FORCE)

###########
# Targets #
###########

# Print a summary of the derived configuration.
print_config()

# Get version number to inject into 'rocprim_version.hpp.in'.
rocm_setup_version(VERSION ${VERSION_STRING})
math(EXPR rocprim_VERSION_NUMBER "${rocprim_VERSION_MAJOR} * 100000 + ${rocprim_VERSION_MINOR} * 100 + ${rocprim_VERSION_PATCH}")

# Detect if we're a top level project and not included by a different project.
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(ROCPRIM_PROJECT_IS_TOP_LEVEL TRUE)
else()
  set(ROCPRIM_PROJECT_IS_TOP_LEVEL FALSE)
endif()

add_subdirectory(rocprim)

if(ROCPRIM_PROJECT_IS_TOP_LEVEL AND (BUILD_TEST OR BUILD_BENCHMARK))
  rocm_package_setup_component(clients)
endif()

if(BUILD_TEST)
  if(USE_SYSTEM_LIB)
    find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/rocprim")
    if (NOT ${rocprim_VERSION} VERSION_EQUAL ${VERSION_STRING})
      message(WARNING "The installed rocprim version, ${rocprim_VERSION}, does not match project version ${VERSION_STRING}. Building tests with USE_SYSTEM_LIB=ON may not work properly.")
    endif()
  endif()

  if (ROCPRIM_PROJECT_IS_TOP_LEVEL)
    rocm_package_setup_client_component(tests)
  endif()
  enable_testing()
  add_subdirectory(test)
endif()

# Benchmarks
if(BUILD_BENCHMARK)
  if (ROCPRIM_PROJECT_IS_TOP_LEVEL)
    rocm_package_setup_client_component(benchmarks)
  endif()
  add_subdirectory(benchmark)
endif()

# Examples
if(BUILD_EXAMPLE)
  add_subdirectory(example)
endif()

# Docs
if(BUILD_DOCS)
  add_subdirectory(docs)
endif()

if (ROCPRIM_PROJECT_IS_TOP_LEVEL)
  # Add dependency on HIP runtime
  set(HIP_RUNTIME_MINIMUM 4.5.0)
  if(BUILD_ADDRESS_SANITIZER)
    set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan" )
  else()
    set(DEPENDS_HIP_RUNTIME "hip-runtime-amd" )
  endif()

  rocm_package_add_dependencies(SHARED_DEPENDS "${DEPENDS_HIP_RUNTIME} >= ${HIP_RUNTIME_MINIMUM}")
  rocm_package_add_deb_dependencies(STATIC_DEPENDS "hip-static-dev >= ${HIP_RUNTIME_MINIMUM}")
  rocm_package_add_rpm_dependencies(STATIC_DEPENDS "hip-static-devel >= ${HIP_RUNTIME_MINIMUM}")

  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
  set(CPACK_RPM_PACKAGE_LICENSE "MIT")
  set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" )

  rocm_create_package(
    NAME rocprim
    DESCRIPTION "rocPRIM is a header-only library that provides HIP parallel primitives." 
    MAINTAINER "rocPRIM Maintainer <rocprim-maintainer@amd.com>"
    HEADER_ONLY
  )
endif()

if(BUILD_CODE_COVERAGE)
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

  target_compile_options(rocprim INTERFACE
    -w
    -fprofile-instr-generate
    -fcoverage-mapping
  )
  target_link_options(rocprim INTERFACE
    -fprofile-instr-generate
  )

  add_custom_target(
    coverage_analysis
    COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
    COMMAND ${CMAKE_COMMAND} -E rm -rf ./coverage-report
    COMMAND ${CMAKE_COMMAND} -E make_directory ./coverage-report/profraw
    COMMAND ctest --output-on-failure -E "rocprim.linking*" --gtest_filter=\"\${GTEST_FILTER}\"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  get_property(LLVM_COV_OBJECT_ARGS GLOBAL PROPERTY LLVM_COV_OBJECT_ARGS)
  add_custom_target(
    coverage
    DEPENDS coverage_analysis
    COMMAND ${LLVM_PROFDATA} merge -sparse ./coverage-report/profraw/rocprim-coverage_*.profraw -o ./coverage-report/rocprim.profdata
    COMMAND ${LLVM_COV} report  ${LLVM_COV_OBJECT_ARGS} --ignore-filename-regex="test_*" -instr-profile=./coverage-report/rocprim.profdata
    COMMAND ${LLVM_COV} show  ${LLVM_COV_OBJECT_ARGS}  --ignore-filename-regex="test_*" -instr-profile=./coverage-report/rocprim.profdata -format=html -output-dir=coverage-report
    COMMAND ${LLVM_COV} export  ${LLVM_COV_OBJECT_ARGS} --ignore-filename-regex="test_*" -instr-profile=./coverage-report/rocprim.profdata -format=lcov > ./coverage-report/coverage.info
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()
