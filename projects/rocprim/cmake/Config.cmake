# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Configures the CXX and HIP languages for this project.
macro(setup_languages)
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
endmacro(setup_languages)

# Derive the device architecture we need to compile for. We prioritize the
# the following:
# 1. GPU_TARGETS (user defined, used by hip-config)
# 2. AMDGPU_TARGETS (user defined, used by hip-config, deprecated)
# 3. HIP_ARCHITECTURES (user defined)
# 4. CMAKE_HIP_ARCHITECTURES (inferred, requires CMake HIP support)
# 5. 'all' (default, no CMake HIP support)
#
# Force sets cache variables 'HIP_ARCHITECTURES' and 'GPU_TARGETS'.
macro(get_gpu_targets)
  if(NOT DEFINED GPU_TARGETS)
    if(DEFINED AMDGPU_TARGETS)
      message(DEPRECATION "The use of 'AMDGPU_TARGETS' is deprecated use 'GPU_TARGETS' or 'CMAKE_HIP_ARCHITECTURES'.")
      set(GPU_TARGETS ${AMDGPU_TARGETS})
    elseif(DEFINED HIP_ARCHITECTURES)
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
      check_target_ids(VERIFIED_GPU_TARGETS
        TARGETS "gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+;gfx950:xnack+"
      )
    else()
      check_target_ids(VERIFIED_GPU_TARGETS
        TARGETS "gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201"
      )
    endif()
  else()
    check_target_ids(VERIFIED_GPU_TARGETS TARGETS "${GPU_TARGETS}")
  endif()

  # Overwrite the targets with the subset of supported archs.
  # HIP_ARCHITECTURES and CMAKE_HIP_ARCHITECTURES are new CMake
  # cache variables. Unfortunately most of the ROCm ecosystem
  # still uses GPU_TARGETS. We'll set both, just in case.
  set(CMAKE_HIP_ARCHITECTURES "${VERIFIED_GPU_TARGETS}" CACHE INTERNAL "" FORCE)
  set(HIP_ARCHITECTURES "${VERIFIED_GPU_TARGETS}" CACHE INTERNAL "" FORCE)
  set(GPU_TARGETS "${VERIFIED_GPU_TARGETS}" CACHE STRING "" FORCE)
endmacro(get_gpu_targets)

include(CheckCompilerFlag)
include(CMakeParseArguments)

# rocm_check_target_ids(...) from
#   https://github.com/ROCm/rocm-cmake/blob/develop/share/rocmcmakebuildtools/cmake/ROCMCheckTargetIds.cmake
# does not support CMake's HIP language since it uses 'check_cxx_compiler_flag'...
function(check_target_ids VARIABLE)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs TARGETS)

  cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(PARSE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown keywords given to rocm_check_target_ids(): \"${PARSE_UNPARSED_ARGUMENTS}\"")
  endif()

  foreach(_target_id ${PARSE_TARGETS})
    if(${_target_id} IN_LIST _supported_target_ids)
      continue()
    endif()
    _rocm_sanitize_target_id("${_target_id}" _result_var)
    set(_result_var "COMPILER_HAS_TARGET_ID_${_result_var}")
    check_compiler_flag(HIP "--offload-arch=${_target_id}" "${_result_var}")
    if(${_result_var})
        list(APPEND _supported_target_ids "${_target_id}")
    endif()
  endforeach()
  set(${VARIABLE} "${_supported_target_ids}" PARENT_SCOPE)
endfunction()
