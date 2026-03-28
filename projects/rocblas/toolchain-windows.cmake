# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(DEFINED ENV{HIP_PATH})
  file(TO_CMAKE_PATH "$ENV{HIP_PATH}" HIP_DIR)
  set(rocm_bin "${HIP_DIR}/bin")
elseif(DEFINED ENV{HIP_DIR})
  file(TO_CMAKE_PATH "$ENV{HIP_DIR}" HIP_DIR)
  set(rocm_bin "${HIP_DIR}/bin")
else()
  set(HIP_DIR "C:/hip")
  set(rocm_bin "C:/hip/bin")
endif()

set(CMAKE_CXX_COMPILER "${rocm_bin}/clang++.exe")
set(CMAKE_C_COMPILER "${rocm_bin}/clang.exe")

if(NOT ROCBLAS_TOOLCHAIN_VARS_APPENDED)
  set(ROCBLAS_TOOLCHAIN_VARS_APPENDED True)

  string(APPEND CMAKE_CXX_FLAGS " -DWIN32 -DWIN32_LEAN_AND_MEAN -DNOMINMAX")
  string(APPEND CMAKE_CXX_FLAGS " -D_CRT_SECURE_NO_WARNINGS -D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-ignored-attributes -fms-extensions -fms-compatibility")
  string(APPEND CMAKE_CXX_FLAGS " -DHIP_CLANG_HCC_COMPAT_MODE=1 -D__HIP_ROCclr__=1 -D__HIP_PLATFORM_AMD__=1")
endif()

if(DEFINED ENV{OPENBLAS_DIR})
  file(TO_CMAKE_PATH "$ENV{OPENBLAS_DIR}" OPENBLAS_DIR)
else()
  set(OPENBLAS_DIR "C:/OpenBLAS/OpenBLAS-0.3.18-x64")
endif()

if(DEFINED ENV{VCPKG_PATH})
  file(TO_CMAKE_PATH "$ENV{VCPKG_PATH}" VCPKG_PATH)
else()
  set(VCPKG_PATH "C:/github/vcpkg")
endif()
include("${VCPKG_PATH}/scripts/buildsystems/vcpkg.cmake")

set(CMAKE_STATIC_LIBRARY_SUFFIX ".a")
set(CMAKE_STATIC_LIBRARY_PREFIX "static_")
set(CMAKE_SHARED_LIBRARY_SUFFIX ".dll")
set(CMAKE_SHARED_LIBRARY_PREFIX "")

set(BUILD_FORTRAN_CLIENTS OFF)
