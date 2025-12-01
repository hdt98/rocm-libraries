# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# This file defines client-specific build options
# It's included from both standalone client builds and superbuild configurations

include(CMakeDependentOption)

# Note: These options are also defined in the root CMakeLists.txt
# This file exists for compatibility with potential standalone client builds

if(NOT DEFINED HIPBLAS_ENABLE_FORTRAN)
    cmake_dependent_option(
        HIPBLAS_ENABLE_FORTRAN
        "Build hipBLAS clients requiring Fortran capabilities"
        ON
        "NOT WIN32"
        OFF
    )
endif()

if(NOT DEFINED HIPBLAS_BUILD_TESTING)
    option(HIPBLAS_BUILD_TESTING "Build hipBLAS unit tests" OFF)
endif()

if(NOT DEFINED HIPBLAS_ENABLE_BENCHMARKS)
    option(HIPBLAS_ENABLE_BENCHMARKS "Build hipBLAS benchmarks" OFF)
endif()

if(NOT DEFINED HIPBLAS_ENABLE_SAMPLES)
    option(HIPBLAS_ENABLE_SAMPLES "Build hipBLAS samples" OFF)
endif()

if(NOT DEFINED HIPBLAS_ENABLE_OPENMP)
    option(HIPBLAS_ENABLE_OPENMP "Enable OpenMP support" ON)
endif()

if(NOT DEFINED HIPBLAS_ENABLE_BLIS)
    if(HIPBLAS_ENABLE_CUDA)
        option(HIPBLAS_ENABLE_BLIS "Link AOCL BLIS reference library" OFF)
    else()
        option(HIPBLAS_ENABLE_BLIS "Link AOCL BLIS reference library" ON)
    endif()
endif()
