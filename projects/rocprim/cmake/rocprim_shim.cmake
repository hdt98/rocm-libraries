# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Backward compatibility shim for rocPRIM option names
# This file maps legacy option names to modern equivalents

macro(_rocprim_deprecation_warning old_var new_var)
    message(DEPRECATION "Use '${new_var}' instead of '${old_var}'. The old option will be removed in a future release.")
endmacro()

macro(_rocprim_check_conflict old_var new_var)
    if(DEFINED ${old_var} AND DEFINED ${new_var})
        if(NOT "${${old_var}}" STREQUAL "${${new_var}}")
            message(FATAL_ERROR "Conflicting options: ${old_var}=${${old_var}} vs ${new_var}=${${new_var}}. Please use only ${new_var}.")
        endif()
    endif()
endmacro()

# Map BUILD_TEST -> ROCPRIM_BUILD_TESTING
if(DEFINED BUILD_TEST)
    _rocprim_check_conflict(BUILD_TEST ROCPRIM_BUILD_TESTING)
    if(NOT DEFINED ROCPRIM_BUILD_TESTING)
        set(ROCPRIM_BUILD_TESTING ${BUILD_TEST} CACHE BOOL "Build tests" FORCE)
        _rocprim_deprecation_warning(BUILD_TEST ROCPRIM_BUILD_TESTING)
    endif()
endif()

# Map WITH_ROCRAND -> ROCPRIM_ENABLE_ROCRAND
if(DEFINED WITH_ROCRAND)
    _rocprim_check_conflict(WITH_ROCRAND ROCPRIM_ENABLE_ROCRAND)
    if(NOT DEFINED ROCPRIM_ENABLE_ROCRAND)
        set(ROCPRIM_ENABLE_ROCRAND ${WITH_ROCRAND} CACHE BOOL "Build tests with device-side data generation (requires rocRAND)" FORCE)
        _rocprim_deprecation_warning(WITH_ROCRAND ROCPRIM_ENABLE_ROCRAND)
    endif()
endif()

# Map BUILD_BENCHMARK -> ROCPRIM_ENABLE_BENCHMARK
if(DEFINED BUILD_BENCHMARK)
    _rocprim_check_conflict(BUILD_BENCHMARK ROCPRIM_ENABLE_BENCHMARK)
    if(NOT DEFINED ROCPRIM_ENABLE_BENCHMARK)
        set(ROCPRIM_ENABLE_BENCHMARK ${BUILD_BENCHMARK} CACHE BOOL "Build benchmarks" FORCE)
        _rocprim_deprecation_warning(BUILD_BENCHMARK ROCPRIM_ENABLE_BENCHMARK)
    endif()
endif()

# Map BUILD_EXAMPLE -> ROCPRIM_ENABLE_EXAMPLES
if(DEFINED BUILD_EXAMPLE)
    _rocprim_check_conflict(BUILD_EXAMPLE ROCPRIM_ENABLE_EXAMPLES)
    if(NOT DEFINED ROCPRIM_ENABLE_EXAMPLES)
        set(ROCPRIM_ENABLE_EXAMPLES ${BUILD_EXAMPLE} CACHE BOOL "Build examples" FORCE)
        _rocprim_deprecation_warning(BUILD_EXAMPLE ROCPRIM_ENABLE_EXAMPLES)
    endif()
endif()

# Map BUILD_DOCS -> ROCPRIM_ENABLE_DOCS
if(DEFINED BUILD_DOCS)
    _rocprim_check_conflict(BUILD_DOCS ROCPRIM_ENABLE_DOCS)
    if(NOT DEFINED ROCPRIM_ENABLE_DOCS)
        set(ROCPRIM_ENABLE_DOCS ${BUILD_DOCS} CACHE BOOL "Build documentation" FORCE)
        _rocprim_deprecation_warning(BUILD_DOCS ROCPRIM_ENABLE_DOCS)
    endif()
endif()

# Map BUILD_CODE_COVERAGE -> ROCPRIM_ENABLE_COVERAGE
if(DEFINED BUILD_CODE_COVERAGE)
    _rocprim_check_conflict(BUILD_CODE_COVERAGE ROCPRIM_ENABLE_COVERAGE)
    if(NOT DEFINED ROCPRIM_ENABLE_COVERAGE)
        set(ROCPRIM_ENABLE_COVERAGE ${BUILD_CODE_COVERAGE} CACHE BOOL "Build with code coverage enabled" FORCE)
        _rocprim_deprecation_warning(BUILD_CODE_COVERAGE ROCPRIM_ENABLE_COVERAGE)
    endif()
endif()

# Map ROCPRIM_INSTALL -> ROCPRIM_ENABLE_INSTALL
if(DEFINED ROCPRIM_INSTALL)
    _rocprim_check_conflict(ROCPRIM_INSTALL ROCPRIM_ENABLE_INSTALL)
    if(NOT DEFINED ROCPRIM_ENABLE_INSTALL)
        set(ROCPRIM_ENABLE_INSTALL ${ROCPRIM_INSTALL} CACHE BOOL "Enable installation of rocPRIM" FORCE)
        _rocprim_deprecation_warning(ROCPRIM_INSTALL ROCPRIM_ENABLE_INSTALL)
    endif()
endif()

# Map BUILD_OFFLOAD_COMPRESS -> ROCPRIM_ENABLE_OFFLOAD_COMPRESS
if(DEFINED BUILD_OFFLOAD_COMPRESS)
    _rocprim_check_conflict(BUILD_OFFLOAD_COMPRESS ROCPRIM_ENABLE_OFFLOAD_COMPRESS)
    if(NOT DEFINED ROCPRIM_ENABLE_OFFLOAD_COMPRESS)
        set(ROCPRIM_ENABLE_OFFLOAD_COMPRESS ${BUILD_OFFLOAD_COMPRESS} CACHE BOOL "Build rocPRIM with offload compression" FORCE)
        _rocprim_deprecation_warning(BUILD_OFFLOAD_COMPRESS ROCPRIM_ENABLE_OFFLOAD_COMPRESS)
    endif()
endif()

# Map BUILD_NAIVE_BENCHMARK -> ROCPRIM_ENABLE_NAIVE_BENCHMARK
if(DEFINED BUILD_NAIVE_BENCHMARK)
    _rocprim_check_conflict(BUILD_NAIVE_BENCHMARK ROCPRIM_ENABLE_NAIVE_BENCHMARK)
    if(NOT DEFINED ROCPRIM_ENABLE_NAIVE_BENCHMARK)
        set(ROCPRIM_ENABLE_NAIVE_BENCHMARK ${BUILD_NAIVE_BENCHMARK} CACHE BOOL "Build naive benchmarks" FORCE)
        _rocprim_deprecation_warning(BUILD_NAIVE_BENCHMARK ROCPRIM_ENABLE_NAIVE_BENCHMARK)
    endif()
endif()

# Map USE_SYSTEM_LIB -> ROCPRIM_USE_SYSTEM_LIBS
if(DEFINED USE_SYSTEM_LIB)
    _rocprim_check_conflict(USE_SYSTEM_LIB ROCPRIM_USE_SYSTEM_LIBS)
    if(NOT DEFINED ROCPRIM_USE_SYSTEM_LIBS)
        set(ROCPRIM_USE_SYSTEM_LIBS ${USE_SYSTEM_LIB} CACHE BOOL "Use installed ROCm libs when building tests" FORCE)
        _rocprim_deprecation_warning(USE_SYSTEM_LIB ROCPRIM_USE_SYSTEM_LIBS)
    endif()
endif()

# Map USE_HIPCXX -> ROCPRIM_USE_HIPCXX
if(DEFINED USE_HIPCXX)
    _rocprim_check_conflict(USE_HIPCXX ROCPRIM_USE_HIPCXX)
    if(NOT DEFINED ROCPRIM_USE_HIPCXX)
        set(ROCPRIM_USE_HIPCXX ${USE_HIPCXX} CACHE BOOL "Use CMake HIP language support" FORCE)
        _rocprim_deprecation_warning(USE_HIPCXX ROCPRIM_USE_HIPCXX)
    endif()
endif()

# Handle ONLY_INSTALL special case
if(DEFINED ONLY_INSTALL)
    if(ONLY_INSTALL)
        message(DEPRECATION "ONLY_INSTALL is deprecated. Set ROCPRIM_BUILD_TESTING=OFF, ROCPRIM_ENABLE_BENCHMARK=OFF, and ROCPRIM_ENABLE_EXAMPLES=OFF instead.")
        if(NOT DEFINED ROCPRIM_BUILD_TESTING)
            set(ROCPRIM_BUILD_TESTING OFF CACHE BOOL "Build tests" FORCE)
        endif()
        if(NOT DEFINED ROCPRIM_ENABLE_BENCHMARK)
            set(ROCPRIM_ENABLE_BENCHMARK OFF CACHE BOOL "Build benchmarks" FORCE)
        endif()
        if(NOT DEFINED ROCPRIM_ENABLE_EXAMPLES)
            set(ROCPRIM_ENABLE_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
        endif()
    endif()
endif()

# Map BENCHMARK_CONFIG_TUNING -> ROCPRIM_ENABLE_CONFIG_TUNING
if(DEFINED BENCHMARK_CONFIG_TUNING)
    _rocprim_check_conflict(BENCHMARK_CONFIG_TUNING ROCPRIM_ENABLE_CONFIG_TUNING)
    if(NOT DEFINED ROCPRIM_ENABLE_CONFIG_TUNING)
        set(ROCPRIM_ENABLE_CONFIG_TUNING ${BENCHMARK_CONFIG_TUNING} CACHE BOOL "Benchmark device-level functions using various configs" FORCE)
        _rocprim_deprecation_warning(BENCHMARK_CONFIG_TUNING ROCPRIM_ENABLE_CONFIG_TUNING)
    endif()
endif()

# Map BENCHMARK_AUTOTUNED_TYPES_ONLY -> ROCPRIM_ENABLE_AUTOTUNED_TYPES_ONLY
if(DEFINED BENCHMARK_AUTOTUNED_TYPES_ONLY)
    _rocprim_check_conflict(BENCHMARK_AUTOTUNED_TYPES_ONLY ROCPRIM_ENABLE_AUTOTUNED_TYPES_ONLY)
    if(NOT DEFINED ROCPRIM_ENABLE_AUTOTUNED_TYPES_ONLY)
        set(ROCPRIM_ENABLE_AUTOTUNED_TYPES_ONLY ${BENCHMARK_AUTOTUNED_TYPES_ONLY} CACHE BOOL "Benchmark autotuned types only" FORCE)
        _rocprim_deprecation_warning(BENCHMARK_AUTOTUNED_TYPES_ONLY ROCPRIM_ENABLE_AUTOTUNED_TYPES_ONLY)
    endif()
endif()

# Map BENCHMARK_USE_AMDSMI -> ROCPRIM_BENCHMARK_USE_AMDSMI
if(DEFINED BENCHMARK_USE_AMDSMI)
    _rocprim_check_conflict(BENCHMARK_USE_AMDSMI ROCPRIM_BENCHMARK_USE_AMDSMI)
    if(NOT DEFINED ROCPRIM_BENCHMARK_USE_AMDSMI)
        set(ROCPRIM_BENCHMARK_USE_AMDSMI ${BENCHMARK_USE_AMDSMI} CACHE BOOL "Let benchmarks use AMD SMI to output more GPU statistics" FORCE)
        _rocprim_deprecation_warning(BENCHMARK_USE_AMDSMI ROCPRIM_BENCHMARK_USE_AMDSMI)
    endif()
endif()
