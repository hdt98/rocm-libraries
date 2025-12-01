# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ==============================================================================
# Backward Compatibility Shims for hipBLAS Build Options
# ==============================================================================
# This file maps legacy build option names to modern project-specific options.
# It provides a transition period for users to update their build scripts while
# maintaining backward compatibility.
#
# Users can suppress deprecation warnings with: -DCMAKE_WARN_DEPRECATED=OFF
#
# Legacy → Modern Mappings:
# ------------------------
# BUILD_CLIENTS_TESTS       → HIPBLAS_BUILD_TESTING
# BUILD_CLIENTS_BENCHMARKS  → HIPBLAS_ENABLE_BENCHMARKS
# BUILD_CLIENTS_SAMPLES     → HIPBLAS_ENABLE_SAMPLES
# BUILD_CLIENTS             → HIPBLAS_ENABLE_CLIENT
# BUILD_FORTRAN_CLIENTS     → HIPBLAS_ENABLE_FORTRAN
# BUILD_WITH_SOLVER         → HIPBLAS_ENABLE_SOLVER
# LINK_BLIS                 → HIPBLAS_ENABLE_BLIS
# BUILD_ADDRESS_SANITIZER   → HIPBLAS_ENABLE_ASAN
# BUILD_CODE_COVERAGE       → HIPBLAS_BUILD_COVERAGE
# BUILD_VERBOSE             → CMAKE_VERBOSE_MAKEFILE
# BUILD_DOCS                → HIPBLAS_BUILD_DOCS
# BUILD_SHARED_LIBS         → HIPBLAS_BUILD_SHARED_LIBS
# USE_CUDA                  → HIPBLAS_ENABLE_CUDA
#
# Modern Options (no legacy equivalent):
# --------------------------------------
# HIPBLAS_ENABLE_HIP        - Build hipBLAS with HIP backend (default)
# HIPBLAS_ENABLE_CUDA       - Build hipBLAS with CUDA backend
# HIPBLAS_ENABLE_CLIENT     - Build hipBLAS clients
# HIPBLAS_ENABLE_OPENMP     - Enable OpenMP support in clients
# CMAKE_VERBOSE_MAKEFILE    - Enable verbose Makefile output
# ==============================================================================

# Helper macro for deprecation warnings using native CMake mechanism
macro(_hipblas_deprecation_warning old_var new_var)
    message(DEPRECATION
        "The option '${old_var}' is deprecated and will be removed in a future release.\n"
        "Please use '${new_var}' instead.\n"
        "To suppress these warnings: cmake -DCMAKE_WARN_DEPRECATED=OFF ...")
endmacro()

# Helper macro for conflict detection
macro(_hipblas_check_conflict old_var new_var)
    # If new variable is defined, unset any cached legacy variable to avoid conflicts
    # This allows smooth migration from legacy to modern options without requiring build dir removal
    if(DEFINED ${new_var})
        unset(${old_var} CACHE)
    endif()

    # Only check for actual conflicts if both are actively being set
    if(DEFINED ${old_var} AND DEFINED ${new_var})
        if(NOT "${${old_var}}" STREQUAL "${${new_var}}")
            message(FATAL_ERROR
                "Conflicting options detected:\n"
                "  ${old_var}=${${old_var}} (deprecated)\n"
                "  ${new_var}=${${new_var}}\n"
                "Please remove ${old_var} from your build configuration and use only ${new_var}.")
        endif()
    endif()
endmacro()

# Consolidated function for mapping legacy options to modern equivalents
# Usage: shim_mapping(<legacy_var> <modern_var> <description> [<type>])
# Type defaults to BOOL if not specified
function(shim_mapping legacy_var modern_var description)
    set(var_type "${ARGV3}")
    if(NOT var_type)
        set(var_type "BOOL")
    endif()

    if(DEFINED ${legacy_var})
        _hipblas_check_conflict(${legacy_var} ${modern_var})
        if(NOT DEFINED ${modern_var})
            set(${modern_var} ${${legacy_var}} CACHE ${var_type} "${description}" FORCE)
            _hipblas_deprecation_warning(${legacy_var} ${modern_var})
            list(APPEND _HIPBLAS_LEGACY_OPTIONS_USED "${legacy_var}=${${legacy_var}}")
            list(APPEND _HIPBLAS_CURRENT_OPTIONS "${modern_var}=${${modern_var}}")
            # Propagate lists to parent scope
            set(_HIPBLAS_LEGACY_OPTIONS_USED "${_HIPBLAS_LEGACY_OPTIONS_USED}" PARENT_SCOPE)
            set(_HIPBLAS_CURRENT_OPTIONS "${_HIPBLAS_CURRENT_OPTIONS}" PARENT_SCOPE)
        endif()
    endif()
endfunction()

# ==============================================================================
# Apply Legacy Option Mappings
# ==============================================================================

# Initialize tracking lists for migration guidance
set(_HIPBLAS_LEGACY_OPTIONS_USED "")
set(_HIPBLAS_CURRENT_OPTIONS "")

# Apply mappings using consolidated function
shim_mapping(BUILD_CLIENTS_TESTS HIPBLAS_BUILD_TESTING "Build test client; master switch.")
shim_mapping(BUILD_CLIENTS_BENCHMARKS HIPBLAS_ENABLE_BENCHMARKS "Build benchmark client.")
shim_mapping(BUILD_CLIENTS_SAMPLES HIPBLAS_ENABLE_SAMPLES "Build client samples.")
shim_mapping(BUILD_FORTRAN_CLIENTS HIPBLAS_ENABLE_FORTRAN "Build Fortran clients.")
shim_mapping(BUILD_WITH_SOLVER HIPBLAS_ENABLE_SOLVER "Add additional functions from rocSOLVER.")
shim_mapping(LINK_BLIS HIPBLAS_ENABLE_BLIS "Link AOCL BLIS reference library.")
shim_mapping(BUILD_ADDRESS_SANITIZER HIPBLAS_ENABLE_ASAN "Build with address sanitizer enabled.")
shim_mapping(BUILD_CODE_COVERAGE HIPBLAS_BUILD_COVERAGE "Build tests with coverage enabled.")
shim_mapping(BUILD_VERBOSE CMAKE_VERBOSE_MAKEFILE "Enable verbose output from Makefile builds.")
shim_mapping(BUILD_DOCS HIPBLAS_BUILD_DOCS "Build documentation.")
shim_mapping(BUILD_SHARED_LIBS HIPBLAS_BUILD_SHARED_LIBS "Build hipBLAS as a shared library.")

# Special case: BUILD_CLIENTS (only map if ON)
if(DEFINED BUILD_CLIENTS AND BUILD_CLIENTS AND NOT DEFINED HIPBLAS_ENABLE_CLIENT)
    _hipblas_check_conflict(BUILD_CLIENTS HIPBLAS_ENABLE_CLIENT)
    set(HIPBLAS_ENABLE_CLIENT ON CACHE BOOL "Build hipBLAS clients." FORCE)
    _hipblas_deprecation_warning(BUILD_CLIENTS HIPBLAS_ENABLE_CLIENT)
    list(APPEND _HIPBLAS_LEGACY_OPTIONS_USED "BUILD_CLIENTS=${BUILD_CLIENTS}")
    list(APPEND _HIPBLAS_CURRENT_OPTIONS "HIPBLAS_ENABLE_CLIENT=${HIPBLAS_ENABLE_CLIENT}")
endif()

# Special case: USE_CUDA with environment variable note
if(DEFINED USE_CUDA)
    _hipblas_check_conflict(USE_CUDA HIPBLAS_ENABLE_CUDA)
    if(NOT DEFINED HIPBLAS_ENABLE_CUDA)
        if(USE_CUDA)
            set(HIPBLAS_ENABLE_CUDA ON CACHE BOOL "Build hipBLAS with CUDA backend" FORCE)
        else()
            set(HIPBLAS_ENABLE_CUDA OFF CACHE BOOL "Build hipBLAS with CUDA backend" FORCE)
        endif()
        message(DEPRECATION
            "The option 'USE_CUDA' is deprecated and will be removed in a future release.\n"
            "Please use 'HIPBLAS_ENABLE_CUDA' instead, or set environment variable HIP_PLATFORM=nvidia.\n"
            "To suppress these warnings: cmake -DCMAKE_WARN_DEPRECATED=OFF ...")
        list(APPEND _HIPBLAS_LEGACY_OPTIONS_USED "USE_CUDA=${USE_CUDA}")
        list(APPEND _HIPBLAS_CURRENT_OPTIONS "HIPBLAS_ENABLE_CUDA=${HIPBLAS_ENABLE_CUDA}")
    endif()
endif()

# ==============================================================================
# Legacy conditional option disabling (DEPRECATED - to be removed)
# ==============================================================================
# The new build system enforces incompatible options as FATAL_ERROR. We provide
# this auto-disabling behavior temporarily to help users transition, but users
# should explicitly set options in their build commands.
# ==============================================================================

if(HIPBLAS_ENABLE_ASAN)
    message(DEPRECATION 
      "LEGACY BUILD BEHAVIOR: Auto-disabling HIPBLAS_ENABLE_FORTRAN (incompatible with ASAN).\n"
      "This behavior is DEPRECATED and will be removed in a future release.\n"
      "Please explicitly set: -DHIPBLAS_ENABLE_FORTRAN=OFF")
    set(HIPBLAS_ENABLE_FORTRAN OFF CACHE BOOL "Fortran disabled (ASAN enabled)" FORCE)
endif()

# Same for coverage
if(HIPBLAS_BUILD_COVERAGE OR BUILD_CODE_COVERAGE)
    message(DEPRECATION 
      "LEGACY BUILD BEHAVIOR: Auto-disabling HIPBLAS_ENABLE_FORTRAN (incompatible with coverage).\n"
      "This behavior is DEPRECATED and will be removed in a future release.\n"
      "Please explicitly set: -DHIPBLAS_ENABLE_FORTRAN=OFF")
    set(HIPBLAS_ENABLE_FORTRAN OFF CACHE BOOL "Fortran disabled (coverage enabled)" FORCE)
endif()

# ==============================================================================
# Infer HIPBLAS_ENABLE_CLIENT from child options
# ==============================================================================

# If any child client option is ON, automatically enable HIPBLAS_ENABLE_CLIENT
# This ensures the client packaging logic is triggered, since all clients depend
# on the clients and clients-common infrastructure
if(HIPBLAS_BUILD_TESTING OR HIPBLAS_ENABLE_BENCHMARKS OR HIPBLAS_ENABLE_SAMPLES)
    if(NOT DEFINED HIPBLAS_ENABLE_CLIENT)
        set(HIPBLAS_ENABLE_CLIENT ON CACHE BOOL "Build hipBLAS clients." FORCE)
        message(STATUS "Automatically enabled HIPBLAS_ENABLE_CLIENT because at least one child client option is ON")
    endif()
endif()

# ==============================================================================
# Display Migration Guidance
# ==============================================================================

if(_HIPBLAS_LEGACY_OPTIONS_USED)
    # Format the current options with -D prefix for each
    string(REPLACE ";" " -D" _formatted_current_options "${_HIPBLAS_CURRENT_OPTIONS}")
    set(_formatted_current_options "-D${_formatted_current_options}")

    message(DEPRECATION
        "\n"
        "  Legacy build options detected:\n"
        "    ${_HIPBLAS_LEGACY_OPTIONS_USED}\n"
        "\n"
        "  To use current options, run:\n"
        "    cmake ${_formatted_current_options} ..\n"
        "\n"
        "  To suppress warnings: cmake -DCMAKE_WARN_DEPRECATED=OFF ...\n"
    )
endif()
