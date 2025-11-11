# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ==============================================================================
# Backward Compatibility Shims for rocSOLVER Build Options
# ==============================================================================
# This file maps legacy build option names to modern project-specific options.
# It provides a transition period for users to update their build scripts while
# maintaining backward compatibility.
#
# Users can suppress deprecation warnings with: -DCMAKE_WARN_DEPRECATED=OFF
#
# Legacy → Modern Mappings:
# ------------------------
# BUILD_CLIENTS_TESTS       → ROCSOLVER_ENABLE_TESTS
# BUILD_CLIENTS_BENCHMARKS  → ROCSOLVER_ENABLE_BENCHMARKS
# BUILD_CLIENTS_SAMPLES     → ROCSOLVER_ENABLE_SAMPLES
# BUILD_CLIENTS_EXTRA_TESTS → ROCSOLVER_ENABLE_EXTRA_TESTS
# BUILD_WITH_SPARSE         → ROCSOLVER_ENABLE_SPARSE
# BUILD_ADDRESS_SANITIZER   → ROCSOLVER_ENABLE_ASAN
# BUILD_CODE_COVERAGE       → ROCSOLVER_ENABLE_COVERAGE
# BUILD_COMPRESSED_DBG      → ROCSOLVER_ENABLE_COMPRESSED_DBG
# BUILD_OFFLOAD_COMPRESS    → ROCSOLVER_ENABLE_OFFLOAD_COMPRESS
# WERROR                    → ROCSOLVER_ENABLE_WERROR
# USE_HIPCXX                → ROCSOLVER_ENABLE_HIPCXX
# BUILD_LIBRARY             → ROCSOLVER_BUILD_LIBRARY
# SKIP_LIBRARY              → ROCSOLVER_BUILD_LIBRARY (inverted)
#
# Modern Options (no legacy equivalent):
# --------------------------------------
# ROCSOLVER_BUILD_LIBRARY           - Build rocSOLVER library
# ROCSOLVER_BUILD_TESTING           - Build rocSOLVER tests (standard CMake option wrapper)
# ROCSOLVER_EMBED_FMT               - Hide libfmt symbols
# ROCSOLVER_FIND_PACKAGE_LAPACK_CONFIG - Skip module mode search for LAPACK
# ROCSOLVER_ENABLE_INTERNAL_BLAS    - Use internal GEMM/TRSM implementations
# ROCSOLVER_ENABLE_REFERENCE_SECULAR_SOLVER - Use LAPACK secular equations solver
# ROCSOLVER_ENABLE_ASYNC_LOGGER     - Enable asynchronous HIP event-based logger
# ==============================================================================

# Helper macro for deprecation warnings using native CMake mechanism
macro(_rocsolver_deprecation_warning old_var new_var)
    message(DEPRECATION
        "The option '${old_var}' is deprecated and will be removed in a future release.\n"
        "Please use '${new_var}' instead.\n"
        "To suppress these warnings: cmake -DCMAKE_WARN_DEPRECATED=OFF ...")
endmacro()

# Helper macro for conflict detection
macro(_rocsolver_check_conflict old_var new_var)
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
        _rocsolver_check_conflict(${legacy_var} ${modern_var})
        if(NOT DEFINED ${modern_var})
            set(${modern_var} ${${legacy_var}} CACHE ${var_type} "${description}" FORCE)
            _rocsolver_deprecation_warning(${legacy_var} ${modern_var})
            list(APPEND _ROCSOLVER_LEGACY_OPTIONS_USED "${legacy_var}=${${legacy_var}}")
            list(APPEND _ROCSOLVER_CURRENT_OPTIONS "${modern_var}=${${modern_var}}")
            # Propagate lists to parent scope
            set(_ROCSOLVER_LEGACY_OPTIONS_USED "${_ROCSOLVER_LEGACY_OPTIONS_USED}" PARENT_SCOPE)
            set(_ROCSOLVER_CURRENT_OPTIONS "${_ROCSOLVER_CURRENT_OPTIONS}" PARENT_SCOPE)
        endif()
    endif()
endfunction()

# ==============================================================================
# Apply Legacy Option Mappings
# ==============================================================================

# Initialize tracking lists for migration guidance
set(_ROCSOLVER_LEGACY_OPTIONS_USED "")
set(_ROCSOLVER_CURRENT_OPTIONS "")

# Apply mappings using consolidated function
shim_mapping(BUILD_CLIENTS_TESTS ROCSOLVER_ENABLE_TESTS "Build rocSOLVER test client.")
shim_mapping(BUILD_CLIENTS_BENCHMARKS ROCSOLVER_ENABLE_BENCHMARKS "Build rocSOLVER benchmark client.")
shim_mapping(BUILD_CLIENTS_SAMPLES ROCSOLVER_ENABLE_SAMPLES "Build rocSOLVER samples.")
shim_mapping(BUILD_CLIENTS_EXTRA_TESTS ROCSOLVER_ENABLE_EXTRA_TESTS "Build extra tests.")
shim_mapping(BUILD_WITH_SPARSE ROCSOLVER_ENABLE_SPARSE "Build with rocsparse support.")
shim_mapping(BUILD_ADDRESS_SANITIZER ROCSOLVER_ENABLE_ASAN "Build with address sanitizer enabled.")
shim_mapping(BUILD_CODE_COVERAGE ROCSOLVER_ENABLE_COVERAGE "Build with code coverage enabled.")
shim_mapping(BUILD_COMPRESSED_DBG ROCSOLVER_ENABLE_COMPRESSED_DBG "Enable compressed debug symbols.")
shim_mapping(BUILD_OFFLOAD_COMPRESS ROCSOLVER_ENABLE_OFFLOAD_COMPRESS "Build with offload compression.")
shim_mapping(WERROR ROCSOLVER_ENABLE_WERROR "Treat warnings as errors.")
shim_mapping(USE_HIPCXX ROCSOLVER_ENABLE_HIPCXX "Use CMake HIP language support.")
shim_mapping(BUILD_LIBRARY ROCSOLVER_BUILD_LIBRARY "Build rocSOLVER library.")

# Map SKIP_LIBRARY → ROCSOLVER_BUILD_LIBRARY (inverted logic)
if(DEFINED SKIP_LIBRARY)
    _rocsolver_check_conflict(SKIP_LIBRARY ROCSOLVER_BUILD_LIBRARY)
    if(NOT DEFINED ROCSOLVER_BUILD_LIBRARY)
        if(SKIP_LIBRARY)
            set(ROCSOLVER_BUILD_LIBRARY OFF CACHE BOOL "Build rocSOLVER library." FORCE)
        else()
            set(ROCSOLVER_BUILD_LIBRARY ON CACHE BOOL "Build rocSOLVER library." FORCE)
        endif()
        _rocsolver_deprecation_warning(SKIP_LIBRARY ROCSOLVER_BUILD_LIBRARY)
        list(APPEND _ROCSOLVER_LEGACY_OPTIONS_USED "SKIP_LIBRARY=${SKIP_LIBRARY}")
        list(APPEND _ROCSOLVER_CURRENT_OPTIONS "ROCSOLVER_BUILD_LIBRARY=${ROCSOLVER_BUILD_LIBRARY}")
    endif()
endif()

# ==============================================================================
# Display Migration Guidance
# ==============================================================================

if(_ROCSOLVER_LEGACY_OPTIONS_USED)
    # Format the current options with -D prefix for each
    string(REPLACE ";" " -D" _formatted_current_options "${_ROCSOLVER_CURRENT_OPTIONS}")
    set(_formatted_current_options "-D${_formatted_current_options}")

    message(DEPRECATION
        "\n"
        "  Legacy build options detected:\n"
        "    ${_ROCSOLVER_LEGACY_OPTIONS_USED}\n"
        "\n"
        "  To use current options, run:\n"
        "    cmake ${_formatted_current_options} ..\n"
        "\n"
        "  To suppress warnings: cmake -DCMAKE_WARN_DEPRECATED=OFF ...\n"
    )
endif()
