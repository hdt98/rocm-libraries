# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(FindPackageHandleStandardArgs)

# ---------------------------------------------------------------------------
# AOCL 5.x unified library (libaocl) — checked first, highest priority
# ---------------------------------------------------------------------------

# Build-tree AOCL 5.x (built by install.sh --deps)
if(EXISTS "${CMAKE_BINARY_DIR}/deps/aocl/install_package/lib/libaocl.a")
    set(BLIS_LIB "${CMAKE_BINARY_DIR}/deps/aocl/install_package/lib/libaocl.a")
    find_path(BLIS_INCLUDE_DIR
        NAMES blis.h
        PATHS "${CMAKE_BINARY_DIR}/deps/aocl/install_package/include_ILP64"
              "${CMAKE_BINARY_DIR}/deps/aocl/install_package/include"
        NO_DEFAULT_PATH
    )
endif()

# System AOCL 5.x — glob across /opt/AMD/aocl/<version>/, prefer newest
if(NOT BLIS_LIB)
    file(GLOB _aocl5_dirs "/opt/AMD/aocl/[0-9]*")
    list(SORT _aocl5_dirs)
    list(REVERSE _aocl5_dirs)
    foreach(_dir IN LISTS _aocl5_dirs)
        foreach(_libdir IN ITEMS lib lib64)
            if(EXISTS "${_dir}/${_libdir}/libaocl.a")
                set(BLIS_LIB "${_dir}/${_libdir}/libaocl.a")
            elseif(EXISTS "${_dir}/${_libdir}/libaocl.so")
                set(BLIS_LIB "${_dir}/${_libdir}/libaocl.so")
            endif()
            if(BLIS_LIB)
                if(EXISTS "${_dir}/include_ILP64")
                    set(BLIS_INCLUDE_DIR "${_dir}/include_ILP64")
                elseif(EXISTS "${_dir}/include")
                    set(BLIS_INCLUDE_DIR "${_dir}/include")
                endif()
                break()
            endif()
        endforeach()
        if(BLIS_LIB)
            break()
        endif()
    endforeach()
    unset(_aocl5_dirs)
    unset(_dir)
    unset(_libdir)
endif()

# ---------------------------------------------------------------------------
# AOCL 4.x BLIS library — glob across installed versions, prefer GCC/ILP64
# ---------------------------------------------------------------------------

# Helper: search for libblis-mt.a for a given compiler toolchain
function(_rocblas_find_libblis compiler out_lib out_inc)
    file(GLOB _dirs "/opt/AMD/aocl/aocl-linux-${compiler}-*")
    list(SORT _dirs)
    list(REVERSE _dirs)
    foreach(_dir IN LISTS _dirs)
        get_filename_component(_dir_name "${_dir}" NAME)
        if(_dir_name MATCHES "aocl-linux-${compiler}-([0-9]+)\\.([0-9]+)\\.([0-9]+)")
            set(_lib "${_dir}/${compiler}/lib_ILP64/libblis-mt.a")
            if(EXISTS "${_lib}")
                set(${out_lib} "${_lib}" PARENT_SCOPE)
                set(${out_inc} "${_dir}/${compiler}/include_ILP64" PARENT_SCOPE)
                return()
            endif()
        endif()
    endforeach()
endfunction()

if(NOT BLIS_LIB)
    _rocblas_find_libblis(gcc _blis_lib _blis_inc)
    if(_blis_lib)
        set(BLIS_LIB "${_blis_lib}")
        set(BLIS_INCLUDE_DIR "${_blis_inc}")
    endif()
    unset(_blis_lib)
    unset(_blis_inc)
endif()

if(NOT BLIS_LIB)
    _rocblas_find_libblis(aocc _blis_lib _blis_inc)
    if(_blis_lib)
        set(BLIS_LIB "${_blis_lib}")
        set(BLIS_INCLUDE_DIR "${_blis_inc}")
        set(BLIS_WARN_NOT_ILP64_PREFERRED TRUE)
    endif()
    unset(_blis_lib)
    unset(_blis_inc)
endif()

# ---------------------------------------------------------------------------
# Build-tree bundled BLIS (older install.sh builds)
# ---------------------------------------------------------------------------
if(NOT BLIS_LIB)
    foreach(_candidate IN ITEMS
            "${CMAKE_BINARY_DIR}/deps/amd-blis/lib/ILP64/libblis-mt.a"
            "${CMAKE_BINARY_DIR}/deps/blis/lib/libblis.a"
            "/usr/local/lib/libblis.a")
        if(EXISTS "${_candidate}")
            set(BLIS_LIB "${_candidate}")
            set(BLIS_WARN_NOT_ILP64_PREFERRED TRUE)
            break()
        endif()
    endforeach()
    if(BLIS_LIB AND NOT BLIS_INCLUDE_DIR)
        find_path(BLIS_INCLUDE_DIR
            NAMES blis.h
            PATHS
                "${CMAKE_BINARY_DIR}/deps/amd-blis/include/ILP64"
                "${CMAKE_BINARY_DIR}/deps/blis/include/blis"
                "/usr/local/include/blis"
            NO_DEFAULT_PATH
        )
    endif()
    unset(_candidate)
endif()

# ---------------------------------------------------------------------------
# User-supplied BLIS_ROOT override
# ---------------------------------------------------------------------------
if(BLIS_ROOT OR DEFINED ENV{BLIS_ROOT})
    find_library(BLIS_LIB
        NAMES libaocl.a libblis-mt.a libblis.a libaocl.so
        PATHS ${BLIS_ROOT} ENV BLIS_ROOT
        NO_DEFAULT_PATH
    )
    find_path(BLIS_INCLUDE_DIR
        NAMES blis.h
        PATHS ${BLIS_ROOT} ENV BLIS_ROOT
        NO_DEFAULT_PATH
    )
endif()

# ---------------------------------------------------------------------------
# Windows: AOCL-Windows static lib
# ---------------------------------------------------------------------------
if(WIN32 AND NOT BLIS_LIB)
    file(TO_CMAKE_PATH
        "C:/Program Files/AMD/AOCL-Windows/amd-blis/lib/ILP64/AOCL-LibBlis-Win-MT.lib"
        _aocl_win_lib)
    if(EXISTS "${_aocl_win_lib}")
        set(BLIS_LIB "${_aocl_win_lib}")
        set(BLIS_INCLUDE_DIR "C:/Program Files/AMD/AOCL-Windows/amd-blis/include/ILP64")
    endif()
    unset(_aocl_win_lib)
endif()

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
find_package_handle_standard_args(BLIS
    REQUIRED_VARS BLIS_LIB BLIS_INCLUDE_DIR
)

if(BLIS_FOUND)
    set(BLIS_LIBRARIES "${BLIS_LIB}")
    set(BLIS_INCLUDE_DIRS "${BLIS_INCLUDE_DIR}")

    if(BLIS_WARN_NOT_ILP64_PREFERRED)
        message(WARNING
            "Using ${BLIS_LIB} as reference BLAS library — 64-bit tests may "
            "fail. Run tests with --gtest_filter=-*stress*")
    endif()

    if(NOT TARGET BLIS::BLIS)
        add_library(BLIS::BLIS UNKNOWN IMPORTED)
        set_target_properties(BLIS::BLIS PROPERTIES
            IMPORTED_LOCATION "${BLIS_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${BLIS_INCLUDE_DIR}"
        )
    endif()

    message(STATUS "Found BLIS library: ${BLIS_LIB}")
    message(STATUS "Found BLIS includes: ${BLIS_INCLUDE_DIR}")
endif()
