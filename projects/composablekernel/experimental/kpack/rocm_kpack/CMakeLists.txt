# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# ROCm Kpack Runtime Library (vendored)

set(KPACK_SOURCES
    src/kpack.cpp
    src/archive.cpp
    src/compression.cpp
    src/toc_parser.cpp
    src/path_resolution.cpp
    src/isa_target_match.cpp
    src/loader.cpp
)

set(KPACK_HEADERS
    include/rocm_kpack/kpack.h
    include/rocm_kpack/kpack_types.h
    include/rocm_kpack/kpack_export.h
)

# Build as static library
add_library(rocm_kpack STATIC ${KPACK_SOURCES} ${KPACK_HEADERS})

# Include directories
target_include_directories(rocm_kpack
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Link dependencies - PRIVATE since they're not exposed in public API
find_package(msgpack-cxx REQUIRED)
find_package(zstd REQUIRED)

# zstd cmake config exports zstd::libzstd_static / zstd::libzstd_shared
if(TARGET zstd::libzstd_static)
    set(ZSTD_TARGET zstd::libzstd_static)
else()
    set(ZSTD_TARGET zstd::libzstd_shared)
endif()

target_link_libraries(rocm_kpack
    PRIVATE
        msgpack-cxx
        ${ZSTD_TARGET}
)

# Static library: define ROCM_KPACK_STATIC for consumers
target_compile_definitions(rocm_kpack PUBLIC ROCM_KPACK_STATIC)

# Symbol visibility control
set_target_properties(rocm_kpack PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
)
