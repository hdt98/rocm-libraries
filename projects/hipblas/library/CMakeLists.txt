# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# This is incremented when the ABI to the library changes
set(hipblas_SOVERSION 3.2)

# Get the git hash of the hipBLAS branch
find_package(Git REQUIRED)

execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    OUTPUT_VARIABLE GIT_HASH_HIPBLAS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(hipblas_VERSION_COMMIT_ID "${GIT_HASH_HIPBLAS}")

message(STATUS "Building hipBLAS commit: ${hipblas_VERSION_COMMIT_ID}")

# Create hipBLAS library target
if(HIPBLAS_BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
    add_library(hipblas SHARED)
else()
    add_library(hipblas STATIC)
endif()

add_library(roc::hipblas ALIAS hipblas)

# Set library versioning
rocm_set_soversion(hipblas ${hipblas_SOVERSION})

# Target compile features
target_compile_features(hipblas PRIVATE cxx_std_17)

# Target properties
set_target_properties(hipblas PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

# Following Boost conventions of prefixing 'lib' on static built libraries
if(NOT HIPBLAS_BUILD_SHARED_LIBS AND NOT BUILD_SHARED_LIBS)
    set_target_properties(hipblas PROPERTIES PREFIX "lib")
endif()

# Allow target_link_libraries in subdirectories (CMP0079)
cmake_policy(SET CMP0079 NEW)

# Generate export header
include(GenerateExportHeader)
generate_export_header(hipblas EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hipblas/hipblas-export.h)

# Code coverage support
if(HIPBLAS_BUILD_COVERAGE)
    target_compile_options(hipblas PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(hipblas PRIVATE -fprofile-instr-generate)
endif()

# Address sanitizer support
if(HIPBLAS_ENABLE_ASAN)
    target_compile_options(hipblas PRIVATE -fsanitize=address -shared-libasan)
    target_link_options(hipblas PRIVATE -fsanitize=address -shared-libasan -fuse-ld=lld)
endif()

# Add solver compile definition
if(HIPBLAS_ENABLE_SOLVER)
    target_compile_definitions(hipblas PRIVATE __HIP_PLATFORM_SOLVER__)
endif()

# Add subdirectories for headers and sources
add_subdirectory(include)
add_subdirectory(src)

# Install library target to runtime component
rocm_install_targets(
    TARGETS hipblas
    COMPONENT runtime
)

# Install headers to devel component
rocm_install(
    DIRECTORY ${CMAKE_BINARY_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT devel
)

# Export targets
set(static_depends)

if(HIPBLAS_ENABLE_HIP)
    list(APPEND static_depends PACKAGE rocblas)
    if(HIPBLAS_ENABLE_SOLVER)
        list(APPEND static_depends PACKAGE rocsolver)
    endif()
    
    rocm_export_targets(
        TARGETS roc::hipblas
        DEPENDS PACKAGE hip
        DEPENDS PACKAGE hipblas-common
        STATIC_DEPENDS ${static_depends}
        NAMESPACE roc::
    )
else()
    rocm_export_targets(
        TARGETS roc::hipblas
        DEPENDS PACKAGE HIP
        DEPENDS PACKAGE hipblas-common
        NAMESPACE roc::
    )
endif()

# Force installation of .f90 module files
rocm_install(
    FILES "src/hipblas_module.f90"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hipblas"
)
