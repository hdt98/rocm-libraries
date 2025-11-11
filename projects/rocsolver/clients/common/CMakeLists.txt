# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Create OBJECT library for common test/benchmark code
# This is used by both rocsolver-test and rocsolver-bench
add_library(rocsolver-test-common OBJECT)

# Enable position-independent code for linking into shared libraries
if(BUILD_SHARED_LIBS)
    set_target_properties(rocsolver-test-common PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endif()

target_compile_features(rocsolver-test-common PRIVATE cxx_std_17)

target_compile_definitions(rocsolver-test-common PRIVATE
    ROCM_USE_FLOAT16
    ROCBLAS_NO_DEPRECATED_WARNINGS
)

target_include_directories(rocsolver-test-common PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../"
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/include>
)

target_link_libraries(rocsolver-test-common PUBLIC
    roc::rocsolver
    ${LAPACK_LIBRARIES}
    fmt::fmt-header-only
    Threads::Threads
    $<$<PLATFORM_ID:Linux>:stdc++fs>
)

if(LAPACK_LINKER_FLAGS)
    target_link_options(rocsolver-test-common PUBLIC
        ${LAPACK_LINKER_FLAGS}
    )
endif()

# Add sources from subdirectories
add_subdirectory(auxiliary)
add_subdirectory(lapack)
add_subdirectory(refact)
add_subdirectory(unit)
add_subdirectory(misc)
add_subdirectory(matrix_utils)
