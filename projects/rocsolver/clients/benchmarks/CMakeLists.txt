# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(rocsolver-bench)

target_sources(rocsolver-bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/client.cpp"
    "${PROJECT_SOURCE_DIR}/common/src/common_host_helpers.cpp"
)

target_compile_features(rocsolver-bench PRIVATE cxx_std_17)

add_armor_flags(rocsolver-bench "${ARMOR_LEVEL}")

target_compile_definitions(rocsolver-bench PRIVATE
    ROCM_USE_FLOAT16
    ROCSOLVER_CLIENTS_BENCH
)

# Explicit dependencies - no INTERFACE library
target_link_libraries(rocsolver-bench PRIVATE
    roc::rocsolver
    rocsolver-test-common
    rocsolver-common
    hip::host
    Threads::Threads
    ${LAPACK_LIBRARIES}
    fmt::fmt-header-only
    $<$<PLATFORM_ID:Linux>:stdc++fs>
    $<$<PLATFORM_ID:Linux>:m>
)

if(NOT USE_HIPCXX)
    target_link_libraries(rocsolver-bench PRIVATE hip::device)
endif()

if(LAPACK_LINKER_FLAGS)
    target_link_options(rocsolver-bench PRIVATE
        ${LAPACK_LINKER_FLAGS}
    )
endif()

rocm_install(TARGETS rocsolver-bench COMPONENT benchmarks)
