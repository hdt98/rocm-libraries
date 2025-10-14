# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

enable_language(HIP)

find_package(rocprofiler-sdk REQUIRED)
find_package(rocprofiler-sdk-roctx REQUIRED)
find_package(
    amd_comgr
    REQUIRED
    CONFIG
    HINTS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATHS
    ${rocm_version_DIR}
    ${ROCM_PATH})
find_library(
    LIBDW_LIBRARY
    NAMES dw
    HINTS ${ROCM_PATH} /usr
    PATHS ${ROCM_PATH} /usr
    PATH_SUFFIXES lib lib64)

add_library(rocprofiler-agent SHARED)

target_sources(rocprofiler-agent
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/agent.cpp"
)

target_link_libraries(rocprofiler-agent
    PRIVATE
        rocprofiler-sdk::rocprofiler-sdk
        rocprofiler-sdk-roctx::rocprofiler-sdk-roctx
        amd_comgr
        ${LIBDW_LIBRARY}
)

# HIP kernel version
add_executable(hip-kernel)

set_source_files_properties(
    hip_main.cpp PROPERTIES LANGUAGE HIP)

target_sources(hip-kernel
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/hip_main.cpp"
)

target_link_libraries(hip-kernel
    PRIVATE
        rocprofiler-sdk-roctx::rocprofiler-sdk-roctx
        ${BLAS_LIBRARIES}
)

# rocRoller version
add_executable(rr-kernel)

target_sources(rr-kernel
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/rr_main.cpp"
)

target_link_libraries(rr-kernel
    PRIVATE
        rocprofiler-sdk-roctx::rocprofiler-sdk-roctx
        ${BLAS_LIBRARIES}
        rocroller::common-test-utilities
)
