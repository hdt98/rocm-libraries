# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

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

find_package(Catch2 QUIET)
if(NOT Catch2_FOUND)
    if(ROCROLLER_ENABLE_FETCH)
        include(FetchContent)
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG devel
        )
        FetchContent_MakeAvailable(Catch2)
    else()
        message(FATAL_ERROR "Failed to find Catch2")
    endif()
endif()

target_link_libraries(rr-kernel-catch
    PRIVATE
        Catch2::Catch2WithMain
        rocprofiler-sdk::rocprofiler-sdk
        rocprofiler-sdk-roctx::rocprofiler-sdk-roctx
        amd_comgr
        ${LIBDW_LIBRARY}
        ${BLAS_LIBRARIES}
        rocroller::common-test-utilities
)

target_sources(rr-kernel-catch
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/rr_main.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/agent.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../catch/TestContext.cpp"
)
