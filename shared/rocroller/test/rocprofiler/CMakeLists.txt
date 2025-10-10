# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

enable_language(HIP)

# rocprofiler-sdk dependencies
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

add_executable(rocprofiler-thread-trace)

set_source_files_properties(
    main.cpp PROPERTIES LANGUAGE HIP)

target_sources(rocprofiler-thread-trace
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/agent.cpp"
)

find_library(
    LIBDW_LIBRARY
    NAMES dw
    HINTS ${ROCM_PATH} /usr
    PATHS ${ROCM_PATH} /usr
    PATH_SUFFIXES lib lib64)

if(NOT LIBDW_LIBRARY)
    message(FATAL_ERROR "libdw not found. Please install elfutils-devel or elfutils-libelf-devel")
endif()

# Link libraries
target_link_libraries(rocprofiler-thread-trace
    PRIVATE
        rocprofiler-sdk::rocprofiler-sdk
        rocprofiler-sdk-roctx::rocprofiler-sdk-roctx
        amd_comgr
        ${LIBDW_LIBRARY}
        ${BLAS_LIBRARIES}
        rocroller::common-test-utilities
)

if(ROCROLLER_ENABLE_ASAN)
    rocroller_target_configure_sanitizers(rocprofiler-thread-trace PRIVATE)
endif()

find_library(
    attdecoder_LIBRARY
    NAMES rocprof-trace-decoder
    HINTS ${ROCM_PATH}
    PATHS ${ROCM_PATH}
    PATH_SUFFIXES lib)

if(attdecoder_LIBRARY)
    cmake_path(GET attdecoder_LIBRARY PARENT_PATH attdecoder_LIB_DIR)
    message(STATUS "Found rocprof-trace-decoder library at: ${attdecoder_LIB_DIR}")
else()
    message(WARNING "rocprof-trace-decoder library not found. Thread trace decoding may not work.")
endif()
