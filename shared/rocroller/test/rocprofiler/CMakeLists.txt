# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

# Find HIP compiler if not already set
if(NOT CMAKE_HIP_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    endif()
endif()

# Enable HIP language
enable_language(HIP)

# Find required packages
find_package(rocprofiler-sdk)
find_package(rocprofiler-sdk-roctx)
find_package(amd_comgr CONFIG)

# Find system elfutils libdw
find_path(LIBDW_INCLUDE_DIR 
    NAMES elfutils/libdw.h
    HINTS /usr/include /usr/local/include)

find_library(LIBDW_LIBRARY
    NAMES dw
    HINTS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

# Check if all required packages are found
if(NOT rocprofiler-sdk_FOUND OR NOT rocprofiler-sdk-roctx_FOUND OR NOT amd_comgr_FOUND)
    message(WARNING "rocprofiler-sdk test dependencies not found. Skipping rocprofiler test.")
    return()
endif()

if(NOT LIBDW_INCLUDE_DIR OR NOT LIBDW_LIBRARY)
    message(WARNING "elfutils libdw not found. Skipping rocprofiler test. Install libdw-dev or elfutils-devel.")
    return()
endif()

# Set C++ and HIP standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# Set HIP flags from CXX flags for different build types
foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

# Set source files properties
set_source_files_properties(
    main.cpp PROPERTIES LANGUAGE HIP COMPILE_FLAGS "${CMAKE_HIP_FLAGS_RELWITHDEBINFO}")

# Create the executable
add_executable(rocprofiler-thread-trace-test main.cpp agent.cpp)

# Add libdw include directories
target_include_directories(rocprofiler-thread-trace-test PRIVATE ${LIBDW_INCLUDE_DIR})

# Link libraries
target_link_libraries(
    rocprofiler-thread-trace-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk 
            rocprofiler-sdk-roctx::rocprofiler-sdk-roctx 
            amd_comgr
            ${LIBDW_LIBRARY})

# Find the trace decoder library (optional)
find_library(
    attdecoder_LIBRARY
    NAMES rocprof-trace-decoder
    HINTS ${ROCM_PATH}
    PATHS ${ROCM_PATH}
    PATH_SUFFIXES lib)

if(attdecoder_LIBRARY)
    cmake_path(GET attdecoder_LIBRARY PARENT_PATH attdecoder_LIB_DIR)
endif()

find_package_handle_standard_args(attdecoder REQUIRED_VARS attdecoder_LIB_DIR
                                                           attdecoder_LIBRARY)

# Determine if test should be disabled
set(IS_DISABLED ON)
if(attdecoder_FOUND)
    set(IS_DISABLED OFF)
endif()

# Create preload environment
set(PRELOAD_ENV)
if(attdecoder_FOUND)
    list(APPEND PRELOAD_ENV "ROCPROFILER_TRACE_DECODER_LIB_PATH=${attdecoder_LIB_DIR}")
endif()

# Add test
add_test(NAME rocprofiler-thread-trace-test COMMAND $<TARGET_FILE:rocprofiler-thread-trace-test>)

# Set test properties
set_tests_properties(
    rocprofiler-thread-trace-test 
    PROPERTIES TIMEOUT 60 
               ENVIRONMENT "${PRELOAD_ENV}" 
               LABELS "GPU;rocprofiler"
               DISABLED ${IS_DISABLED})

# If using address sanitizer, configure it for this target
if(ROCROLLER_ENABLE_ASAN)
    rocroller_target_configure_sanitizers(rocprofiler-thread-trace-test PRIVATE)
    set_target_properties(rocprofiler-thread-trace-test
        PROPERTIES
            CROSSCOMPILING_EMULATOR "${CMAKE_COMMAND};-E;env;${asan_opts}"
    )
endif()
