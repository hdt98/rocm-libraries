# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(print_configuration_summary)
    find_package(Git)
    if(GIT_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} show --format=%H --no-patch
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            OUTPUT_VARIABLE COMMIT_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        execute_process(
            COMMAND ${GIT_EXECUTABLE} show --format=%s --no-patch
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            OUTPUT_VARIABLE COMMIT_SUBJECT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()

    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} --version
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE CMAKE_CXX_COMPILER_VERBOSE_DETAILS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    find_program(UNAME_EXECUTABLE uname)
    if(UNAME_EXECUTABLE)
        execute_process(
            COMMAND ${UNAME_EXECUTABLE} -a
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            OUTPUT_VARIABLE LINUX_KERNEL_DETAILS
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()

    string(REPLACE "\n" ";" CMAKE_CXX_COMPILER_VERBOSE_DETAILS "${CMAKE_CXX_COMPILER_VERBOSE_DETAILS}")
    list(TRANSFORM CMAKE_CXX_COMPILER_VERBOSE_DETAILS PREPEND "--     ")
    string(REPLACE ";" "\n" CMAKE_CXX_COMPILER_VERBOSE_DETAILS "${CMAKE_CXX_COMPILER_VERBOSE_DETAILS}")

    message(STATUS "")
    message(STATUS "******** Summary ********")
    message(STATUS "General:")
    message(STATUS "  System                      : ${CMAKE_SYSTEM_NAME}")
    if(ROCPRIM_USE_HIPCXX)
        message(STATUS "  HIP compiler                : ${CMAKE_HIP_COMPILER}")
        message(STATUS "  HIP compiler version        : ${CMAKE_HIP_COMPILER_VERSION}")
        string(STRIP "${CMAKE_HIP_FLAGS}" CMAKE_HIP_FLAGS_STRIP)
        message(STATUS "  HIP flags                   : ${CMAKE_HIP_FLAGS_STRIP}")
    else()
        message(STATUS "  C++ compiler                : ${CMAKE_CXX_COMPILER}")
        message(STATUS "  C++ compiler version        : ${CMAKE_CXX_COMPILER_VERSION}")
        string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS_STRIP)
        message(STATUS "  CXX flags                   : ${CMAKE_CXX_FLAGS_STRIP}")
    endif()
    get_property(GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(GENERATOR_IS_MULTI_CONFIG)
        message(STATUS "  Build types                 : ${CMAKE_CONFIGURATION_TYPES}")
    else()
        message(STATUS "  Build type                  : ${CMAKE_BUILD_TYPE}")
    endif()
    message(STATUS "  Install prefix              : ${CMAKE_INSTALL_PREFIX}")
    if(ROCPRIM_USE_HIPCXX)
        message(STATUS "  Device targets              : ${CMAKE_HIP_ARCHITECTURES}")
    else()
        message(STATUS "  Device targets              : ${GPU_TARGETS}")
    endif()
    message(STATUS "")
    message(STATUS "  ROCPRIM_ENABLE_INSTALL       : ${ROCPRIM_ENABLE_INSTALL}")
    message(STATUS "  ROCPRIM_BUILD_TESTING        : ${ROCPRIM_BUILD_TESTING}")
    message(STATUS "  ROCPRIM_ENABLE_ROCRAND       : ${ROCPRIM_ENABLE_ROCRAND}")
    message(STATUS "  ROCPRIM_ENABLE_BENCHMARK     : ${ROCPRIM_ENABLE_BENCHMARK}")
    message(STATUS "  ROCPRIM_ENABLE_NAIVE_BENCHMARK : ${ROCPRIM_ENABLE_NAIVE_BENCHMARK}")
    message(STATUS "  ROCPRIM_ENABLE_EXAMPLES      : ${ROCPRIM_ENABLE_EXAMPLES}")
    message(STATUS "  ROCPRIM_ENABLE_DOCS          : ${ROCPRIM_ENABLE_DOCS}")
    message(STATUS "  ROCPRIM_ENABLE_OFFLOAD_COMPRESS : ${ROCPRIM_ENABLE_OFFLOAD_COMPRESS}")
    message(STATUS "  ROCPRIM_USE_SYSTEM_LIBS      : ${ROCPRIM_USE_SYSTEM_LIBS}")
    message(STATUS "")
    message(STATUS "Detailed:")
    message(STATUS "  C++ compiler details        : \n${CMAKE_CXX_COMPILER_VERBOSE_DETAILS}")
    if(GIT_FOUND)
        message(STATUS "  Commit                      : ${COMMIT_HASH}")
        message(STATUS "                                ${COMMIT_SUBJECT}")
    endif()
    if(UNAME_EXECUTABLE)
        message(STATUS "  Unix name                   : ${LINUX_KERNEL_DETAILS}")
    endif()
endfunction()
