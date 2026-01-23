# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# API tests are built into the rocroller-tests-api target (created by the parent test CMake).
# This file wires up dependencies and registers sources under test/api/.

find_package(Catch2 QUIET)
if(NOT Catch2_FOUND)
    if(ROCROLLER_ENABLE_FETCH)
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

target_link_libraries(rocroller-tests-api
    PRIVATE
        rocroller::common-test-utilities
        Catch2::Catch2WithMain
        ${BLAS_LIBRARIES}
        ${msgpack_libs}
)

target_sources(rocroller-tests-api
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/LazySingletonAPITest.cpp"
)

target_include_directories(rocroller-tests-api
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../client/include
)
