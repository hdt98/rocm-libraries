# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################
cmake_minimum_required(VERSION 3.25.0)
project(hostlibrarytests LANGUAGES CXX)

set(HOST_LIB_TESTS_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite/HostLibraryTests")

# find_package(GTest REQUIRED)
enable_testing()
find_package( GTest )
if(GTest_FOUND)
    option(GTEST_INTEGRATION "Enable Googletest integration with CMake." ON)
else()
    option(GTEST_INTEGRATION "Enable Googletest integration with CMake." OFF)
endif()



find_package(Boost REQUIRED COMPONENTS filesystem)

add_executable(hostlib-tests)

add_subdirectory(configs)

# from old cmakelists.txt(did not know how to modernize it)
set(TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
file(MAKE_DIRECTORY "${TEST_DATA_DIR}")

foreach(FILE ${SOLUTION_LIBRARY_FILES})
    if(EXISTS "${FILE}.gz")
        execute_process(COMMAND gzip --keep --decompress --force "${FILE}.gz")
    endif()
    file(COPY ${FILE} DESTINATION "${TEST_DATA_DIR}")
endforeach()

add_subdirectory(testlib)

target_sources(hostlib-tests
    PRIVATE
        "${HOST_LIB_TESTS_SRC_DIR}/test.cpp"
        "${HOST_LIB_TESTS_SRC_DIR}/ContractionSelectionLibrary_test.cpp"
        "${HOST_LIB_TESTS_SRC_DIR}/DataTypes_test.cpp"
        "${HOST_LIB_TESTS_SRC_DIR}/Predicates_test.cpp"
)

if(HIPBLASLT_ENABLE_LLVM)
    target_sources(hostlib-tests
        PRIVATE
            "${HOST_LIB_TESTS_SRC_DIR}/llvm/LibraryPerformance_test.cpp"
    )
endif()

target_link_libraries(hostlib-tests
    PUBLIC
        tensilelite::tensilelite-host
        Boost::filesystem
)

if(NOT TENSILE_DISABLE_CTEST)
    if(GTEST_INTEGRATION)
        gtest_discover_tests(hostlib-tests WORKING_DIRECTORY ${CMAKE_BINARY_DIR} TIMEOUT 60)
    else()
        add_test(NAME TensileHostLibraryTests COMMAND hostlib-tests WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    endif()
endif()


target_link_libraries(hostlib-tests PUBLIC GTest::gtest)


if(HIPBLASLT_ENABLE_OPENMP)
    target_link_libraries(hostlib-tests PRIVATE OpenMP::OpenMP_CXX)
endif()

if(HIPBLASLT_ENABLE_LLVM)
    target_link_libraries(hostlib-tests PRIVATE LLVM)
endif()

# include(GoogleTest)
# gtest_discover_tests(hostlib-tests)
