# MIT License
#
# Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.21...3.25 FATAL_ERROR)

# This project includes tests that should be run after
# rocPRIM is installed from package or using `make install`
project(rocprim_package_install_test LANGUAGES CXX HIP)
set(VERSION_STRING "4.2.0")

find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/rocprim")

include(../../cmake/Dependencies.cmake)
fetch_rocmcmake()

include(../../cmake/Config.cmake)
setup_languages()
get_gpu_targets()

include(../../cmake/ConfigUtils.cmake)

# Enable testing (ctest)
enable_testing()
print_config()

# Build
function(add_rocprim_test TEST_NAME TEST_SOURCES)
  list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
  add_executable(${TEST_TARGET} ${TEST_SOURCES})
  target_link_libraries(${TEST_TARGET}
    PRIVATE
      ${rocprim_LIBRARIES} # roc::rocprim_hip
  )
  add_test(${TEST_NAME} ${TEST_TARGET})
endfunction()

add_rocprim_test("test_rocprim_package" test_rocprim_package.hip)
