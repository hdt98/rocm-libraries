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

function(add_example EXAMPLE_SOURCE)
  get_filename_component(EXAMPLE_TARGET ${EXAMPLE_SOURCE} NAME_WE)

  if(USE_HIPCXX)
    set_source_files_properties(${EXAMPLE_SOURCE} PROPERTIES LANGUAGES HIP)
  endif()

  add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE})

  target_link_libraries(${EXAMPLE_TARGET}
    PRIVATE
    rocprim
  )

  if(USE_HIPCXX)
    target_link_libraries(${EXAMPLE_TARGET}
      PRIVATE
      $<IF:$<LINK_LANGUAGE:HIP>,hip::host,hip::device>
    )
  else()
    target_link_libraries(${EXAMPLE_TARGET}
      PRIVATE
      hip::device)
  endif()

  set_target_properties(${EXAMPLE_TARGET}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example"
  )

  # Use an EXAMPLE_SOURCE like "rocprim/device/example_device_search.cpp"
  # to create a test named "example.rocprim.device.example_device_search"
  get_filename_component(DIR_PATH "${EXAMPLE_SOURCE}" PATH)
  string(REPLACE "/" "." DIR_PATH_DOTS "${DIR_PATH}")
  set(TEST_NAME "example.${DIR_PATH_DOTS}.${EXAMPLE_TARGET}")
  add_test(NAME ${TEST_NAME} COMMAND ${EXAMPLE_TARGET})

endfunction()

# Copy DLLs and test support files at configure time on Windows.
if(WIN32)
  file(GLOB _test_support_files
    ${HIP_DIR}/bin/*.dll
    ${CMAKE_SOURCE_DIR}/rtest.*
  )
  if(_test_support_files)
    file(COPY ${_test_support_files} DESTINATION ${PROJECT_BINARY_DIR}/example)
  endif()
endif()

# ****************************************************************************
# rocPRIM examples
# ****************************************************************************

add_example(extra/example_temporary_storage.cpp)
add_example(extra/example_type_traits_interface.cpp)
add_example(rocprim/device/example_device_search.cpp)
