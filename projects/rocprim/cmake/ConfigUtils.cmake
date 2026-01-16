# MIT License
#
# Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

include(CMakeDependentOption)

set(SUMMARY_KEY_PADDING 32 CACHE INTERNAL "Padding width of the key value when printing summary.")

# Pads a '_str' with spaces to length '_length' and
# stores it to '_var'.
function(pad_string _var _str _length)
  string(LENGTH "${_str}" _strlen)
  math(EXPR _strlen "${_length} - ${_strlen}")

  if(_strlen GREATER_EQUAL 0)
    string(REPEAT " " ${_strlen} _padding)
    string(APPEND _str ${_padding})
  endif()
  set(${_var} "  ${_str}" PARENT_SCOPE)
endfunction(pad_string)

# For each variable in '_vars'. Print the padded
# variable name and value.
function(print_summary_vars _vars)
  foreach(_var ${${_vars}})
    pad_string(_key ${_var} ${SUMMARY_KEY_PADDING})
    message(STATUS "${_key}: ${${_var}}")
  endforeach()
endfunction()

# Wraps around 'option(...)', but also registers the
# variable to 'SUMMARY_OPTION_VARS'.
macro(register_option _var _doc _default)
  option(${_var} ${_doc} ${_default})
  list(APPEND SUMMARY_OPTION_VARS ${_var})
endmacro()

# Wraps around 'set(...)', but also registers the
# variable to 'SUMMARY_OPTION_VARS'.
macro(register_set _var _default _type _doc)
  set(${_var} ${_default} CACHE ${_type} ${_doc})
  list(APPEND SUMMARY_OPTION_VARS ${_var})
endmacro()

# Wraps around 'cmake_dependent_option(...)', but also 
# registers the variable to 'SUMMARY_OPTION_VARS'.
macro(register_dep_option _var _doc _default _cond _else)
  cmake_dependent_option(${_var} ${_doc} ${_default} ${_cond} ${_else})
  list(APPEND SUMMARY_OPTION_VARS ${_var})
endmacro()

# Prints the current configuration and some extra verbose details.
function(print_config)
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

  # Prettify CMAKE_CXX_COMPILER_VERBOSE_DETAILS
  string(REPLACE "\n" ";" CMAKE_CXX_COMPILER_VERBOSE_DETAILS "${CMAKE_CXX_COMPILER_VERBOSE_DETAILS}")
  list(TRANSFORM CMAKE_CXX_COMPILER_VERBOSE_DETAILS PREPEND "\n--     ")
  string(REPLACE ";" "" CMAKE_CXX_COMPILER_VERBOSE_DETAILS "${CMAKE_CXX_COMPILER_VERBOSE_DETAILS}")

  # Configure what variables to print.
  # Basic info:
  list(APPEND SUMMARY_CMAKE_VARS CMAKE_PROJECT_NAME CMAKE_SYSTEM_NAME)
  # Compiler details:
  list(APPEND SUMMARY_CMAKE_VARS CMAKE_HIP_COMPILER CMAKE_HIP_COMPILER_VERSION CMAKE_HIP_FLAGS)
  list(APPEND SUMMARY_CMAKE_VARS CMAKE_CXX_COMPILER CMAKE_CXX_COMPILER_VERSION CMAKE_CXX_FLAGS)
  # Build type:
  list(APPEND SUMMARY_CMAKE_VARS CMAKE_CONFIGURATION_TYPES CMAKE_BUILD_TYPE)
  # Archs:
  list(APPEND SUMMARY_CMAKE_VARS CMAKE_HIP_ARCHITECTURES GPU_TARGETS)
  # Compiler version:
  list(APPEND SUMMARY_DETAIL_VARS CMAKE_CXX_COMPILER_VERBOSE_DETAILS)
  # Commit:
  if(GIT_FOUND)
    list(APPEND SUMMARY_DETAIL_VARS COMMIT_HASH COMMIT_SUBJECT)
  endif()
  # Host system.
  if(UNAME_EXECUTABLE)
    list(APPEND SUMMARY_DETAIL_VARS LINUX_KERNEL_DETAILS)
  endif()

  # Print variables
  message(STATUS "******** Summary ********")
  message(STATUS "General:")
  print_summary_vars(SUMMARY_CMAKE_VARS)
  message(STATUS "Options:")
  print_summary_vars(SUMMARY_OPTION_VARS)
  message(STATUS "Details:")
  print_summary_vars(SUMMARY_DETAIL_VARS)
endfunction()
