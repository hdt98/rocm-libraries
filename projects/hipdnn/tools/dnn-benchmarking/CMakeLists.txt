# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.18)

find_package(Python COMPONENTS Interpreter REQUIRED)

# Build dnn-benchmarking wheel and install to staging area
install(CODE "
    set(_wheel_dir \"\${CMAKE_INSTALL_PREFIX}/share/hipdnn/wheels\")
    file(MAKE_DIRECTORY \"\${_wheel_dir}\")

    message(STATUS \"Building dnn-benchmarking wheel\")
    execute_process(
        COMMAND \"${Python_EXECUTABLE}\" -m pip wheel --no-deps
            -w \"\${_wheel_dir}\"
            \"${CMAKE_CURRENT_SOURCE_DIR}\"
        RESULT_VARIABLE _result
    )
    if(_result)
        message(FATAL_ERROR \"Failed to build dnn-benchmarking wheel\")
    endif()
")
