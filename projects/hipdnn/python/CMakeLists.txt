# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.18)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(hipdnn_python_bindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and nanobind
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# FetchContent to get nanobind if not found
include(FetchContent)
fetchcontent_declare(nanobind GIT_REPOSITORY https://github.com/wjakob/nanobind GIT_TAG v1.8.0)
fetchcontent_makeavailable(nanobind)

# Find hipDNN frontend headers
set(HIPDNN_FRONTEND_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../frontend/include"
    CACHE PATH "Path to hipDNN frontend includes"
)

# Find HIP (if needed)
find_package(hip REQUIRED)

# Only find_package for hipDNN targets when built standalone
if(NOT TARGET hipdnn_frontend)
    find_package(hipdnn_frontend CONFIG REQUIRED)
endif()
if(NOT TARGET hipdnn_backend)
    find_package(hipdnn_backend REQUIRED)
endif()

# Create the Python module
nanobind_add_module(
    hipdnn_frontend_python
    src/module.cpp
    src/graph_bindings.cpp
    src/tensor_bindings.cpp
    src/attributes_bindings.cpp
    src/types_bindings.cpp
    src/handle_bindings.cpp
    src/memory_bindings.cpp
)

# Include directories
target_include_directories(
    hipdnn_frontend_python PRIVATE ${HIPDNN_FRONTEND_INCLUDE_DIR} ${HIP_INCLUDE_DIRS}
)

set_target_properties(hipdnn_frontend_python PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

# Link against hipDNN frontend library if it exists as a separate library Adjust this based on your
# actual library structure
target_link_libraries(hipdnn_frontend_python PRIVATE hipdnn_frontend hipdnn_backend hip::host)

# ============================================================================
# Installation
# ============================================================================

# Use a relative path so the install respects CMAKE_INSTALL_PREFIX.
set(HIPDNN_PYTHON_SITE_DIR
    "lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/dist-packages"
    CACHE PATH "Python module install directory (relative to CMAKE_INSTALL_PREFIX)")

# scikit-build-core (pip wheel / pip install) install rules.
# SKBUILD is set by scikit-build-core when building a wheel.
if(DEFINED SKBUILD)
    install(TARGETS hipdnn_frontend_python DESTINATION hipdnn_frontend)
    install(FILES hipdnn_frontend/__init__.py DESTINATION hipdnn_frontend)
else()
    # Regular cmake install — place in dist-packages under CMAKE_INSTALL_PREFIX.
    install(TARGETS hipdnn_frontend_python
        LIBRARY DESTINATION ${HIPDNN_PYTHON_SITE_DIR}/hipdnn_frontend
    )
    install(FILES hipdnn_frontend/__init__.py
        DESTINATION ${HIPDNN_PYTHON_SITE_DIR}/hipdnn_frontend
    )
endif()

# Build wheels and stage them in the install prefix.
# Skipped when scikit-build-core drives cmake (SKBUILD) to avoid recursion.
if(NOT DEFINED SKBUILD)
    install(CODE "
        set(_wheel_dir \"\${CMAKE_INSTALL_PREFIX}/share/hipdnn/wheels\")
        file(MAKE_DIRECTORY \"\${_wheel_dir}\")

        message(STATUS \"Building hipdnn-frontend wheel\")
        execute_process(
            COMMAND \"${Python_EXECUTABLE}\" -m pip wheel --no-deps
                -w \"\${_wheel_dir}\"
                \"${CMAKE_CURRENT_SOURCE_DIR}\"
                -C \"cmake.define.CMAKE_PREFIX_PATH=\${CMAKE_INSTALL_PREFIX}\"
            RESULT_VARIABLE _result
        )
        if(_result)
            message(FATAL_ERROR \"Failed to build hipdnn-frontend wheel\")
        endif()

    ")
endif()
