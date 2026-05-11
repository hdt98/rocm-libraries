# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.18)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(hipdnn_python_bindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)
fetchcontent_declare(nanobind GIT_REPOSITORY https://github.com/wjakob/nanobind GIT_TAG v1.8.0)
fetchcontent_makeavailable(nanobind)

set(HIPDNN_FRONTEND_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../frontend/include"
    CACHE PATH "Path to hipDNN frontend includes"
)

if(NOT TARGET hip::host)
    find_package(hip REQUIRED)
endif()
if(NOT TARGET hipdnn_frontend)
    find_package(hipdnn_frontend CONFIG REQUIRED)
endif()
if(NOT TARGET hipdnn_backend)
    find_package(hipdnn_backend REQUIRED)
endif()

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

target_include_directories(
    hipdnn_frontend_python PRIVATE ${HIPDNN_FRONTEND_INCLUDE_DIR} ${HIP_INCLUDE_DIRS}
)

target_link_libraries(hipdnn_frontend_python PRIVATE hipdnn_frontend hipdnn_backend hip::host)

set_target_properties(hipdnn_frontend_python PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

if(DEFINED SKBUILD)
    # pip install: SKBUILD is set by scikit-build-core. DESTINATION . because
    # pyproject.toml has wheel.install-dir = "hipdnn_frontend" which maps
    # the install prefix root to hipdnn_frontend/ in the wheel.
    install(TARGETS hipdnn_frontend_python DESTINATION .)
    install(FILES hipdnn_frontend/__init__.py DESTINATION .)
else()
    # Superbuild / cmake --install: install to python_bindings/ under install prefix.
    # TheRock's wheel builder detects python_bindings/*/ and promotes to top-level
    # packages in the rocm-sdk-libraries wheel.
    install(TARGETS hipdnn_frontend_python
        LIBRARY DESTINATION python_bindings/hipdnn_frontend
        COMPONENT python_bindings)

    install(FILES hipdnn_frontend/__init__.py
        DESTINATION python_bindings/hipdnn_frontend
        COMPONENT python_bindings)
endif()
