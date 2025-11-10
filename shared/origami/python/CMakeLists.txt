# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)

## Build Origami nanobind library (core)
include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG        v1.9.2  # or latest stable version
)
FetchContent_MakeAvailable(nanobind)

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(origami_core 
    "${CMAKE_CURRENT_SOURCE_DIR}/origami_module.cpp"
)

set_target_properties(origami_core PROPERTIES
    OUTPUT_NAME "_core"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/origami"
)

target_link_libraries(origami_core PRIVATE roc::origami)

## Build Origami ATen library (torch_ops)
if(ORIGAMI_ENABLE_ATEN)
    # Get PyTorch's CMake directory from Python
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.__path__[0])"
        OUTPUT_VARIABLE TORCH_PYTHON_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    # Add PyTorch's CMake path to CMAKE_PREFIX_PATH
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_PYTHON_DIR}/share/cmake/Torch")
    message(STATUS "Added PyTorch CMake path: ${TORCH_PYTHON_DIR}/share/cmake/Torch")

    find_package(Torch REQUIRED)

    add_library(origami_aten MODULE
        "${CMAKE_CURRENT_SOURCE_DIR}/aten_module.cpp"
    )

    set_target_properties(origami_aten PROPERTIES
        OUTPUT_NAME "_aten"
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/origami"
    )

    target_include_directories(origami_aten
        PRIVATE ${Python_INCLUDE_DIRS}
    )

    target_link_libraries(origami_aten
        PRIVATE roc::origami
        PRIVATE "${TORCH_LIBRARIES}"
    )
endif()

## Copy Python package files to build directory
add_custom_command(
    TARGET origami_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/origami/__init__.py"
            "${CMAKE_CURRENT_BINARY_DIR}/origami/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/setup.py"
            "${CMAKE_CURRENT_BINARY_DIR}/setup.py"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml"
            "${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/MANIFEST.in"
            "${CMAKE_CURRENT_BINARY_DIR}/MANIFEST.in"
    COMMENT "Copying Python package files to build directory"
)
