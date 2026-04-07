cmake_minimum_required(VERSION 3.21)
project(hipconv LANGUAGES HIP CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(hip REQUIRED)

# ---- hipconv static library ----
add_library(hipconv STATIC
    src/registry.cpp
    src/grouped/grouped_conv.cpp
    src/conv3d/conv3d.cpp
)
set_target_properties(hipconv PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    HIP_ARCHITECTURES gfx950
)
set_source_files_properties(
    src/registry.cpp
    src/grouped/grouped_conv.cpp
    src/conv3d/conv3d.cpp
    PROPERTIES LANGUAGE HIP
)

target_include_directories(hipconv
    PUBLIC  include
    PRIVATE src
)
target_link_libraries(hipconv PUBLIC hip::device)
target_compile_options(hipconv PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
target_compile_definitions(hipconv PRIVATE HIPCONV_BUILDING_LIBRARY)

# ---- Tests ----
add_subdirectory(test)
