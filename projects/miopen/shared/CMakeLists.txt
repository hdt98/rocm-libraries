cmake_policy(SET CMP0057 NEW)

message("### MIOPEN_BACKEND_HIP: ${MIOPEN_BACKEND_HIP}")
message("### MIOPEN_BACKEND: ${MIOPEN_BACKEND}")

add_library(MIOpen_shared STATIC
    env.cpp
    sysinfo_utils.cpp
    logger.cpp
)

if(MIOPEN_BACKEND STREQUAL "HIPOC" OR MIOPEN_BACKEND STREQUAL "HIP")
    target_link_libraries( MIOpen_shared PRIVATE hip::device )
    target_link_libraries( MIOpen_shared INTERFACE hip::host )
endif()

set_target_properties(MIOpen_shared PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(MIOpen_shared INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

message("Include directory: ${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(MIOpen_shared PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)

# miopen_generate_export_header(MIOpen_shared)