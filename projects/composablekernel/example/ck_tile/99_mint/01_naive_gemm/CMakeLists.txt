# MINT Hierarchical Naive GEMM Tutorial
# Demonstrates MINT's hierarchical programming model with grid/block/warp levels

project(mint_naive_gemm_tutorial)

# Target name
set(TARGET_NAME example_mint_naive_gemm)

# Create executable
add_executable(${TARGET_NAME} practice_gemm.cpp)

# Include directories
target_include_directories(${TARGET_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(${TARGET_NAME} PRIVATE
    hip::device
)

# Compiler options
target_compile_options(${TARGET_NAME} PRIVATE
    -DMINT_BACKEND_ROCM
    -fgpu-rdc
    -Wno-unused-result
    -Wno-error
    -Wno-undefined-func-template
    -Wno-float-equal
    -Wno-sign-compare
)

# Linker options
target_link_options(${TARGET_NAME} PRIVATE
    -fgpu-rdc
)

# Set output directory
set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install
install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION bin)
