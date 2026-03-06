# MINT Naive GEMM Example
# Demonstrates MINT tensor views and basic GEMM computation

project(mint_naive_gemm)

# Target name
set(TARGET_NAME example_mint_naive_gemm)

# Create executable
add_executable(${TARGET_NAME} gemm_mint.cpp)

# Include directories
target_include_directories(${TARGET_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# Link CK_Tile host utilities
target_link_libraries(${TARGET_NAME} PRIVATE
    hip::device
)

# Compiler options for HIP
target_compile_options(${TARGET_NAME} PRIVATE
    -DMINT_BACKEND_ROCM
    -fgpu-rdc
    -Wno-unused-result
    -Wno-error
    -Wno-undefined-func-template
    -Wno-float-equal
)

# Linker options for HIP device code
target_link_options(${TARGET_NAME} PRIVATE
    -fgpu-rdc
)

# Set output directory to top-level build/bin
set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install
install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION bin)
