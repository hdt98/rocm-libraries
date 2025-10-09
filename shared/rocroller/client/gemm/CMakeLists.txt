# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(rocroller-gemm)

add_subdirectory(include)
add_subdirectory(src)

target_link_libraries(rocroller-gemm
    PRIVATE
        rocroller::common-test-utilities
        CLI11::CLI11
)
target_include_directories(rocroller-gemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(ROCROLLER_BUILD_SHARED_LIBS) # why?
    target_link_libraries(rocroller-gemm PUBLIC ${llvm_yaml_libs})
endif()

if(ROCROLLER_ENABLE_ASAN)
    rocroller_target_configure_sanitizers(rocroller-gemm PRIVATE)
endif()

if(ROCROLLER_ENABLE_COVERAGE)
    target_compile_options(rocroller-gemm PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(rocroller-gemm PRIVATE -fprofile-instr-generate)
endif()
