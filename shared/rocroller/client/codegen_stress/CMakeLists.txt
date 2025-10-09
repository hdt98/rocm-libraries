# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(rocroller-codegen-stress "${CMAKE_CURRENT_SOURCE_DIR}/codegen_stress_test.cpp")
target_link_libraries(rocroller-codegen-stress
    PRIVATE
        rocroller::common-test-utilities
        CLI11::CLI11
)
if(ROCROLLER_BUILD_SHARED_LIBS) # why are we using this logic
    target_link_libraries(rocroller-codegen-stress PUBLIC ${llvm_yaml_libs})
endif()

if(ROCROLLER_ENABLE_ASAN)
    rocroller_target_configure_sanitizers(rocroller-codegen-stress PRIVATE)
endif()

if(ROCROLLER_ENABLE_COVERAGE)
    target_compile_options(rocroller-codegen-stress PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(rocroller-codegen-stress PRIVATE -fprofile-instr-generate)
endif()
