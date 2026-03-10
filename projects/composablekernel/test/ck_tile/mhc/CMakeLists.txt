# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx90a|gfx940|gfx941|gfx942|gfx9|gfx11|gfx12")
    # Simple compilation test
    add_gtest_executable(test_ck_tile_mhc_simple test_mhc_simple.cpp)
    target_compile_options(test_ck_tile_mhc_simple PRIVATE -Wno-unused-parameter)
    
    # Comprehensive reference implementation tests
    add_gtest_executable(test_ck_tile_mhc_comprehensive test_mhc_comprehensive.cpp)
    target_compile_options(test_ck_tile_mhc_comprehensive PRIVATE -Wno-unused-parameter)
    
    # Full fused pipeline test (currently has compilation issues to be fixed)
    # add_gtest_executable(test_ck_tile_mhc_fused_pipeline test_mhc_fused_pipeline.cpp)
    # target_compile_options(test_ck_tile_mhc_fused_pipeline PRIVATE -Wno-unused-parameter)
endif()
