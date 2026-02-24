# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx95")
    set(TEST_MX_FLATMM_COMPILE_OPTIONS)
    if(CK_USE_OCP_FP8)
        list(APPEND TEST_MX_FLATMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
    endif()
    list(APPEND TEST_MX_FLATMM_COMPILE_OPTIONS "SHELL: -mllvm -greedy-reverse-local-assignment=1")

    # FP4 tests
    add_gtest_executable(test_tile_mx_flatmm_fp4
        test_mx_flatmm_fp4.cpp
    )
    target_compile_options(test_tile_mx_flatmm_fp4 PRIVATE ${TEST_MX_FLATMM_COMPILE_OPTIONS})

    # FP6 tests
    add_gtest_executable(test_tile_mx_flatmm_fp6
        test_mx_flatmm_fp6.cpp
    )
    target_compile_options(test_tile_mx_flatmm_fp6 PRIVATE ${TEST_MX_FLATMM_COMPILE_OPTIONS})

    # FP8 tests
    add_gtest_executable(test_tile_mx_flatmm_fp8
        test_mx_flatmm_fp8.cpp
    )
    target_compile_options(test_tile_mx_flatmm_fp8 PRIVATE ${TEST_MX_FLATMM_COMPILE_OPTIONS})

    # Mixed precision tests (FP8xFP4, FP4xFP8)
    add_gtest_executable(test_tile_mx_flatmm_mixed
        test_mx_flatmm_mixed.cpp
    )
    target_compile_options(test_tile_mx_flatmm_mixed PRIVATE ${TEST_MX_FLATMM_COMPILE_OPTIONS})

    # Umbrella target
    add_custom_target(test_tile_mx_flatmm_all)
    add_dependencies(test_tile_mx_flatmm_all
        test_tile_mx_flatmm_fp4
        test_tile_mx_flatmm_fp6
        test_tile_mx_flatmm_fp8
        test_tile_mx_flatmm_mixed
    )
else()
    message(DEBUG "Skipping MX FLATMM tests: requires gfx95 target")
endif()
