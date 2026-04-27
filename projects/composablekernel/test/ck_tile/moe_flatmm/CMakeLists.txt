# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(TEST_MOE_FLATMM_COMPILE_OPTIONS)
list(APPEND TEST_MOE_FLATMM_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0)

if(CK_USE_OCP_FP8)
    list(APPEND TEST_MOE_FLATMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

set(TEST_MOE_FLATMM_DTYPES)

if(GPU_TARGETS MATCHES "gfx942|gfx950")
    list(APPEND TEST_MOE_FLATMM_DTYPES fp16 bf16 fp8 bf8)
endif()

if(GPU_TARGETS MATCHES "gfx95")
    # FP4/MX and A16W4 paths use gfx950-specific MX instructions/configs.
    list(APPEND TEST_MOE_FLATMM_DTYPES fp4fp4 fp8fp4 bf16fp4 fp16fp4)
endif()

if(TEST_MOE_FLATMM_DTYPES)
    # Base-pipeline dtype tests share a single fixture; MX/A16W4 tests use
    # gfx950-only fixtures. Each executable instantiates its kernel set directly
    # (per_test_direct) so there's no generated instance machinery.
    set(TEST_MOE_FLATMM_TARGETS)
    foreach(DTYPE IN LISTS TEST_MOE_FLATMM_DTYPES)
        if(DTYPE MATCHES "fp4fp4|fp8fp4")
            set(TEST_SRC test_mx_moe_flatmm_${DTYPE}.cpp)
        elseif(DTYPE MATCHES "bf16fp4|fp16fp4")
            set(TEST_SRC test_a16w4_moe_flatmm_${DTYPE}.cpp)
        else()
            set(TEST_SRC test_moe_flatmm_${DTYPE}.cpp)
        endif()

        add_gtest_executable(test_tile_moe_flatmm_${DTYPE} ${TEST_SRC})
        target_include_directories(test_tile_moe_flatmm_${DTYPE} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_options(test_tile_moe_flatmm_${DTYPE}
                               PRIVATE ${TEST_MOE_FLATMM_COMPILE_OPTIONS})
        list(APPEND TEST_MOE_FLATMM_TARGETS test_tile_moe_flatmm_${DTYPE})
    endforeach()

    # Umbrella target to build all moe_flatmm tests at once
    add_custom_target(test_tile_moe_flatmm_all)
    add_dependencies(test_tile_moe_flatmm_all ${TEST_MOE_FLATMM_TARGETS})
else()
    message(DEBUG "Skipping ck_tile moe_flatmm tests for current target")
endif()
