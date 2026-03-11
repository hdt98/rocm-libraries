# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Integration test for GroupedConvolutionForwardIm2winKernel.
# Requires a GPU target that supports XDL (gfx94x / gfx95x).
if(GPU_TARGETS MATCHES "gfx94|gfx95")
    add_gtest_executable(test_ck_tile_grouped_conv_fwd_im2win
        test_grouped_conv_fwd_im2win.cpp)

    target_include_directories(test_ck_tile_grouped_conv_fwd_im2win PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})

    set_tests_properties(test_ck_tile_grouped_conv_fwd_im2win
        PROPERTIES LABELS "CK_TILE_CONV_IM2WIN_TESTS")
else()
    message(DEBUG "Skipping im2win conv test: no supported GPU target (gfx94x/gfx95x)")
endif()
