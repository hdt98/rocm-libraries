# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# im2win forward convolution example targets gfx94/gfx95 (CDNA3/CDNA4)
# where the small-channel grouped conv use case originates.
if(GPU_TARGETS MATCHES "gfx94|gfx95")
    set(EXAMPLE_IM2WIN_COMPILE_OPTIONS)
    list(APPEND EXAMPLE_IM2WIN_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0)

    add_executable(tile_example_grouped_conv_fwd_im2win
        grouped_convolution_fwd_im2win.cpp)

    target_compile_options(tile_example_grouped_conv_fwd_im2win
        PRIVATE ${EXAMPLE_IM2WIN_COMPILE_OPTIONS})

    # The run harness includes headers from the sibling example directory.
    target_include_directories(tile_example_grouped_conv_fwd_im2win PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../20_grouped_convolution)
endif()
