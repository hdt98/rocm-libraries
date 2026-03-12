# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx94|gfx95")
    set(EXAMPLE_IM2WIN_COMPILE_OPTIONS)
    list(APPEND EXAMPLE_IM2WIN_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0)

    set(IM2WIN_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../20_grouped_convolution)

    # ── Config registry ──────────────────────────────────────────────────────
    # Parallel lists: IM2WIN_CONFIG_IDS[i] ↔ IM2WIN_CONFIG_NAMES[i]
    # IM2WIN_CONFIG_ID matches the #if branch in im2win_conv_configs.hpp.
    # True im2win configs use negative IDs (IM2WIN_CONFIG_ID < 0)
    # CMake does not allow negative values in lists, so pass them as strings.
    set(IM2WIN_CONFIG_IDS
        -4 -3 -2 -1
        0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17
        18 19 20 21 22)
    set(IM2WIN_CONFIG_NAMES
        Ti_M16N64K64
        Ti_M8N128K64
        Ti_M4N128K64
        Ti_M4N256K64
        CV3_M16N64K64
        CV3_M64N64K64
        CV3_M64N32K64
        Mem_M128N32K64
        Mem_M128N32K64_IW
        CV3_M128N128K64_Occ2
        CV3_M64N16K64
        CV3_M128N16K64
        Mem_M128N16K64
        Mem_M128N16K64_IW
        CV3_M64N16K64_Occ2
        Mem_M128N4K64
        Mem_M256N4K64
        CV3_M128N16K64_Gm2
        CV3_M128N16K64_Gm4
        CV3_M128N16K64_Gm8
        CV3_M128N16K64_Gm32
        CV3_M64N16K64_Gm32
        Merge_Gm2_M8N128K64
        Merge_Gm4_M16N64K64
        Merge_Gm8_M32N128K64
        Merge_Gm32_M128N32K64
        Merge_Gm32_M128N64K64
    )

    list(LENGTH IM2WIN_CONFIG_IDS _num_configs)
    math(EXPR _last "${_num_configs} - 1")

    add_custom_target(tile_example_grouped_conv_fwd_im2win_all)

    foreach(_i RANGE 0 ${_last})
        list(GET IM2WIN_CONFIG_IDS   ${_i} _id)
        list(GET IM2WIN_CONFIG_NAMES ${_i} _name)
        set(_tgt "tile_example_grouped_conv_fwd_im2win_${_name}")

        add_executable(${_tgt} grouped_convolution_fwd_im2win.cpp)
        target_compile_definitions(${_tgt} PRIVATE IM2WIN_CONFIG_ID=${_id})
        target_compile_options(${_tgt} PRIVATE ${EXAMPLE_IM2WIN_COMPILE_OPTIONS})
        target_include_directories(${_tgt} PRIVATE ${IM2WIN_INCLUDE_DIRS})

        add_dependencies(tile_example_grouped_conv_fwd_im2win_all ${_tgt})
    endforeach()

    # Convenience alias without a suffix (config 0 = CV3_M16N64K64).
    add_executable(tile_example_grouped_conv_fwd_im2win
        grouped_convolution_fwd_im2win.cpp)
    target_compile_definitions(tile_example_grouped_conv_fwd_im2win PRIVATE
        IM2WIN_CONFIG_ID=0)
    target_compile_options(tile_example_grouped_conv_fwd_im2win PRIVATE
        ${EXAMPLE_IM2WIN_COMPILE_OPTIONS})
    target_include_directories(tile_example_grouped_conv_fwd_im2win PRIVATE
        ${IM2WIN_INCLUDE_DIRS})
endif()
