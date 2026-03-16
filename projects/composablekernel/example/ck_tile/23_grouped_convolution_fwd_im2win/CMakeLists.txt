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
        0  1  2  3  4  5  6  7  8  9  10  11  12
        #13  14  15  16  17
        #18 19 20 21 22 23
        24
        # Large-K configs (K=2376, C=256)
        100 101 102 103 104 105 106 107
        )
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
        # CV3_M128N16K64_Gm2
        # CV3_M128N16K64_Gm4
        # CV3_M128N16K64_Gm8
        # CV3_M128N16K64_Gm32
        # CV3_M64N16K64_Gm32
        # Merge_Gm2_M8N128K64
        # Merge_Gm4_M16N64K64
        # Merge_Gm8_M32N128K64
        # Merge_Gm32_M128N32K64
        # Merge_Gm32_M128N64K64
        # Merge_Gm8_M128N32K32
        Merge_Gm32_M128N64K64_512T
        # Large-K configs (compute-bound, K=2376, C=256)
        LK_CV3_M128N32K64
        LK_CV3_M64N64K64
        LK_CV3_M32N128K64
        LK_CV3_M16N64K64
        LK_CV3_M64N16K64
        LK_CV3_M128N32K64_Occ2
        LK_CV3_M64N64K64_Occ2
        LK_Mem_M128N32K64
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

    # ── NHWGC im2col configs ──────────────────────────────────────────────────
    # Standard im2col GEMM shape (M=N×Ho×Wo, N=K) with channels-last NHWGC input.
    # Binary prefix: tile_example_nhwgc_im2col_<name>
    # Compare these against Merge_Gm* im2win configs for algorithm comparison.
    # IDs 0-6: standard im2col (UseIm2Win=false)
    # IDs 7-10: im2win A descriptor variants (UseIm2Win=true, same tile shapes as 0-3)
    set(NHWGC_IM2COL_CONFIG_IDS   0  1  2  3  4  5  6   7  8  9  10)
    set(NHWGC_IM2COL_CONFIG_NAMES
        IC_CV3_M16N64K64
        IC_Mem_M128N32K64
        IC_CV3_M128N128K64
        IC_CV3_M64N64K64
        IC_Mem_M128N16K64
        IC_Mem_M128N32K64_Occ2
        IC_CV3_M64N32K64
        IW_CV3_M16N64K64
        IW_Mem_M128N32K64
        IW_CV3_M128N128K64
        IW_CV3_M64N64K64
    )

    list(LENGTH NHWGC_IM2COL_CONFIG_IDS _num_ic_configs)
    math(EXPR _ic_last "${_num_ic_configs} - 1")

    add_custom_target(tile_example_nhwgc_im2col_all)

    foreach(_i RANGE 0 ${_ic_last})
        list(GET NHWGC_IM2COL_CONFIG_IDS   ${_i} _id)
        list(GET NHWGC_IM2COL_CONFIG_NAMES ${_i} _name)
        set(_tgt "tile_example_nhwgc_im2col_${_name}")

        add_executable(${_tgt} grouped_convolution_fwd_nhwgc_im2col.cpp)
        target_compile_definitions(${_tgt} PRIVATE NHWGC_IM2COL_CONFIG_ID=${_id})
        target_compile_options(${_tgt} PRIVATE ${EXAMPLE_IM2WIN_COMPILE_OPTIONS})
        target_include_directories(${_tgt} PRIVATE ${IM2WIN_INCLUDE_DIRS})

        add_dependencies(tile_example_nhwgc_im2col_all ${_tgt})
        add_dependencies(tile_example_grouped_conv_fwd_im2win_all ${_tgt})
    endforeach()

    # Convenience alias (config 1 = IC_Mem_M128N32K64 — typically fastest).
    add_executable(tile_example_nhwgc_im2col
        grouped_convolution_fwd_nhwgc_im2col.cpp)
    target_compile_definitions(tile_example_nhwgc_im2col PRIVATE
        NHWGC_IM2COL_CONFIG_ID=1)
    target_compile_options(tile_example_nhwgc_im2col PRIVATE
        ${EXAMPLE_IM2WIN_COMPILE_OPTIONS})
    target_include_directories(tile_example_nhwgc_im2col PRIVATE
        ${IM2WIN_INCLUDE_DIRS})

    # ── Two-stage im2win: explicit I' materialisation + tunable Stage-2 GEMM ──
    # Supports both GNCHW and NHWGC input layouts.
    # IM2WIN_STAGE2_CONFIG selects the Stage-2 GEMM tile (0-5).
    # ACTIVE_IM2WIN_TRANSFORM_CONFIG selects the Stage-1 tile (0-3).
    #
    # Stage-2 configs:
    #   0: Mem_M128N32K64  — memory pipeline, small K/C
    #   1: CV3_M64N64K64   — ComputeV3, good for large K
    #   2: CV3_M128N64K64  — ComputeV3, large M tile
    #   3: CV3_M64N64K64_Occ2  — CV3, higher occupancy (default/best)
    #   4: CV3_M128N64K64_Occ2 — CV3, large M + high occupancy
    #   5: CV3_M64N128K64  — CV3, large N tile (for K=256)
    # IDs 0-5:  VectorSize=4 (small or large C)
    # IDs 6-10: VectorSize=8 (large C=256/K=256 only — faster loads)
    set(IM2WIN_2STAGE_S2_IDS   0  1  2  3  4  5   6  7  8  9  10)
    set(IM2WIN_2STAGE_S2_NAMES
        Mem_M128N32K64
        CV3_M64N64K64
        CV3_M128N64K64
        CV3_M64N64K64_Occ2
        CV3_M128N64K64_Occ2
        CV3_M64N128K64
        CV3_M64N64K64_VS8
        CV3_M128N64K64_VS8
        CV3_M64N64K64_Occ2_VS8
        CV3_M128N64K64_Occ2_VS8
        CV3_M64N128K64_VS8)

    list(LENGTH IM2WIN_2STAGE_S2_IDS _num_s2)
    math(EXPR _s2_last "${_num_s2} - 1")

    add_custom_target(tile_example_grouped_conv_fwd_im2win_2stage_all)

    foreach(_i RANGE 0 ${_s2_last})
        list(GET IM2WIN_2STAGE_S2_IDS   ${_i} _s2id)
        list(GET IM2WIN_2STAGE_S2_NAMES ${_i} _s2name)
        set(_tgt "tile_example_grouped_conv_fwd_im2win_2stage_${_s2name}")

        add_executable(${_tgt} grouped_convolution_fwd_gnchw_im2win_2stage.cpp)
        target_compile_definitions(${_tgt} PRIVATE
            ACTIVE_IM2WIN_TRANSFORM_CONFIG=0
            IM2WIN_STAGE2_CONFIG=${_s2id})
        target_compile_options(${_tgt} PRIVATE ${EXAMPLE_IM2WIN_COMPILE_OPTIONS})
        target_include_directories(${_tgt} PRIVATE ${IM2WIN_INCLUDE_DIRS})

        add_dependencies(tile_example_grouped_conv_fwd_im2win_2stage_all ${_tgt})
        add_dependencies(tile_example_grouped_conv_fwd_im2win_all ${_tgt})
    endforeach()

    # Default convenience alias (config 3 = CV3_M64N64K64_Occ2 — best for large K)
    add_executable(tile_example_grouped_conv_fwd_im2win_2stage
        grouped_convolution_fwd_gnchw_im2win_2stage.cpp)
    target_compile_definitions(tile_example_grouped_conv_fwd_im2win_2stage PRIVATE
        ACTIVE_IM2WIN_TRANSFORM_CONFIG=0
        IM2WIN_STAGE2_CONFIG=3)
    target_compile_options(tile_example_grouped_conv_fwd_im2win_2stage PRIVATE
        ${EXAMPLE_IM2WIN_COMPILE_OPTIONS})
    target_include_directories(tile_example_grouped_conv_fwd_im2win_2stage PRIVATE
        ${IM2WIN_INCLUDE_DIRS})

    add_dependencies(tile_example_grouped_conv_fwd_im2win_all
        tile_example_grouped_conv_fwd_im2win_2stage)
endif()
