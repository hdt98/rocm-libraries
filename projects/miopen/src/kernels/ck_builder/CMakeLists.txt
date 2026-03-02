# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# =============================================================================
# CK Builder Kernel Instances with Architecture-Aware Filtering
# =============================================================================
#
# This CMakeLists.txt implements architecture-aware filtering to compile only
# the kernel instances compatible with the target GPU architectures.
#
# Architecture Detection:
#   - Files in **/xdl/** directories → CDNA only (gfx908/90a/942/950)
#   - Files in **/wmma/** directories → RDNA only (gfx1030/1100/1200/1201)
#   - Files in **/dl/** directories → All architectures
#   - Static test files → All architectures
#
# Benefits:
#   - Reduces binary size by 20-30% for single-arch builds
#   - Reduces compile time by skipping unused architectures
#   - Automatic: follows directory structure from develop
# =============================================================================

include(cmake/ShardInstantiation.cmake)

# Grouped Conv2D Forward - DL (4 files) - ARCHITECTURE AGNOSTIC
set(GROUPED_CONV2D_FWD_DL_SOURCES
    factories/grouped_conv2d_fwd/dl/device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
    factories/grouped_conv2d_fwd/dl/device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f32_instance.cpp
    factories/grouped_conv2d_fwd/dl/device_grouped_conv2d_fwd_dl_nhwgc_gkyxc_nhwgk_f16_instance.cpp
    factories/grouped_conv2d_fwd/dl/device_grouped_conv2d_fwd_dl_nhwgc_gkyxc_nhwgk_f32_instance.cpp
)

# Grouped Conv2D Forward - WMMA Base (16 files) - RDNA ONLY
set(GROUPED_CONV2D_FWD_WMMA_SOURCES
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_1x1p0_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_1x1s1p0_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_oddc_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_1x1p0_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_1x1s1p0_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_oddc_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_1x1p0_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_1x1s1p0_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_oddc_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_1x1p0_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_1x1s1p0_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_oddc_instance.cpp
)

# Grouped Conv2D Forward - XDL Comp (15 files) - CDNA ONLY
set(GROUPED_CONV2D_FWD_XDL_COMP_SOURCES
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_comp_2x_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_comp_part2_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_2x_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_part2_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_comp_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_2x_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_part2_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_comp_2x_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_comp_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_comp_part2_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_comp_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_comp_instance.cpp
    factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_comp_instance.cpp
)

# Grouped Conv2D Forward - XDL Large Tensor (5 files) - CDNA ONLY
set(GROUPED_CONV2D_FWD_XDL_LARGE_TENSOR_SOURCES
    factories/grouped_conv2d_fwd/xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f32_tf32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_int8_instance.cpp
)

# Grouped Conv2D Forward - XDL Merged Groups (8 files) - CDNA ONLY
set(GROUPED_CONV2D_FWD_XDL_MERGED_GROUPS_SOURCES
    factories/grouped_conv2d_fwd/xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkcyx_ngkhw_bf16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkcyx_ngkhw_f16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkcyx_ngkhw_f32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f32_tf32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_int8_instance.cpp
)

# Grouped Conv2D Forward - WMMA CShuffleV3 (10 files) - RDNA ONLY
set(GROUPED_CONV2D_FWD_WMMA_CSHUFFLEV3_SOURCES
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part1.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part2.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part3.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part4.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part1.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part2.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part3.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part4.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_wave_transfer_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
    factories/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_wave_transfer_nhwgc_gkyxc_nhwgk_f16_instance.cpp
)

# Grouped Conv2D Forward - WMMA Large Tensor (2 files) - RDNA ONLY
set(GROUPED_CONV2D_FWD_WMMA_LARGE_TENSOR_SOURCES
    factories/grouped_conv2d_fwd/wmma/large_tensor/device_grouped_conv2d_fwd_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
    factories/grouped_conv2d_fwd/wmma/large_tensor/device_grouped_conv2d_fwd_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_f16_instance.cpp
)

# Grouped Conv2D Forward - XDL Base (bf16/f16 sharded separately) - CDNA ONLY
set(GROUPED_CONV2D_FWD_XDL_SOURCES
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_bf16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_16x16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_16x16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_16x16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_bf16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_f16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_f32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_int8_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_16x16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_16x16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_16x16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_16x16_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_instance.cpp
    factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_instance.cpp
)

# Grouped Conv2D Forward - XDL MEM (14 files, int8 inter/intra sharded separately) - CDNA ONLY
set(GROUPED_CONV2D_FWD_XDL_MEM_SOURCES
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_mem_inter_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_mem_intra_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_mem_inter_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_mem_intra_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_mem_inter_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_mem_intra_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_direct_load_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_inter_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_intra_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_direct_load_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_mem_inter_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_mem_intra_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_mem_inter_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_mem_intra_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_mem_inter_instance.cpp
    factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_mem_intra_instance.cpp
)

# =============================================================================
# Architecture-Aware Source List Construction
# =============================================================================

set(CK_BUILDER_SOURCES "")

# Static test files - ALWAYS INCLUDED (architecture agnostic)
set(CK_BUILDER_STATIC_TEST_SOURCES
    static_tests/grouped_conv_2d_fwd_xdl_f32.cpp
    static_tests/grouped_conv_2d_fwd_xdl_v3_f32.cpp
    static_tests/grouped_conv_2d_fwd_xdl_large_tensor_f32.cpp
    static_tests/grouped_conv_2d_fwd_wmma_f32.cpp
    static_tests/grouped_conv_2d_fwd_dl_f32.cpp
    static_tests/grouped_conv_fwd_xdl_test.cpp
)

# Always add architecture-agnostic sources
list(APPEND CK_BUILDER_SOURCES ${CK_BUILDER_STATIC_TEST_SOURCES})
list(APPEND CK_BUILDER_SOURCES ${GROUPED_CONV2D_FWD_DL_SOURCES})

# Detect CDNA architectures
set(CDNA_ENABLED FALSE)
if(DEFINED MIOPEN_GPU_TARGET_GFX908 OR
   DEFINED MIOPEN_GPU_TARGET_GFX90A OR
   DEFINED MIOPEN_GPU_TARGET_GFX942 OR
   DEFINED MIOPEN_GPU_TARGET_GFX950)
    set(CDNA_ENABLED TRUE)
endif()

# Detect RDNA architectures
set(RDNA_ENABLED FALSE)
if(DEFINED MIOPEN_GPU_TARGET_GFX1030 OR
   DEFINED MIOPEN_GPU_TARGET_GFX1100 OR
   DEFINED MIOPEN_GPU_TARGET_GFX1200 OR
   DEFINED MIOPEN_GPU_TARGET_GFX1201)
    set(RDNA_ENABLED TRUE)
endif()

# Sharded instantiations for large instance files
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)

# Add CDNA (XDL) sources if building for CDNA architectures
if(CDNA_ENABLED)
    list(APPEND CK_BUILDER_SOURCES ${GROUPED_CONV2D_FWD_XDL_SOURCES})
    list(APPEND CK_BUILDER_SOURCES ${GROUPED_CONV2D_FWD_XDL_COMP_SOURCES})
    list(APPEND CK_BUILDER_SOURCES ${GROUPED_CONV2D_FWD_XDL_LARGE_TENSOR_SOURCES})
    list(APPEND CK_BUILDER_SOURCES ${GROUPED_CONV2D_FWD_XDL_MEM_SOURCES})
    list(APPEND CK_BUILDER_SOURCES ${GROUPED_CONV2D_FWD_XDL_MERGED_GROUPS_SOURCES})

    # Generate sharded XDL instantiations
    generate_sharded_instantiations(
        INSTANCES_NAME device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_instances
        TEMPLATE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_instance.in
        NUM_SHARDS 16
        SRC_LIST CK_BUILDER_XDL_SHARDED_SOURCES
        OUTPUT_DIR ${GENERATED_DIR}/xdl
    )

    generate_sharded_instantiations(
        INSTANCES_NAME device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_instances
        TEMPLATE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/factories/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_instance.in
        NUM_SHARDS 16
        SRC_LIST CK_BUILDER_XDL_SHARDED_SOURCES
        OUTPUT_DIR ${GENERATED_DIR}/xdl
    )

    generate_sharded_instantiations(
        INSTANCES_NAME device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_comp_instances
        TEMPLATE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/factories/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_comp_instance.in
        NUM_SHARDS 16
        SRC_LIST CK_BUILDER_XDL_SHARDED_SOURCES
        OUTPUT_DIR ${GENERATED_DIR}/xdl/comp
    )

    generate_sharded_instantiations(
        INSTANCES_NAME device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_mem_inter_instances
        TEMPLATE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_mem_inter_instance.in
        NUM_SHARDS 21
        SRC_LIST CK_BUILDER_XDL_SHARDED_SOURCES
        OUTPUT_DIR ${GENERATED_DIR}/xdl/mem
    )

    generate_sharded_instantiations(
        INSTANCES_NAME device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_mem_intra_instances
        TEMPLATE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/factories/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_mem_intra_instance.in
        NUM_SHARDS 21
        SRC_LIST CK_BUILDER_XDL_SHARDED_SOURCES
        OUTPUT_DIR ${GENERATED_DIR}/xdl/mem
    )

    list(APPEND CK_BUILDER_SOURCES ${CK_BUILDER_XDL_SHARDED_SOURCES})

    list(LENGTH GROUPED_CONV2D_FWD_XDL_SOURCES XDL_BASE_COUNT)
    list(LENGTH GROUPED_CONV2D_FWD_XDL_COMP_SOURCES XDL_COMP_COUNT)
    list(LENGTH GROUPED_CONV2D_FWD_XDL_LARGE_TENSOR_SOURCES XDL_LT_COUNT)
    list(LENGTH GROUPED_CONV2D_FWD_XDL_MEM_SOURCES XDL_MEM_COUNT)
    list(LENGTH GROUPED_CONV2D_FWD_XDL_MERGED_GROUPS_SOURCES XDL_MG_COUNT)
    list(LENGTH CK_BUILDER_XDL_SHARDED_SOURCES XDL_SHARDED_COUNT)
    math(EXPR XDL_TOTAL "${XDL_BASE_COUNT} + ${XDL_COMP_COUNT} + ${XDL_LT_COUNT} + ${XDL_MEM_COUNT} + ${XDL_MG_COUNT} + ${XDL_SHARDED_COUNT}")
    message(STATUS "CK Builder: Adding ${XDL_TOTAL} XDL instance files (CDNA: gfx908, gfx90a, gfx942, gfx950)")
endif()

# Add RDNA (WMMA) sources if building for RDNA architectures
if(RDNA_ENABLED)
    list(APPEND CK_BUILDER_SOURCES ${GROUPED_CONV2D_FWD_WMMA_SOURCES})
    list(APPEND CK_BUILDER_SOURCES ${GROUPED_CONV2D_FWD_WMMA_CSHUFFLEV3_SOURCES})
    list(APPEND CK_BUILDER_SOURCES ${GROUPED_CONV2D_FWD_WMMA_LARGE_TENSOR_SOURCES})

    list(LENGTH GROUPED_CONV2D_FWD_WMMA_SOURCES WMMA_BASE_COUNT)
    list(LENGTH GROUPED_CONV2D_FWD_WMMA_CSHUFFLEV3_SOURCES WMMA_CSV3_COUNT)
    list(LENGTH GROUPED_CONV2D_FWD_WMMA_LARGE_TENSOR_SOURCES WMMA_LT_COUNT)
    math(EXPR WMMA_TOTAL "${WMMA_BASE_COUNT} + ${WMMA_CSV3_COUNT} + ${WMMA_LT_COUNT}")
    message(STATUS "CK Builder: Adding ${WMMA_TOTAL} WMMA instance files (RDNA: gfx1030, gfx1100, gfx1200, gfx1201)")
endif()

# Verify we have at least some sources
if(NOT CK_BUILDER_SOURCES)
    message(FATAL_ERROR "CK Builder: No kernel instance files selected! This indicates a configuration error.\n"
                        "  GPU_TARGETS: ${GPU_TARGETS}\n"
                        "  CDNA enabled: ${CDNA_ENABLED}, RDNA enabled: ${RDNA_ENABLED}\n"
                        "Please set GPU_TARGETS to target GPU architectures or disable MIOPEN_CK_BUILDER_EXPERIMENTAL.")
endif()

# Create library from filtered sources
add_library(ck_builder STATIC ${CK_BUILDER_SOURCES})

set_target_properties(ck_builder PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(ck_builder PRIVATE -DCK_EXPERIMENTAL_BUILDER)
target_include_directories(ck_builder PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link against Composable Kernel device operations
find_package(composable_kernel 1.0.0 COMPONENTS device_conv_operations)
target_link_libraries(ck_builder composable_kernel::device_conv_operations)

# CK template instantiations can produce fold expressions exceeding the default bracket-depth limit of 256
target_compile_options(ck_builder PRIVATE $<$<CXX_COMPILER_ID:Clang>:-fbracket-depth=2048>)

# This static library contains a large amount of GPU code that will benefit from offload-compress if it's available
check_cxx_compiler_flag("--offload-compress -x hip" CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
if(CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
    target_compile_options(ck_builder PRIVATE --offload-compress)
    message(STATUS "CK Builder: Enabling --offload-compress for ck_builder library")
endif()

# Summary message
list(LENGTH CK_BUILDER_SOURCES TOTAL_COUNT)
list(LENGTH GROUPED_CONV2D_FWD_DL_SOURCES DL_COUNT)
message(STATUS "==========================================")
message(STATUS "CK Builder Architecture Filter Summary:")
if(CDNA_ENABLED)
    message(STATUS "  CDNA (XDL): ${XDL_TOTAL} files")
endif()
if(RDNA_ENABLED)
    message(STATUS "  RDNA (WMMA): ${WMMA_TOTAL} files")
endif()
message(STATUS "  Arch-agnostic (DL + tests): ${DL_COUNT} + static tests")
message(STATUS "  Total compiled: ${TOTAL_COUNT} files")
message(STATUS "==========================================")
