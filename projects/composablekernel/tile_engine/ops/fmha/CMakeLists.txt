# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# FMHA Tile Engine -- Pure Python benchmarking via the CK dispatcher.
# No C++ per-kernel targets; all compilation is JIT via the dispatcher.

set(FMHA_TE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FMHA_TE_CONFIGS ${FMHA_TE_DIR}/configs)

# Main benchmark target (runs forward sweep by default)
add_custom_target(benchmark_fmha
    COMMAND ${Python3_EXECUTABLE} ${FMHA_TE_DIR}/fmha_benchmark.py
        ${FMHA_TE_CONFIGS}/fwd.json
        --arch ${USER_GPU_TARGETS}
        --workers 128
        --best
        --json ${CMAKE_CURRENT_BINARY_DIR}/fmha_fwd_results.json
    WORKING_DIRECTORY ${FMHA_TE_DIR}
    COMMENT "FMHA tile engine benchmark (forward)"
)

# Per-variant convenience targets
foreach(variant fwd bwd splitkv appendkv pagedkv batch_prefill)
    if(EXISTS ${FMHA_TE_CONFIGS}/${variant}.json)
        add_custom_target(benchmark_fmha_${variant}
            COMMAND ${Python3_EXECUTABLE} ${FMHA_TE_DIR}/fmha_benchmark.py
                ${FMHA_TE_CONFIGS}/${variant}.json
                --arch ${USER_GPU_TARGETS}
                --workers 128
                --best
                --json ${CMAKE_CURRENT_BINARY_DIR}/fmha_${variant}_results.json
            WORKING_DIRECTORY ${FMHA_TE_DIR}
            COMMENT "FMHA tile engine benchmark (${variant})"
        )
    endif()
endforeach()

# CI target (minimal sweep for quick validation)
if(EXISTS ${FMHA_TE_CONFIGS}/fwd_ci.json)
    add_custom_target(benchmark_fmha_ci
        COMMAND ${Python3_EXECUTABLE} ${FMHA_TE_DIR}/fmha_benchmark.py
            ${FMHA_TE_CONFIGS}/fwd_ci.json
            --arch ${USER_GPU_TARGETS}
            --workers 8
            --verify
        WORKING_DIRECTORY ${FMHA_TE_DIR}
        COMMENT "FMHA tile engine CI benchmark"
    )
endif()

# All-variants target
add_custom_target(benchmark_fmha_all
    COMMAND ${Python3_EXECUTABLE} ${FMHA_TE_DIR}/fmha_benchmark.py
        ${FMHA_TE_CONFIGS}/fwd.json
        ${FMHA_TE_CONFIGS}/bwd.json
        ${FMHA_TE_CONFIGS}/splitkv.json
        --arch ${USER_GPU_TARGETS}
        --workers 128
        --best
        --json ${CMAKE_CURRENT_BINARY_DIR}/fmha_all_results.json
    WORKING_DIRECTORY ${FMHA_TE_DIR}
    COMMENT "FMHA tile engine benchmark (all variants)"
)
