# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(NOT GPU_TARGETS MATCHES "gfx13")
    add_example_executable(example_batched_gemm_add_add_relu_gemm_add_xdl_fp16 batched_gemm_add_add_relu_gemm_add_xdl_fp16.cpp)
    add_example_executable(example_batched_gemm_add_add_relu_gemm_add_wmma_fp16 batched_gemm_add_add_relu_gemm_add_wmma_fp16.cpp)
endif()
