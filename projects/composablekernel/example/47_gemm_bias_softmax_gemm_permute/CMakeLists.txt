# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(NOT GPU_TARGETS MATCHES "gfx13")
    add_example_executable(example_gemm_bias_softmax_gemm_permute gemm_bias_softmax_gemm_permute_xdl.cpp)
endif()