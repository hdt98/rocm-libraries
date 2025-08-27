# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_basic_matmul_for_cov "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_basic_matmul_for_cov.cpp")
target_link_libraries(sample_hipblaslt_basic_matmul_for_cov PRIVATE sample-helper)
