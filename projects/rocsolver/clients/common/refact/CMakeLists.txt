# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(rocsolver-test-common PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/testing_csrrf_analysis.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/testing_csrrf_sumlu.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/testing_csrrf_splitlu.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/testing_csrrf_refactlu.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/testing_csrrf_refactchol.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/testing_csrrf_solve.cpp"
)
