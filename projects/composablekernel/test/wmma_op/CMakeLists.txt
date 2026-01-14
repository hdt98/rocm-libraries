# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_test_executable(test_wmma_op wmma_op.cpp)
target_compile_options(test_wmma_op PRIVATE -save-temps=obj -Wno-gnu-line-marker)
target_link_libraries(test_wmma_op PRIVATE utility)
