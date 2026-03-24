# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9")    
    # Comprehensive reference implementation tests
    add_gtest_executable(test_ck_tile_mhc test_mhc.cpp)
endif()
