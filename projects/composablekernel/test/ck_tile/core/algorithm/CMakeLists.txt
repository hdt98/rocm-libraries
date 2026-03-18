# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(ck_tile_unit_gf2_linear_transform unit_gf2_linear_transform.cpp)
if(result EQUAL 0)
  target_link_libraries(ck_tile_unit_gf2_linear_transform PRIVATE utility)
endif()
