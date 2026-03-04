# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(TARGET_NAME example_mhc)
add_executable(${TARGET_NAME} example_mhc.cpp)

set(TARGET_NAME_FUSED example_mhc_fused)
add_executable(${TARGET_NAME_FUSED} example_mhc_fused.cpp)

set(TARGET_NAME_OPTIMIZED example_mhc_optimized)
add_executable(${TARGET_NAME_OPTIMIZED} example_mhc_optimized.cpp)

set(TARGET_NAME_LOW_REGISTER example_mhc_low_register)
add_executable(${TARGET_NAME_LOW_REGISTER} example_mhc_low_register.cpp)

set(TARGET_NAME_SMALL_TILES example_mhc_small_tiles)
add_executable(${TARGET_NAME_SMALL_TILES} example_mhc_small_tiles.cpp)

set(TARGET_NAME_SEQUENTIAL_NORMS example_mhc_sequential_norms)
add_executable(${TARGET_NAME_SEQUENTIAL_NORMS} example_mhc_sequential_norms.cpp)

set(TARGET_NAME_ULTRA_LOW_SPILL example_mhc_ultra_low_spill)
add_executable(${TARGET_NAME_ULTRA_LOW_SPILL} example_mhc_ultra_low_spill.cpp)
