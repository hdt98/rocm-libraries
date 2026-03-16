# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# MHC Fused Pipeline - Winning Version
set(TARGET_NAME_MHC mhc_example)
add_executable(${TARGET_NAME_MHC} mhc_example.cpp)
# Add compiler flag to report kernel resource usage (register usage, LDS, etc.)
#target_compile_options(${TARGET_NAME_MHC} PRIVATE -Rpass-analysis=kernel-resource-usage)
