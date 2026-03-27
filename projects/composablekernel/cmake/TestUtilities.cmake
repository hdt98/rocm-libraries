# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Helper macro to conditionally link libraries only if they exist as targets.
# This is useful when device_conv libraries may be filtered out based on GPU targets,
# DTYPES, or build configuration flags.
#
# Usage:
#   target_link_libraries_if_exist(my_test PRIVATE utility device_conv2d_nhwgc_operations ...)
#
# Only device_conv* libraries are checked with if(TARGET).
# All other libraries (utility, gtest_main, etc.) are always linked.
macro(target_link_libraries_if_exist TARGET_NAME VISIBILITY)
    foreach(lib IN LISTS ARGN)
        if(lib MATCHES "^device_conv")
            # Only check device_conv libraries conditionally
            if(TARGET ${lib})
                target_link_libraries(${TARGET_NAME} ${VISIBILITY} ${lib})
            endif()
        else()
            # Always link non-device_conv libraries (utility, gtest_main, etc.)
            target_link_libraries(${TARGET_NAME} ${VISIBILITY} ${lib})
        endif()
    endforeach()
endmacro()
