# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Helper macro to conditionally link libraries only if they exist as targets.
# This is useful when libraries may be filtered out based on GPU targets,
# DTYPES, or build configuration flags.
#
# Usage:
#   target_link_libraries_if_exist(my_test PRIVATE utility lib1 lib2 lib3)
#
# Each library in the list is checked with if(TARGET), and only linked if it exists.
macro(target_link_libraries_if_exist TARGET_NAME VISIBILITY)
    foreach(lib IN LISTS ARGN)
        if(TARGET ${lib})
            target_link_libraries(${TARGET_NAME} ${VISIBILITY} ${lib})
        endif()
    endforeach()
endmacro()
