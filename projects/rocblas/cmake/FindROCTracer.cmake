# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# FindROCTracer
# -------------
# Finds the ROCTracer roctx64 library and headers.
#
# Imported Targets:
#   ROCTracer::roctx  - The roctx64 library with include directories
#
# Result Variables:
#   ROCTracer_FOUND           - True if roctx64 was found
#   ROCTRACER_INCLUDE_DIR     - Path to roctracer/roctx.h
#   ROCTRACER_ROCTX_LIBRARY   - Path to libroctx64

include(FindPackageHandleStandardArgs)

find_path(ROCTRACER_INCLUDE_DIR
    NAMES roctracer/roctx.h
)

find_library(ROCTRACER_ROCTX_LIBRARY
    NAMES roctx64
)

find_package_handle_standard_args(ROCTracer
    REQUIRED_VARS ROCTRACER_INCLUDE_DIR ROCTRACER_ROCTX_LIBRARY
    FAIL_MESSAGE "Could not find ROCTracer roctx64 library or headers"
)

if(ROCTracer_FOUND AND NOT TARGET ROCTracer::roctx)
    add_library(ROCTracer::roctx UNKNOWN IMPORTED)
    set_target_properties(ROCTracer::roctx PROPERTIES
        IMPORTED_LOCATION "${ROCTRACER_ROCTX_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ROCTRACER_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(ROCTRACER_INCLUDE_DIR ROCTRACER_ROCTX_LIBRARY)
