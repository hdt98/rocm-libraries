# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Need to support Release/Debug

string(REGEX MATCHALL "gfx[a-z0-9]+" archs "${GPU_TARGETS}")
list(REMOVE_DUPLICATES archs)

# Create separate --offload-arch flags for each architecture
set(arch_flags "")
foreach(arch IN LISTS archs)
    list(APPEND arch_flags "--offload-arch=${arch}")
endforeach()

set(matrix_transform_output_library "${HIPBLASLT_TENSILE_LIBPATH}/library/hipblasltTransform.hsaco")
set(matrix_transform_stamp "${CMAKE_CURRENT_SOURCE_DIR}/matrix_transform.stamp")

add_custom_command(
<<<<<<< HEAD
    COMMAND ${CMAKE_CXX_COMPILER} -x hip ${matrix_transform_cpp} ${arch_flags} -c --offload-device-only -Xoffload-linker --build-id=sha1 -O3 -o ${output_library}
    OUTPUT ${output_library}
=======
    COMMAND ${CMAKE_CXX_COMPILER} -x hip "${CMAKE_CURRENT_SOURCE_DIR}/matrix_transform.cpp" --offload-arch="${archs}" -c --offload-device-only -Xoffload-linker --build-id=sha1 -O3 -o ${matrix_transform_output_library}
    COMMAND ${CMAKE_COMMAND} -E touch ${matrix_transform_stamp}
    OUTPUT ${matrix_transform_output_library} ${matrix_transform_stamp}
>>>>>>> Add stamp to output
    COMMENT "Building hipblasltTransform.hsaco"
)

add_custom_target(matrix-transform ALL
    DEPENDS ${matrix_transform_stamp}
)
