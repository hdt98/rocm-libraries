# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Install pre-built custom kernel code objects into the Tensile library
# directory so they are packaged with the runtime component.

set(custom_kernel_co_files
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256.co"
)

set(custom_kernel_outputs)
foreach(co_file IN LISTS custom_kernel_co_files)
    get_filename_component(co_filename "${co_file}" NAME)
    add_custom_command(
        OUTPUT "${HIPBLASLT_TENSILE_LIBPATH}/library/${co_filename}"
        COMMAND ${CMAKE_COMMAND} -E copy "${co_file}" "${HIPBLASLT_TENSILE_LIBPATH}/library/"
        DEPENDS "${co_file}"
        COMMENT "Copying custom kernel ${co_filename} to Tensile library directory"
    )
    list(APPEND custom_kernel_outputs "${HIPBLASLT_TENSILE_LIBPATH}/library/${co_filename}")
endforeach()

add_custom_target(custom-kernels ALL
    DEPENDS ${custom_kernel_outputs}
)
