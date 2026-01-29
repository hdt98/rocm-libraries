# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Custom kernel wrapping

set(aiter_kernel_files
    "${CMAKE_CURRENT_SOURCE_DIR}/custom_kernels/f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256.co"
)

set(clang_offload_wrapper_exe /opt/rocm/lib/llvm/bin/clang-offload-wrapper)
set(wrapper_bc "${CMAKE_CURRENT_BINARY_DIR}/multi_kernel_wrapper.bc")
set(wrapper_obj "${CMAKE_CURRENT_BINARY_DIR}/multi_kernel_wrapper.o")

add_custom_command(
    OUTPUT ${wrapper_bc}
    COMMAND ${clang_offload_wrapper_exe} 
            -o ${wrapper_bc}
            --target=amdgcn-amd-amdhsa
            --offload-arch=gfx950
            ${aiter_kernel_files}  # Passes all files in the list
    DEPENDS ${aiter_kernel_files}
    COMMENT "Wrapping multiple GPU Code Objects into a single host bitcode"
)

add_custom_command(
    OUTPUT ${wrapper_obj}
    COMMAND ${CMAKE_CXX_COMPILER} -c ${wrapper_bc} -o ${wrapper_obj}
    DEPENDS ${wrapper_bc}
    COMMENT "Compiling multi-kernel wrapper"
)

set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/aiter_kernels.hip"
    PROPERTIES 
        LANGUAGE HIP
)

add_library(hipblaslt-rocroller-kernels OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/aiter_kernels.hip" ${wrapper_obj})
set_target_properties(hipblaslt-rocroller-kernels
        PROPERTIES 
            HIP_ARCHITECTURES "gfx950"
            POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(hipblaslt-rocroller-kernels PRIVATE hip::device)
target_include_directories(hipblaslt-rocroller-kernels
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)


# HipBLASLt-RocRoller layer

add_library(hipblaslt-rocroller OBJECT)

target_compile_features(hipblaslt-rocroller PUBLIC cxx_std_20)
set_target_properties(hipblaslt-rocroller PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hipblaslt-rocroller
    PUBLIC
        roc::origami-headers
        $<TARGET_OBJECTS:hipblaslt-rocroller-kernels>
)
if(TARGET roc::rocroller-interface)
    target_link_libraries(hipblaslt-rocroller PUBLIC roc::rocroller-interface)
else()
    target_link_libraries(hipblaslt-rocroller PUBLIC roc::rocroller_interface)
endif()
target_link_libraries(hipblaslt-rocroller PUBLIC hip::host roc::hipblas-common)
target_compile_definitions(hipblaslt-rocroller PUBLIC HIPBLASLT_USE_ROCROLLER)
target_include_directories(hipblaslt-rocroller
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
        "${PROJECT_BINARY_DIR}/library/include"
        "${PROJECT_SOURCE_DIR}/library/include"
        "${PROJECT_SOURCE_DIR}/tensilelite/include"
)

target_sources(hipblaslt-rocroller
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/custom_kernels.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gemm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/parameter_selection.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/runtime_args_selection.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocroller_host.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/solution_cache.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/solution_selection.cpp"
)

target_link_libraries(hipblaslt PRIVATE $<TARGET_OBJECTS:hipblaslt-rocroller> $<TARGET_OBJECTS:hipblaslt-rocroller-kernels>)
if(TARGET roc::rocroller-interface)
    target_link_libraries(hipblaslt PRIVATE roc::rocroller-interface)
else()
    target_link_libraries(hipblaslt PRIVATE roc::rocroller_interface)
endif()
