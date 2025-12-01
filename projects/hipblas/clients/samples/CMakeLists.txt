# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Create a small OBJECT library for sample utility code
add_library(hipblas-samples-common OBJECT)

target_sources(hipblas-samples-common PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/utility.cpp"
)

target_include_directories(hipblas-samples-common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

target_compile_features(hipblas-samples-common PRIVATE cxx_std_17)

target_compile_definitions(hipblas-samples-common PUBLIC
    HIPBLAS_NO_DEPRECATED_WARNINGS
    HIPBLAS_BFLOAT16_CLASS
)

if(HIPBLAS_ENABLE_HIP)
    target_compile_definitions(hipblas-samples-common PUBLIC __HIP_PLATFORM_AMD__)
elseif(HIPBLAS_ENABLE_CUDA)
    target_compile_definitions(hipblas-samples-common PUBLIC
        __HIP_PLATFORM_NVCC__
        __HIP_PLATFORM_NVIDIA__
    )
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(hipblas-samples-common PUBLIC -mf16c)
endif()

target_link_libraries(hipblas-samples-common PUBLIC roc::hipblas)

# Sample: sscal
add_executable(hipblas-example-sscal)
target_sources(hipblas-example-sscal PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_sscal.cpp")
target_link_libraries(hipblas-example-sscal PRIVATE hipblas-samples-common)
set_target_properties(hipblas-example-sscal PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)
target_include_directories(hipblas-example-sscal SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)
if(HIPBLAS_ENABLE_HIP)
    target_link_libraries(hipblas-example-sscal PRIVATE hip::host)
    if(CUSTOM_TARGET)
        target_link_libraries(hipblas-example-sscal PRIVATE hip::${CUSTOM_TARGET})
    endif()
elseif(HIPBLAS_ENABLE_CUDA)
    target_include_directories(hipblas-example-sscal PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )
    target_link_libraries(hipblas-example-sscal PRIVATE ${CUDA_LIBRARIES})
endif()
rocm_install(TARGETS hipblas-example-sscal COMPONENT samples)

# Sample: scal-ex
add_executable(hipblas-example-scal-ex)
target_sources(hipblas-example-scal-ex PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_scal_ex.cpp")
target_link_libraries(hipblas-example-scal-ex PRIVATE hipblas-samples-common)
set_target_properties(hipblas-example-scal-ex PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)
target_include_directories(hipblas-example-scal-ex SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)
if(HIPBLAS_ENABLE_HIP)
    target_link_libraries(hipblas-example-scal-ex PRIVATE hip::host)
    if(CUSTOM_TARGET)
        target_link_libraries(hipblas-example-scal-ex PRIVATE hip::${CUSTOM_TARGET})
    endif()
elseif(HIPBLAS_ENABLE_CUDA)
    target_include_directories(hipblas-example-scal-ex PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )
    target_link_libraries(hipblas-example-scal-ex PRIVATE ${CUDA_LIBRARIES})
endif()
rocm_install(TARGETS hipblas-example-scal-ex COMPONENT samples)

# Sample: sgemm
add_executable(hipblas-example-sgemm)
target_sources(hipblas-example-sgemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_sgemm.cpp")
target_link_libraries(hipblas-example-sgemm PRIVATE hipblas-samples-common)
set_target_properties(hipblas-example-sgemm PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)
target_include_directories(hipblas-example-sgemm SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)
if(HIPBLAS_ENABLE_HIP)
    target_link_libraries(hipblas-example-sgemm PRIVATE hip::host)
    if(CUSTOM_TARGET)
        target_link_libraries(hipblas-example-sgemm PRIVATE hip::${CUSTOM_TARGET})
    endif()
elseif(HIPBLAS_ENABLE_CUDA)
    target_include_directories(hipblas-example-sgemm PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )
    target_link_libraries(hipblas-example-sgemm PRIVATE ${CUDA_LIBRARIES})
endif()
rocm_install(TARGETS hipblas-example-sgemm COMPONENT samples)

# Sample: strmm
add_executable(hipblas-example-strmm)
target_sources(hipblas-example-strmm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_strmm.cpp")
target_link_libraries(hipblas-example-strmm PRIVATE hipblas-samples-common)
set_target_properties(hipblas-example-strmm PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)
target_include_directories(hipblas-example-strmm SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)
if(HIPBLAS_ENABLE_HIP)
    target_link_libraries(hipblas-example-strmm PRIVATE hip::host)
    if(CUSTOM_TARGET)
        target_link_libraries(hipblas-example-strmm PRIVATE hip::${CUSTOM_TARGET})
    endif()
elseif(HIPBLAS_ENABLE_CUDA)
    target_include_directories(hipblas-example-strmm PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )
    target_link_libraries(hipblas-example-strmm PRIVATE ${CUDA_LIBRARIES})
endif()
rocm_install(TARGETS hipblas-example-strmm COMPONENT samples)

# Sample: sgemm-strided-batched
add_executable(hipblas-example-sgemm-strided-batched)
target_sources(hipblas-example-sgemm-strided-batched PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_sgemm_strided_batched.cpp")
target_link_libraries(hipblas-example-sgemm-strided-batched PRIVATE hipblas-samples-common)
set_target_properties(hipblas-example-sgemm-strided-batched PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)
target_include_directories(hipblas-example-sgemm-strided-batched SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)
if(HIPBLAS_ENABLE_HIP)
    target_link_libraries(hipblas-example-sgemm-strided-batched PRIVATE hip::host)
    if(CUSTOM_TARGET)
        target_link_libraries(hipblas-example-sgemm-strided-batched PRIVATE hip::${CUSTOM_TARGET})
    endif()
elseif(HIPBLAS_ENABLE_CUDA)
    target_include_directories(hipblas-example-sgemm-strided-batched PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )
    target_link_libraries(hipblas-example-sgemm-strided-batched PRIVATE ${CUDA_LIBRARIES})
endif()
rocm_install(TARGETS hipblas-example-sgemm-strided-batched COMPONENT samples)

# Sample: c (test C99 compatibility)
add_executable(hipblas-example-c)
target_sources(hipblas-example-c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_c.c")
set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/example_c.c" PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "-xc -std=c99 -I${HIP_INCLUDE_DIRS}"
)
target_link_libraries(hipblas-example-c PRIVATE hipblas-samples-common)
set_target_properties(hipblas-example-c PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)
target_include_directories(hipblas-example-c SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)
if(HIPBLAS_ENABLE_HIP)
    target_link_libraries(hipblas-example-c PRIVATE hip::host)
    if(CUSTOM_TARGET)
        target_link_libraries(hipblas-example-c PRIVATE hip::${CUSTOM_TARGET})
    endif()
elseif(HIPBLAS_ENABLE_CUDA)
    target_include_directories(hipblas-example-c PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )
    target_link_libraries(hipblas-example-c PRIVATE ${CUDA_LIBRARIES})
endif()
rocm_install(TARGETS hipblas-example-c COMPONENT samples)

# Sample: hip-complex-her2
add_executable(hipblas-example-hip-complex-her2)
target_sources(hipblas-example-hip-complex-her2 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hip_complex_her2.cpp")
target_link_libraries(hipblas-example-hip-complex-her2 PRIVATE hipblas-samples-common)
set_target_properties(hipblas-example-hip-complex-her2 PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)
target_include_directories(hipblas-example-hip-complex-her2 SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)
if(HIPBLAS_ENABLE_HIP)
    target_link_libraries(hipblas-example-hip-complex-her2 PRIVATE hip::host)
    if(CUSTOM_TARGET)
        target_link_libraries(hipblas-example-hip-complex-her2 PRIVATE hip::${CUSTOM_TARGET})
    endif()
elseif(HIPBLAS_ENABLE_CUDA)
    target_include_directories(hipblas-example-hip-complex-her2 PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )
    target_link_libraries(hipblas-example-hip-complex-her2 PRIVATE ${CUDA_LIBRARIES})
endif()
rocm_install(TARGETS hipblas-example-hip-complex-her2 COMPONENT samples)

# Sample: hgemm-half
add_executable(hipblas-example-hgemm-half)
target_sources(hipblas-example-hgemm-half PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hgemm_hip_half.cpp")
target_compile_definitions(hipblas-example-hgemm-half PRIVATE HIPBLAS_USE_HIP_HALF)
if(WIN32)
    target_compile_options(hipblas-example-hgemm-half PRIVATE -xhip)
endif()
target_link_libraries(hipblas-example-hgemm-half PRIVATE hipblas-samples-common)
set_target_properties(hipblas-example-hgemm-half PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)
target_include_directories(hipblas-example-hgemm-half SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)
if(HIPBLAS_ENABLE_HIP)
    target_link_libraries(hipblas-example-hgemm-half PRIVATE hip::host)
    if(CUSTOM_TARGET)
        target_link_libraries(hipblas-example-hgemm-half PRIVATE hip::${CUSTOM_TARGET})
    endif()
elseif(HIPBLAS_ENABLE_CUDA)
    target_include_directories(hipblas-example-hgemm-half PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )
    target_link_libraries(hipblas-example-hgemm-half PRIVATE ${CUDA_LIBRARIES})
endif()
rocm_install(TARGETS hipblas-example-hgemm-half COMPONENT samples)

# Sample: gemmEx
add_executable(hipblas-example-gemmEx)
target_sources(hipblas-example-gemmEx PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_gemm_ex.cpp")
target_compile_definitions(hipblas-example-gemmEx PRIVATE HIPBLAS_USE_HIP_HALF)
if(WIN32)
    target_compile_options(hipblas-example-gemmEx PRIVATE -xhip)
endif()
target_link_libraries(hipblas-example-gemmEx PRIVATE hipblas-samples-common)
set_target_properties(hipblas-example-gemmEx PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)
target_include_directories(hipblas-example-gemmEx SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)
if(HIPBLAS_ENABLE_HIP)
    target_link_libraries(hipblas-example-gemmEx PRIVATE hip::host)
    if(CUSTOM_TARGET)
        target_link_libraries(hipblas-example-gemmEx PRIVATE hip::${CUSTOM_TARGET})
    endif()
elseif(HIPBLAS_ENABLE_CUDA)
    target_include_directories(hipblas-example-gemmEx PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )
    target_link_libraries(hipblas-example-gemmEx PRIVATE ${CUDA_LIBRARIES})
endif()
rocm_install(TARGETS hipblas-example-gemmEx COMPONENT samples)

# Sample: hgemm (only for hipcc or amdclang++)
if(CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" OR CMAKE_CXX_COMPILER MATCHES ".*/amdclang\\+\\+$")
    add_executable(hipblas-example-hgemm)
    target_sources(hipblas-example-hgemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hgemm.cpp")
    target_link_libraries(hipblas-example-hgemm PRIVATE hipblas-samples-common)
    set_target_properties(hipblas-example-hgemm PROPERTIES
        LINKER_LANGUAGE CXX
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS ON
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    )
    target_include_directories(hipblas-example-hgemm SYSTEM PRIVATE
        $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    )
    if(HIPBLAS_ENABLE_HIP)
        target_link_libraries(hipblas-example-hgemm PRIVATE hip::host)
        if(CUSTOM_TARGET)
            target_link_libraries(hipblas-example-hgemm PRIVATE hip::${CUSTOM_TARGET})
        endif()
    elseif(HIPBLAS_ENABLE_CUDA)
        target_include_directories(hipblas-example-hgemm PRIVATE
            $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
        )
        target_link_libraries(hipblas-example-hgemm PRIVATE ${CUDA_LIBRARIES})
    endif()
    rocm_install(TARGETS hipblas-example-hgemm COMPONENT samples)
endif()

# Fortran samples (non-Windows only)
if(NOT WIN32 AND HIPBLAS_ENABLE_FORTRAN)
    add_executable(hipblas-example-sscal-fortran)
    target_sources(hipblas-example-sscal-fortran PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_sscal_fortran.F90"
        $<TARGET_OBJECTS:hipblas-fortran-client>
    )
    target_link_libraries(hipblas-example-sscal-fortran PRIVATE roc::hipblas)
    rocm_install(TARGETS hipblas-example-sscal-fortran COMPONENT samples)
    
    add_executable(hipblas-example-gemmEx-fortran)
    target_sources(hipblas-example-gemmEx-fortran PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_gemm_ex_fortran.F90"
        $<TARGET_OBJECTS:hipblas-fortran-client>
    )
    target_link_libraries(hipblas-example-gemmEx-fortran PRIVATE roc::hipblas)
    rocm_install(TARGETS hipblas-example-gemmEx-fortran COMPONENT samples)
endif()

# Sample: bfdot-hip-bfloat16 (HIP backend only)
if(HIPBLAS_ENABLE_HIP)
    add_executable(hipblas-example-bfdot-hip-bfloat16)
    target_sources(hipblas-example-bfdot-hip-bfloat16 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_bfdot_hip_bfloat16.cpp")
    target_compile_definitions(hipblas-example-bfdot-hip-bfloat16 PRIVATE HIPBLAS_USE_HIP_BFLOAT16)
    target_link_libraries(hipblas-example-bfdot-hip-bfloat16 PRIVATE hipblas-samples-common)
    set_target_properties(hipblas-example-bfdot-hip-bfloat16 PROPERTIES
        LINKER_LANGUAGE CXX
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS ON
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    )
    target_include_directories(hipblas-example-bfdot-hip-bfloat16 SYSTEM PRIVATE
        $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    )
    target_link_libraries(hipblas-example-bfdot-hip-bfloat16 PRIVATE hip::host)
    if(CUSTOM_TARGET)
        target_link_libraries(hipblas-example-bfdot-hip-bfloat16 PRIVATE hip::${CUSTOM_TARGET})
    endif()
    rocm_install(TARGETS hipblas-example-bfdot-hip-bfloat16 COMPONENT samples)
endif()
