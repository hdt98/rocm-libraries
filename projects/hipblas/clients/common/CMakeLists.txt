# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Create OBJECT library for common client sources
# Uses PUBLIC dependencies to propagate BLIS libraries and definitions to consuming targets
add_library(hipblas-clients-common OBJECT)

# Compile features
target_compile_features(hipblas-clients-common PRIVATE cxx_std_17)

# Add common source files
target_sources(hipblas-clients-common PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/utility.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cblas_interface.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/clients_common.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipblas_arguments.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipblas_parse_data.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipblas_datatype2string.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/norm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/near.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/arg_check.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/argument_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/host_alloc.cpp"
)

# Conditionally add BLIS interface if BLIS is enabled
if(HIPBLAS_ENABLE_BLIS)
    target_sources(hipblas-clients-common PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/blis_interface.cpp"
    )
    
    # Add BLIS include directory and link libraries (static AOCL libs)
    # Use PUBLIC so dependencies propagate to targets that link hipblas-clients-common
    target_include_directories(hipblas-clients-common PUBLIC ${BLIS_INCLUDE_DIR})
    target_link_libraries(hipblas-clients-common PUBLIC ${BLAS_LIBRARY})
    
    # Add FLA_ENABLE_ILP64 define for ILP64 BLIS interface
    # Use PUBLIC so cblas_interface.cpp compiles the same way in all targets
    target_compile_definitions(hipblas-clients-common PUBLIC FLA_ENABLE_ILP64)
endif()

# CRITICAL: Enable PIC when building shared libraries
# Without this, linking OBJECT library into shared lib fails with relocation errors
if(HIPBLAS_BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
    set_target_properties(hipblas-clients-common PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# Include directories
target_include_directories(hipblas-clients-common
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

# Compile definitions
target_compile_definitions(hipblas-clients-common PUBLIC
    HIPBLAS_BFLOAT16_CLASS
    ROCM_USE_FLOAT16
    HIPBLAS_NO_DEPRECATED_WARNINGS
)

if(HIPBLAS_ENABLE_HIP)
    target_compile_definitions(hipblas-clients-common PUBLIC __HIP_PLATFORM_AMD__)
elseif(HIPBLAS_ENABLE_CUDA)
    target_compile_definitions(hipblas-clients-common PUBLIC
        __HIP_PLATFORM_NVCC__
        __HIP_PLATFORM_NVIDIA__
    )
endif()

# Platform-specific compile options
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Need mf16c flag for float->half conversion
    target_compile_options(hipblas-clients-common PUBLIC -mf16c)
endif()

# Link against hipblas library
target_link_libraries(hipblas-clients-common PUBLIC roc::hipblas)

# Add solver support define if enabled
# This is required to compile solver LAPACK wrapper functions (ref_getrf, ref_gels, etc.)
# in cblas_interface.cpp which are guarded by #ifdef __HIP_PLATFORM_SOLVER__
if(HIPBLAS_ENABLE_SOLVER)
    target_compile_definitions(hipblas-clients-common PUBLIC __HIP_PLATFORM_SOLVER__)
endif()
