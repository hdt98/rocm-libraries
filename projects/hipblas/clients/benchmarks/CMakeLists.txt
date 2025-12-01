# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Create benchmark executable
add_executable(hipblas-bench)

# Add source files
target_sources(hipblas-bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/client.cpp"
)

# Target compile features
target_compile_features(hipblas-bench PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type)

# Internal header includes
target_include_directories(hipblas-bench PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

# External header includes as system files
target_include_directories(hipblas-bench SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)

# Link libraries with visibility keywords
target_link_libraries(hipblas-bench PRIVATE
    roc::hipblas
    hipblas-clients-common
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

# Add BLAS/LAPACK libraries
# Note: BLIS dependencies now propagate from hipblas-clients-common (PUBLIC linkage)
# Only add system BLAS if BLIS is not used
if(NOT WIN32 AND NOT HIPBLAS_ENABLE_BLIS)
    target_link_libraries(hipblas-bench PRIVATE BLAS::BLAS)
    
    if(LAPACK_FOUND)
        target_link_libraries(hipblas-bench PRIVATE LAPACK::LAPACK)
    endif()
elseif(WIN32 AND NOT HIPBLAS_ENABLE_BLIS)
    # Windows BLAS libraries (BLIS dependencies propagate from clients-common)
    target_link_libraries(hipblas-bench PRIVATE ${BLAS_LIBRARY})
endif()

# Add BLIS-specific compile definitions for benchmark target
if(HIPBLAS_ENABLE_BLIS)
    target_compile_definitions(hipblas-bench PRIVATE
        BLIS_ENABLE_NO_UNDERSCORE_API
        BLIS_ENABLE_CBLAS
    )
endif()

# Add OpenMP support
if(HIPBLAS_ENABLE_OPENMP)
    target_link_libraries(hipblas-bench PRIVATE OpenMP::OpenMP_CXX)
    
    # Handle OpenMP RPATH for HIP/Clang
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND HIPBLAS_ENABLE_HIP)
        if(TARGET OpenMP::OpenMP_CXX)
            set_target_properties(hipblas-bench PROPERTIES
                BUILD_RPATH "${HIP_CLANG_ROOT}/lib"
                INSTALL_RPATH "$ORIGIN/../llvm/lib"
            )
        elseif(TARGET OpenMP::omp)
            set_target_properties(hipblas-bench PROPERTIES
                BUILD_RPATH "${HIP_CLANG_ROOT}/${openmp_LIB_DIR}"
                INSTALL_RPATH "$ORIGIN/../llvm/${openmp_LIB_DIR}"
            )
        endif()
    endif()
endif()

# Compile definitions
target_compile_definitions(hipblas-bench PRIVATE HIPBLAS_BENCH)

# Platform-specific linking
if(HIPBLAS_ENABLE_HIP)
    target_link_libraries(hipblas-bench PRIVATE hip::host)
    
    if(CUSTOM_TARGET)
        target_link_libraries(hipblas-bench PRIVATE hip::${CUSTOM_TARGET})
    endif()
    
    if(NOT WIN32)
        target_link_libraries(hipblas-bench PRIVATE stdc++fs)
    endif()
elseif(HIPBLAS_ENABLE_CUDA)
    target_include_directories(hipblas-bench PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )
    target_link_libraries(hipblas-bench PRIVATE ${CUDA_LIBRARIES})
endif()

# Fortran client library support
if(HIPBLAS_ENABLE_FORTRAN)
    target_link_libraries(hipblas-bench PRIVATE hipblas-fortran-client)
    target_compile_definitions(hipblas-bench PRIVATE HIPBLAS_FORTRAN_CLIENTS)
endif()

# Solver support define
if(HIPBLAS_ENABLE_SOLVER)
    target_compile_definitions(hipblas-bench PRIVATE __HIP_PLATFORM_SOLVER__)
endif()

# Target properties
set_target_properties(hipblas-bench PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

# Add dependency on common files target
add_dependencies(hipblas-bench hipblas-clients-common-files)

# Install benchmark executable
rocm_install(TARGETS hipblas-bench COMPONENT benchmarks)
