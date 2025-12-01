# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# NOTE: Clients are NOT a separate project - they build as part of hipBLAS
# No cmake_minimum_required() or project() call here

# Find required dependencies for clients
find_package(Threads REQUIRED)

if(HIPBLAS_BUILD_TESTING OR HIPBLAS_ENABLE_BENCHMARKS)
    # Find OpenMP if enabled
    if(HIPBLAS_ENABLE_OPENMP)
        find_package(OpenMP REQUIRED)
    endif()
    
    find_package(GTest REQUIRED)
    
    # Find BLAS/LAPACK for reference implementations
    # Default to static host libraries
    if(NOT DEFINED BLA_STATIC)
        set(BLA_STATIC ON)
    endif()
    
    if(NOT WIN32)
        # Set 64-bit integer size for BLAS/LAPACK
        set(BLA_SIZEOF_INTEGER 8)
        
        # FindBLAS requires Fortran, so only find if Fortran is enabled
        # When Fortran is disabled (e.g., for ASAN builds), BLIS will be used if available
        if(HIPBLAS_ENABLE_FORTRAN)
            find_package(BLAS REQUIRED)
            find_package(LAPACK)
            
            if(NOT LAPACK_FOUND)
                message(WARNING "LAPACK not found - reference implementations may be limited")
            endif()
        endif()
        
        # Find BLIS if enabled - fail if enabled but not found
        if(HIPBLAS_ENABLE_BLIS)
            # Try to find BLIS static libraries (libflame, libblis-mt, libaoclutils)
            # Following the same logic as the old build system
            set(BLIS_FOUND FALSE)
            
            # Try AOCL 4.2.0 GCC
            if(EXISTS "/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib_ILP64/libflame.a" AND
               EXISTS "/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib_ILP64/libblis-mt.a" AND
               EXISTS "/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib_ILP64/libaoclutils.a")
                set(BLAS_LIBRARY 
                    -Wl,--allow-multiple-definition
                    /opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib_ILP64/libflame.a
                    /opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib_ILP64/libblis-mt.a
                    /opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib_ILP64/libaoclutils.a
                )
                set(BLIS_INCLUDE_DIR /opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/include_ILP64)
                set(BLIS_FOUND TRUE)
                message(STATUS "Found BLIS (AOCL 4.2.0 GCC): ${BLAS_LIBRARY}")
            # Try AOCL 4.1.0 AOCC
            elseif(EXISTS "/opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/lib_ILP64/libflame.a" AND
                   EXISTS "/opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/lib_ILP64/libblis-mt.a")
                set(BLAS_LIBRARY
                    /opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/lib_ILP64/libflame.a
                    /opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/lib_ILP64/libblis-mt.a
                )
                set(BLIS_INCLUDE_DIR /opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/include_ILP64)
                set(BLIS_FOUND TRUE)
                message(STATUS "Found BLIS (AOCL 4.1.0 AOCC): ${BLAS_LIBRARY}")
            # Try AOCL 4.0 AOCC
            elseif(EXISTS "/opt/AMD/aocl/aocl-linux-aocc-4.0/lib_ILP64/libflame.a" AND
                   EXISTS "/opt/AMD/aocl/aocl-linux-aocc-4.0/lib_ILP64/libblis-mt.a")
                set(BLAS_LIBRARY
                    /opt/AMD/aocl/aocl-linux-aocc-4.0/lib_ILP64/libflame.a
                    /opt/AMD/aocl/aocl-linux-aocc-4.0/lib_ILP64/libblis-mt.a
                )
                set(BLIS_INCLUDE_DIR /opt/AMD/aocl/aocl-linux-aocc-4.0/include_ILP64)
                set(BLIS_FOUND TRUE)
                message(STATUS "Found BLIS (AOCL 4.0 AOCC): ${BLAS_LIBRARY}")
            endif()
            
            if(NOT BLIS_FOUND)
                message(FATAL_ERROR "HIPBLAS_ENABLE_BLIS is ON but BLIS static libraries (libflame.a, libblis-mt.a) were not found. "
                        "Please install AOCL BLIS or set HIPBLAS_ENABLE_BLIS=OFF.")
            endif()
        endif()
    else()
        # Windows: handle AOCL or OpenBLAS
        if(HIPBLAS_ENABLE_BLIS)
            file(TO_CMAKE_PATH "C:/Program Files/AMD/AOCL-Windows" AOCL_ROOT)
            find_library(BLIS_LIBRARY
                NAMES AOCL-LibBlis-Win-MT
                PATHS "${AOCL_ROOT}/amd-blis/lib/ILP64"
                NO_DEFAULT_PATH
            )
            
            find_path(BLIS_INCLUDE_DIR blis.h
                PATHS "${AOCL_ROOT}/amd-blis/include/ILP64"
                NO_DEFAULT_PATH
            )
            
            if(NOT BLIS_LIBRARY OR NOT BLIS_INCLUDE_DIR)
                message(FATAL_ERROR "HIPBLAS_ENABLE_BLIS is ON but BLIS library was not found. "
                        "Please install BLIS or set HIPBLAS_ENABLE_BLIS=OFF.")
            endif()
            message(STATUS "Found BLIS: ${BLIS_LIBRARY}")
        else()
            find_library(BLAS_LIBRARY
                NAMES openblas libopenblas
                PATHS ${OPENBLAS_DIR}/lib
                NO_DEFAULT_PATH
            )
            if(NOT BLAS_LIBRARY)
                find_package(OpenBLAS CONFIG REQUIRED)
                set(BLAS_LIBRARY OpenBLAS::OpenBLAS)
            endif()
        endif()
    endif()
endif()

# Build common client library
add_subdirectory(common)

# Build Fortran module if enabled
if(HIPBLAS_ENABLE_FORTRAN)
    set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include/hipblas)
    
    add_library(hipblas-fortran OBJECT
        "${CMAKE_CURRENT_SOURCE_DIR}/../library/src/hipblas_module.f90"
    )
    
    if(HIPBLAS_ENABLE_SOLVER)
        add_library(hipblas-fortran-client STATIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblas_fortran_module.f90"
            $<TARGET_OBJECTS:hipblas-fortran>
        )
    else()
        add_library(hipblas-fortran-client STATIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblas_fortran_no_solver_module.f90"
            $<TARGET_OBJECTS:hipblas-fortran>
        )
    endif()
endif()

# Build samples
if(HIPBLAS_ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()

# Build tests and benchmarks
if(HIPBLAS_BUILD_TESTING)
    add_subdirectory(gtest)
endif()

if(HIPBLAS_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Install common YAML and Python files for tests/benchmarks
if(HIPBLAS_BUILD_TESTING OR HIPBLAS_ENABLE_BENCHMARKS)
    set(HIPBLAS_COMMON "${PROJECT_BINARY_DIR}/staging/hipblas_common.yaml")
    add_custom_command(
        OUTPUT "${HIPBLAS_COMMON}"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblas_common.yaml"
            "${HIPBLAS_COMMON}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblas_common.yaml"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    
    set(HIPBLAS_TEMPLATE "${PROJECT_BINARY_DIR}/staging/hipblas_template.yaml")
    add_custom_command(
        OUTPUT "${HIPBLAS_TEMPLATE}"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblas_template.yaml"
            "${HIPBLAS_TEMPLATE}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblas_template.yaml"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    
    set(HIPBLAS_SMOKE "${PROJECT_BINARY_DIR}/staging/hipblas_smoke.yaml")
    add_custom_command(
        OUTPUT "${HIPBLAS_SMOKE}"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblas_smoke.yaml"
            "${HIPBLAS_SMOKE}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblas_smoke.yaml"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    
    set(HIPBLAS_GENTEST "${PROJECT_BINARY_DIR}/staging/hipblas_gentest.py")
    add_custom_command(
        OUTPUT "${HIPBLAS_GENTEST}"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/common/hipblas_gentest.py"
            "${HIPBLAS_GENTEST}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/common/hipblas_gentest.py"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    
    set(HIPBLAS_CLIENTS_README "${PROJECT_BINARY_DIR}/staging/hipblas_clients_readme.txt")
    add_custom_command(
        OUTPUT "${HIPBLAS_CLIENTS_README}"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/hipblas_clients_readme.txt"
            "${HIPBLAS_CLIENTS_README}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/hipblas_clients_readme.txt"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    
    add_custom_target(hipblas-clients-common-files
        DEPENDS
            "${HIPBLAS_COMMON}"
            "${HIPBLAS_TEMPLATE}"
            "${HIPBLAS_SMOKE}"
            "${HIPBLAS_GENTEST}"
            "${HIPBLAS_CLIENTS_README}"
    )
    
    rocm_install(
        FILES ${HIPBLAS_COMMON} ${HIPBLAS_TEMPLATE} ${HIPBLAS_SMOKE}
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT clients-common
    )
    
    rocm_install(
        PROGRAMS ${HIPBLAS_GENTEST}
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT clients-common
    )
    
    # This readme also serves to prevent an empty package hipblas-clients
    # which dpkg may auto-remove entire hipblas-clients and non empty children
    rocm_install(
        FILES ${HIPBLAS_CLIENTS_README}
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT clients
    )
endif()
