# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Add source files to hipblas target
target_sources(hipblas PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/hipblas_auxiliary.cpp"
)

# Platform-specific sources
if(HIPBLAS_ENABLE_HIP)
    target_sources(hipblas PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/hipblas.cpp"
    )
    
    # NOTE: amd_detail/hipblas.cpp is compiled with the verbose flag -v for QC purposes
    # This flag enables verbose compiler output for quality control
    set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/hipblas.cpp"
        PROPERTIES COMPILE_OPTIONS "-v"
    )
elseif(HIPBLAS_ENABLE_CUDA)
    target_sources(hipblas PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/nvidia_detail/hipblas.cpp"
    )
endif()

# Target include directories
target_include_directories(hipblas
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/hipblas>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Find and link hipblas-common
find_package(hipblas-common REQUIRED CONFIG PATHS ${ROCM_PATH})
target_link_libraries(hipblas PUBLIC roc::hipblas-common)

# Platform-specific dependencies
if(HIPBLAS_ENABLE_HIP)
    # Find rocBLAS
    if(CUSTOM_ROCBLAS)
        set(ENV{rocblas_DIR} ${CUSTOM_ROCBLAS})
        find_package(rocblas REQUIRED CONFIG NO_CMAKE_PATH)
    elseif(WIN32)
        find_package(rocblas REQUIRED CONFIG PATHS ${ROCBLAS_PATH} ${ROCM_PATH})
    else()
        find_package(rocblas REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm /opt/rocm/rocblas /usr/local/rocblas)
    endif()
    
    target_link_libraries(hipblas PRIVATE roc::rocblas)
    target_link_libraries(hipblas PUBLIC hip::host)
    
    # Add rocSOLVER as a dependency if HIPBLAS_ENABLE_SOLVER is ON
    if(HIPBLAS_ENABLE_SOLVER)
        if(CUSTOM_ROCSOLVER)
            set(ENV{rocsolver_DIR} ${CUSTOM_ROCSOLVER})
            find_package(rocsolver REQUIRED CONFIG NO_CMAKE_PATH)
            
            # In case of using custom rocsolver and not custom rocblas, we need to have
            # custom rocsolver include directories before rocblas/hip include directories
            # in case there is a rocsolver installed on the system
            target_include_directories(hipblas
                SYSTEM PRIVATE $<BUILD_INTERFACE:${ROCSOLVER_INCLUDE_DIRS}>
            )
        elseif(WIN32)
            find_package(rocsolver REQUIRED CONFIG PATHS ${ROCSOLVER_PATH} ${ROCM_PATH})
        else()
            find_package(rocsolver REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm /opt/rocm/rocsolver /usr/local/rocsolver)
        endif()
        
        target_link_libraries(hipblas PRIVATE roc::rocsolver)
    endif()
    
    if(CUSTOM_TARGET)
        target_link_libraries(hipblas PRIVATE hip::${CUSTOM_TARGET})
    endif()
    
    # External header includes as system files
    target_include_directories(hipblas
        SYSTEM PRIVATE
            $<BUILD_INTERFACE:${ROCBLAS_INCLUDE_DIRS}>
            $<BUILD_INTERFACE:${ROCSOLVER_INCLUDE_DIRS}>
            $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    )
elseif(HIPBLAS_ENABLE_CUDA)
    target_compile_definitions(hipblas PRIVATE
        __HIP_PLATFORM_NVCC__
        __HIP_PLATFORM_NVIDIA__
    )
    
    target_link_libraries(hipblas PRIVATE ${CUDA_CUBLAS_LIBRARIES})
    
    target_include_directories(hipblas
        SYSTEM PRIVATE
            $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )
endif()

# Windows-specific post-build commands
if(WIN32)
    add_custom_command(
        TARGET hipblas POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${PROJECT_BINARY_DIR}/staging/$<TARGET_FILE_NAME:hipblas>
            ${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_FILE_NAME:hipblas>
    )
    
    # Use generator expression for config-specific logic
    add_custom_command(
        TARGET hipblas POST_BUILD
        COMMAND $<$<CONFIG:Debug>:${CMAKE_COMMAND}> $<$<CONFIG:Debug>:-E> $<$<CONFIG:Debug>:copy>
            $<$<CONFIG:Debug>:${PROJECT_BINARY_DIR}/staging/hipblas.pdb>
            $<$<CONFIG:Debug>:${PROJECT_BINARY_DIR}/clients/staging/hipblas.pdb>
    )
endif()
