# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Enable Fortran for LAPACK on UNIX systems
if(UNIX)
    enable_language(Fortran)
endif()

# Specify where to put the client binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/staging")

# The rocsolver target will exist if the library is being built along with the clients,
# but if this is a clients-only build, we'll have to search for it.
if(NOT TARGET rocsolver)
    find_package(rocsolver REQUIRED CONFIG PATHS ${ROCM_PATH}/rocsolver /opt/rocm/rocsolver)
    get_imported_target_location(location roc::rocsolver)
    message(STATUS "Found rocSOLVER: ${location}")
endif()

if(ROCSOLVER_ENABLE_BENCHMARKS OR ROCSOLVER_ENABLE_TESTS)
    if(ROCSOLVER_FIND_PACKAGE_LAPACK_CONFIG)
        find_package(LAPACK 3.7 REQUIRED CONFIG)
    else()
        find_package(LAPACK 3.7 REQUIRED)
    endif()

    if(NOT LAPACK_LIBRARIES)
        set(LAPACK_LIBRARIES
            ${LAPACK_blas_LIBRARIES}
            ${LAPACK_lapack_LIBRARIES}
        )
    endif()

    # Add common test infrastructure
    add_subdirectory(common)

    # Copy and point to sparse test data
    file(COPY
        "${CMAKE_CURRENT_SOURCE_DIR}/sparsedata/"
        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/staging/sparsedata/"
    )
    install(DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/sparsedata/"
        DESTINATION ${CMAKE_INSTALL_DATADIR}/rocsolver/test
        COMPONENT tests
    )

    if(ROCSOLVER_ENABLE_BENCHMARKS)
        add_subdirectory(benchmarks)
    endif()

    if(ROCSOLVER_ENABLE_TESTS)
        add_subdirectory(gtest)
    endif()
endif()

if(ROCSOLVER_ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()

if(ROCSOLVER_ENABLE_EXTRA_TESTS)
    add_subdirectory(extras)
endif()
