# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(GTest REQUIRED)

add_executable(rocsolver-test)

target_sources(rocsolver-test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/rocsolver_gtest_main.cpp"
    "${PROJECT_SOURCE_DIR}/common/src/common_host_helpers.cpp"
    
    # LAPACK tests - linear systems solvers
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/getri_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/getrs_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/gesv_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/potrs_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/posv_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/potri_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/trtri_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/geblttrs_gtest.cpp"
    
    # LAPACK tests - least squares solvers
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/gels_gtest.cpp"
    
    # LAPACK tests - triangular factorizations
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/getf2_getrf_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/getrf_large_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/potf2_potrf_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/sytf2_sytrf_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/geblttrf_gtest.cpp"
    
    # LAPACK tests - orthogonal factorizations
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/geqr2_geqrf_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/gerq2_gerqf_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/geql2_geqlf_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/gelq2_gelqf_gtest.cpp"
    
    # LAPACK tests - problem and matrix reductions
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/gebd2_gebrd_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/sytxx_hetxx_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/sygsx_hegsx_gtest.cpp"
    
    # LAPACK tests - singular value decomposition
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/gesvd_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/gesdd_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/gesvdj_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/gesvdx_gtest.cpp"
    
    # LAPACK tests - symmetric eigensolvers
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/syev_heev_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/syevd_heevd_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/sygv_hegv_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/sygvd_hegvd_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/syevj_heevj_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/syevdj_heevdj_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/sygvj_hegvj_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/sygvdj_hegvdj_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/syevx_heevx_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/syevdx_heevdx_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/sygvx_hegvx_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/sygvdx_hegvdx_gtest.cpp"
    
    # Auxiliary tests - vector & matrix manipulations
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/lacgv_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/laswp_gtest.cpp"
    
    # Auxiliary tests - householder reflections
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/larf_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/larfg_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/larft_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/larfb_gtest.cpp"
    
    # Auxiliary tests - plane rotations
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/lasr_gtest.cpp"
    
    # Auxiliary tests - orthonormal/unitary matrices
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/orgxr_ungxr_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/orglx_unglx_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/orgxl_ungxl_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/orgbr_ungbr_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/orgtr_ungtr_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/ormxr_unmxr_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/ormlx_unmlx_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/ormxl_unmxl_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/ormbr_unmbr_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/ormtr_unmtr_gtest.cpp"
    
    # Auxiliary tests - bidiagonal matrices
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/labrd_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/bdsqr_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/bdsvdx_gtest.cpp"
    
    # Auxiliary tests - tridiagonal matrices
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/sterf_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/steqr_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/stedc_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/stedcj_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/stedcx_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/stebz_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/stein_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/latrd_gtest.cpp"
    
    # Auxiliary tests - symmetric matrices
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/lasyf_gtest.cpp"
    
    # Auxiliary tests - triangular matrices
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/lauum_gtest.cpp"
    
    # Refactorization tests
    "${CMAKE_CURRENT_SOURCE_DIR}/refact/csrrf_analysis_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/refact/csrrf_workflow_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/refact/csrrf_sumlu_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/refact/csrrf_splitlu_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/refact/csrrf_refactlu_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/refact/csrrf_refactchol_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/refact/csrrf_solve_gtest.cpp"
    
    # Unit tests
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/gemm_gtest.cpp"
    
    # Other tests
    "${CMAKE_CURRENT_SOURCE_DIR}/managed_malloc_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/memory_model_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/logging_gtest.cpp"
)

target_compile_features(rocsolver-test PRIVATE cxx_std_17)

add_armor_flags(rocsolver-test "${ARMOR_LEVEL}")

target_compile_definitions(rocsolver-test PRIVATE
    ROCM_USE_FLOAT16
    ROCSOLVER_CLIENTS_TEST
)

# Explicit dependencies - no INTERFACE library
target_link_libraries(rocsolver-test PRIVATE
    $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,GTest::GTest>
    roc::rocsolver
    roc::rocblas
    rocsolver-test-common
    rocsolver-common
    hip::host
    Threads::Threads
    ${LAPACK_LIBRARIES}
    $<$<PLATFORM_ID:Linux>:stdc++fs>
    $<$<PLATFORM_ID:Linux>:m>
)

if(NOT USE_HIPCXX)
    target_link_libraries(rocsolver-test PRIVATE hip::device)
endif()

if(LAPACK_LINKER_FLAGS)
    target_link_options(rocsolver-test PRIVATE
        ${LAPACK_LINKER_FLAGS}
    )
endif()

# Windows-specific: copy third-party DLLs to staging
if(WIN32)
    file(GLOB third_party_dlls
        LIST_DIRECTORIES OFF
        CONFIGURE_DEPENDS
        ${ROCSOLVER_LAPACK_PATH}/bin/*.dll
        ${GTest_DIR}/bin/*.dll
        $ENV{rocblas_DIR}/bin/*.dll
        $ENV{HIP_DIR}/bin/*.dll
        $ENV{HIP_DIR}/bin/hipinfo.exe
        ${CMAKE_SOURCE_DIR}/rtest.*
    )
    foreach(file_i ${third_party_dlls})
        add_custom_command(TARGET rocsolver-test
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${file_i} $<TARGET_FILE_DIR:rocsolver-test>
        )
    endforeach()
    if(DEFINED ENV{rocblas_DIR})
        add_custom_command(TARGET rocsolver-test
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory 
                $ENV{rocblas_DIR}/bin/rocblas/library 
                $<TARGET_FILE_DIR:rocsolver-test>/library
        )
    else()
        message(WARNING "rocblas_DIR not set. rocBLAS Tensile runtime kernels will not be copied to client staging directory (insitu testing affected).")
    endif()
endif()

add_test(
    NAME rocsolver-test
    COMMAND rocsolver-test
)

rocm_install(TARGETS rocsolver-test COMPONENT tests)
