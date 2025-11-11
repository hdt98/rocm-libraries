# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Define the main library target
add_library(rocsolver)
add_library(roc::rocsolver ALIAS rocsolver)

# Set library version
rocm_set_soversion(rocsolver ${rocsolver_SOVERSION})

# Add all source files directly (no variables)
target_sources(rocsolver PRIVATE
    # Common source from PROJECT_SOURCE_DIR/common/src
    "${PROJECT_SOURCE_DIR}/common/src/common_host_helpers.cpp"
    
    # Auxiliaries
    "${CMAKE_CURRENT_SOURCE_DIR}/common/buildinfo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/rocsolver_handle.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/rocsolver_logger.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/rocsparse.cpp"
    
    # Auxiliary functions - vector & matrix manipulations
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_aliases.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_lacgv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_laswp.cpp"
    
    # Auxiliary functions - householder reflections
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_larfg.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_larf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_larft.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_larfb.cpp"
    
    # Auxiliary functions - plane rotations
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_lasr.cpp"
    
    # Auxiliary functions - orthonormal/unitary matrices
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_org2r_ung2r.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_orgqr_ungqr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_orgl2_ungl2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_orglq_unglq.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_org2l_ung2l.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_orgql_ungql.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_orgbr_ungbr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_orgtr_ungtr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_orm2r_unm2r.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_ormqr_unmqr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_orml2_unml2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_ormlq_unmlq.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_orm2l_unm2l.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_ormql_unmql.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_ormbr_unmbr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_ormtr_unmtr.cpp"
    
    # Auxiliary functions - bidiagonal matrices
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_bdsqr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_bdsvdx.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_labrd.cpp"
    
    # Auxiliary functions - tridiagonal matrices
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_sterf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_stebz.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_steqr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_stedc.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_stedcj.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_stedcx.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_stein.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_latrd.cpp"
    
    # Auxiliary functions - symmetric matrices
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_lasyf.cpp"
    
    # Auxiliary functions - triangular matrices
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary/rocauxiliary_lauum.cpp"
    
    # LAPACK functions - linear systems solvers - general inverse
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_getri.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_getri_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_getri_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_getri_outofplace.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_getri_outofplace_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_getri_outofplace_strided_batched.cpp"
    
    # LAPACK functions - symmetric positive definite inverse
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_potri.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_potri_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_potri_strided_batched.cpp"
    
    # LAPACK functions - triangular inverse
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_trtri.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_trtri_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_trtri_strided_batched.cpp"
    
    # LAPACK functions - general systems
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_getrs.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_getrs_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_getrs_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gesv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gesv_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gesv_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gesv_outofplace.cpp"
    
    # LAPACK functions - symmetric positive definite systems
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_potrs.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_potrs_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_potrs_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_posv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_posv_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_posv_strided_batched.cpp"
    
    # LAPACK functions - block-tridiagonal systems
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geblttrs_npvt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geblttrs_npvt_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geblttrs_npvt_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geblttrs_npvt_interleaved_batched.cpp"
    
    # LAPACK functions - least-squares solvers
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gels.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gels_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gels_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gels_outofplace.cpp"
    
    # LAPACK functions - triangular factorizations - general matrices
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_getf2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_getf2_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_getf2_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_getrf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_getrf_info32.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_getrf_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_getrf_strided_batched.cpp"
    
    # LAPACK functions - symmetric positive definite matrices
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_potf2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_potf2_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_potf2_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_potrf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_potrf_info32.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_potrf_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_potrf_strided_batched.cpp"
    
    # LAPACK functions - symmetric indefinite matrices
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sytf2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sytf2_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sytf2_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sytrf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sytrf_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sytrf_strided_batched.cpp"
    
    # LAPACK functions - block-tridiagonal matrices
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geblttrf_npvt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geblttrf_npvt_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geblttrf_npvt_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geblttrf_npvt_interleaved_batched.cpp"
    
    # LAPACK functions - orthogonal factorizations - bottom row compression
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geqr2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geqr2_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geqr2_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geqrf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geqrf_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geqrf_ptr_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geqrf_strided_batched.cpp"
    
    # LAPACK functions - top row compression
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geql2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geql2_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geql2_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geqlf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geqlf_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_geqlf_strided_batched.cpp"
    
    # LAPACK functions - right column compression
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gelq2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gelq2_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gelq2_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gelqf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gelqf_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gelqf_strided_batched.cpp"
    
    # LAPACK functions - left column compression
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gerq2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gerq2_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gerq2_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gerqf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gerqf_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gerqf_strided_batched.cpp"
    
    # LAPACK functions - bidiagonalization
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gebd2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gebd2_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gebd2_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gebrd.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gebrd_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gebrd_strided_batched.cpp"
    
    # LAPACK functions - tridiagonalization
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sytd2_hetd2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sytd2_hetd2_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sytd2_hetd2_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sytrd_hetrd.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sytrd_hetrd_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sytrd_hetrd_strided_batched.cpp"
    
    # LAPACK functions - reduction of generalized eigenproblems
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygs2_hegs2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygs2_hegs2_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygs2_hegs2_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygst_hegst.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygst_hegst_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygst_hegst_strided_batched.cpp"
    
    # LAPACK functions - SVD - classic
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gesvd.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gesvd_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gesvd_strided_batched.cpp"
    
    # LAPACK functions - SVD - divide-and-conquer
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gesdd.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gesdd_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gesdd_strided_batched.cpp"
    
    # LAPACK functions - SVD - jacobi
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gesvdj.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gesvdj_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gesvdj_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gesvdj_notransv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gesvdj_notransv_strided_batched.cpp"
    
    # LAPACK functions - SVD - bisection
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gesvdx.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gesvdx_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gesvdx_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_gesvdx_notransv_strided_batched.cpp"
    
    # LAPACK functions - symmetric eigensolvers - classic
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_syev_heev.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_syev_heev_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_syev_heev_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygv_hegv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygv_hegv_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygv_hegv_strided_batched.cpp"
    
    # LAPACK functions - symmetric eigensolvers - divide & conquer
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_syevd_heevd.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_syevd_heevd_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_syevd_heevd_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygvd_hegvd.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygvd_hegvd_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygvd_hegvd_strided_batched.cpp"
    
    # LAPACK functions - symmetric eigensolvers - jacobi
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_syevj_heevj.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_syevj_heevj_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_syevj_heevj_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygvj_hegvj.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygvj_hegvj_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygvj_hegvj_strided_batched.cpp"
    
    # LAPACK functions - symmetric eigensolvers - bisection
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_syevx_heevx.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_syevx_heevx_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_syevx_heevx_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygvx_hegvx.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygvx_hegvx_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygvx_hegvx_strided_batched.cpp"
    
    # LAPACK functions - symmetric eigensolvers - d&c + jacobi
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_syevdj_heevdj.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_syevdj_heevdj_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_syevdj_heevdj_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygvdj_hegvdj.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygvdj_hegvdj_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygvdj_hegvdj_strided_batched.cpp"
    
    # LAPACK functions - symmetric eigensolvers - d&c + bisection
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_syevdx_heevdx.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_syevdx_heevdx_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_syevdx_heevdx_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygvdx_hegvdx.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygvdx_hegvdx_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygvdx_hegvdx_strided_batched.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_syevdx_heevdx_inplace.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack/roclapack_sygvdx_hegvdx_inplace.cpp"
    
    # Refactorization functions - rfinfo analysis
    "${CMAKE_CURRENT_SOURCE_DIR}/refact/rocsolver_rfinfo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/refact/rocrefact_csrrf_analysis.cpp"
    
    # Refactorization functions - re-factorization
    "${CMAKE_CURRENT_SOURCE_DIR}/refact/rocrefact_csrrf_sumlu.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/refact/rocrefact_csrrf_splitlu.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/refact/rocrefact_csrrf_refactlu.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/refact/rocrefact_csrrf_refactchol.cpp"
    
    # Refactorization functions - direct solver
    "${CMAKE_CURRENT_SOURCE_DIR}/refact/rocrefact_csrrf_solve.cpp"
    
    # Specialized kernels - larf
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/rocauxiliary_larf_specialized_kernels_s.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/rocauxiliary_larf_specialized_kernels_d.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/rocauxiliary_larf_specialized_kernels_c.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/rocauxiliary_larf_specialized_kernels_z.cpp"
    
    # Specialized kernels - larfg
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/rocauxiliary_larfg_specialized_kernels_s.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/rocauxiliary_larfg_specialized_kernels_d.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/rocauxiliary_larfg_specialized_kernels_c.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/rocauxiliary_larfg_specialized_kernels_z.cpp"
    
    # Specialized kernels - gemm
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_gemm_specialized_kernels_s.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_gemm_specialized_kernels_d.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_gemm_specialized_kernels_c.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_gemm_specialized_kernels_z.cpp"
    
    # Specialized kernels - ger
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_ger_specialized_kernels_s.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_ger_specialized_kernels_d.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_ger_specialized_kernels_c.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_ger_specialized_kernels_z.cpp"
    
    # Specialized kernels - trsm
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_trsm_specialized_kernels_s.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_trsm_specialized_kernels_d.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_trsm_specialized_kernels_c.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_trsm_specialized_kernels_z.cpp"
    
    # Specialized kernels - potf2
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_potf2_specialized_kernels_s.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_potf2_specialized_kernels_d.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_potf2_specialized_kernels_c.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_potf2_specialized_kernels_z.cpp"
)

# Add OPTIMAL specialized kernels conditionally
if(OPTIMAL)
    target_sources(rocsolver PRIVATE
        # Specialized kernels - getf2
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_getf2_specialized_kernels_s.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_getf2_specialized_kernels_d.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_getf2_specialized_kernels_c.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_getf2_specialized_kernels_z.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_getf2_small_s.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_getf2_small_d.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_getf2_small_c.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_getf2_small_z.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_getf2_small_sb.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_getf2_small_db.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_getf2_small_cb.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_getf2_small_zb.cpp"
        
        # Specialized kernels - getri
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_getri_specialized_kernels_s.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_getri_specialized_kernels_d.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_getri_specialized_kernels_c.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_getri_specialized_kernels_z.cpp"
        
        # Specialized kernels - trtri
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_trtri_specialized_kernels_s.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_trtri_specialized_kernels_d.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_trtri_specialized_kernels_c.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/specialized/roclapack_trtri_specialized_kernels_z.cpp"
    )
    target_compile_definitions(rocsolver PRIVATE OPTIMAL)
endif()

# Target link libraries with explicit PUBLIC/PRIVATE
target_link_libraries(rocsolver
    PUBLIC
        roc::rocblas
    PRIVATE
        $<BUILD_INTERFACE:rocsolver-common>
        roc::rocprim
        $<$<PLATFORM_ID:Linux>:--rtlib=compiler-rt>
        $<$<PLATFORM_ID:Linux>:--unwindlib=libgcc>
)

if(NOT ROCSOLVER_ENABLE_HIPCXX)
    target_link_libraries(rocsolver PRIVATE hip::device)
endif()

if(NOT ROCSOLVER_ENABLE_SPARSE)
    target_link_libraries(rocsolver PRIVATE $<$<PLATFORM_ID:Linux>:${CMAKE_DL_LIBS}>)
endif()

# Conditional rocsparse support
if(ROCSOLVER_ENABLE_SPARSE)
    target_link_libraries(rocsolver PRIVATE roc::rocsparse)
    target_compile_definitions(rocsolver PRIVATE HAVE_ROCSPARSE)
endif()

# fmt library handling
if(ROCSOLVER_EMBED_FMT)
    target_link_libraries(rocsolver PRIVATE $<BUILD_INTERFACE:fmt::fmt-header-only>)
else()
    target_link_libraries(rocsolver PRIVATE fmt::fmt)
endif()

# Compiler options for HIP
if(ROCSOLVER_ENABLE_HIPCXX)
    target_compile_options(rocsolver PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:--gpu-max-threads-per-block=1024>
    )
else()
    # ROCm 4.0 and earlier had 256 max threads per block default
    target_compile_options(rocsolver PRIVATE --gpu-max-threads-per-block=1024)
endif()

# Ignore loop unrolling failures
target_compile_options(rocsolver PRIVATE -Wno-pass-failed)

# Compiler optimization options for better performance
# hipcc adds these automatically; add them for direct clang usage
if(USE_HIPCXX)
    target_compile_options(rocsolver PRIVATE
        "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-mllvm -amdgpu-early-inline-all=true>"
        "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-mllvm -amdgpu-function-calls=false>"
    )
elseif(CMAKE_CXX_COMPILER MATCHES ".*clang\\+\\+.*")
    target_compile_options(rocsolver PRIVATE
        "SHELL:-mllvm -amdgpu-early-inline-all=true"
        "SHELL:-mllvm -amdgpu-function-calls=false"
    )
endif()

if(ROCSOLVER_ENABLE_OFFLOAD_COMPRESS)
    target_compile_options(rocsolver PRIVATE "--offload-compress")
endif()

# Include directories
target_include_directories(rocsolver
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/rocsolver>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Output directory
set_target_properties(rocsolver PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

# Export header generation
include(GenerateExportHeader)
set_target_properties(rocsolver PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
)
generate_export_header(rocsolver 
    EXPORT_FILE_NAME "${PROJECT_BINARY_DIR}/include/rocsolver/rocsolver-export.h"
)

# Static library configuration
if(NOT BUILD_SHARED_LIBS)
    set_target_properties(rocsolver PROPERTIES PREFIX "lib")
    target_compile_definitions(rocsolver PRIVATE ROCSOLVER_STATIC_LIB)
endif()

# Compile definitions
if(ROCSOLVER_ENABLE_INTERNAL_BLAS)
    target_compile_definitions(rocsolver PRIVATE USE_INTERNAL_GEMM USE_INTERNAL_TRSM)
endif()

if(ROCSOLVER_ENABLE_REFERENCE_SECULAR_SOLVER)
    target_compile_definitions(rocsolver PRIVATE ROCSOLVER_USE_REFERENCE_SECULAR_EQUATIONS_SOLVER)
endif()

# Asynchronous logger option
option(ROCSOLVER_ENABLE_ASYNC_LOGGER "Enable asynchronous HIP event-based logger (reduces GPU synchronizations)" ON)

# Configure asynchronous logger compile definition
if(ROCSOLVER_ENABLE_ASYNC_LOGGER)
    set(ROCSOLVER_ASYNC_LOGGER_DEFINITION "ROCSOLVER_USE_ASYNC_LOGGER=1")
else()
    set(ROCSOLVER_ASYNC_LOGGER_DEFINITION "ROCSOLVER_USE_ASYNC_LOGGER=0")
endif()

target_compile_definitions(rocsolver PRIVATE
    ROCM_USE_FLOAT16
    ROCBLAS_INTERNAL_API
    ROCSOLVER_LIBRARY
    ${ROCSOLVER_ASYNC_LOGGER_DEFINITION}
)

# Apply armor (security hardening) flags
add_armor_flags(rocsolver "${ARMOR_LEVEL}")

# Windows-specific: copy library to client staging directory
if(WIN32)
    if(ROCSOLVER_ENABLE_BENCHMARKS OR ROCSOLVER_ENABLE_TESTS)
        add_custom_command(TARGET rocsolver
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy 
                "${PROJECT_BINARY_DIR}/staging/$<TARGET_FILE_NAME:rocsolver>" 
                "${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_FILE_NAME:rocsolver>"
        )
        if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
            add_custom_command(TARGET rocsolver
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy 
                    "${PROJECT_BINARY_DIR}/staging/rocsolver.pdb" 
                    "${PROJECT_BINARY_DIR}/clients/staging/rocsolver.pdb"
            )
        endif()
    endif()
endif()
