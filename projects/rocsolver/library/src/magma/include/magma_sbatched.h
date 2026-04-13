/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Tingxing Dong

       @generated from include/magma_zbatched.h, normal z -> s, Wed Apr  8 05:20:24 2026
*/

#ifndef MAGMA_SBATCHED_H
#define MAGMA_SBATCHED_H

#include "magma_types.h"

#define MAGMA_REAL

#ifdef __cplusplus
extern "C" {
#endif
  /*
   *  local auxiliary routines
   */
void
magma_sset_pointer(
    float **output_array,
    float *input,
    magma_int_t lda,
    magma_int_t row, magma_int_t column,
    magma_int_t batch_offset,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_sdisplace_pointers(
    float **output_array,
    float **input_array, magma_int_t lda,
    magma_int_t row, magma_int_t column,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_srecommend_cublas_gemm_batched(
    magma_trans_t transa, magma_trans_t transb,
    magma_int_t m, magma_int_t n, magma_int_t k);

magma_int_t
magma_srecommend_cublas_gemm_stream(
    magma_trans_t transa, magma_trans_t transb,
    magma_int_t m, magma_int_t n, magma_int_t k);

void magma_get_spotrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb);

magma_int_t magma_get_spotrf_batched_crossover();

void magma_get_sgetrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb);
magma_int_t magma_get_sgetrf_batched_ntcol(magma_int_t m, magma_int_t n);
magma_int_t magma_get_sgemm_batched_ntcol(magma_int_t n);
magma_int_t magma_get_sgemm_batched_smallsq_limit(magma_int_t n);
magma_int_t magma_get_sgeqrf_batched_nb(magma_int_t m);
magma_int_t magma_use_sgeqrf_batched_fused_update(magma_int_t m, magma_int_t n, magma_int_t batchCount);
magma_int_t magma_get_sgeqr2_fused_sm_batched_nthreads(magma_int_t m, magma_int_t n);
magma_int_t magma_get_sgeqrf_batched_ntcol(magma_int_t m, magma_int_t n);
magma_int_t magma_get_sgetri_batched_ntcol(magma_int_t m, magma_int_t n);
magma_int_t magma_get_strsm_batched_stop_nb(magma_side_t side, magma_int_t m, magma_int_t n);
magma_int_t magma_sorm2r_batched_kernel_sm_size(magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t ib, magma_int_t nb);
magma_int_t magma_get_ssyevj_batched_small_nthreads(magma_int_t n);
void        magma_get_sgbtrf_batched_params(magma_int_t m, magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t *nb, magma_int_t *threads);
bool        magma_sgesvj_batched_use_fused( magma_vec_t jobu, magma_vec_t jobv, magma_int_t m, magma_int_t n );
magma_int_t magma_get_sgesvj_batched_nb( magma_int_t m, magma_int_t n );

#ifdef MAGMA_REAL
magma_int_t
magmablas_ssort_batched(
    magma_sort_t sort, magma_int_t n,
    float const * const * dx_array, magma_int_t incx,
    float               **dy_array, magma_int_t incy,
    magma_int_t** dindex_array,
    magma_int_t batchCount, magma_queue_t queue);
#endif

  /*
   *  LAPACK batched routines
   */

  /*
   *  BLAS batched routines
   */
void
magmablas_sgemm_batched_core(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    float const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
    float beta,
    float **dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magma_sgemm_batched_core(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    float const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
    float beta,
    float **dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magma_sgemm_batched(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t ldda,
    float const * const * dB_array, magma_int_t lddb,
    float beta,
    float **dC_array, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_sgemm_batched(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t ldda,
    float const * const * dB_array, magma_int_t lddb,
    float beta,
    float **dC_array, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_sgemm_batched_strided(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    float const * dA, magma_int_t ldda, magma_int_t strideA,
    float const * dB, magma_int_t lddb, magma_int_t strideB,
    float beta,
    float       * dC, magma_int_t lddc, magma_int_t strideC,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_sgemm_batched_smallsq(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    float const * const * dB_array, magma_int_t bi, magma_int_t bj, magma_int_t lddb,
    float beta,
    float **dC_array, magma_int_t ci, magma_int_t cj, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyrk_batched_core(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    float const * const * dB_array, magma_int_t bi, magma_int_t bj, magma_int_t lddb,
    float beta,
    float **dC_array, magma_int_t ci, magma_int_t cj, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyrk_batched_core(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    float const * const * dB_array, magma_int_t bi, magma_int_t bj, magma_int_t lddb,
    float beta,
    float **dC_array, magma_int_t ci, magma_int_t cj, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyrk_batched(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t ldda,
    float beta,
    float **dC_array, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magma_ssyrk_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t ldda,
    float beta,
    float **dC_array, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyrk_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t ldda,
    float beta,
    float **dC_array, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyr2k_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t ldda,
    float const * const * dB_array, magma_int_t lddb,
    float beta, float **dC_array, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_ssyevj_batched_expert_small_sm(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n, float** dA_array, magma_int_t ldda,
    float **dW_array, magma_int_t* info_array,
    int* batch_mask, int *num_sweeps,
    float heevj_tol, magma_int_t sort_flag, magma_int_t max_sweeps,
    magma_int_t nthreads, magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_ssyevj_batched_strided_expert_small_sm(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n, float* dA, magma_int_t ldda, magma_int_t strideA,
    float *dW, magma_int_t strideW,
    magma_int_t* info_array, int* batch_mask, int *num_sweeps,
    float heevj_tol, magma_int_t sort_flag, magma_int_t max_sweeps,
    magma_int_t nthreads, magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_ssyevj_batched_small_sm(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    float** dA_array, magma_int_t ldda, float **dW_array,
    magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_ssyevj_batched_strided_small_sm(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    float* dA, magma_int_t ldda, magma_int_t strideA,
    float *dW, magma_int_t strideW,
    magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyr2k_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t ldda,
    float const * const * dB_array, magma_int_t lddb,
    float beta, float **dC_array, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_strtri_diag_batched(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    float const * const *dA_array, magma_int_t ldda,
    float **dinvA_array,
    magma_int_t resetozero,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_strsm_small_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        float alpha,
        float **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        float **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_strsm_recursive_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        float alpha,
        float **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        float **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_strsm_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        float alpha,
        float **dA_array, magma_int_t ldda,
        float **dB_array, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_strsm_inv_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    float** dA_array,    magma_int_t ldda,
    float** dB_array,    magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_strsm_inv_work_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, magma_int_t m, magma_int_t n,
    float alpha,
    float** dA_array,    magma_int_t ldda,
    float** dB_array,    magma_int_t lddb,
    float** dX_array,    magma_int_t lddx,
    float** dinvA_array, magma_int_t dinvA_length,
    float** dA_displ, float** dB_displ,
    float** dX_displ, float** dinvA_displ,
    magma_int_t resetozero,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_strsm_inv_outofplace_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, magma_int_t m, magma_int_t n,
    float alpha,
    float** dA_array,    magma_int_t ldda,
    float** dB_array,    magma_int_t lddb,
    float** dX_array,    magma_int_t lddx,
    float** dinvA_array, magma_int_t dinvA_length,
    float** dA_displ, float** dB_displ,
    float** dX_displ, float** dinvA_displ,
    magma_int_t resetozero,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_strsv_batched(
    magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t n,
    float** dA_array,    magma_int_t ldda,
    float** dB_array,    magma_int_t incb,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_strsv_recursive_batched(
        magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t n,
        float **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        float **dx_array, magma_int_t xi, magma_int_t incx,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_strmm_batched_core(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        float alpha,
        float **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        float **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_strmm_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        float alpha,
        float **dA_array, magma_int_t ldda,
        float **dB_array, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssymm_batched_core(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t m, magma_int_t n,
        float alpha,
        float **dA_array, magma_int_t ldda,
        float **dB_array, magma_int_t lddb,
        float beta,
        float **dC_array, magma_int_t lddc,
        magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssymm_batched(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t m, magma_int_t n,
        float alpha,
        float **dA_array, magma_int_t ldda,
        float **dB_array, magma_int_t lddb,
        float beta,
        float **dC_array, magma_int_t lddc,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssymv_batched_core(
        magma_uplo_t uplo, magma_int_t n,
        float alpha,
        float **dA_array, magma_int_t ldda,
        float **dX_array, magma_int_t incx,
        float beta,
        float **dY_array, magma_int_t incy,
        magma_int_t offA, magma_int_t offX, magma_int_t offY,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssymv_batched(
        magma_uplo_t uplo, magma_int_t n,
        float alpha,
        float **dA_array, magma_int_t ldda,
        float **dX_array, magma_int_t incx,
        float beta,
        float **dY_array, magma_int_t incy,
        magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_spotrf_batched(
    magma_uplo_t uplo, magma_int_t n,
    float **dA_array, magma_int_t lda,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_spotrf_expert_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t recnb,
    float **dA_array, magma_int_t ldda,
    magma_int_t *info_array,  magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_spotf2_batched(
    magma_uplo_t uplo, magma_int_t n,
    float **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_spotrf_panel_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    float** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_spotrf_recpanel_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t min_recpnb,
    float** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_spotrf_rectile_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t min_recpnb,
    float** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_spotrs_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    float **dA_array, magma_int_t ldda,
    float **dB_array, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sposv_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    float **dA_array, magma_int_t ldda,
    float **dB_array, magma_int_t lddb,
    magma_int_t *dinfo_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgetrs_batched(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    float **dA_array, magma_int_t ldda,
    magma_int_t **dipiv_array,
    float **dB_array, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_slaswp_rowparallel_batched(
    magma_int_t n,
    float**  input_array, magma_int_t  input_i, magma_int_t  input_j, magma_int_t ldi,
    float** output_array, magma_int_t output_i, magma_int_t output_j, magma_int_t ldo,
    magma_int_t k1, magma_int_t k2,
    magma_int_t **pivinfo_array,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_slaswp_rowserial_batched(
    magma_int_t n, float** dA_array, magma_int_t lda,
    magma_int_t k1, magma_int_t k2,
    magma_int_t **ipiv_array,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_slaswp_columnserial_batched(
    magma_int_t n, float** dA_array, magma_int_t lda,
    magma_int_t k1, magma_int_t k2,
    magma_int_t **ipiv_array,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_stranspose_batched(
    magma_int_t m, magma_int_t n,
    float **dA_array,  magma_int_t ldda,
    float **dAT_array, magma_int_t lddat,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_stranspose_batched(
    magma_int_t m, magma_int_t n,
    float **dA_array,  magma_int_t ldda,
    float **dAT_array, magma_int_t lddat,
    magma_int_t batchCount, magma_queue_t queue );

void magmablas_slaset_internal_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float offdiag, float diag,
    magmaFloat_ptr dAarray[], magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_slaset_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float offdiag, float diag,
    magmaFloat_ptr dAarray[], magma_int_t ldda,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgbsv_batched(
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    float **dA_array, magma_int_t ldda, magma_int_t **dipiv_array,
    float** dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgbsv_batched_fused_sm(
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    float** dA_array, magma_int_t ldda, magma_int_t** ipiv_array,
    float** dB_array, magma_int_t lddb, magma_int_t* info_array,
    magma_int_t nthreads, magma_int_t ntcol,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sgbsv_batched_strided_work(
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    float* dA, magma_int_t ldda, magma_int_t strideA,
    magma_int_t* dipiv, magma_int_t stride_piv,
    float* dB, magma_int_t lddb, magma_int_t strideB,
    magma_int_t *info_array,
    void* device_work, magma_int_t *lwork,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgbsv_batched_strided(
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    float* dA, magma_int_t ldda, magma_int_t strideA,
    magma_int_t* dipiv, magma_int_t stride_piv,
    float* dB, magma_int_t lddb, magma_int_t strideB,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgbsv_batched_work(
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    float** dA_array, magma_int_t ldda, magma_int_t **dipiv_array,
    float** dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    void* device_work, magma_int_t *lwork,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_sgbtrf_set_fillin(
    magma_int_t n, magma_int_t kl, magma_int_t ku,
    float** dAB_array, magma_int_t lddab,
    magma_int_t** dipiv_array, int* ju_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgbtf2_sswap_batched(
    magma_int_t kl, magma_int_t ku,
    float **dAB_array, magma_int_t ai, magma_int_t aj, magma_int_t lddab,
    magma_int_t** dipiv_array, magma_int_t ipiv_offset,
    int* ju_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgbtf2_scal_ger_batched(
    magma_int_t m, magma_int_t n, magma_int_t kl, magma_int_t ku,
    float **dAB_array, magma_int_t ai, magma_int_t aj, magma_int_t lddab,
    int* ju_array, magma_int_t gbstep, magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgbtrf_batched_fused_sm(
    magma_int_t m,  magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    float** dAB_array, magma_int_t lddab,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t nthreads, magma_int_t ntcol,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sgbtrf_batched_sliding_window_loopout(
    magma_int_t m,  magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    float** dAB_array, magma_int_t lddab,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    void* device_work, magma_int_t *lwork,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sgbtrf_batched_sliding_window_loopin(
    magma_int_t m,  magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    float** dAB_array, magma_int_t lddab,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sgbtrf_batched_strided(
        magma_int_t m, magma_int_t n,
        magma_int_t kl, magma_int_t ku,
        float* dAB, magma_int_t lddab, magma_int_t strideAB,
        magma_int_t* dipiv, magma_int_t stride_piv,
        magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgbtrf_batched_strided_work(
        magma_int_t m, magma_int_t n,
        magma_int_t kl, magma_int_t ku,
        float* dAB, magma_int_t lddab, magma_int_t strideAB,
        magma_int_t* dipiv, magma_int_t stride_piv,
        magma_int_t *info_array,
        void* device_work, magma_int_t *lwork,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgbtrf_batched_work(
        magma_int_t m, magma_int_t n,
        magma_int_t kl, magma_int_t ku,
        float **dAB_array, magma_int_t lddab,
        magma_int_t **dipiv_array, magma_int_t *info_array,
        void* device_work, magma_int_t *lwork,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgbtrf_batched(
    magma_int_t m, magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    float **dAB_array, magma_int_t lddab,
    magma_int_t **dipiv_array, magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgbtrs_batched(
    magma_trans_t transA,
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    float **dA_array, magma_int_t ldda, magma_int_t **dipiv_array,
    float** dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgbtrs_lower_batched(
    magma_trans_t transA,
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    float** dA_array, magma_int_t ldda, magma_int_t **dipiv_array,
    float** dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magmablas_sgbtrs_lower_blocked_batched(
        magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
        float** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
        float** dB_array, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_sgbtrs_swap_batched(
    magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, magma_int_t j,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgbtrs_upper_batched(
    magma_trans_t transA,
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    float** dA_array, magma_int_t ldda, magma_int_t **dipiv_array,
    float** dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magmablas_sgbtrs_upper_blocked_batched(
        magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
        float** dA_array, magma_int_t ldda,
        float** dB_array, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_sgbtrs_upper_columnwise_batched(
    magma_int_t n, magma_int_t kl, magma_int_t ku,
    magma_int_t nrhs, magma_int_t j,
    float** dA_array, magma_int_t ldda,
    float** dB_array, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_sger_batched_core(
    magma_int_t m, magma_int_t n,
    float alpha,
    float** dX_array, magma_int_t xi, magma_int_t xj, magma_int_t lddx, magma_int_t incx,
    float** dY_array, magma_int_t yi, magma_int_t yj, magma_int_t lddy, magma_int_t incy,
    float** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sgesv_batched_small(
    magma_int_t n, magma_int_t nrhs,
    float** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array,
    float **dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sgesvj_batched(
    magma_vec_t jobu, magma_vec_t jobv,
    magma_int_t m, magma_int_t n,
    float** dA_array, magma_int_t ldda, float **dS_array,
    float** dU_array, magma_int_t lddu,
    float** dV_array, magma_int_t lddv,
    magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sgesvj_blocked_expert_batched(
    magma_vec_t jobu, magma_vec_t jobv,
    magma_int_t morg, magma_int_t norg,
    float** dA_array, magma_int_t ldda, float **dS_array,
    float** dU_array, magma_int_t lddu,
    float** dV_array, magma_int_t lddv,
    magma_int_t* info_array,
    magma_int_t nb, magma_int_t max_sweeps,
    magma_int_t heevj_max_sweeps, float heevj_tol, float heevj_tol_min, float heevj_tol_scal,
    void *device_work, int64_t *device_lwork,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sgesvj_expert_batched(
    magma_vec_t jobu, magma_vec_t jobv,
    magma_int_t morg, magma_int_t norg,
    float** dA_array, magma_int_t ldda, float **dS_array,
    float** dU_array, magma_int_t lddu,
    float** dV_array, magma_int_t lddv,
    magma_int_t* info_array,
    void *device_work, int64_t *device_lwork,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sgesvj_expert_batched_strided(
    magma_vec_t jobu, magma_vec_t jobv,
    magma_int_t morg, magma_int_t norg,
    magmaFloat_ptr dA, magma_int_t ldda, magma_int_t strideA,
    magmaFloat_ptr        dS, magma_int_t strideS,
    magmaFloat_ptr dU, magma_int_t lddu, magma_int_t strideU,
    magmaFloat_ptr dV, magma_int_t lddv, magma_int_t strideV,
    magmaInt_ptr dinfo_array,
    void *device_work, int64_t *device_lwork,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sgesvj_qr_expert_batched(
    magma_vec_t jobu_org, magma_vec_t jobv_org,
    magma_int_t morg, magma_int_t norg,
    float** dA_array, magma_int_t ldda, float **dS_array,
    float** dU_array, magma_int_t lddu,
    float** dV_array, magma_int_t lddv,
    magma_int_t* info_array,
    void *device_work, int64_t *device_lwork,
    magma_int_t batchCount, magma_queue_t queue );

void
magma_sgesvj_batched_finalize_values(
    magma_int_t m, magma_int_t n,
    float const* const* dA_array, magma_int_t ldda,
    float** dSigma_array, magma_int_t batchCount, magma_queue_t queue);

void
magma_sgesvj_batched_finalize_vectors(
    magma_vec_t jobu, magma_vec_t jobv, magma_int_t m, magma_int_t n,
    float const* const* dUi_array, magma_int_t lddui,
    float const* const* dVi_array, magma_int_t lddvi,
    float            ** dUo_array, magma_int_t ldduo,
    float            ** dVo_array, magma_int_t lddvo,
    float** dSigma_array, magma_int_t** index_array,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_sgesvj_batched_setup_ptr_arrays(
    magma_vec_t jobv, magma_int_t i_gesvj_sweep, magma_int_t nb, magma_int_t nblk_col2,
    float** dUi_array, float** dUo_array, magma_int_t lddu,
    float** dVi_array, float** dVo_array, magma_int_t lddv,
    float** dAgemm0_array, float** dAgemm1_array,
    float** dUjVj_input_array, float** dUkVk_input_array, float** dUjkVjk_output_array,
    magma_int_t flat_batchCount, magma_queue_t queue);

void
magma_sgesvj_batched_test_convergence(
    magma_int_t gesvj_iters_per_sweep, magma_int_t sub_batch, magma_int_t batchCount,
    magma_int_t* dheevj_info, int* dheevj_nsweeps,
    int* dheevj_mask, int* all_converged,
    magma_queue_t queue );

magma_int_t
magma_sgesvj_batched_small_sm(
    magma_vec_t jobu, magma_vec_t jobv,
    magma_int_t m, magma_int_t n,
    float** dA_array, magma_int_t ldda, float **dS_array,
    float** dU_array, magma_int_t lddu,
    float** dV_array, magma_int_t lddv,
    magma_int_t* info_array, magma_int_t batchCount,
    magma_queue_t queue );

magma_int_t
magma_sgesvj_batched_update_vectors(
    magma_int_t m, magma_int_t nb,
    float **dU0array, magma_int_t lddu0,
    float **dU1array, magma_int_t lddu1,
    float **dGarray,  magma_int_t lddg,
    magma_int_t *heevj_info, int *heevj_nsweeps,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sgetf2_batched(
    magma_int_t m, magma_int_t n,
    float **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda,
    magma_int_t **ipiv_array,
    magma_int_t **dpivinfo_array,
    magma_int_t *info_array,
    magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgetrf_recpanel_batched(
    magma_int_t m, magma_int_t n, magma_int_t min_recpnb,
    float** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t** dipiv_array, magma_int_t** dpivinfo_array,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount,  magma_queue_t queue);

magma_int_t
magma_sgetrf_batched(
    magma_int_t m, magma_int_t n,
    float **dA_array,
    magma_int_t lda,
    magma_int_t **ipiv_array,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgetf2_fused_batched(
    magma_int_t m, magma_int_t n,
    float **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t **dipiv_array,
    magma_int_t *info_array, magma_int_t batchCount,
    magma_queue_t queue);

magma_int_t
magma_sgetrf_batched_smallsq_noshfl(
    magma_int_t n,
    float** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sgetri_outofplace_batched(
    magma_int_t n,
    float **dA_array, magma_int_t ldda,
    magma_int_t **dipiv_array,
    float **dinvA_array, magma_int_t lddia,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_sdisplace_intpointers(
    magma_int_t **output_array,
    magma_int_t **input_array, magma_int_t lda,
    magma_int_t row, magma_int_t column,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_isamax_atomic_batched(
    magma_int_t n,
    float** x_array, magma_int_t incx,
    magma_int_t **max_id_array,
    magma_int_t batchCount);

void
magmablas_isamax_tree_batched(
    magma_int_t n,
    float** x_array, magma_int_t incx,
    magma_int_t **max_id_array,
    magma_int_t batchCount);

void
magmablas_isamax_batched(
    magma_int_t n,
    float** x_array, magma_int_t incx,
    magma_int_t **max_id_array,
    magma_int_t batchCount);

void
magmablas_isamax(
    magma_int_t n,
    float* x, magma_int_t incx,
    magma_int_t *max_id);

magma_int_t
magma_isamax_batched(
        magma_int_t length,
        float **x_array, magma_int_t xi, magma_int_t xj, magma_int_t lda, magma_int_t incx,
        magma_int_t** ipiv_array, magma_int_t ipiv_i,
        magma_int_t step, magma_int_t gbstep, magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sswap_batched(
    magma_int_t n, float **x_array, magma_int_t xi, magma_int_t xj, magma_int_t incx,
    magma_int_t step, magma_int_t** ipiv_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sscal_sger_batched(
    magma_int_t m, magma_int_t n,
    float **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda,
    magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_scomputecolumn_batched(
    magma_int_t m, magma_int_t paneloffset, magma_int_t step,
    float **dA_array,  magma_int_t lda,
    magma_int_t ai, magma_int_t aj,
    magma_int_t **ipiv_array,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_sgetf2trsm_batched(
    magma_int_t ib, magma_int_t n,
    float **dA_array,  magma_int_t j, magma_int_t lda,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgetf2_nopiv_internal_batched(
    magma_int_t m, magma_int_t n,
    float** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sgetf2_nopiv_batched(
    magma_int_t m, magma_int_t n,
    float **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgetrf_recpanel_nopiv_batched(
    magma_int_t m, magma_int_t n, magma_int_t min_recpnb,
    float** dA_array,    magma_int_t ldda,
    float** dX_array,    magma_int_t dX_length,
    float** dinvA_array, magma_int_t dinvA_length,
    float** dW1_displ, float** dW2_displ,
    float** dW3_displ, float** dW4_displ,
    float** dW5_displ,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgetrf_nopiv_batched(
    magma_int_t m, magma_int_t n,
    float **dA_array,
    magma_int_t lda,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgetrs_nopiv_batched(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    float **dA_array, magma_int_t ldda,
    float **dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgesv_nopiv_batched(
    magma_int_t n, magma_int_t nrhs,
    float **dA_array, magma_int_t ldda,
    float **dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgesv_rbt_batched(
    magma_int_t n, magma_int_t nrhs,
    float **dA_array, magma_int_t ldda,
    float **dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgesv_batched(
    magma_int_t n, magma_int_t nrhs,
    float **dA_array, magma_int_t ldda,
    magma_int_t **dipiv_array,
    float **dB_array, magma_int_t lddb,
    magma_int_t *dinfo_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgerbt_batched(
    magma_bool_t gen, magma_int_t n, magma_int_t nrhs,
    float **dA_array, magma_int_t ldda,
    float **dB_array, magma_int_t lddb,
    float *U, float *V,
    magma_int_t *info,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_sprbt_batched(
    magma_int_t n,
    float **dA_array, magma_int_t ldda,
    float *du, float *dv,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_sprbt_mv_batched(
    magma_int_t n, magma_int_t nrhs,
    float *dv, float **db_array, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_sprbt_mtv_batched(
    magma_int_t n, magma_int_t nrhs,
    float *du, float **db_array, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue);

void
setup_pivinfo(
    magma_int_t *pivinfo, magma_int_t *ipiv,
    magma_int_t m, magma_int_t nb,
    magma_queue_t queue);

void
magmablas_sgeadd_batched(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr  const dAarray[], magma_int_t ldda,
    magmaFloat_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_slacpy_internal_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr const dAarray[], magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaFloat_ptr             dBarray[], magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_slacpy_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr  const dAarray[], magma_int_t ldda,
    magmaFloat_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_sgemv_batched_core(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    const float alpha,
    float const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    float const * const * dx_array, magma_int_t xi, magma_int_t incx,
    const float beta,
    float** dy_array, magma_int_t yi, magma_int_t incy,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_sgemv_batched_internal(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    const float alpha,
    float const * const * dA_array, const float* dA, magma_int_t ldda, magma_int_t strideA, magma_int_t Ai, magma_int_t Aj,
    float const * const * dx_array, const float* dx, magma_int_t incx, magma_int_t stridex, magma_int_t xi,
    const float beta,
    float** dy_array, float* dy, magma_int_t incy, magma_int_t stridey, magma_int_t yi,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_sgemv_batched_strided_core(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    const float alpha,
    const float* dA, magma_int_t ldda, magma_int_t Ai, magma_int_t Aj, magma_int_t strideA,
    const float* dx, magma_int_t incx, magma_int_t xi, magma_int_t stridex,
    const float beta,
          float* dy, magma_int_t incy, magma_int_t yi, magma_int_t stridey,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_sgemv_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    const float alpha,
    float const * const * dA_array, magma_int_t ldda,
    float const * const * dx_array, magma_int_t incx,
    const float beta,
    float** dy_array, magma_int_t incy,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_sgemv_batched_strided(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    const float alpha,
    const float* dA, magma_int_t ldda, magma_int_t strideA,
    const float* dx, magma_int_t incx, magma_int_t stridex,
    const float beta,
    float* dy, magma_int_t incy, magma_int_t stridey,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magmablas_sgemv_batched_smallsq(
    magma_trans_t trans, magma_int_t n,
    const float alpha,
    float const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    float const * const * dx_array, magma_int_t xi, magma_int_t incx,
    const float beta,
    float** dy_array, magma_int_t yi, magma_int_t incy,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magmablas_sgemv_batched_strided_smallsq(
    magma_trans_t transA, magma_int_t n,
    const float alpha,
    const float* dA, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magma_int_t strideA,
    const float* dx, magma_int_t xi, magma_int_t incx, magma_int_t stridex,
    const float beta,
    float* dy, magma_int_t yi, magma_int_t incy, magma_int_t stridey,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgeqrf_batched_smallsq(
    magma_int_t n,
    float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    float **dtau_array, magma_int_t taui, magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sgeqrf_batched(
    magma_int_t m, magma_int_t n,
    float **dA_array,
    magma_int_t lda,
    float **dtau_array,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgeqrf_batched_work(
    magma_int_t m, magma_int_t n,
    float **dA_array, magma_int_t ldda,
    float **dtau_array,
    magma_int_t *info_array,
    void* device_work, magma_int_t* device_lwork,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgeqrf_expert_batched(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    float **dA_array, magma_int_t ldda,
    float **dR_array, magma_int_t lddr,
    float **dT_array, magma_int_t lddt,
    float **dtau_array, magma_int_t provide_RT,
    float **dW_array,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgeqrf_batched_v4(
    magma_int_t m, magma_int_t n,
    float **dA_array,
    magma_int_t lda,
    float **tau_array,
    magma_int_t *info_array,
    magma_int_t batchCount);

magma_int_t
magma_sgeqrf_panel_fused_update_batched(
        magma_int_t m, magma_int_t n, magma_int_t nb,
        float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        float** tau_array, magma_int_t taui,
        float** dR_array, magma_int_t Ri, magma_int_t Rj, magma_int_t lddr,
        magma_int_t *info_array, magma_int_t separate_R_V,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgeqrf_panel_internal_batched(
        magma_int_t m, magma_int_t n, magma_int_t nb,
        float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        float** tau_array, magma_int_t taui,
        float** dT_array, magma_int_t Ti, magma_int_t Tj, magma_int_t lddt,
        float** dR_array, magma_int_t Ri, magma_int_t Rj, magma_int_t lddr,
        float** dwork_array,
        magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgeqrf_panel_batched(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    float** dA_array,    magma_int_t ldda,
    float** tau_array,
    float** dT_array, magma_int_t ldt,
    float** dR_array, magma_int_t ldr,
    float** dwork_array,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgels_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    float **dA_array, magma_int_t ldda,
    float **dB_array, magma_int_t lddb,
    float *hwork, magma_int_t lwork,
    magma_int_t *info,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgeqr2_fused_reg_tall_batched(
    magma_int_t m, magma_int_t n,
    float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    float **dtau_array, magma_int_t taui,
    magma_int_t* info_array, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sgeqr2_fused_reg_medium_batched(
    magma_int_t m, magma_int_t n,
    float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    float **dtau_array, magma_int_t taui,
    magma_int_t* info_array, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sgeqr2_fused_reg_batched(
    magma_int_t m, magma_int_t n,
    float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    float **dtau_array, magma_int_t taui,
    magma_int_t* info_array, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sgeqr2_fused_sm_batched(
    magma_int_t m, magma_int_t n,
    float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    float **dtau_array, magma_int_t taui,
    magma_int_t* info_array, magma_int_t nthreads, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sgeqr2_batched(
    magma_int_t m, magma_int_t n,
    float **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    float **dtau_array, magma_int_t taui,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sorm2r_reg_tall_batched(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t ib,
    float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    float** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv,
    float **dtau_array, magma_int_t taui,
    magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sorm2r_reg_medium_batched(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t ib,
    float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    float** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv,
    float **dtau_array, magma_int_t taui,
    magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sorm2r_reg_batched(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t k,
    float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    float** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv,
    float **dtau_array, magma_int_t taui,
    magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sorm2r_sm_batched(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t k,
    float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    float** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv,
    float **dtau_array, magma_int_t taui,
    magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_slarfb_gemm_internal_batched(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_const_ptr dV_array[],    magma_int_t vi, magma_int_t vj, magma_int_t lddv,
    magmaFloat_const_ptr dT_array[],    magma_int_t Ti, magma_int_t Tj, magma_int_t lddt,
    magmaFloat_ptr dC_array[],          magma_int_t Ci, magma_int_t Cj, magma_int_t lddc,
    magmaFloat_ptr dwork_array[],       magma_int_t ldwork,
    magmaFloat_ptr dworkvt_array[],     magma_int_t ldworkvt,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_slarfb_gemm_batched(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_const_ptr dV_array[],    magma_int_t lddv,
    magmaFloat_const_ptr dT_array[],    magma_int_t lddt,
    magmaFloat_ptr dC_array[],          magma_int_t lddc,
    magmaFloat_ptr dwork_array[],       magma_int_t ldwork,
    magmaFloat_ptr dworkvt_array[],     magma_int_t ldworkvt,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_slarft_internal_batched(
        magma_int_t n, magma_int_t k, magma_int_t stair_T,
        float **v_array,   magma_int_t vi, magma_int_t vj, magma_int_t ldv,
        float **tau_array, magma_int_t taui,
        float **T_array,   magma_int_t Ti, magma_int_t Tj, magma_int_t ldt,
        float **work_array, magma_int_t lwork,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_slarft_batched(
    magma_int_t n, magma_int_t k, magma_int_t stair_T,
    float **v_array, magma_int_t ldv,
    float **tau_array,
    float **T_array, magma_int_t ldt,
    float **work_array, magma_int_t lwork,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_slarft_sm32x32_batched(
    magma_int_t n, magma_int_t k,
    float **v_array, magma_int_t vi, magma_int_t vj, magma_int_t ldv,
    float **tau_array, magma_int_t taui,
    float **T_array, magma_int_t Ti, magma_int_t Tj, magma_int_t ldt,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_slarft_recstrmv_sm32x32(
    magma_int_t m, magma_int_t n,
    float *tau,
    float *Trec, magma_int_t ldtrec,
    float *Ttri, magma_int_t ldttri,
    magma_queue_t queue);

void magmablas_slarft_recstrmv_sm32x32_batched(
    magma_int_t m, magma_int_t n,
    float **tau_array,  magma_int_t taui,
    float **Trec_array, magma_int_t Treci, magma_int_t Trecj, magma_int_t ldtrec,
    float **Ttri_array, magma_int_t Ttrii, magma_int_t Ttrij, magma_int_t ldttri,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_slarft_strmv_sm32x32(
    magma_int_t m, magma_int_t n,
    float *tau,
    float *Tin, magma_int_t ldtin,
    float *Tout, magma_int_t ldtout,
    magma_queue_t queue);

void magmablas_slarft_strmv_sm32x32_batched(
    magma_int_t m, magma_int_t n,
    float **tau_array, magma_int_t taui,
    float **Tin_array, magma_int_t Tini, magma_int_t Tinj, magma_int_t ldtin,
    float **Tout_array, magma_int_t Touti, magma_int_t Toutj, magma_int_t ldtout,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_snrm2_cols_batched(
    magma_int_t m, magma_int_t n,
    float **dA_array, magma_int_t lda,
    float **dxnorm_array,
    magma_int_t batchCount);

void
magma_slarfgx_batched(
    magma_int_t n, float **dx0_array, float **dx_array,
    float **dtau_array, float **dxnorm_array,
    float **dR_array, magma_int_t it,
    magma_int_t batchCount);

void
magma_slarfx_batched_v4(
    magma_int_t m, magma_int_t n,
    float **v_array,
    float **tau_array,
    float **C_array, magma_int_t ldc, float **xnorm_array,
    magma_int_t step,
    magma_int_t batchCount);

void
magmablas_slarfg_batched(
    magma_int_t n,
    float** dalpha_array,
    float** dx_array, magma_int_t incx,
    float** dtau_array,
    magma_int_t batchCount );

magma_int_t
magma_spotrf_lpout_batched(
    magma_uplo_t uplo, magma_int_t n,
    float **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda, magma_int_t gbstep,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_spotrf_lpin_batched(
    magma_uplo_t uplo, magma_int_t n,
    float **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda, magma_int_t gbstep,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

// host interface
void
blas_slacpy_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float const * const * hA_array, magma_int_t lda,
    float               **hB_array, magma_int_t ldb,
    magma_int_t batchCount );

void
blas_sgemm_batched(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * hA_array, magma_int_t lda,
    float const * const * hB_array, magma_int_t ldb,
    float beta,
    float **hC_array, magma_int_t ldc,
    magma_int_t batchCount );

void
blas_strsm_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        float alpha,
        float **hA_array, magma_int_t lda,
        float **hB_array, magma_int_t ldb,
        magma_int_t batchCount );

void
blas_strmm_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        float alpha,
        float **hA_array, magma_int_t lda,
        float **hB_array, magma_int_t ldb,
        magma_int_t batchCount );

void
blas_ssymm_batched(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t m, magma_int_t n,
        float alpha,
        float **hA_array, magma_int_t lda,
        float **hB_array, magma_int_t ldb,
        float beta,
        float **hC_array, magma_int_t ldc,
        magma_int_t batchCount );

void
blas_ssyrk_batched(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha, float const * const * hA_array, magma_int_t lda,
    float beta,  float               **hC_array, magma_int_t ldc,
    magma_int_t batchCount );

void
blas_ssyr2k_batched(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha, float const * const * hA_array, magma_int_t lda,
                              float const * const * hB_array, magma_int_t ldb,
    float beta,              float               **hC_array, magma_int_t ldc,
    magma_int_t batchCount );

magma_int_t
magma_sormqr_batched(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float **dA_array, magma_int_t ldda,
    float **dtau_array,
    float **dC_array, magma_int_t lddc,
    void *device_work, int64_t *lwork_device,
    magma_int_t *dinfo_array, magma_int_t batchCount,
    magma_queue_t queue);
// for debugging purpose
void
sset_stepinit_ipiv(
    magma_int_t **ipiv_array,
    magma_int_t pm,
    magma_int_t batchCount);

#ifdef __cplusplus
}
#endif

#undef MAGMA_REAL

#endif  /* MAGMA_SBATCHED_H */
