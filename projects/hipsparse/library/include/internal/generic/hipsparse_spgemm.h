/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#ifndef HIPSPARSE_SPGEMM_H
#define HIPSPARSE_SPGEMM_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \details
*  \p hipsparseSpGEMM_createDescr creates a sparse matrix sparse matrix product descriptor. It should be
*  destroyed at the end using \ref hipsparseSpGEMM_destroyDescr().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_createDescr(hipsparseSpGEMMDescr_t* descr);
#endif

/*! \ingroup generic_module
*  \details
*  \p hipsparseSpGEMM_destroyDescr destroys a sparse matrix sparse matrix product descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_destroyDescr(hipsparseSpGEMMDescr_t descr);
#endif

/*! \ingroup generic_module
*  \brief Work estimation step of the sparse matrix sparse matrix product:
*  \f[
*    C' := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  where \f$C'\f$, \f$A\f$, \f$B\f$, \f$C\f$ are sparse matrices and \f$C'\f$ and \f$C\f$ have the same sparsity pattern.
*
*  \details
*  \p hipsparseSpGEMM_workEstimation is called twice. We call it to compute the size of the first required user allocated
*  buffer. After this buffer size is determined, the user allocates it and calls \p hipsparseSpGEMM_workEstimation
*  a second time with the newly allocated buffer passed in. This second call inspects the matrices \f$A\f$ and \f$B\f$ to 
*  determine the number of intermediate products that will result from multipltying \f$A\f$ and \f$B\f$ together.
*
*  \p hipsparseSpGEMM_workEstimation supports multiple combinations of data types and compute types. See \ref hipsparseSpGEMM_copy 
*  for a complete listing of all the data type and compute type combinations available.
*  
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  opA              sparse matrix \f$A\f$ operation type.
*  @param[in]
*  opB              sparse matrix \f$B\f$ operation type.
*  @param[in]
*  alpha            scalar \f$\alpha\f$.
*  @param[in]
*  matA             sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  matB             sparse matrix \f$B\f$ descriptor.
*  @param[in]
*  beta             scalar \f$\beta\f$.
*  @param[out]
*  matC             sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  computeType      floating point precision for the SpGEMM computation.
*  @param[in]
*  alg              SpGEMM algorithm for the SpGEMM computation.
*  @param[in]
*  spgemmDescr      SpGEMM descriptor.
*  @param[out]
*  bufferSize1      number of bytes of the temporary storage buffer. 
*  @param[in]
*  externalBuffer1  temporary storage buffer allocated by the user.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p beta, \p matA, \p matB, \p matC 
*                                         or \p bufferSize1 pointer is invalid.
*  \retval HIPSPARSE_STATUS_ALLOC_FAILED additional buffer for long rows could not be
*          allocated.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \p opA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE or
*          \p opB != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE.
*
*  \par Example (See full example below)
*  \code{.c}
*    void*  dBuffer1  = NULL; 
*    size_t bufferSize1 = 0;
*
*    hipsparseSpGEMMDescr_t spgemmDesc;
*    hipsparseSpGEMM_createDescr(&spgemmDesc);
*
*    size_t bufferSize1 = 0;
*    hipsparseSpGEMM_workEstimation(handle, opA, opB,
*                                  &alpha, matA, matB, &beta, matC,
*                                  computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                  spgemmDesc, &bufferSize1, NULL);
*    hipMalloc((void**) &dBuffer1, bufferSize1);
*
*    // Determine number of intermediate product when computing A * B
*    hipsparseSpGEMM_workEstimation(handle, opA, opB,
*                                    &alpha, matA, matB, &beta, matC,
*                                    computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                    spgemmDesc, &bufferSize1, dBuffer1);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_workEstimation(hipsparseHandle_t          handle,
                                                 hipsparseOperation_t       opA,
                                                 hipsparseOperation_t       opB,
                                                 const void*                alpha,
                                                 hipsparseConstSpMatDescr_t matA,
                                                 hipsparseConstSpMatDescr_t matB,
                                                 const void*                beta,
                                                 hipsparseSpMatDescr_t      matC,
                                                 hipDataType                computeType,
                                                 hipsparseSpGEMMAlg_t       alg,
                                                 hipsparseSpGEMMDescr_t     spgemmDescr,
                                                 size_t*                    bufferSize1,
                                                 void*                      externalBuffer1);
#elif(CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_workEstimation(hipsparseHandle_t      handle,
                                                 hipsparseOperation_t   opA,
                                                 hipsparseOperation_t   opB,
                                                 const void*            alpha,
                                                 hipsparseSpMatDescr_t  matA,
                                                 hipsparseSpMatDescr_t  matB,
                                                 const void*            beta,
                                                 hipsparseSpMatDescr_t  matC,
                                                 hipDataType            computeType,
                                                 hipsparseSpGEMMAlg_t   alg,
                                                 hipsparseSpGEMMDescr_t spgemmDescr,
                                                 size_t*                bufferSize1,
                                                 void*                  externalBuffer1);
#endif

/*! \ingroup generic_module
*  \brief Description: Estimate memory step of the sparse matrix sparse matrix product C' = alpha * A * B + beta * C 
*  where C', A, B, C are sparse matrices and C' and C have the same sparsity pattern.
*
*  \details
*  When using HIPSPARSE_SPGEMM_ALG2 or HIPSPARSE_SPGEMM_ALG3, \p hipsparseSpGEMM_estimateMemory is called twice. 
*  First to determine the size of the third temporary user allocated buffer. Once this has been determined, the buffer
*  is allocated and \p hipsparseSpGEMM_estimateMemory called a second time. The second time it determines the size of the 
*  second temporary user allocated buffer. Once this second buffer is allocated we call \p hipsparseSpGEMM_compute to perform
*  the actual computation of C' = alpha * A * B (the result is stored in the temporary buffers). 
* 
*  \note \p hipsparseSpGEMM_estimateMemory is only used with HIPSPARSE_SPGEMM_ALG2 and HIPSPARSE_SPGEMM_ALG3 and it replaces 
*  the first call to \p hipsparseSpGEMM_compute.
*
*  \par Example using HIPSPARSE_SPGEMM_ALG2(3) (See full example below)
*  \code{.c}
*    hipsparseSpGEMMAlg_t alg = HIPSPARSE_SPGEMM_ALG2;
*    float chunk_fraction = 0.2f;
*    void*  dBuffer2  = NULL; 
*    void*  dBuffer3  = NULL; 
* 
*    size_t bufferSize2 = 0;
*    size_t bufferSize3 = 0;
*
*    // Determine size of dBuffer3
*    hipsparseSpGEMM_estimateMemory(handle, opA, opB,
*                            &alpha, matA, matB, &beta, matC,
*                            computeType, alg,
*                            spgemmDesc, chunk_fraction, &bufferSize3, NULL, NULL);
*    hipMalloc((void**) &dBuffer3, bufferSize3);
*
*    // Determine size of dBuffer2
*    hipsparseSpGEMM_estimateMemory(handle, opA, opB,
*                            &alpha, matA, matB, &beta, matC,
*                            computeType, alg,
*                            spgemmDesc, chunk_fraction, &bufferSize3, dBuffer3, &bufferSize2);
*
*    // We can now free dBuffer3 to save memory
*    hipFree(dBuffer3);
*
*    // Allocate second buffer
*    hipMalloc((void**) &dBuffer2, bufferSize2);
*
*    // compute the intermediate product of A * B
*    hipsparseSpGEMM_compute(handle, opA, opB,
*                            &alpha, matA, matB, &beta, matC,
*                            computeType, alg,
*                            spgemmDesc, &bufferSize2, dBuffer2);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12001)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_estimateMemory(hipsparseHandle_t          handle,
                                                 hipsparseOperation_t       opA,
                                                 hipsparseOperation_t       opB,
                                                 const void*                alpha,
                                                 hipsparseConstSpMatDescr_t matA,
                                                 hipsparseConstSpMatDescr_t matB,
                                                 const void*                beta,
                                                 hipsparseSpMatDescr_t      matC,
                                                 hipDataType                computeType,
                                                 hipsparseSpGEMMAlg_t       alg,
                                                 hipsparseSpGEMMDescr_t     spgemmDescr,
                                                 float                      chunk_fraction,
                                                 size_t*                    bufferSize3,
                                                 void*                      externalBuffer3,
                                                 size_t*                    bufferSize2);
#endif

/*! \ingroup generic_module
*  \brief Description: Compute step of the sparse matrix sparse matrix product C' = alpha * A * B + beta * C 
*  where C', A, B, C are sparse matrices and C' and C have the same sparsity pattern.
*
*  \details
*  When using HIPSPARSE_SPGEMM_ALG1 (or HIPSPARSE_SPGEMM_DEFAULT), \p hipsparseSpGEMM_compute is called twice. 
*  First to compute the size of the second required user allocated buffer. After this buffer size is determined, 
*  the user allocates it and calls \p hipsparseSpGEMM_compute a second time with the newly allocated buffer passed 
*  in. This second call performs the actual computation of C' = alpha * A * B (the result is stored in the temporary 
*  buffers). 
*
*  \note \p hipsparseSpGEMM_compute is only used to determine the size of the second buffer when using HIPSPARSE_SPGEMM_ALG1
*  (or HIPSPARSE_SPGEMM_DEFAULT). When using HIPSPARSE_SPGEMM_ALG2 and HIPSPARSE_SPGEMM_ALG3, 
*  \p hipsparseSpGEMM_estimateMemory must be used instead when determining the size of the second buffer. 
*  
*  \par Example using HIPSPARSE_SPGEMM_ALG1 (See full example below)
*  \code{.c}
*    hipsparseSpGEMMAlg_t alg = HIPSPARSE_SPGEMM_ALG1;
*    void*  dBuffer2  = NULL; 
*    size_t bufferSize2 = 0;
*
*    size_t bufferSize2 = 0;
*    hipsparseSpGEMM_compute(handle, opA, opB,
*                            &alpha, matA, matB, &beta, matC,
*                            computeType, alg,
*                            spgemmDesc, &bufferSize2, NULL);
*    hipMalloc((void**) &dBuffer2, bufferSize2);
*
*    // compute the intermediate product of A * B
*    hipsparseSpGEMM_compute(handle, opA, opB,
*                            &alpha, matA, matB, &beta, matC,
*                            computeType, alg,
*                            spgemmDesc, &bufferSize2, dBuffer2);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_compute(hipsparseHandle_t          handle,
                                          hipsparseOperation_t       opA,
                                          hipsparseOperation_t       opB,
                                          const void*                alpha,
                                          hipsparseConstSpMatDescr_t matA,
                                          hipsparseConstSpMatDescr_t matB,
                                          const void*                beta,
                                          hipsparseSpMatDescr_t      matC,
                                          hipDataType                computeType,
                                          hipsparseSpGEMMAlg_t       alg,
                                          hipsparseSpGEMMDescr_t     spgemmDescr,
                                          size_t*                    bufferSize2,
                                          void*                      externalBuffer2);
#elif(CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_compute(hipsparseHandle_t      handle,
                                          hipsparseOperation_t   opA,
                                          hipsparseOperation_t   opB,
                                          const void*            alpha,
                                          hipsparseSpMatDescr_t  matA,
                                          hipsparseSpMatDescr_t  matB,
                                          const void*            beta,
                                          hipsparseSpMatDescr_t  matC,
                                          hipDataType            computeType,
                                          hipsparseSpGEMMAlg_t   alg,
                                          hipsparseSpGEMMDescr_t spgemmDescr,
                                          size_t*                bufferSize2,
                                          void*                  externalBuffer2);
#endif
/*! \ingroup generic_module
*  \brief Description: Copy step of the sparse matrix sparse matrix product C' = alpha * A * B + beta * C 
*  where C', A, B, C are sparse matrices and C' and C have the same sparsity pattern.
*
*  \details
*  \p hipsparseSpGEMM_copy is called once to copy the results (that are currently stored in the temporary arrays) 
*  to the output sparse matrix. If beta != 0, then the beta * C portion of the computation: C' = alpha * A * B + beta * C
*  is handled. This is possible because C' and C must have the same sparsity pattern.
*
*  \note The two user allocated temporary buffers can only be freed after the call to \p hipsparseSpGEMM_copy
*  
*  \par Example using HIPSPARSE_SPGEMM_ALG1 (Full example)
*  \code{.c}
*    hipsparseHandle_t     handle = NULL;
*    hipsparseSpMatDescr_t matA, matB, matC;
*    void*  dBuffer1  = NULL; 
*    void*  dBuffer2  = NULL;
*    size_t bufferSize1 = 0;  
*    size_t bufferSize2 = 0;
*
*    hipsparseCreate(&handle);
*
*    // Create sparse matrix A in CSR format
*    hipsparseCreateCsr(&matA, m, k, nnzA,
*                                        dcsr_row_ptrA, dcsr_col_indA, dcsr_valA,
*                                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*    hipsparseCreateCsr(&matB, k, n, nnzB,
*                                        dcsr_row_ptrB, dcsr_col_indB, dcsr_valB,
*                                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*    hipsparseCreateCsr(&matC, m, n, 0,
*                                        dcsr_row_ptrC, NULL, NULL,
*                                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*
*    hipsparseSpGEMMDescr_t spgemmDesc;
*    hipsparseSpGEMM_createDescr(&spgemmDesc);
*
*    // Determine size of first user allocated buffer
*    hipsparseSpGEMM_workEstimation(handle, opA, opB,
*                                        &alpha, matA, matB, &beta, matC,
*                                        computeType, HIPSPARSE_SPGEMM_ALG1,
*                                        spgemmDesc, &bufferSize1, NULL);
*    hipMalloc((void**) &dBuffer1, bufferSize1);
*
*    // Inspect the matrices A and B to determine the number of intermediate product in 
*    // C = alpha * A * B
*    hipsparseSpGEMM_workEstimation(handle, opA, opB,
*                                        &alpha, matA, matB, &beta, matC,
*                                        computeType, HIPSPARSE_SPGEMM_ALG1,
*                                        spgemmDesc, &bufferSize1, dBuffer1);
*
*    // Determine size of second user allocated buffer
*    hipsparseSpGEMM_compute(handle, opA, opB,
*                                &alpha, matA, matB, &beta, matC,
*                                computeType, HIPSPARSE_SPGEMM_ALG1,
*                                spgemmDesc, &bufferSize2, NULL);
*    hipMalloc((void**) &dBuffer2, bufferSize2);
*
*    // Compute C = alpha * A * B and store result in temporary buffers
*    hipsparseSpGEMM_compute(handle, opA, opB,
*                                        &alpha, matA, matB, &beta, matC,
*                                        computeType, HIPSPARSE_SPGEMM_ALG1,
*                                        spgemmDesc, &bufferSize2, dBuffer2);
*
*    // Get matrix C non-zero entries C_nnz1
*    int64_t C_num_rows1, C_num_cols1, C_nnz1;
*    hipsparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1);
*
*    // Allocate the CSR structures for the matrix C
*    hipMalloc((void**) &dcsr_col_indC, C_nnz1 * sizeof(int));
*    hipMalloc((void**) &dcsr_valC,  C_nnz1 * sizeof(float));
*
*    // Update matC with the new pointers
*    hipsparseCsrSetPointers(matC, dcsr_row_ptrC, dcsr_col_indC, dcsr_valC);
*
*    // Copy the final products to the matrix C
*    hipsparseSpGEMM_copy(handle, opA, opB,
*                            &alpha, matA, matB, &beta, matC,
*                            computeType, HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc);
*
*    // Destroy matrix descriptors and handles
*    hipsparseSpGEMM_destroyDescr(spgemmDesc);
*    hipsparseDestroySpMat(matA);
*    hipsparseDestroySpMat(matB);
*    hipsparseDestroySpMat(matC);
*    hipsparseDestroy(handle);
* 
*    // Free device memory
*    hipFree(dBuffer1);
*    hipFree(dBuffer2);
*  \endcode
*
*  \par Example using HIPSPARSE_SPGEMM_ALG2 (Full example)
*  \code{.c}
*    hipsparseHandle_t     handle = NULL;
*    hipsparseSpMatDescr_t matA, matB, matC;
*    void*  dBuffer1  = NULL; 
*    void*  dBuffer2  = NULL;
*    void*  dBuffer3  = NULL;
*    size_t bufferSize1 = 0;  
*    size_t bufferSize2 = 0;
*    size_t bufferSize3 = 0;
*
*    hipsparseCreate(&handle);
*
*    // Create sparse matrix A in CSR format
*    hipsparseCreateCsr(&matA, m, k, nnzA,
*                                        dcsr_row_ptrA, dcsr_col_indA, dcsr_valA,
*                                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*    hipsparseCreateCsr(&matB, k, n, nnzB,
*                                        dcsr_row_ptrB, dcsr_col_indB, dcsr_valB,
*                                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*    hipsparseCreateCsr(&matC, m, n, 0,
*                                        dcsr_row_ptrC, NULL, NULL,
*                                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*
*    hipsparseSpGEMMDescr_t spgemmDesc;
*    hipsparseSpGEMM_createDescr(&spgemmDesc);
*
*    // Determine size of first user allocated buffer
*    hipsparseSpGEMM_workEstimation(handle, opA, opB,
*                                        &alpha, matA, matB, &beta, matC,
*                                        computeType, HIPSPARSE_SPGEMM_ALG2,
*                                        spgemmDesc, &bufferSize1, NULL);
*    hipMalloc((void**) &dBuffer1, bufferSize1);
*
*    // Inspect the matrices A and B to determine the number of intermediate product in 
*    // C = alpha * A * B
*    hipsparseSpGEMM_workEstimation(handle, opA, opB,
*                                        &alpha, matA, matB, &beta, matC,
*                                        computeType, HIPSPARSE_SPGEMM_ALG2,
*                                        spgemmDesc, &bufferSize1, dBuffer1);
*
*    // Determine size of second user allocated buffer
*    float chunk_fraction = 0.2f;
*    hipsparseSpGEMM_estimateMemory(handle, opA, opB,
*                                &alpha, matA, matB, &beta, matC,
*                                computeType, HIPSPARSE_SPGEMM_ALG2,
*                                spgemmDesc, chunk_fraction, &bufferSize3, NULL, NULL);
*    hipMalloc((void**) &dBuffer3, bufferSize3);
*    hipsparseSpGEMM_estimateMemory(handle, opA, opB,
*                                &alpha, matA, matB, &beta, matC,
*                                computeType, HIPSPARSE_SPGEMM_ALG2,
*                                spgemmDesc, chunk_fraction, &bufferSize3, dBuffer3, &bufferSize2);
*    hipFree(dBuffer3);
*
*    // Allocate second user allocated buffer
*    hipMalloc((void**) &dBuffer2, bufferSize2);
*
*    // Compute C = alpha * A * B and store result in temporary buffers
*    hipsparseSpGEMM_compute(handle, opA, opB,
*                                        &alpha, matA, matB, &beta, matC,
*                                        computeType, HIPSPARSE_SPGEMM_ALG2,
*                                        spgemmDesc, &bufferSize2, dBuffer2);
*
*    // Get matrix C non-zero entries C_nnz1
*    int64_t C_num_rows1, C_num_cols1, C_nnz1;
*    hipsparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1);
*
*    // Allocate the CSR structures for the matrix C
*    hipMalloc((void**) &dcsr_col_indC, C_nnz1 * sizeof(int));
*    hipMalloc((void**) &dcsr_valC,  C_nnz1 * sizeof(float));
*
*    // Update matC with the new pointers
*    hipsparseCsrSetPointers(matC, dcsr_row_ptrC, dcsr_col_indC, dcsr_valC);
*
*    // Copy the final products to the matrix C
*    hipsparseSpGEMM_copy(handle, opA, opB,
*                            &alpha, matA, matB, &beta, matC,
*                            computeType, HIPSPARSE_SPGEMM_ALG2, spgemmDesc);
*
*    // Destroy matrix descriptors and handles
*    hipsparseSpGEMM_destroyDescr(spgemmDesc);
*    hipsparseDestroySpMat(matA);
*    hipsparseDestroySpMat(matB);
*    hipsparseDestroySpMat(matC);
*    hipsparseDestroy(handle);
*
*    // Free device memory
*    hipFree(dBuffer1);
*    hipFree(dBuffer2);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_copy(hipsparseHandle_t          handle,
                                       hipsparseOperation_t       opA,
                                       hipsparseOperation_t       opB,
                                       const void*                alpha,
                                       hipsparseConstSpMatDescr_t matA,
                                       hipsparseConstSpMatDescr_t matB,
                                       const void*                beta,
                                       hipsparseSpMatDescr_t      matC,
                                       hipDataType                computeType,
                                       hipsparseSpGEMMAlg_t       alg,
                                       hipsparseSpGEMMDescr_t     spgemmDescr);
#elif(CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_copy(hipsparseHandle_t      handle,
                                       hipsparseOperation_t   opA,
                                       hipsparseOperation_t   opB,
                                       const void*            alpha,
                                       hipsparseSpMatDescr_t  matA,
                                       hipsparseSpMatDescr_t  matB,
                                       const void*            beta,
                                       hipsparseSpMatDescr_t  matC,
                                       hipDataType            computeType,
                                       hipsparseSpGEMMAlg_t   alg,
                                       hipsparseSpGEMMDescr_t spgemmDescr);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_SPGEMM_H */
