/* **************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

#include <iomanip>
#include <iostream>

// Helper for conjugate that works for both real and complex types
template <typename T>
inline T conj_helper(const T& x)
{
    if constexpr(rocblas_is_complex<T>)
        return std::conj(x);
    else
        return x;
}

template <bool STRIDED, typename T, typename I, typename S, typename U, typename V>
void cholqr_checkBadArgs(const rocblas_handle handle,
                         const I m,
                         const I n,
                         T dA,
                         const I lda,
                         const rocblas_stride stA,
                         U dR,
                         const I ldr,
                         const rocblas_stride stR,
                         S dSigma,
                         const rocsolver_cholqr_algo algo,
                         V dInfo,
                         const I bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, nullptr, m, n, dA, lda, stA, dR, ldr, stR,
                                           dSigma, algo, dInfo, bc),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, n, dA, lda, stA, dR, ldr, stR,
                                               dSigma, algo, dInfo, (I)-1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, n, (T) nullptr, lda, stA, dR, ldr,
                                           stR, dSigma, algo, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, n, dA, lda, stA, (U) nullptr, ldr,
                                           stR, dSigma, algo, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, n, dA, lda, stA, dR, ldr, stR,
                                           dSigma, algo, (V) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, (I)0, n, (T) nullptr, lda, stA,
                                           (U) nullptr, ldr, stR, dSigma, algo, dInfo, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, (I)0, (T) nullptr, lda, stA,
                                           (U) nullptr, ldr, stR, dSigma, algo, dInfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, n, dA, lda, stA, dR, ldr, stR,
                                               dSigma, algo, dInfo, (I)0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
void testing_cholqr_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    I m = 1;
    I n = 1;
    I lda = 1;
    I ldr = 1;
    rocblas_stride stA = 1;
    rocblas_stride stR = 1;
    I bc = 1;
    rocsolver_cholqr_algo algo = rocsolver_cholqr_default;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<T> dR(1, 1, stR, 1);
        device_strided_batch_vector<S> dSigma(1, 1, 1, 1);
        device_strided_batch_vector<I> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dR.memcheck());
        CHECK_HIP_ERROR(dSigma.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        cholqr_checkBadArgs<STRIDED>(handle, m, n, dA.data(), lda, stA, dR.data(), ldr, stR,
                                     dSigma.data(), algo, dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dR(1, 1, 1, 1);
        device_strided_batch_vector<S> dSigma(1, 1, 1, 1);
        device_strided_batch_vector<I> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dR.memcheck());
        CHECK_HIP_ERROR(dSigma.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        cholqr_checkBadArgs<STRIDED>(handle, m, n, dA.data(), lda, stA, dR.data(), ldr, stR,
                                     dSigma.data(), algo, dInfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename I, typename Td, typename Ud, typename Sd, typename Th, typename Uh, typename Sh>
void cholqr_initData(const rocblas_handle handle,
                     const I m,
                     const I n,
                     Td& dA,
                     const I lda,
                     const rocblas_stride stA,
                     Ud& dR,
                     const I ldr,
                     const rocblas_stride stR,
                     Sd& dSigma,
                     const rocsolver_cholqr_algo algo,
                     const I bc,
                     Th& hA,
                     Uh& hR,
                     Sh& hSigma)
{
    using S = decltype(std::real(T{}));

    if(CPU)
    {
        rocblas_init<T>(hA, true);

        for(I b = 0; b < bc; ++b)
        {
            for(I i = 0; i < m; i++)
            {
                for(I j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }
        }

        // Initialize sigma array based on algorithm mode
        // For cholqr3_user, we compute sigma using the same formula as cholqr3_compute:
        //   sigma = 11 * n * eps * (m + (n+1)) * ||A||_F^2
        // For other modes, sigma is either not used or computed internally
        if(algo == rocsolver_cholqr_cholqr3_user)
        {
            S eps = std::numeric_limits<S>::epsilon();
            for(I b = 0; b < bc; ++b)
            {
                // Compute ||A||_F^2 (Frobenius norm squared)
                S gnorm_sq = S(0);
                for(I j = 0; j < n; ++j)
                {
                    for(I i = 0; i < m; ++i)
                    {
                        T val = hA[b][i + j * lda];
                        gnorm_sq += std::real(val * conj_helper(val));
                    }
                }
                // sigma = 11 * n * eps * (m + (n+1)) * gnorm_sq
                hSigma[b][0] = S(11.0) * S(n) * eps * (S(m) + S(n + 1)) * gnorm_sq;
            }
        }
        else
        {
            // For other algorithms, sigma is not used or computed internally
            for(I b = 0; b < bc; ++b)
            {
                hSigma[b][0] = S(0.0);
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dSigma.transfer_from(hSigma));
    }
}

// Helper function to normalize QR signs: make R's diagonal positive
// For each k, if R[k,k] < 0 (or has negative real part for complex),
// flip signs of column k of Q and row k of R
template <typename T, typename I>
void normalize_qr_signs(T* Q, I m, I ldq, T* R, I n, I ldr, I min_mn)
{
    using S = decltype(std::real(T{}));
    
    for(I k = 0; k < min_mn; ++k)
    {
        // Check if diagonal of R is negative
        T diag = R[k + k * ldr];
        S real_diag = std::real(diag);
        
        if(real_diag < S(0))
        {
            // Flip sign of column k of Q
            for(I i = 0; i < m; ++i)
            {
                Q[i + k * ldq] = -Q[i + k * ldq];
            }
            
            // Flip sign of row k of R (only upper triangular part matters)
            for(I j = k; j < n; ++j)
            {
                R[k + j * ldr] = -R[k + j * ldr];
            }
        }
    }
}

// Helper function to print a matrix for debugging (first 10x10 elements)
template <typename T, typename I>
void print_matrix(const char* name, const T* A, I rows, I cols, I lda)
{
    const I max_print = 10;
    I print_rows = std::min(rows, max_print);
    I print_cols = std::min(cols, max_print);
    
    std::cout << "\n=== " << name << " (" << rows << " x " << cols << ", showing " 
              << print_rows << " x " << print_cols << ") ===" << std::endl;
    
    for(I i = 0; i < print_rows; ++i)
    {
        for(I j = 0; j < print_cols; ++j)
        {
            if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
                std::cout << std::setw(12) << std::fixed << std::setprecision(std::numeric_limits<T>::digits10) << A[i + j * lda] << " ";
            else
                std::cout << "(" << std::setw(9) << std::fixed << std::setprecision(std::numeric_limits<T>::digits10) << std::real(A[i + j * lda]) 
                          << "," << std::setw(9) << std::fixed << std::setprecision(std::numeric_limits<T>::digits10) << std::imag(A[i + j * lda]) << ") ";
        }
        if(cols > max_print)
            std::cout << "...";
        std::cout << std::endl;
    }
    if(rows > max_print)
        std::cout << "  ... (" << (rows - max_print) << " more rows)" << std::endl;
}

template <bool STRIDED, typename T, typename I, typename Td, typename Ud, typename Sd, typename Id, typename Th, typename Uh, typename Sh, typename Ih>
void cholqr_getError(const rocblas_handle handle,
                     const I m,
                     const I n,
                     Td& dA,
                     const I lda,
                     const rocblas_stride stA,
                     Ud& dR,
                     const I ldr,
                     const rocblas_stride stR,
                     Sd& dSigma,
                     const rocsolver_cholqr_algo algo,
                     Id& dInfo,
                     const I bc,
                     Th& hA,
                     Th& hARes,
                     Uh& hR,
                     Uh& hRRes,
                     Sh& hSigma,
                     Ih& hInfo,
                     double* max_err)
{
    using S = decltype(std::real(T{}));
    I min_mn = std::min(m, n);

    // Work arrays for CPU reference (GEQRF + ORGQR)
    I lwork = std::max(I(1), std::max(m, n) * 32);
    std::vector<T> hW(lwork);
    std::vector<T> hTau(min_mn);

    // input data initialization
    cholqr_initData<true, true, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo, bc, hA,
                                   hR, hSigma);

    for (I b = 0; b < bc; ++b)
    {   
        print_matrix("A", hA[b], m, n, lda);
    }

    // execute GPU cholqr
    CHECK_ROCBLAS_ERROR(rocsolver_cholqr(STRIDED, handle, m, n, dA.data(), lda, stA, dR.data(), ldr,
                                         stR, dSigma.data(), algo, dInfo.data(), bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA)); // Q_gpu is stored in A
    CHECK_HIP_ERROR(hRRes.transfer_from(dR)); // R_gpu
    CHECK_HIP_ERROR(hInfo.transfer_from(dInfo));

    // Check info for factorization success
    // (We expect the used input matrices to always succeed. The test matrices
    // are well-conditioned with large diagonal dominance.)
    *max_err = 0;
    for(I b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hInfo[b][0], 0) << "where b = " << b;
        if(hInfo[b][0] != 0)
            *max_err += 1;
    }

    // Compute Error Option 1: Check factorization quality
    // ||Q^T Q - I||_F (orthogonality) and ||A - QR||_F (reconstruction)
    double err = 0;
    for(I b = 0; b < bc; ++b)
    {
        // Only compute numerical error if factorization succeeded
        if(hInfo[b][0] == 0)
        {
            // Save a copy of original A before CPU reference computation overwrites it
            std::vector<T> A_orig(size_t(lda) * n);
            for(I j = 0; j < n; ++j)
            {
                for(I i = 0; i < m; ++i)
                {
                    A_orig[i + j * lda] = hA[b][i + j * lda];
                }
            }

            // CPU GEQRF: compute QR factorization (for comparison/debugging)
            // After geqrf, hA contains Householder vectors (below diagonal) and R (on and above diagonal)
            cpu_geqrf(m, n, hA[b], lda, hTau.data(), hW.data(), lwork);

            // Extract R_cpu from the upper triangular part of hA
            std::vector<T> R_cpu(size_t(ldr) * n, T(0));
            for(I j = 0; j < n; ++j)
            {
                for(I i = 0; i <= j && i < min_mn; ++i)
                {
                    R_cpu[i + j * ldr] = hA[b][i + j * lda];
                }
            }

            // Reconstruct Q_cpu from Householder vectors using ORGQR/UNGQR
            // cpu_orgqr_ungqr overwrites hA with Q
            cpu_orgqr_ungqr(m, min_mn, min_mn, hA[b], lda, hTau.data(), hW.data(), lwork);

            normalize_qr_signs(hA[b], m, lda, R_cpu.data(), n, ldr, min_mn);
            normalize_qr_signs(hARes[b], m, lda, hRRes[b], n, ldr, min_mn);

            // Debug: Print matrices for this batch (after sign normalization)
            std::cout << "\n========== BATCH " << b << " (signs normalized) ==========" << std::endl;
            print_matrix("Q_CPU", hA[b], m, min_mn, lda);
            print_matrix("Q_GPU", hARes[b], m, min_mn, lda);
            print_matrix("R_CPU", R_cpu.data(), n, n, ldr);
            print_matrix("R_GPU", hRRes[b], n, n, ldr);
            std::cout << "================================\n" << std::endl;

            // Compute ||A||_F for relative error
            S norm_A = snorm('F', m, n, A_orig.data(), lda);

            // Compute Q_gpu^H * Q_gpu using GEMM (should be identity for orthogonal Q)
            // QtQ = Q^H * Q, result is min_mn x min_mn
            std::vector<T> QtQ(size_t(min_mn) * min_mn, T(0));
            rocblas_operation transQ = rocblas_is_complex<T> ? rocblas_operation_conjugate_transpose 
                                                              : rocblas_operation_transpose;
            cpu_gemm(transQ, rocblas_operation_none, min_mn, min_mn, m, 
                     T(1), hARes[b], lda, hARes[b], lda, T(0), QtQ.data(), min_mn);

            // Subtract identity: QtQ - I
            for(I i = 0; i < min_mn; ++i)
                QtQ[i + i * min_mn] -= T(1);

            // Compute ||Q^H Q - I||_F
            S orth_err = snorm('F', min_mn, min_mn, QtQ.data(), min_mn);

            // Compute Q_gpu * R_gpu using GEMM (should equal original A)
            // QR = Q * R, result is m x n
            // Note: R may have garbage below diagonal from GPU, so extract only upper triangular
            std::vector<T> R_gpu_clean(size_t(ldr) * n, T(0));
            for(I j = 0; j < n; ++j)
                for(I i = 0; i <= j && i < min_mn; ++i)
                    R_gpu_clean[i + j * ldr] = hRRes[b][i + j * ldr];

            std::vector<T> QR(size_t(lda) * n, T(0));
            cpu_gemm(rocblas_operation_none, rocblas_operation_none, m, n, min_mn,
                     T(1), hARes[b], lda, R_gpu_clean.data(), ldr, T(0), QR.data(), lda);

            // Compute ||A - QR||_F
            // Note: Must iterate column by column to respect leading dimension lda
            S recon_err_sq = S(0);
            for(I j = 0; j < n; ++j)
            {
                for(I i = 0; i < m; ++i)
                {
                    T diff = A_orig[i + j * lda] - QR[i + j * lda];
                    recon_err_sq += std::real(diff * conj_helper(diff));
                }
            }
            S recon_err = std::sqrt(recon_err_sq);

            // ============================================================
            // Same checks for CPU GEQRF reference
            // ============================================================

            // Compute Q_cpu^H * Q_cpu using GEMM (should be identity)
            std::vector<T> QtQ_cpu(size_t(min_mn) * min_mn, T(0));
            cpu_gemm(transQ, rocblas_operation_none, min_mn, min_mn, m,
                     T(1), hA[b], lda, hA[b], lda, T(0), QtQ_cpu.data(), min_mn);

            // Subtract identity
            for(I i = 0; i < min_mn; ++i)
                QtQ_cpu[i + i * min_mn] -= T(1);

            S orth_err_cpu = snorm('F', min_mn, min_mn, QtQ_cpu.data(), min_mn);

            // Compute Q_cpu * R_cpu using GEMM
            // R_cpu is already properly upper triangular (zero below diagonal)
            std::vector<T> QR_cpu(size_t(lda) * n, T(0));
            cpu_gemm(rocblas_operation_none, rocblas_operation_none, m, n, min_mn,
                     T(1), hA[b], lda, R_cpu.data(), ldr, T(0), QR_cpu.data(), lda);

            // Compute ||A - Q_cpu R_cpu||_F
            S recon_err_cpu_sq = S(0);
            for(I j = 0; j < n; ++j)
            {
                for(I i = 0; i < m; ++i)
                {
                    T diff = A_orig[i + j * lda] - QR_cpu[i + j * lda];
                    recon_err_cpu_sq += std::real(diff * conj_helper(diff));
                }
            }
            S recon_err_cpu = std::sqrt(recon_err_cpu_sq);

            // ============================================================
            // GPU GEQRF reference (for comparison)
            // ============================================================
            S orth_err_gpu_geqrf = S(0);
            S recon_err_gpu_geqrf = S(0);
            {
                // Allocate device memory for GPU GEQRF
                size_t size_A_bytes = size_t(lda) * n * sizeof(T);
                size_t size_tau_bytes = size_t(min_mn) * sizeof(T);
                T* dA_geqrf = nullptr;
                T* dTau = nullptr;
                CHECK_HIP_ERROR(hipMalloc(&dA_geqrf, size_A_bytes));
                CHECK_HIP_ERROR(hipMalloc(&dTau, size_tau_bytes));

                // Copy original A to device
                CHECK_HIP_ERROR(hipMemcpy(dA_geqrf, A_orig.data(), size_A_bytes, hipMemcpyHostToDevice));

                // Run GPU GEQRF (non-strided, single matrix)
                CHECK_ROCBLAS_ERROR(rocsolver_geqr2_geqrf(false, true, handle, m, n, dA_geqrf, lda, rocblas_stride(0), dTau, rocblas_stride(0), I(1)));

                // Copy results back to extract R
                std::vector<T> hA_geqrf(size_t(lda) * n);
                std::vector<T> hTau_geqrf(min_mn);
                CHECK_HIP_ERROR(hipMemcpy(hA_geqrf.data(), dA_geqrf, size_A_bytes, hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(hipMemcpy(hTau_geqrf.data(), dTau, size_tau_bytes, hipMemcpyDeviceToHost));

                // Extract R from upper triangular part
                std::vector<T> R_gpu_geqrf(size_t(ldr) * n, T(0));
                for(I j = 0; j < n; ++j)
                {
                    for(I i = 0; i <= j && i < min_mn; ++i)
                    {
                        R_gpu_geqrf[i + j * ldr] = hA_geqrf[i + j * lda];
                    }
                }

                // Run GPU ORGQR/UNGQR to reconstruct Q
                CHECK_ROCBLAS_ERROR(rocsolver_orgxr_ungxr(true, handle, m, min_mn, min_mn, dA_geqrf, lda, dTau));

                // Copy Q back to host
                std::vector<T> Q_gpu_geqrf(size_t(lda) * n);
                CHECK_HIP_ERROR(hipMemcpy(Q_gpu_geqrf.data(), dA_geqrf, size_A_bytes, hipMemcpyDeviceToHost));

                // Free device memory
                CHECK_HIP_ERROR(hipFree(dA_geqrf));
                CHECK_HIP_ERROR(hipFree(dTau));

                // Compute Q^H * Q for GPU GEQRF using GEMM
                std::vector<T> QtQ_gpu_geqrf(size_t(min_mn) * min_mn, T(0));
                cpu_gemm(transQ, rocblas_operation_none, min_mn, min_mn, m,
                         T(1), Q_gpu_geqrf.data(), lda, Q_gpu_geqrf.data(), lda, T(0), QtQ_gpu_geqrf.data(), min_mn);

                // Subtract identity
                for(I i = 0; i < min_mn; ++i)
                    QtQ_gpu_geqrf[i + i * min_mn] -= T(1);

                orth_err_gpu_geqrf = snorm('F', min_mn, min_mn, QtQ_gpu_geqrf.data(), min_mn);

                // Compute Q * R for GPU GEQRF using GEMM
                // R_gpu_geqrf is already properly upper triangular (extracted above)
                std::vector<T> QR_gpu_geqrf(size_t(lda) * n, T(0));
                cpu_gemm(rocblas_operation_none, rocblas_operation_none, m, n, min_mn,
                         T(1), Q_gpu_geqrf.data(), lda, R_gpu_geqrf.data(), ldr, T(0), QR_gpu_geqrf.data(), lda);

                // Compute ||A - QR||_F for GPU GEQRF
                S recon_err_gpu_geqrf_sq = S(0);
                for(I j = 0; j < n; ++j)
                {
                    for(I i = 0; i < m; ++i)
                    {
                        T diff = A_orig[i + j * lda] - QR_gpu_geqrf[i + j * lda];
                        recon_err_gpu_geqrf_sq += std::real(diff * conj_helper(diff));
                    }
                }
                recon_err_gpu_geqrf = std::sqrt(recon_err_gpu_geqrf_sq);
            }

            // Print results for CPU GEQRF, GPU GEQRF, and GPU CHOLQR
            std::cout << "  CPU GEQRF:" << std::endl;
            std::cout << "    Orthogonality error ||Q^T Q - I||_F = " << orth_err_cpu << std::endl;
            std::cout << "    Reconstruction error ||A - QR||_F = " << recon_err_cpu << std::endl;
            std::cout << "    Relative reconstruction error = " << (norm_A > 0 ? recon_err_cpu / norm_A : recon_err_cpu) << std::endl;

            std::cout << "  GPU GEQRF:" << std::endl;
            std::cout << "    Orthogonality error ||Q^T Q - I||_F = " << orth_err_gpu_geqrf << std::endl;
            std::cout << "    Reconstruction error ||A - QR||_F = " << recon_err_gpu_geqrf << std::endl;
            std::cout << "    Relative reconstruction error = " << (norm_A > 0 ? recon_err_gpu_geqrf / norm_A : recon_err_gpu_geqrf) << std::endl;

            std::cout << "  GPU CHOLQR:" << std::endl;
            std::cout << "    Orthogonality error ||Q^T Q - I||_F = " << orth_err << std::endl;
            std::cout << "    Reconstruction error ||A - QR||_F = " << recon_err << std::endl;
            std::cout << "    Relative reconstruction error = " << (norm_A > 0 ? recon_err / norm_A : recon_err) << std::endl;

            // Use relative errors for the test check:
            // - Orthogonality error normalized by sqrt(min_mn) (size of identity matrix)
            // - Reconstruction error normalized by ||A||_F
            // S rel_orth_err = orth_err;
            // S rel_orth_err = (min_mn > 0) ? orth_err / std::sqrt(S(min_mn)) : orth_err;
            // S rel_recon_err = (norm_A > 0) ? recon_err / norm_A : recon_err;
            // err = rel_orth_err + rel_recon_err;
            err = orth_err + recon_err;
            std::cout << "Error: " << err << std::endl;
            *max_err = err > *max_err ? err : *max_err;

            // ============================================================
            // Compute Error Option 2: Compare Q and R directly with CPU reference
            // Requires sign normalization since QR signs are arbitrary
            // ============================================================
            /*
            // Compute ||Q_cpu - Q_gpu||_F
            S norm_Q_diff_sq = S(0);
            for(I j = 0; j < min_mn; ++j)
            {
                for(I i = 0; i < m; ++i)
                {
                    T diff = hA[b][i + j * lda] - hARes[b][i + j * lda];
                    norm_Q_diff_sq += std::real(diff * conj_helper(diff));
                }
            }

            // Compute ||R_cpu - R_gpu||_F
            S norm_R_diff_sq = S(0);
            for(I j = 0; j < n; ++j)
            {
                for(I i = 0; i <= j && i < n; ++i)
                {
                    T diff = R_cpu[i + j * ldr] - hRRes[b][i + j * ldr];
                    norm_R_diff_sq += std::real(diff * conj_helper(diff));
                }
            }

            err = std::sqrt(norm_Q_diff_sq) + std::sqrt(norm_R_diff_sq);
            *max_err = err > *max_err ? err : *max_err;
            */
        }
    }
}

template <bool STRIDED, typename T, typename I, typename Td, typename Ud, typename Sd, typename Id, typename Th, typename Uh, typename Sh, typename Ih>
void cholqr_getPerfData(const rocblas_handle handle,
                        const I m,
                        const I n,
                        Td& dA,
                        const I lda,
                        const rocblas_stride stA,
                        Ud& dR,
                        const I ldr,
                        const rocblas_stride stR,
                        Sd& dSigma,
                        const rocsolver_cholqr_algo algo,
                        Id& dInfo,
                        const I bc,
                        Th& hA,
                        Uh& hR,
                        Sh& hSigma,
                        Ih& hInfo,
                        double* gpu_time_used,
                        double* cpu_time_used,
                        const rocblas_int hot_calls,
                        const int profile,
                        const bool profile_kernels,
                        const bool perf)
{
    using S = decltype(std::real(T{}));
    std::vector<T> hW(n);
    std::vector<T> hTau(std::min(m, n));

    if(!perf)
    {
        cholqr_initData<true, false, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo, bc,
                                        hA, hR, hSigma);

        // cpu-lapack performance (using GEQRF as reference, only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(I b = 0; b < bc; ++b)
        {
            cpu_geqrf(m, n, hA[b], lda, hTau.data(), hW.data(), n);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    cholqr_initData<true, false, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo, bc, hA,
                                    hR, hSigma);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        cholqr_initData<false, true, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo, bc,
                                        hA, hR, hSigma);

        CHECK_ROCBLAS_ERROR(rocsolver_cholqr(STRIDED, handle, m, n, dA.data(), lda, stA, dR.data(),
                                             ldr, stR, dSigma.data(), algo, dInfo.data(), bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    rocsolver_timer timer;

    if(profile > 0)
    {
        if(profile_kernels)
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                         | rocblas_layer_mode_ex_log_kernel);
        else
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        cholqr_initData<false, true, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo, bc,
                                        hA, hR, hSigma);

        timer.start(stream);
        rocsolver_cholqr(STRIDED, handle, m, n, dA.data(), lda, stA, dR.data(), ldr, stR,
                         dSigma.data(), algo, dInfo.data(), bc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
void testing_cholqr(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    I m = argus.get<I>("m");
    I n = argus.get<I>("n", m);
    I lda = argus.get<I>("lda", m);
    I ldr = argus.get<I>("ldr", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stR = argus.get<rocblas_stride>("strideR", ldr * n);

    // Get cholqr algorithm from command line
    char algo_char = argus.get<char>("cholqr_algo", 'D');
    rocsolver_cholqr_algo algo = char2rocsolver_cholqr_algo(algo_char);

    I bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;
    rocblas_stride stRRes = (argus.unit_check || argus.norm_check) ? stR : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_R = size_t(ldr) * n;
    size_t size_Sigma = 1; // one sigma per batch
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_RRes = (argus.unit_check || argus.norm_check) ? size_R : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || lda < m || ldr < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_cholqr(STRIDED, handle, m, n, (T* const*)nullptr, lda, stA,
                                 (T*)nullptr, ldr, stR, (S*)nullptr, algo, (I*)nullptr, bc),
                rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(
                rocsolver_cholqr(STRIDED, handle, m, n, (T*)nullptr, lda, stA, (T*)nullptr, ldr,
                                 stR, (S*)nullptr, algo, (I*)nullptr, bc),
                rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        if(BATCHED)
            CHECK_ALLOC_QUERY(rocsolver_cholqr(STRIDED, handle, m, n, (T* const*)nullptr, lda, stA,
                                               (T*)nullptr, ldr, stR, (S*)nullptr, algo,
                                               (I*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_cholqr(STRIDED, handle, m, n, (T*)nullptr, lda, stA,
                                               (T*)nullptr, ldr, stR, (S*)nullptr, algo,
                                               (I*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    if(BATCHED && STRIDED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        host_strided_batch_vector<T> hR(size_R, 1, stR, bc);
        host_strided_batch_vector<T> hRRes(size_RRes, 1, stR, bc);
        host_strided_batch_vector<S> hSigma(size_Sigma, 1, size_Sigma, bc);
        host_strided_batch_vector<I> hInfo(1, 1, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_strided_batch_vector<T> dR(size_R, 1, stR, bc);
        device_strided_batch_vector<S> dSigma(size_Sigma, 1, size_Sigma, bc);
        device_strided_batch_vector<I> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_R)
            CHECK_HIP_ERROR(dR.memcheck());
        CHECK_HIP_ERROR(dSigma.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, n, dA.data(), lda, stA,
                                                   dR.data(), ldr, stR, dSigma.data(), algo,
                                                   dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            cholqr_getError<STRIDED, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo,
                                        dInfo, bc, hA, hARes, hR, hRRes, hSigma, hInfo, &max_error);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            cholqr_getPerfData<STRIDED, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo,
                                           dInfo, bc, hA, hR, hSigma, hInfo, &gpu_time_used,
                                           &cpu_time_used, hot_calls, argus.profile,
                                           argus.profile_kernels, argus.perf);
    }

    else if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        host_strided_batch_vector<T> hR(size_R, 1, stR, bc);
        host_strided_batch_vector<T> hRRes(size_RRes, 1, stR, bc);
        host_strided_batch_vector<S> hSigma(size_Sigma, 1, size_Sigma, bc);
        host_strided_batch_vector<I> hInfo(1, 1, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_strided_batch_vector<T> dR(size_R, 1, stR, bc);
        device_strided_batch_vector<S> dSigma(size_Sigma, 1, size_Sigma, bc);
        device_strided_batch_vector<I> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_R)
            CHECK_HIP_ERROR(dR.memcheck());
        CHECK_HIP_ERROR(dSigma.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, n, dA.data(), lda, stA,
                                                   dR.data(), ldr, stR, dSigma.data(), algo,
                                                   dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            cholqr_getError<STRIDED, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo,
                                        dInfo, bc, hA, hARes, hR, hRRes, hSigma, hInfo, &max_error);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            cholqr_getPerfData<STRIDED, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo,
                                           dInfo, bc, hA, hR, hSigma, hInfo, &gpu_time_used,
                                           &cpu_time_used, hot_calls, argus.profile,
                                           argus.profile_kernels, argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<T> hR(size_R, 1, stR, bc);
        host_strided_batch_vector<T> hRRes(size_RRes, 1, stRRes, bc);
        host_strided_batch_vector<S> hSigma(size_Sigma, 1, size_Sigma, bc);
        host_strided_batch_vector<I> hInfo(1, 1, 1, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dR(size_R, 1, stR, bc);
        device_strided_batch_vector<S> dSigma(size_Sigma, 1, size_Sigma, bc);
        device_strided_batch_vector<I> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_R)
            CHECK_HIP_ERROR(dR.memcheck());
        CHECK_HIP_ERROR(dSigma.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, n, dA.data(), lda, stA,
                                                   dR.data(), ldr, stR, dSigma.data(), algo,
                                                   dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            cholqr_getError<STRIDED, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo,
                                        dInfo, bc, hA, hARes, hR, hRRes, hSigma, hInfo, &max_error);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            cholqr_getPerfData<STRIDED, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo,
                                           dInfo, bc, hA, hR, hSigma, hInfo, &gpu_time_used,
                                           &cpu_time_used, hot_calls, argus.profile,
                                           argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    // Using m * n * machine_precision as tolerance for CHOLQR
    // For ill-conditioned matrices, tolerance may need to be larger
    double eps = get_epsilon<T>();
    double tol = static_cast<double>(m) * static_cast<double>(n) * eps;
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Tolerance: " << tol << std::endl;
    std::cout << "Machine precision: " << eps << std::endl;
    std::cout << "M*N: " << m * n << std::endl;
    std::cout << std::defaultfloat;
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 10 * m * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("m", "n", "lda", "ldr", "strideR", "algo", "batch_c");
                rocsolver_bench_output(m, n, lda, ldr, stR, algo_char, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideA", "ldr", "strideR", "algo",
                                       "batch_c");
                rocsolver_bench_output(m, n, lda, stA, ldr, stR, algo_char, bc);
            }
            else
            {
                rocsolver_bench_output("m", "n", "lda", "ldr", "algo");
                rocsolver_bench_output(m, n, lda, ldr, algo_char);
            }
            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            rocsolver_bench_endl();
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, max_error);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

#define EXTERN_TESTING_CHOLQR(...) extern template void testing_cholqr<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_CHOLQR,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_SCALAR_TYPE,
            FOREACH_INT_TYPE,
            APPLY_STAMP)

