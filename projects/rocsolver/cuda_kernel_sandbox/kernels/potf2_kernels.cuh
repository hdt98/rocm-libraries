/* **************************************************************************
 * rocSOLVER Kernel Sandbox - POTF2 Kernels
 *
 * Cholesky factorization kernels for symmetric positive definite matrices.
 * potf2_kernel_small performs Cholesky factorization using shared memory
 * for small matrices that fit entirely in LDS cache.
 *
 * Derived from:
 * - roclapack_potf2_specialized_kernels.hpp
 *
 * Adapted for CUDA compilation with nvcc.
 * *************************************************************************/

#pragma once

#include "../cuda_compat.cuh"
#include "../rocsolver_types.cuh"
#include "../device_helpers.cuh"

#include <algorithm>
#include <cmath>

ROCSOLVER_BEGIN_NAMESPACE

// Block size for POTF2 small kernel (threads per dimension)
#ifndef POTF2_BS
#define POTF2_BS 16
#endif

// Maximum matrix size for small kernel (n*(n+1)/2 elements must fit in shared memory)
#ifndef POTF2_MAX_SMALL_SIZE
#define POTF2_MAX_SMALL_SIZE 64
#endif

/**
 * indexing for packed storage
 * for upper triangular
 *
 * ---------------------------
 * 0 1 3
 *   2 4
 *     5
 * ---------------------------
 *
 **/

template <typename I>
__device__ static I idx_upper(I i, I j, I n)
{
    return (i + (j * (j + 1)) / 2);
}

/**
 * indexing for packed storage
 * for lower triangular
 *
 * ---------------------------
 * 0
 * 1      n
 * *      (n+1)
 * *
 * (n-1)  ...        n*(n+1)/2
 * ---------------------------
 **/
template <typename I>
__device__ static I idx_lower(I i, I j, I n)
{
    return ((i - j) + (j * (2 * n + 1 - j)) / 2);
}

/**
 * Conjugate helper for real types (no-op)
 */
template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ __host__ __forceinline__ T conj(T val)
{
    return val;
}

/**
 * Conjugate helper for complex types
 */
template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__device__ __host__ __forceinline__ T conj(T val)
{
    return T(val.real(), -val.imag());
}

/**
 * ------------------------------------------------------
 * Perform Cholesky factorization for small n by n matrix.
 * The function executes in a single thread block.
 *
 * Uses packed storage in shared memory to minimize LDS usage.
 *
 * Algorithm:
 * For lower triangular (A = L*L'):
 *   [  l11     ]  * [ l11'   vl21' ]  =  [ a11       ]
 *   [ vl21  L22]    [        L22' ]     [ va21, A22 ]
 *
 *   (1) l11 = sqrt(a11), scalar computation
 *   (2) vl21 = va21 / l11', scale vector
 *   (3) A22 = A22 - vl21 * vl21', symmetric rank-1 update
 *   (4) Recurse on L22
 *
 * For upper triangular (A = U'*U):
 *   Similar with upper triangular storage.
 * ------------------------------------------------------
**/
template <typename T, typename I, typename INFO>
__device__ static void potf2_simple(bool const is_upper, I const n, T* const A, INFO* const info)
{
    auto const lda = n;
    bool const is_lower = (!is_upper);

    auto const i_start = hipThreadIdx_x;
    auto const i_inc = hipBlockDim_x;
    auto const j_start = hipThreadIdx_y;
    auto const j_inc = hipBlockDim_y;

    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);
    auto const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;

    auto const j0_start = tid;
    auto const j0_inc = nthreads;

    if(is_lower)
    {
        // ---------------------------------------------------
        // [  l11     ]  * [ l11'   vl21' ]  =  [ a11       ]
        // [ vl21  L22]    [        L22' ]     [ va21, A22 ]
        //
        //
        //   assume l11 is scalar 1x1 matrix
        //
        //   (1) l11 * l11' = a11 =>  l11 = sqrt( abs(a11) ), scalar computation
        //   (2) vl21 * l11' = va21 =>  vl21 = va21/ l11', scale vector
        //   (3) L22 * L22' + vl21 * vl21' = A22
        //
        //   (3a) A22 = A22 - vl21 * vl21',  symmetric rank-1 update
        //   (3b) L22 * L22' = A22,   cholesky factorization, tail recursion
        // ---------------------------------------------------

        for(I kcol = 0; kcol < n; kcol++)
        {
            auto kk = idx_lower(kcol, kcol, lda);
            auto const akk = A[kk];  // For real types, this is just the real value
            bool const isok = (akk > 0) && (isfinite(akk));
            if(!isok)
            {
                if(tid == 0)
                {
                    A[kk] = akk;
                    // Fortran 1-based index
                    if(*info == 0)
                        *info = kcol + 1;
                }
                break;
            }

            auto const lkk = sqrtf(akk);
            if(tid == 0)
            {
                A[kk] = lkk;
            }

            __syncthreads();

            // ------------------------------------------------------------
            //   (2) vl21 * l11' = va21 =>  vl21 = va21/ l11', scale vector
            // ------------------------------------------------------------

            auto const conj_lkk = conj(lkk);
            for(I j0 = (kcol + 1) + j0_start; j0 < n; j0 += j0_inc)
            {
                auto const j0k = idx_lower(j0, kcol, lda);

                A[j0k] = (A[j0k] / conj_lkk);
            }

            __syncthreads();

            // ------------------------------------------------------------
            //   (3a) A22 = A22 - vl21 * vl21',  symmetric rank-1 update
            //
            //   note: update lower triangular part
            // ------------------------------------------------------------

            for(I j = (kcol + 1) + j_start; j < n; j += j_inc)
            {
                auto const vj = A[idx_lower(j, kcol, lda)];
                for(I i = (kcol + 1) + i_start; i < n; i += i_inc)
                {
                    bool const lower_part = (i >= j);
                    if(lower_part)
                    {
                        auto const vi = A[idx_lower(i, kcol, lda)];
                        auto const ij = idx_lower(i, j, lda);

                        A[ij] = A[ij] - vi * conj(vj);
                    }
                }
            }

            __syncthreads();

        } // end for kcol
    }
    else
    {
        // --------------------------------------------------
        // [u11'        ] * [u11    vU12 ] = [ a11     vA12 ]
        // [vU12'   U22']   [       U22  ]   [ vA12'   A22  ]
        //
        // (1) u11' * u11 = a11 =?  u11 = sqrt( abs( a11 ) )
        // (2) vU12' * u11 = vA12', or u11' * vU12 = vA12
        //     or vU12 = vA12/u11'
        // (3) vU12' * vU12 + U22'*U22 = A22
        //
        // (3a) A22 = A22 - vU12' * vU12
        // (3b) U22' * U22 = A22,  cholesky factorization, tail recursion
        // --------------------------------------------------

        for(I kcol = 0; kcol < n; kcol++)
        {
            auto const kk = idx_upper(kcol, kcol, lda);
            auto const akk = A[kk];
            bool const isok = (akk > 0) && (isfinite(akk));
            if(!isok)
            {
                if(tid == 0)
                {
                    A[kk] = akk;
                    // Fortran 1-based index
                    if(*info == 0)
                        *info = kcol + 1;
                }

                break;
            }

            auto const ukk = sqrtf(akk);
            if(tid == 0)
            {
                A[kk] = ukk;
            }
            __syncthreads();

            // ----------------------------------------------
            // (2) vU12' * u11 = vA12', or u11' * vU12 = vA12
            // ----------------------------------------------
            for(I j0 = (kcol + 1) + j0_start; j0 < n; j0 += j0_inc)
            {
                auto const kj0 = idx_upper(kcol, j0, lda);

                A[kj0] = A[kj0] / ukk;
            }

            __syncthreads();

            // -----------------------------
            // (3a) A22 = A22 - vU12' * vU12
            //
            // note: update upper triangular part
            // -----------------------------
            for(I j = (kcol + 1) + j_start; j < n; j += j_inc)
            {
                auto const vj = A[idx_upper(kcol, j, lda)];
                for(I i = (kcol + 1) + i_start; i < n; i += i_inc)
                {
                    bool const upper_part = (i <= j);
                    if(upper_part)
                    {
                        auto const vi = A[idx_upper(kcol, i, lda)];
                        auto const ij = idx_upper(i, j, lda);

                        A[ij] = A[ij] - conj(vi) * vj;
                    }
                }
            }

            __syncthreads();

        } // end for kcol
    }
}

/**
 * ------------------------------------------------------
 * Main POTF2 kernel for small matrices.
 *
 * Loads the triangular matrix into shared memory using packed storage,
 * performs the Cholesky factorization, then writes results back.
 *
 * Block dimensions: (POTF2_BS, POTF2_BS, 1)
 * Grid dimensions: (1, 1, batch_count)
 *
 * Shared memory: sizeof(T) * n * (n+1) / 2 bytes (packed storage)
 * ------------------------------------------------------
**/
template <typename T, typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void potf2_kernel_small(const bool is_upper,
                                         const I n,
                                         U AA,
                                         const rocblas_stride shiftA,
                                         const I lda,
                                         const rocblas_stride strideA,
                                         INFO* const info)
{
    bool const is_lower = (!is_upper);

    auto const i_start = hipThreadIdx_x;
    auto const i_inc = hipBlockDim_x;
    auto const j_start = hipThreadIdx_y;
    auto const j_inc = hipBlockDim_y;

    // --------------------------------
    // note hipGridDim_z == batch_count
    // --------------------------------
    auto const bid = hipBlockIdx_z;

    T* const A = load_ptr_batch(AA, bid, shiftA, strideA);
    INFO* const info_bid = info + bid;

    // -----------------------------------------
    // assume n by n matrix will fit in LDS cache
    // -----------------------------------------
    extern __shared__ rocblas_int lsmem[];
    T* Ash = reinterpret_cast<T*>(lsmem);

    // --------------------------------------------------------
    // factoring Lower triangular matrix may be slightly faster
    // due to simpler index calculation down a column
    // --------------------------------------------------------
    bool const use_compute_lower = true;

    // ------------------------------------
    // copy n by n packed matrix into shared memory
    // ------------------------------------
    __syncthreads();

    if(is_lower)
    {
        for(I j = j_start; j < n; j += j_inc)
        {
            for(I i = j + i_start; i < n; i += i_inc)
            {
                auto const ij = i + j * static_cast<int64_t>(lda);
                auto const ij_packed = idx_lower(i, j, n);

                Ash[ij_packed] = A[ij];
            }
        }
    }
    else
    {
        for(I j = j_start; j < n; j += j_inc)
        {
            for(I i = i_start; i <= j; i += i_inc)
            {
                auto const ij = i + j * static_cast<int64_t>(lda);
                auto const ij_packed = (use_compute_lower) ? idx_lower(j, i, n) : idx_upper(i, j, n);

                auto const aij = A[ij];
                Ash[ij_packed] = (use_compute_lower) ? conj(aij) : aij;
            }
        }
    }

    __syncthreads();

    bool const is_up = (use_compute_lower) ? false : is_upper;
    potf2_simple<T>(is_up, n, Ash, info_bid);

    __syncthreads();

    // -------------------------------------
    // copy n by n packed matrix into global memory
    // -------------------------------------
    if(is_lower)
    {
        for(I j = j_start; j < n; j += j_inc)
        {
            for(I i = j + i_start; i < n; i += i_inc)
            {
                auto const ij = i + j * static_cast<int64_t>(lda);
                auto const ij_packed = idx_lower(i, j, n);

                A[ij] = Ash[ij_packed];
            }
        }
    }
    else
    {
        for(I j = j_start; j < n; j += j_inc)
        {
            for(I i = i_start; i <= j; i += i_inc)
            {
                auto const ij = i + j * static_cast<int64_t>(lda);
                auto const ij_packed = (use_compute_lower) ? idx_lower(j, i, n) : idx_upper(i, j, n);

                auto const aij_packed = Ash[ij_packed];
                A[ij] = (use_compute_lower) ? conj(aij_packed) : aij_packed;
            }
        }
    }

    __syncthreads();
}

/**
 * ------------------------------------------------------
 * sqrtDiagOnward kernel for non-small POTF2.
 *
 * Computes the square root of the diagonal element and tests
 * for non-positive definiteness. Used in the iterative POTF2
 * algorithm for larger matrices.
 *
 * For each batch instance:
 *   t = M[loc] - res[id]  (res contains dot product result)
 *   if t <= 0: set info and mark as non-positive definite
 *   else: M[loc] = sqrt(t), res[id] = 1/sqrt(t)
 *
 * Block dimensions: (1, 1, 1)
 * Grid dimensions: (batch_count, 1, 1)
 * ------------------------------------------------------
**/
template <typename T, typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void sqrtDiagOnward(U A,
                                     const rocblas_stride shiftA,
                                     const rocblas_stride strideA,
                                     const size_t loc,
                                     const I j,
                                     T* res,
                                     INFO* info)
{
    I id = hipBlockIdx_x;

    T* M = load_ptr_batch<T>(A, id, shiftA, strideA);
    T t = M[loc] - res[id];

    if(t <= T(0))
    {
        // error for non-positive definiteness
        if(info[id] == 0)
            info[id] = j + 1; // use fortran 1-based index
        M[loc] = t;
        res[id] = T(0);
    }
    else
    {
        // minor is positive definite
        M[loc] = sqrtf(t);
        res[id] = T(1) / M[loc];
    }
}

/**
 * ------------------------------------------------------
 * Alternative unrolled POTF2 kernel that processes the entire
 * factorization without external BLAS calls.
 *
 * This version is useful for testing and can be used to verify
 * the shared-memory kernel implementation.
 *
 * Uses dense column-major storage (not packed).
 * ------------------------------------------------------
**/
template <typename T, typename I, typename INFO>
ROCSOLVER_KERNEL void potf2_unrolled_kernel(const bool is_upper,
                                            const I n,
                                            T* A,
                                            const I lda,
                                            const rocblas_stride strideA,
                                            INFO* info)
{
    auto const bid = hipBlockIdx_z;
    auto const tid = hipThreadIdx_x;
    auto const nthreads = hipBlockDim_x;

    T* M = A + bid * strideA;
    INFO* info_bid = info + bid;

    // Shared memory for storing the current column's computed values
    extern __shared__ char smem[];
    T* scol = reinterpret_cast<T*>(smem);

    if(is_upper)
    {
        // Upper triangular: A = U' * U
        for(I j = 0; j < n; j++)
        {
            // Compute U(j,j)
            T sum = T(0);
            for(I k = tid; k < j; k += nthreads)
            {
                T ukj = M[k + j * lda];
                sum += ukj * ukj;
            }

            // Reduce sum across threads
            scol[tid] = sum;
            __syncthreads();

            for(I s = nthreads / 2; s > 0; s >>= 1)
            {
                if(tid < s)
                    scol[tid] += scol[tid + s];
                __syncthreads();
            }

            if(tid == 0)
            {
                T ajj = M[j + j * lda] - scol[0];
                if(ajj <= T(0) || !isfinite(ajj))
                {
                    M[j + j * lda] = ajj;
                    if(*info_bid == 0)
                        *info_bid = j + 1;
                    // Continue to signal error, but don't proceed
                }
                else
                {
                    M[j + j * lda] = sqrtf(ajj);
                }
                scol[0] = M[j + j * lda];  // Store for other threads to use
            }
            __syncthreads();

            if(*info_bid != 0)
                break;

            T ujj = scol[0];

            // Update U(j, j+1:n)
            for(I i = j + 1 + tid; i < n; i += nthreads)
            {
                T sum_i = T(0);
                for(I k = 0; k < j; k++)
                {
                    sum_i += M[k + j * lda] * M[k + i * lda];
                }
                M[j + i * lda] = (M[j + i * lda] - sum_i) / ujj;
            }
            __syncthreads();
        }
    }
    else
    {
        // Lower triangular: A = L * L'
        for(I j = 0; j < n; j++)
        {
            // Compute L(j,j)
            T sum = T(0);
            for(I k = tid; k < j; k += nthreads)
            {
                T ljk = M[j + k * lda];
                sum += ljk * ljk;
            }

            // Reduce sum across threads
            scol[tid] = sum;
            __syncthreads();

            for(I s = nthreads / 2; s > 0; s >>= 1)
            {
                if(tid < s)
                    scol[tid] += scol[tid + s];
                __syncthreads();
            }

            if(tid == 0)
            {
                T ajj = M[j + j * lda] - scol[0];
                if(ajj <= T(0) || !isfinite(ajj))
                {
                    M[j + j * lda] = ajj;
                    if(*info_bid == 0)
                        *info_bid = j + 1;
                }
                else
                {
                    M[j + j * lda] = sqrtf(ajj);
                }
                scol[0] = M[j + j * lda];
            }
            __syncthreads();

            if(*info_bid != 0)
                break;

            T ljj = scol[0];

            // Update L(j+1:n, j)
            for(I i = j + 1 + tid; i < n; i += nthreads)
            {
                T sum_i = T(0);
                for(I k = 0; k < j; k++)
                {
                    sum_i += M[i + k * lda] * M[j + k * lda];
                }
                M[i + j * lda] = (M[i + j * lda] - sum_i) / ljj;
            }
            __syncthreads();
        }
    }
}

ROCSOLVER_END_NAMESPACE
