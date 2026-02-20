/* **************************************************************************
 * rocSOLVER Kernel Sandbox - POTF2 Kernels
 *
 * Cholesky factorization kernels for symmetric positive definite matrices.
 * potf2_kernel_small performs Cholesky factorization using a blocked
 * algorithm with register-based storage and shared memory panels.
 *
 * Derived from:
 * - rocm-libraries-angelo-potf2/projects/rocsolver/library/src/specialized/
 *   roclapack_potf2_specialized_kernels.hpp
 * - rocm-libraries-angelo-potf2/projects/rocsolver/library/src/lapack/
 *   roclapack_potf2.hpp
 *
 * This version uses a blocked algorithm with NB panels of size PANEL_SIZE.
 * The matrix is loaded into registers in packed triangular format, and
 * panels are processed through shared memory.
 *
 * Adapted for CUDA compilation with nvcc.
 * *************************************************************************/

#pragma once

#include "../cuda_compat.cuh"
#include "../rocsolver_types.cuh"
#include "../device_helpers.cuh"

#include <algorithm>
#include <cmath>
#include <array>

ROCSOLVER_BEGIN_NAMESPACE

// Panel size for POTF2 small kernel (threads per dimension)
// This defines the panel width for the blocked algorithm
#ifndef POTF2_PANEL_SIZE
#define POTF2_PANEL_SIZE BS2  // BS2 = 32
#endif

// Maximum matrix size for small kernel
// Complex (16 bytes): max 128x128 = 4 panels
// Real (4-8 bytes): max 256x256 = 8 panels
#ifndef POTF2_MAX_SMALL_SIZE
#define POTF2_MAX_SMALL_SIZE 256
#endif

// Maximum number of panels (NB) supported
#define POTF2_MAX_NB 8

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
 * std::real for non-complex types
 */
template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ __host__ __forceinline__ T real_part(T val)
{
    return val;
}

/**
 * std::real for complex types
 */
template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__device__ __host__ __forceinline__ auto real_part(T val)
{
    return val.real();
}

/**
 * ------------------------------------------------------
 * Perform Cholesky factorization for small n by n matrix.
 * The function executes in a single thread block per matrix.
 * ------------------------------------------------------
 *
 * NB           Number of panels to perform blocked decomposition.
 *              ceildiv(n, PANEL_SIZE)
 * PANEL_SIZE   Size of panel to perform non-blocked decomposition.
 *              PANEL_SIZE == BlockDim.x == BlockDim.y
 *
 * Algorithm:
 * 1. Load A to registers in compact triangular storage: Arg[(NB*(NB+1))/2]
 * 2. For each panel kb = 0 to NB-1:
 *    a. Write panel to LDS shared memory (as lower triangular)
 *    b. Factorize panel using column-wise Cholesky
 *    c. Update trailing matrix in registers
 *    d. Load factored panel back to registers
 * 3. Write A from registers back to global memory
 *
 * This blocked approach enables:
 * - Register reuse for trailing matrix updates
 * - Shared memory for panel factorization with full thread parallelism
 * - Support for matrices up to NB*PANEL_SIZE x NB*PANEL_SIZE
**/
template <int NB, int PANEL_SIZE, typename T, typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void potf2_kernel_small(const bool is_upper,
                                         const I n,
                                         U AA,
                                         const rocblas_stride shiftA,
                                         const I lda,
                                         const rocblas_stride strideA,
                                         INFO* const info)
{
    auto const tid = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;
    auto const inc = hipBlockDim_y * hipBlockDim_x;
    auto const tidx = hipThreadIdx_x;
    auto const tidy = hipThreadIdx_y;

    // get batch index
    auto const bid = hipBlockIdx_z;

    T* const A = load_ptr_batch(AA, bid, shiftA, strideA);
    INFO* const info_bid = info + bid;

    extern __shared__ rocblas_int lsmem[];
    T* Ash = reinterpret_cast<T*>(lsmem);
    auto constexpr ldash = NB * PANEL_SIZE;

    bool failed = false;

    // load A to registers
    T Arg[(NB * (NB + 1)) / 2] = {0};

    I arg_idx = 0;
    for(I jb = 0; jb < NB; jb++)
    {
        for(I i = jb; i < NB; i++)
        {
            if(is_upper)
            {
                const auto col = i * PANEL_SIZE + tidy;
                const auto row = jb * PANEL_SIZE + tidx;
                if(col < n && row < n && row <= col)
                {
                    const auto idx = col * lda + row;
                    Arg[arg_idx] = A[idx];
                }
            }
            else
            {
                const auto col = jb * PANEL_SIZE + tidy;
                const auto row = i * PANEL_SIZE + tidx;
                if(col < n && row < n && row >= col)
                {
                    const auto idx = col * lda + row;
                    Arg[arg_idx] = A[idx];
                }
            }

            arg_idx++;
        }
    }

    // Panel Cholesky decomposition
    arg_idx = 0;
    for(I kb = 0; kb < NB; kb++)
    {
        // write panel to lds
        for(I i = 0; i < NB - kb; i++)
        {
            // write to lds as lower and compute as lower
            if(is_upper)
            {
                const auto col = i * PANEL_SIZE + tidy;
                const auto idx = tidx * ldash + col;
                Ash[idx] = Arg[arg_idx + i];
            }
            else
            {
                const auto row = i * PANEL_SIZE + tidx;
                const auto idx = tidy * ldash + row;
                Ash[idx] = Arg[arg_idx + i];
            }
        }

        __syncthreads();

        I nn = n - kb * PANEL_SIZE;

        // factorize panel
        for(I kcol = 0; kcol < PANEL_SIZE; kcol++)
        {
            if(kcol >= nn)
                break;

            auto kk = kcol * ldash + kcol;
            auto const akk = real_part(Ash[kk]);
            bool const isok = (akk > 0) && (isfinite(akk));

            __syncthreads();

            if(!isok)
            {
                if(tid == 0)
                {
                    Ash[kk] = akk;
                    // Fortran 1-based index
                    if(*info_bid == 0)
                        *info_bid = kb * PANEL_SIZE + kcol + 1;
                }
                failed = true;
                __syncthreads();
                break;
            }

            auto const lkk = sqrtf(akk);
            if(tid == 0)
            {
                Ash[kk] = lkk;
            }

            // ------------------------------------------------------------
            //   (2) vl21 * l11' = va21 =>  vl21 = va21/ l11', scale vector
            // ------------------------------------------------------------

            auto const conj_lkk = conj(lkk);
            for(I j0 = (kcol + 1) + tid; j0 < nn; j0 += inc)
            {
                auto const j0k = j0 + kcol * ldash;

                Ash[j0k] = (Ash[j0k] / conj_lkk);
            }

            __syncthreads();

            // ------------------------------------------------------------
            //   (3a) A22 = A22 - vl21 * vl21',  symmetric rank-1 update
            //
            //   note: update lower triangular part
            // ------------------------------------------------------------

            for(I j = (kcol + 1) + tidy; j < PANEL_SIZE; j += hipBlockDim_y)
            {
                auto const vj = Ash[j + kcol * ldash];
                for(I i = j + tidx; i < nn; i += hipBlockDim_x)
                {
                    auto const vi = Ash[i + kcol * ldash];
                    auto const ij = i + j * ldash;

                    Ash[ij] = Ash[ij] - vi * conj(vj);
                }
            }
            __syncthreads();
        }

        // update trailing matrix
        I upd_arg_idx = arg_idx + NB - kb;
        for(I j = kb + 1; j < NB; j++)
        {
            for(I i = j; i < NB; i++)
            {
                if(is_upper)
                {
                    const auto col = (i - kb) * PANEL_SIZE + tidy;
                    const auto row = (j - kb) * PANEL_SIZE + tidx;

                    for(I p = 0; p < PANEL_SIZE; p++)
                    {
                        Arg[upd_arg_idx + i - j] -= conj(Ash[row + p * ldash]) * Ash[col + p * ldash];
                    }
                }
                else
                {
                    const auto col = (j - kb) * PANEL_SIZE + tidy;
                    const auto row = (i - kb) * PANEL_SIZE + tidx;

                    for(I p = 0; p < PANEL_SIZE; p++)
                    {
                        Arg[upd_arg_idx + i - j] -= Ash[row + p * ldash] * conj(Ash[col + p * ldash]);
                    }
                }
            }
            upd_arg_idx += NB - j;
        }

        // load panel back to registers
        for(I i = 0; i < NB - kb; i++)
        {
            if(is_upper)
            {
                const auto col = i * PANEL_SIZE + tidy;
                const auto idx = tidx * ldash + col;
                Arg[arg_idx + i] = Ash[idx];
            }
            else
            {
                const auto row = i * PANEL_SIZE + tidx;
                const auto idx = tidy * ldash + row;
                Arg[arg_idx + i] = Ash[idx];
            }
        }
        arg_idx += NB - kb;

        __syncthreads();

        if(failed)
            break;
    }

    // write A from registers
    arg_idx = 0;
    for(I jb = 0; jb < NB; jb++)
    {
        for(I i = jb; i < NB; i++)
        {
            if(is_upper)
            {
                const auto col = i * PANEL_SIZE + tidy;
                const auto row = jb * PANEL_SIZE + tidx;
                if(col < n && row < n && row <= col)
                {
                    const auto idx = col * lda + row;
                    A[idx] = Arg[arg_idx];
                }
            }
            else
            {
                const auto col = jb * PANEL_SIZE + tidy;
                const auto row = i * PANEL_SIZE + tidx;
                if(col < n && row < n && row >= col)
                {
                    const auto idx = col * lda + row;
                    A[idx] = Arg[arg_idx];
                }
            }

            arg_idx++;
        }
    }
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
template <typename T, typename I, typename INFO, typename U,
          std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
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
 * sqrtDiagOnward for complex types
 */
template <typename T, typename I, typename INFO, typename U,
          std::enable_if_t<rocblas_is_complex<T>, int> = 0>
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
    auto t = M[loc].real() - res[id].real();

    if(t <= 0.0)
    {
        // error for non-positive definiteness
        if(info[id] == 0)
            info[id] = j + 1; // use fortran 1-based index
        M[loc] = t;
        res[id] = T(0, 0);
    }
    else
    {
        // minor is positive definite
        M[loc] = sqrtf(t);
        res[id] = T(1, 0) / M[loc];
    }
}

/**
 * Helper function to launch the appropriate kernel based on matrix size.
 * Selects the right NB template parameter based on n.
 */
template <typename T, typename I, typename INFO>
void launch_potf2_kernel_small(bool is_upper,
                               I n,
                               T* A,
                               rocblas_stride shiftA,
                               I lda,
                               rocblas_stride strideA,
                               INFO* info,
                               I batch_count,
                               cudaStream_t stream = 0)
{
    const int nb = (n + POTF2_PANEL_SIZE - 1) / POTF2_PANEL_SIZE;
    size_t shared_mem_size = sizeof(T) * nb * POTF2_PANEL_SIZE * POTF2_PANEL_SIZE;

    dim3 grid(1, 1, batch_count);
    dim3 block(POTF2_PANEL_SIZE, POTF2_PANEL_SIZE, 1);

    // Select kernel based on nb
    switch(nb)
    {
    case 1:
        potf2_kernel_small<1, POTF2_PANEL_SIZE, T, I, INFO, T*>
            <<<grid, block, shared_mem_size, stream>>>(
                is_upper, n, A, shiftA, lda, strideA, info);
        break;
    case 2:
        potf2_kernel_small<2, POTF2_PANEL_SIZE, T, I, INFO, T*>
            <<<grid, block, shared_mem_size, stream>>>(
                is_upper, n, A, shiftA, lda, strideA, info);
        break;
    case 3:
        potf2_kernel_small<3, POTF2_PANEL_SIZE, T, I, INFO, T*>
            <<<grid, block, shared_mem_size, stream>>>(
                is_upper, n, A, shiftA, lda, strideA, info);
        break;
    case 4:
        potf2_kernel_small<4, POTF2_PANEL_SIZE, T, I, INFO, T*>
            <<<grid, block, shared_mem_size, stream>>>(
                is_upper, n, A, shiftA, lda, strideA, info);
        break;
    case 5:
        potf2_kernel_small<5, POTF2_PANEL_SIZE, T, I, INFO, T*>
            <<<grid, block, shared_mem_size, stream>>>(
                is_upper, n, A, shiftA, lda, strideA, info);
        break;
    case 6:
        potf2_kernel_small<6, POTF2_PANEL_SIZE, T, I, INFO, T*>
            <<<grid, block, shared_mem_size, stream>>>(
                is_upper, n, A, shiftA, lda, strideA, info);
        break;
    case 7:
        potf2_kernel_small<7, POTF2_PANEL_SIZE, T, I, INFO, T*>
            <<<grid, block, shared_mem_size, stream>>>(
                is_upper, n, A, shiftA, lda, strideA, info);
        break;
    case 8:
        potf2_kernel_small<8, POTF2_PANEL_SIZE, T, I, INFO, T*>
            <<<grid, block, shared_mem_size, stream>>>(
                is_upper, n, A, shiftA, lda, strideA, info);
        break;
    default:
        // Unsupported size - should not reach here if n <= POTF2_MAX_SMALL_SIZE
        break;
    }
}

ROCSOLVER_END_NAMESPACE
