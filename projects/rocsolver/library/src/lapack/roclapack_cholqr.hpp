
/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas.hpp"
#include "roclapack_potrf.hpp"

ROCSOLVER_BEGIN_NAMESPACE

#define IS_POINTER_BATCHED(A) (std::is_pointer_v<std::remove_reference_t<decltype((A)[0])>>)

static int get_num_cu(int deviceId = 0)
{
    int ival = 0;
    auto const attr = hipDeviceAttributeMultiprocessorCount;
    HIP_CHECK(hipDeviceGetAttribute(&ival, attr, deviceId));
    return (ival);
}

static int get_warp_size(int deviceId = 0)
{
    int ival = 0;
    auto const attr = hipDeviceAttributeWarpSize;
    HIP_CHECK(hipDeviceGetAttribute(&ival, attr, deviceId));
    return (ival);
}

template <typename T, typename I>
__device__ T reduce_sum_shfl_wsize(I const wsize, T val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    if(wsize == 64)
    {
        val += __shfl_down(val, 32); // offset = 32
        val += __shfl_down(val, 16); // offset = 16
        val += __shfl_down(val, 8); // offset = 8
        val += __shfl_down(val, 4); // offset = 4
        val += __shfl_down(val, 2); // offset = 2
        val += __shfl_down(val, 1); // offset = 1
    }
    else if(wsize == 32)
    {
        val += __shfl_down(val, 16); // offset = 16
        val += __shfl_down(val, 8); // offset = 8
        val += __shfl_down(val, 4); // offset = 4
        val += __shfl_down(val, 2); // offset = 2
        val += __shfl_down(val, 1); // offset = 1
    }
    else
    {
        for(auto offset = wsize / 2; offset > 0; offset /= 2)
        {
            val += __shfl_down(val, offset);
            // g.sync();
        }
    }
    return val; // note: only thread 0 will return full sum
}

// kernel to compute the square of g-norm
// which is the max 2-norm square of the columns
//
// max_j  norm( A(:,j),2)^2
//
//
// launch as dim3(1,nby,nbz), dim3(nx,ny,1)
//
// all threads in x-direction in thread block work on
// computing the 2-norm square of a single column
//
// assume nx <= warpsize
// to use DPP instructions
template <typename T, typename I, typename Istride, typename UA, typename S = decltype(std::real(T{}))>
static __device__ void cal_gnorm_kernel(I const m,
                                        I const n,
                                        I const batch_count,

                                        UA A,
                                        Istride const shiftA,
                                        I const lda,
                                        Istride const strideA,

                                        S* const gnorm_array)
{
    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1) && (gnorm_array != nullptr);
        if(!has_work)
        {
            return;
        };
    }

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    I const nx = blockDim.x;
    I const ny = blockDim.y;
    I const nz = blockDim.z;

    I const nbx = gridDim.x;
    I const nby = gridDim.y;
    I const nbz = gridDim.z;

    assert(nbx == 1);
    assert(nx <= warpSize);
    assert(nz == 1);

    I const ibx = blockIdx.x;
    I const iby = blockIdx.y;
    I const ibz = blockIdx.z;

    I const tx = threadIdx.x;
    I const ty = threadIdx.y;
    I const tz = threadIdx.z;

    I const i_start = tx;
    I const i_inc = nx;

    I const j_start = ty + iby * ny;
    I const j_inc = ny * nby;

    I const bid_start = ibz;
    I const bid_inc = nbz;

    __shared__ S gnorm_block;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const Ap = load_ptr_batch(A, bid, shiftA, strideA);

        S* const gnorm_bid = &(gnorm_array[bid]);

        bool const use_simple = true;
        if(use_simple)
        {
            // -------------------------
            // use one thread per column
            // -------------------------
            I const txyz = tx + ty * nx + tz * (nx * ny);
            I const nxyz = (nx * ny) * nz;
            I const jcol_start = txyz + iby * nxyz;
            I const jcol_inc = nxyz * nby;

            S gnorm_j = 0;
            for(I jcol = jcol_start; jcol < n; jcol += jcol_inc)
            {
                S norm_j = 0;
                for(I i = 0; i < m; i++)
                {
                    auto const ij = idx2D(i, jcol, lda);
                    auto const aij = Ap[ij];
                    norm_j += std::norm(aij);
                }
                gnorm_j = std::max(gnorm_j, norm_j);
            }
            atomicMax(gnorm_bid, gnorm_j);
        }
        else
        {
            if((tx == 0) && (ty == 0))
            {
                gnorm_block = 0;
            }
            __syncthreads();

            S gnorm_j = 0;
            for(I j = j_start; j < n; j += j_inc)
            {
                S norm_j = 0;
                for(I i = i_start; i < m; i += i_inc)
                {
                    auto const ij = idx2D(i, j, lda);
                    auto const aij = Ap[ij];

                    norm_j += std::norm(aij);
                }

                // only tx == 0 has the correct value
                norm_j = reduce_sum_shfl_wsize(nx, norm_j);
                if(tx == 0)
                {
                    gnorm_j = std::max(gnorm_j, norm_j);
                }
            } // end for j
            if(tx == 0)
            {
                atomicMax(&gnorm_block, gnorm_j);
            }
            __syncthreads();

            if((tx == 0) && (ty == 0))
            {
                atomicMax(gnorm_bid, gnorm_block);
            }
        }
    } // end for bid
}

// ----------------------------------------
// compute the gnorm of matrix,
// which is    max_j  norm( A(:,j), 2 )^2
// ----------------------------------------
template <typename T, typename I, typename Istride, typename UA, typename S = decltype(std::real(T{}))>
static void cal_gnorm(hipStream_t stream,
                      I const m,
                      I const n,
                      I const batch_count,

                      UA A,
                      Istride const shiftA,
                      I const lda,
                      Istride const strideA,

                      S* const gnorm_array)
{
    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return;
        };
    }

    {
        size_t const size_gnorm_array = sizeof(S) * batch_count;
        hipError_t const istat = hipMemsetAsync(gnorm_array, 0, size_gnorm_array, stream);
        if(istat != hipSuccess)
        {
            return;
        };
    }

    auto ceil = [](auto n, auto base) { return ((n - 1) / base + 1); };

    auto const num_cu = get_num_cu();
    auto const warp_size = get_warp_size();
    I const lds_size = sizeof(S);

    I const max_threads = 1024;
    I const nx = warp_size; // note nx <= warp_size is necessary for correctness
    I const ny = max_threads / nx;
    I const nz = 1;

    I const max_blocks = num_cu;
    I const nbx = 1; // note nbx == 1 is necessary for corretness
    I const nby = std::min(max_blocks, ceil(n, ny));
    I const nbz = std::min(max_blocks, batch_count);

    cal_gnorm_kernel<T, I, Istride>
        <<<dim3(nbx, nby, nbz), dim3(nx, ny, nz), lds_size, stream>>>(m, n, batch_count,

                                                                      A, shiftA, lda, strideA,

                                                                      gnorm_array);
}

// -------------------------------------
// scale an array
// launch as dim3(nbx,1,1), dim3(nx,1,1)
// -------------------------------------
template <typename S, typename I, typename Istride>
static __global__ void scale_kernel(I const batch_count, S const dscale, S* const gnorm_array)
{
    I const bid_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const bid_inc = blockDim.x * gridDim.x;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        gnorm_array[bid] *= dscale;
    }
}

// ---------------------------------------
// routine to compute the sigma values
//
// sigma values computed based on paper
// "An improved Shifted CholeskyQR based on columns"
// by Yuwei Fan, Haoran Guan, Zhonghua Qiao
//
// s = 11 * n * u (m + (n+1) ) * gnorm(A)^2
// ---------------------------------------
template <typename T, typename I, typename Istride, typename UA, typename S = decltype(std::real(T{}))>
static void cal_sigma(hipStream_t stream,
                      I const m,
                      I const n,
                      I const batch_count,

                      UA A,
                      Istride const shiftA,
                      I const lda,
                      Istride const strideA,

                      S* const sigma_array,
                      void* work,
                      size_t const size_work)
{
    // -----------------
    // compute square of gnorm
    // reuse sigma_array
    // -----------------
    auto const gnorm_array = sigma_array;
    cal_gnorm<T, I, Istride>(stream, m, n, batch_count,

                             A, shiftA, lda, strideA,

                             gnorm_array);

    auto ceil = [](auto n, auto base) { return ((n - 1) / base + 1); };

    S const dscale = std::numeric_limits<S>::epsilon();

    {
        I const warp_size = get_warp_size();
        I const nx = warp_size;
        I const nbx = ceil(batch_count, nx);
        scale_kernel<S, I><<<dim3(nbx, 1, 1), dim3(nx, 1, 1), 0, stream>>>(

            batch_count, dscale, gnorm_array);
    }
}

// ---------------------------------
// kernel to perform B <- B + sigma * identity
//
// launch as dim3(nbx,1,batch_count), dim3(nx,1,1)
//
// ---------------------------------
template <typename T, typename I, typename Istride, typename UB, typename S = decltype(std::real(T{}))>
static __global__ void add_shift_kernel(I const m,
                                        I const n,
                                        I const batch_count,
                                        S* const sigma_array,

                                        UB B,
                                        Istride const shiftB,
                                        I const ldb,
                                        Istride const strideB

)
{
    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1) && (sigma_array != nullptr);
        if(!has_work)
        {
            return;
        };
    }

    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const i_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const i_inc = blockDim.x * gridDim.x;

    auto const idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    I const min_mn = std::min(m, n);

    S const zero = 0;
    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        auto const Bp = load_ptr_batch(B, bid, shiftB, strideB);
        S const sigma_bid = (sigma_array == nullptr) ? zero : sigma_array[bid];

        // ----------------------------
        // note: ignore negative shifts
        // ----------------------------
        S const sigma = std::max(sigma_bid, zero);

        if(sigma != zero)
        {
            for(I i = i_start; i < min_mn; i += i_inc)
            {
                // diagonal entry
                I const j = i;

                auto const ij = idx2D(i, j, ldb);
                Bp[ij] += sigma;
            }
        }
    }
}

// --------------------------------------------
// routine to perform B <- B + sigma * identity
// --------------------------------------------
template <typename T, typename I, typename Istride, typename UB, typename S = decltype(std::real(T{}))>
static void add_shift(hipStream_t stream,
                      I const m,
                      I const n,
                      I const batch_count,
                      S* const sigma_array,

                      UB B,
                      Istride const shiftB,
                      I const ldb,
                      Istride const strideB

)
{
    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1) && (sigma_array != nullptr);
        if(!has_work)
        {
            return;
        };
    }

    auto const ceil = [](auto n, auto base) { return ((n - 1) / base + 1); };

    I const nx = 64;
    I const ny = 1;
    I const nz = 1;

    auto const num_cu = get_num_cu();
    I const max_blocks = num_cu;
    I const min_mn = std::min(m, n);

    I const nbx = std::min(max_blocks, ceil(min_mn, nx));
    I const nby = 1;
    I const nbz = std::min(max_blocks, batch_count);

    add_shift_kernel<T, I, Istride, decltype(B), S>
        <<<dim3(nbx, nby, nbz), dim3(nx, ny, nz), 0, stream>>>(m, n, batch_count, sigma_array,

                                                               B, shiftB, ldb, strideB);
}

template <typename T, typename I, typename Istride, typename UA>
static __global__ void set_triangular_kernel(char const uplo,
                                             I const m,
                                             I const n,
                                             T const alpha,

                                             UA A_arg,
                                             Istride shiftA,
                                             I const lda,
                                             Istride const strideA,

                                             I const batch_count)
{
    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return;
        }
    }
    bool const set_strictly_lower = (uplo == 'L') || (uplo == 'l');
    bool const set_strictly_upper = (uplo == 'U') || (uplo == 'u');
    bool const set_all = (!set_strictly_lower) && (!set_strictly_upper);

    I const i_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const j_start = threadIdx.y + blockIdx.y * blockDim.y;

    I const i_inc = blockDim.x * gridDim.x;
    I const j_inc = blockDim.y * gridDim.y;

    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T* const A = load_ptr_batch<T>(A_arg, bid, shiftA, strideA);

        for(I j = j_start; j < n; j += j_inc)
        {
            for(I i = i_start; i < m; i += i_inc)
            {
                bool const is_strictly_lower = (i > j);
                bool const is_strictly_upper = (j < i);

                bool const do_assign = set_all || (set_strictly_lower && is_strictly_lower)
                    || (set_strictly_upper && is_strictly_upper);
                if(do_assign)
                {
                    auto const ij = idx2D(i, j, lda);
                    A[ij] = alpha;
                }
            }
        }

    } // end for bid
}

// -----------------------------------------------------
// convert from strided batched storage to pointer batch
//
// launch as <<< dim3( nbx, 1, 1), dim3(nx,1,1), 0, stream >>>
// where nbx = ceil( batch_count, nx )
// -----------------------------------------------------
template <typename T, typename I, typename Istride, typename UB>
__global__ static void copy_array_to_ptrs_kernel(I batch_count,

                                                 UB B,
                                                 Istride const shiftB,
                                                 I const ldb,
                                                 Istride const strideB,

                                                 T** const B_ptr)
{
    I const bid_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const bid_inc = blockDim.x * gridDim.x;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        B_ptr[bid] = load_ptr_batch(B, bid, shiftB, strideB);
    }
}

template <typename T, typename I, typename Istride>
static void copy_array_to_ptr(hipStream_t stream,

                              I const batch_count,

                              T* const B,
                              Istride const shiftB,
                              I const ldb,
                              Istride const strideB,

                              T** const B_ptr)
{
    auto ceil = [](auto n, auto base) { return ((n - 1) / base + 1); };

    // -----------------------------------------------
    // convert from strided batched to pointer batched
    // -----------------------------------------------
    auto const warp_size = get_warp_size();
    auto const nx = warp_size;
    auto const nbx = ceil(batch_count, nx);

    copy_array_to_ptrs_kernel<T, I, Istride><<<dim3(nbx, 1, 1), dim3(nx, 1, 1), 0, stream>>>(
        batch_count, B, shiftB, ldb, strideB, B_ptr);
}

template <typename T, typename I, typename Istride, typename UA>
static void set_triangular(hipStream_t stream,
                           char const uplo,
                           I const m,
                           I const n,
                           T const alpha,

                           UA A_arg,
                           Istride const shiftA,
                           I const lda,
                           Istride const strideA,
                           I const batch_count)
{
    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return;
        }
    }

    I const nx = 32;
    I const ny = 32;

    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    I const max_blocks = 1024;
    I const nbx = std::min(max_blocks, ceil(m, nx));
    I const nby = std::min(max_blocks, ceil(n, ny));
    I const nbz = std::max(I{1}, std::min(max_blocks, batch_count));

    set_triangular_kernel<<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream>>>(uplo, m, n, alpha,

                                                                               A_arg, shiftA, lda,
                                                                               strideA,

                                                                               batch_count);
}

template <typename I>
rocblas_status
    rocsolver_cholqr1_argCheck(rocblas_handle handle, I const m, I const n, I const lda, I const ldr)
{
    bool const isok = (m >= 0) && (n >= 0) && (lda >= m) && (ldr >= n);
    return (isok ? rocblas_status_continue : rocblas_status_invalid_value);
}

template <typename T, typename I>
rocblas_status rocsolver_cholqr1_strided_batched_argCheck(rocblas_handle handle,

                                                          I const m,
                                                          I const n,
                                                          I const lda,
                                                          I const ldr,

                                                          T* const A,
                                                          T* const R,

                                                          I const batch_count)
{
    bool const isok_values = (m >= 0) && (n >= 0) && (batch_count >= 0) && (lda >= m) && (ldr >= n);
    if(!isok_values)
    {
        return (rocblas_status_invalid_value);
    }

    bool const isok_pointer = (A != nullptr) && (R != nullptr);
    if(!isok_pointer)
    {
        return (rocblas_status_invalid_pointer);
    }

    return (rocblas_status_continue);
}

template <typename T, typename I>
rocblas_status rocsolver_cholqr2_strided_batched_argCheck(rocblas_handle handle,

                                                          I const m,
                                                          I const n,
                                                          I const lda,
                                                          I const ldr,

                                                          T* const A,
                                                          T* const R,

                                                          I const batch_count)
{
    bool const isok_values = (m >= 0) && (n >= 0) && (batch_count >= 0) && (lda >= m) && (ldr >= n);
    if(!isok_values)
    {
        return (rocblas_status_invalid_value);
    }

    bool const isok_pointer = (A != nullptr) && (R != nullptr);
    if(!isok_pointer)
    {
        return (rocblas_status_invalid_pointer);
    }

    return (rocblas_status_continue);
}

template <typename I, typename UA, typename UR>
rocblas_status rocsolver_cholqr1_batched_argCheck(rocblas_handle handle,

                                                  I const m,
                                                  I const n,
                                                  I const lda,
                                                  I const ldr,

                                                  UA A,
                                                  UR R,

                                                  I const batch_count)
{
    bool const isok_values = (m >= 0) && (n >= 0) && (batch_count >= 0) && (lda >= m) && (ldr >= n);
    if(!isok_values)
    {
        return (rocblas_status_invalid_value);
    }

    bool const isok_pointer = (A != nullptr) && (R != nullptr);
    if(!isok_pointer)
    {
        return (rocblas_status_invalid_pointer);
    }

    return (rocblas_status_continue);
}

template <typename I>
rocblas_status
    rocsolver_cholqr2_argCheck(rocblas_handle handle, I const m, I const n, I const lda, I const ldr)
{
    bool const isok = (m >= 0) && (n >= 0) && (lda >= m) && (ldr >= n);
    return (isok ? rocblas_status_continue : rocblas_status_invalid_value);
}

template <typename T, typename I>
void rocsolver_cholqr1_getMemorySize(I const m, I const n, I const batch_count, size_t* p_size_work)
{
    size_t size_work = 0;
    *p_size_work = 0;

    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return;
        }
    }

    // ----------------------------------
    // storage for computing  B = A' * A
    // ----------------------------------
    size_t size_syrk_herk = sizeof(T*) * batch_count;
    {
        // ------------------------
        // storage for A_ptr, B_ptr
        // ------------------------
        size_t const size_A_ptr = sizeof(T*) * batch_count;
        size_t const size_B_ptr = sizeof(T*) * batch_count;
        size_syrk_herk += size_A_ptr + size_B_ptr;
    }

    // ----------------------------------------------
    // storage for Cholesky factorization R = chol(B)
    // ----------------------------------------------
    size_t size_potrf = 0;
    {
        rocblas_fill const uplo = rocblas_fill_upper;

        size_t size_scalars = 0;
        size_t size_work1 = 0;
        size_t size_work2 = 0;
        size_t size_work3 = 0;
        size_t size_work4 = 0;
        size_t size_pivots = 0;
        size_t size_iinfo = 0;

        bool optim_mem = true;
        bool constexpr LBATCHED = true;
        bool constexpr LSTRIDED = false;
        rocsolver_potrf_getMemorySize<LBATCHED, LSTRIDED, T, I>(
            n, uplo, batch_count,

            &size_scalars, &size_work1, &size_work2, &size_work3, &size_work4, &size_pivots,
            &size_iinfo, &optim_mem);

        size_potrf += size_scalars + size_work1 + size_work2 + size_work3 + size_work4 + size_pivots
            + size_iinfo;
    }

    // -------------------------------
    // storage for computing Q = A / R
    // -------------------------------
    size_t size_trsm = 0;
    {
        size_t size_x_temp = 0;
        size_t size_x_temp_arr = 0;
        size_t size_invA = 0;
        size_t size_invA_arr = 0;

        bool optimal_mem = true;

        auto const mm = m;
        auto const nn = n;
        auto const ld1 = m;
        auto const ld2 = n;

        rocblas_side const side = rocblas_side_right;
        rocblas_fill const uplo = rocblas_fill_upper;
        rocblas_diagonal const diag = rocblas_diagonal_non_unit;
        rocblas_operation const trans1 = rocblas_operation_none;

        bool constexpr LBATCHED = true;
        auto const istat_trsm_mem = rocblasCall_trsm_mem<LBATCHED, T, I>(
            side, trans1, mm, nn, ld1, ld2, batch_count, &size_x_temp, &size_x_temp_arr, &size_invA,
            &size_invA_arr);

        bool const isok_trsm_mem = (istat_trsm_mem == rocblas_status_success)
            || (istat_trsm_mem == rocblas_status_continue);
        assert(isok_trsm_mem);

        size_trsm += size_x_temp + size_x_temp_arr + size_invA + size_invA_arr;
    }
    {
        size_t const size_A_ptr = sizeof(T*) * batch_count;
        size_t const size_B_ptr = sizeof(T*) * batch_count;
        size_trsm += size_A_ptr + size_B_ptr;
    }

    size_work = std::max(size_work, size_syrk_herk);
    size_work = std::max(size_work, size_potrf);
    size_work = std::max(size_work, size_trsm);

    *p_size_work = size_work;
}

template <typename T, typename I>
void rocsolver_cholqr2_getMemorySize(I const m, I const n, I const batch_count, size_t* p_size_work)
{
    using INFO = I;

    size_t size_work = 0;
    *p_size_work = 0;

    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return;
        }
    }

    // --------------------------------
    // storage for computing [Q,R1] = cholqr1(A)
    // --------------------------------
    I const ldr = n;
    size_t const strideR = ldr * n;
    size_t const size_R1 = sizeof(T) * strideR * batch_count;
    size_work += size_R1;

    // --------------------------------------------
    // storage for iinfo for 2nd call to cholqr1(A)
    // --------------------------------------------
    size_t const size_iinfo = sizeof(INFO) * batch_count;
    size_work += size_iinfo;

    size_t size_cholqr1 = 0;
    {
        rocsolver_cholqr1_getMemorySize<T, I>(m, n, batch_count, &size_cholqr1);
    }
    size_work += size_cholqr1;

    size_t const size_trmm = sizeof(T*) * batch_count;
    size_work += size_trmm;

    *p_size_work = size_work;
}

// -------------------------------------------------
// compute A = Q * R,  using Cholesky factorization
//
// B = A' * A
//
// R = chol(B)
//
// Q = A / R
//
// Q will over-write A
// -------------------------------------------------
template <typename T,
          typename I,
          typename Istride,
          typename UA,
          typename UR,
          typename INFO = I,
          typename S = decltype(std::real(T{}))>
rocblas_status rocsolver_cholqr1_template(

    rocblas_handle handle,

    I const m,
    I const n,

    UA A,
    Istride const shiftA,
    I const lda,
    Istride strideA,

    UR R,
    Istride const shiftR,
    I const ldr,
    Istride strideR,

    I const batch_count,

    I* const info,

    void* work,
    size_t const size_work,

    S* const sigma_array = nullptr)
{
    bool constexpr is_complex = rocblas_is_complex<T>;

    bool constexpr is_pointer_batched_A = IS_POINTER_BATCHED(A);
    bool constexpr is_pointer_batched_R = IS_POINTER_BATCHED(R);

    bool constexpr is_both_same_type = (is_pointer_batched_A == is_pointer_batched_R);

    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return (rocblas_status_success);
        }
    }

    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    int constexpr idebug = 0;
    auto print_mat = [=](auto name, auto m, auto n, auto A, auto shiftA, auto lda, auto strideA) {
        std::vector<T> h_A(lda * n);
        I const bid = 0;
        T* Ap = load_ptr_batch(A, bid, shiftA, strideA);

        auto const istat = hipMemcpy(&(h_A[0]), Ap, sizeof(T) * lda * n, hipMemcpyDeviceToHost);
        assert(istat == hipSuccess);

        for(auto j = 0; j < n; j++)
        {
            for(auto i = 0; i < m; i++)
            {
                auto const ij = i + j * static_cast<int64_t>(lda);
                auto const aij = h_A[ij];
                std::cout << name << "(" << i + 1 << "," << j + 1 << ") = " << aij << ";"
                          << std::endl;
            }
        }
    };

    {
        // ----------
        // reset info
        // ----------
        auto const istat = hipMemsetAsync(info, 0, sizeof(INFO) * batch_count, stream);
        if(istat != hipSuccess)
        {
            std::cout << "internal error: FILE " << __FILE__ << " line " << __LINE__ << std::endl;
            return (rocblas_status_internal_error);
        }
    }

    std::byte* const pwork = (std::byte*)work;
    std::byte* pfree = pwork;

    // -------------------------------------
    // form B = A' * A,  reuse storage for R
    // -------------------------------------
    auto const B = R;
    Istride const shiftB = shiftR;
    Istride const strideB = strideR;
    I const ldb = ldr;

    bool constexpr is_pointer_batched_B = is_pointer_batched_R;

    {
        auto const pfree_saved = pfree;

        rocblas_operation const trans1
            = (is_complex) ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose;

        size_t size_work_arr = sizeof(T*) * batch_count;

        T** const work_arr = (T**)pfree;
        pfree += size_work_arr;

        bool is_mem_ok = (pfree <= (pwork + size_work));
        if(!is_mem_ok)
        {
            return (rocblas_status_memory_error);
        }

        // -------------------------------
        // Note output  matrix for SYRK is nn by nn
        // -------------------------------
        I const nn = n;
        I const kk = m;
        rocblas_fill const uplo = rocblas_fill_upper;

        S alpha = 1;
        S beta = 0;

        if(idebug >= 1)
        {
            std::cout << "before SYRK" << std::endl;
            print_mat("A", m, n, A, shiftA, lda, strideA);
        }

        // ------------------
        // compute B = A' * A
        // ------------------

        rocblas_status istat = rocblas_status_success;

        if constexpr(is_both_same_type)
        {
            bool constexpr LBATCHED = is_pointer_batched_A;
            istat = rocblasCall_syrk_herk<LBATCHED, T>(handle, uplo, trans1,

                                                       nn, kk,

                                                       &alpha,

                                                       A, shiftA, lda, strideA,

                                                       &beta,

                                                       B, shiftB, ldb, strideB,

                                                       batch_count, work_arr);
        }
        else
        {
            auto const pfree_saved = pfree;
            // -------------------------
            // mixed pointer and strided
            // convert to using both pointer batched
            // -------------------------
            T** B_ptr = nullptr;
            T** A_ptr = nullptr;

            if constexpr(is_pointer_batched_A)
            {
                A_ptr = A;
            }
            else
            {
                size_t const size_A_ptr = sizeof(T*) * batch_count;
                A_ptr = (T**)pfree;
                pfree += size_A_ptr;

                copy_array_to_ptr<T, I, Istride>(stream, batch_count,

                                                 A, shiftA, lda, strideA,

                                                 A_ptr);
            }

            if constexpr(is_pointer_batched_B)
            {
                B_ptr = B;
            }
            else
            {
                size_t const size_B_ptr = sizeof(T*) * batch_count;

                B_ptr = (T**)pfree;
                pfree += size_B_ptr;

                copy_array_to_ptr<T, I, Istride>(stream, batch_count,

                                                 B, shiftB, ldb, strideB,

                                                 B_ptr);
            }

            // ------------
            // check memory
            // ------------
            bool const is_mem_ok = (pfree <= (pwork + size_work));
            if(!is_mem_ok)
            {
                return (rocblas_status_memory_error);
            };

            bool constexpr LBATCHED = true;
            istat = rocblasCall_syrk_herk<LBATCHED, T>(handle, uplo, trans1,

                                                       nn, kk,

                                                       &alpha,

                                                       A_ptr, shiftA, lda, strideA,

                                                       &beta,

                                                       B_ptr, shiftB, ldb, strideB,

                                                       batch_count, work_arr);

            pfree = pfree_saved;
        }

        if(istat != rocblas_status_success)
        {
            return (istat);
        }

        if(idebug >= 1)
        {
            std::cout << "after SYRK" << std::endl;
            print_mat("B", n, n, B, shiftB, ldb, strideB);
        }

        pfree = pfree_saved;
    }

    // -------------------------
    // optional, if sigma != 0
    // B <- B + sigma * identity
    // -------------------------
    if(sigma_array != nullptr)
    {
        add_shift<T, I, Istride, decltype(B), S>(stream, m, n, batch_count, sigma_array,

                                                 B, shiftB, ldb, strideB);
    }

    // -----------------------------------
    // perform Cholesky factorization
    // B = R' * R,   R is upper triangular
    //
    // R will over-write B
    // -----------------------------------
    {
        auto const pfree_saved = pfree;

        rocblas_fill const uplo = rocblas_fill_upper;

        size_t size_scalars = 0;
        size_t size_work1 = 0;
        size_t size_work2 = 0;
        size_t size_work3 = 0;
        size_t size_work4 = 0;
        size_t size_pivots = 0;
        size_t size_iinfo = 0;

        bool optim_mem = true;

        bool constexpr LBATCHED = is_pointer_batched_B;
        bool constexpr LSTRIDED = !LBATCHED;
        rocsolver_potrf_getMemorySize<LBATCHED, LSTRIDED, T, I>(
            n, uplo, batch_count,

            &size_scalars, &size_work1, &size_work2, &size_work3, &size_work4, &size_pivots,
            &size_iinfo, &optim_mem);

        // -----------------------------------------------------
        // allocate temporary storage for Cholesky factorization
        // -----------------------------------------------------

        T* const scalars = (T*)pfree;
        pfree += size_scalars;
        T* const pivots = (T*)pfree;
        pfree += size_pivots;
        INFO* const iinfo = (INFO*)pfree;
        pfree += size_iinfo;

        void* const work1 = (void*)pfree;
        pfree += size_work1;
        void* const work2 = (void*)pfree;
        pfree += size_work2;
        void* const work3 = (void*)pfree;
        pfree += size_work3;
        void* const work4 = (void*)pfree;
        pfree += size_work4;

        bool const is_mem_ok = (pfree <= (pwork + size_work));
        if(!is_mem_ok)
        {
            return (rocblas_status_memory_error);
        }

        auto const istat = rocsolver_potrf_template<LBATCHED, LSTRIDED, T, I, INFO, S>(
            handle, uplo, n,

            B, shiftB, ldb, strideB,

            info, batch_count,

            scalars, work1, work2, work3, work4, pivots, iinfo, optim_mem);

        if(istat != rocblas_status_success)
        {
            return (istat);
        }

        if(idebug >= 1)
        {
            std::cout << "after POTRF" << std::endl;
            print_mat("B", n, n, B, shiftB, ldb, strideB);
        }

        pfree = pfree_saved;
    }

    // --------------------------------------
    // compute Q = A / R
    //
    // note Q over-writes original matrix A
    // --------------------------------------

    {
        auto const pfree_saved = pfree;

        size_t size_x_temp = 0;
        size_t size_x_temp_arr = 0;
        size_t size_invA = 0;
        size_t size_invA_arr = 0;

        bool optimal_mem = true;

        auto const mm = m;
        auto const nn = n;
        auto const ld1 = lda;
        auto const ld2 = ldr;

        rocblas_side const side = rocblas_side_right;
        rocblas_fill const uplo = rocblas_fill_upper;
        rocblas_diagonal const diag = rocblas_diagonal_non_unit;
        rocblas_operation const trans1 = rocblas_operation_none;

        bool constexpr LBATCHED = is_pointer_batched_A;

        auto const istat_trsm_mem = rocblasCall_trsm_mem<LBATCHED, T, I>(
            side, trans1, mm, nn, ld1, ld2, batch_count,

            &size_x_temp, &size_x_temp_arr, &size_invA, &size_invA_arr);

        bool const isok_trsm_mem = (istat_trsm_mem == rocblas_status_success)
            || (istat_trsm_mem == rocblas_status_continue);

        if(idebug >= 1)
        {
            std::cout << "after TRSM_mem "
                      << " istat_trsm_mem = " << istat_trsm_mem << std::endl;
        }

        if(!isok_trsm_mem)
        {
            return (istat_trsm_mem);
        }

        // -------------------------------------------------
        // allocate scratch memory for trsm triangular solve
        // -------------------------------------------------

        void* const x_temp = (void*)pfree;
        pfree += size_x_temp;
        void* const x_temp_arr = (void*)pfree;
        pfree += size_x_temp_arr;
        void* const invA = (void*)pfree;
        pfree += size_invA;
        void* const invA_arr = (void*)pfree;
        pfree += size_invA_arr;

        bool is_mem_ok = (pfree <= (pwork + size_work));
        if(!is_mem_ok)
        {
            return (rocblas_status_memory_error);
        }

        T alpha = 1;
        rocblas_status istat_trsm = rocblas_status_success;

        if constexpr(is_both_same_type)
        {
            istat_trsm = rocblasCall_trsm(handle,

                                          side, uplo, trans1, diag,

                                          mm, nn, &alpha,

                                          R, shiftR, ldr, strideR,

                                          A, shiftA, lda, strideA,

                                          batch_count, optimal_mem,

                                          x_temp, x_temp_arr, invA, invA_arr);
        }
        else
        {
            T** R_ptr = nullptr;
            T** A_ptr = nullptr;

            if constexpr(is_pointer_batched_A)
            {
                A_ptr = A;
            }
            else
            {
                size_t const size_A_ptr = sizeof(T*) * batch_count;

                A_ptr = (T**)pfree;
                pfree += size_A_ptr;

                copy_array_to_ptr<T, I, Istride>(stream, batch_count,

                                                 A, shiftA, lda, strideA,

                                                 A_ptr);
            }

            if constexpr(is_pointer_batched_R)
            {
                R_ptr = R;
            }
            else
            {
                size_t const size_R_ptr = sizeof(T*) * batch_count;

                R_ptr = (T**)pfree;
                pfree += size_R_ptr;

                copy_array_to_ptr<T, I, Istride>(stream, batch_count,

                                                 R, shiftR, ldr, strideR,

                                                 R_ptr);
            }

            istat_trsm = rocblasCall_trsm(handle,

                                          side, uplo, trans1, diag,

                                          mm, nn, &alpha,

                                          R_ptr, shiftR, ldr, strideR,

                                          A_ptr, shiftA, lda, strideA,

                                          batch_count, optimal_mem,

                                          x_temp, x_temp_arr, invA, invA_arr);
        }

        bool const isok_trsm
            = (istat_trsm == rocblas_status_success) || (istat_trsm == rocblas_status_continue);
        if(!isok_trsm)
        {
            return (istat_trsm);
        }

        if(idebug >= 1)
        {
            std::cout << "after TRSM " << std::endl;
            print_mat("Q", m, n, A, shiftA, lda, strideA);
        }

        pfree = pfree_saved;
    }

    return (rocblas_status_success);
}

// -----------------------------------------------------
// perform QR factorization using cholesky factorization
// (1) [Q,R1] = cholqr1( A )
// (2) [Q,R] = cholqr1( Q )
// (3) R = R * R1
//
//
// “Roundoff error analysis of the CholeskyQR2 algorithm”,
// by Yamamoto et al, Electronic Transactions on Numerical Analysis,
// Vol 44, p 306-326, 2015.
// -----------------------------------------------------

template <typename T, typename I, typename Istride, typename UA, typename UR, typename INFO = I>
rocblas_status rocsolver_cholqr2_template(rocblas_handle handle,
                                          I const m,
                                          I const n,

                                          UA A,
                                          Istride const shiftA,
                                          I const lda,
                                          Istride strideA,

                                          UR R,
                                          Istride const shiftR,
                                          I const ldr,
                                          Istride strideR,

                                          I const batch_count,
                                          INFO* const info,

                                          void* work,
                                          size_t const size_work)
{
    bool constexpr is_pointer_batched_A = IS_POINTER_BATCHED(A);
    bool constexpr is_pointer_batched_R = IS_POINTER_BATCHED(R);

    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return (rocblas_status_success);
        };
    }

    hipStream_t stream;
    {
        auto const istat = rocblas_get_stream(handle, &stream);
        if(istat != rocblas_status_success)
        {
            return (istat);
        }
    }

    {
        // ----------
        // reset info
        // ----------
        auto const istat = hipMemsetAsync(info, 0, sizeof(INFO) * batch_count, stream);
        if(istat != hipSuccess)
        {
            std::cout << "internal error: FILE " << __FILE__ << " line " << __LINE__ << std::endl;
            return (rocblas_status_internal_error);
        }
    }

    std::byte* const pwork = (std::byte*)work;
    std::byte* pfree = pwork;

    // -------------------------
    // (1) [Q,R1] = cholqr1( A )
    // -------------------------
    I const ldr1 = n;
    size_t const size_R1 = sizeof(T) * ldr1 * n * batch_count;
    Istride const shiftR1 = 0;
    Istride const strideR1 = static_cast<Istride>(ldr1) * n;

    T* const R1 = (T*)pfree;
    pfree += size_R1;

    {
        auto const pfree_saved = pfree;
        size_t const lwork_remain = (pwork + size_work) - pfree;

        {
            size_t size_cholqr1 = 0;

            rocsolver_cholqr1_getMemorySize<T, I>(m, n, batch_count, &size_cholqr1);

            bool const is_memory_ok = (lwork_remain >= size_cholqr1);
            if(!is_memory_ok)
            {
                std::cout << "memory for 1st cholqr1 "
                          << " lwork_remain = " << lwork_remain
                          << " size_cholqr1 = " << size_cholqr1 << std::endl;

                return (rocblas_status_memory_error);
            }
        }

        auto const istat = rocsolver_cholqr1_template<T, I, Istride>(handle,

                                                                     m, n,

                                                                     A, shiftA, lda, strideA,

                                                                     R1, shiftR1, ldr1, strideR1,

                                                                     batch_count, info,

                                                                     (void*)pfree, lwork_remain);

        if(istat != rocblas_status_success)
        {
            return (istat);
        }
        pfree = pfree_saved;
    }

    // ------------------------
    // (2) [Q,R] = cholqr1( Q )
    // ------------------------

    // -----------------------------
    // Note: matrix Q over-writes matrix A
    // -----------------------------
    auto const Q = A;
    auto const shiftQ = shiftA;
    auto const ldq = lda;
    auto const strideQ = strideA;
    {
        auto const pfree_saved = pfree;
        size_t const size_iinfo = sizeof(INFO) * batch_count;
        INFO* iinfo = (INFO*)pfree;
        pfree += size_iinfo;

        bool const is_memory_ok = (pfree <= (pwork + size_work));
        if(!is_memory_ok)
        {
            std::cout << "memory for 2nd cholqr1 "
                      << " pfree = " << pfree << " pwork = " << pwork
                      << " size_work = " << size_work << std::endl;

            return (rocblas_status_memory_error);
        }

        size_t const lwork_remain = (pwork + size_work) - pfree;

        auto const istat = rocsolver_cholqr1_template<T, I, Istride>(handle,

                                                                     m, n,

                                                                     Q, shiftQ, ldq, strideQ,

                                                                     R, shiftR, ldr, strideR,

                                                                     batch_count, iinfo,

                                                                     (void*)pfree, lwork_remain);

        if(istat != rocblas_status_success)
        {
            return (istat);
        }

        pfree = pfree_saved;
    }

    // --------------
    // (3) R = R * R1
    // --------------

    // -------------------
    // update  R <- R * R1
    //
    // (i)  set strictly lower triangular part of R be zero
    // (ii) perform TRMM
    // -------------------

    {
        // ----------------------------------------------------
        // (i)  set strictly lower triangular part of R be zero
        // ----------------------------------------------------

        char uplo = 'L';
        T alpha = 0;
        set_triangular(stream, uplo, n, n, alpha,

                       R, shiftR, ldr, strideR,

                       batch_count);
    }

    {
        // -----------------
        // (ii) perform TRMM
        // R <- R * R1
        // -----------------

        auto const pfree_saved = pfree;

        rocblas_side const side = rocblas_side_right;
        rocblas_fill const uplo = rocblas_fill_upper;
        rocblas_operation const trans1 = rocblas_operation_none;
        rocblas_diagonal const diag = rocblas_diagonal_non_unit;
        I const mm = n;
        I const nn = n;
        T alpha = 1;

        size_t const size_workArr = sizeof(T*) * batch_count;
        T** const workArr = (T**)pfree;
        pfree += size_workArr;

        bool const is_memory_ok = (pfree <= (pwork + size_work));
        if(!is_memory_ok)
        {
            std::cout << "mem for TRMM: pfree = " << pfree << " pwork = " << pwork
                      << " size_work = " << size_work << std::endl;

            return (rocblas_status_memory_error);
        }

        Istride const stride_alpha = 0;

        auto const istat = rocblasCall_trmm<T>(

            handle,

            side, uplo, trans1, diag,

            mm, nn,

            &alpha, stride_alpha,

            R1, shiftR1, ldr1, strideR1,

            R, shiftR, ldr, strideR,

            batch_count, workArr);

        if(istat != rocblas_status_success)
        {
            return (istat);
        }

        pfree = pfree_saved;
    }

    return (rocblas_status_success);
}

//
// shifted CholeskyQR3
//
// (1)  R1 * R1' = A' * A + s * I, where s is the shift
// (2)  Q1 = A/R1
// (3)  [Q2, R2]  = cholQR2(Q1)
// (4)  R = R2 * R1
//
// “Shifted CholeskyQR for computing QR factorization of
// ill-conditioned matrices”, Fukaya et al,
// SIAM J Sci Comp, Vol 42, No 1, pp A477-A503, 2020
//
// “An improved Shifted CholeskyQR based on columns”,
// by Fan et al, arXiv:2408.06311v4 [math.NA] 07 Feb 2025
//
template <typename T,
          typename I,
          typename Istride,
          typename UA,
          typename UR,
          typename INFO = I,
          typename S = decltype(std::real(T{}))>
rocblas_status rocsolver_cholqr3_template(rocblas_handle handle,
                                          I const m,
                                          I const n,

                                          UA A,
                                          Istride const shiftA,
                                          I const lda,
                                          Istride strideA,

                                          UR R,
                                          Istride const shiftR,
                                          I const ldr,
                                          Istride strideR,

                                          I const batch_count,
                                          INFO* const info,

                                          void* work,
                                          size_t const size_work)
{
    bool constexpr is_pointer_batched_A = IS_POINTER_BATCHED(A);
    bool constexpr is_pointer_batched_R = IS_POINTER_BATCHED(R);

    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return (rocblas_status_success);
        }
    }

    hipStream_t stream;
    {
        auto const istat = rocblas_get_stream(handle, &stream);
        if(istat != rocblas_status_success)
        {
            return (istat);
        }
    }

    {
        // ----------
        // reset info
        // ----------
        auto const istat = hipMemsetAsync(info, 0, sizeof(INFO) * batch_count, stream);
        if(istat != hipSuccess)
        {
            std::cout << "internal error: FILE " << __FILE__ << " line " << __LINE__ << std::endl;
            return (rocblas_status_internal_error);
        }
    }

    std::byte* const pwork = (std::byte*)work;
    std::byte* pfree = pwork;

    // -----------------------------
    // Note: matrix Q over-writes matrix A
    // -----------------------------
    auto const Q = A;
    auto const shiftQ = shiftA;
    auto const ldq = lda;
    auto const strideQ = strideA;

    I const ldr1 = n;
    Istride const strideR1 = static_cast<Istride>(ldr1) * n;
    Istride const shiftR1 = 0;
    size_t const size_R1 = sizeof(T) * strideR1 * batch_count;
    T* const R1 = (T*)pfree;
    pfree += size_R1;

    {
        bool const is_mem_ok = (pfree <= (pwork + size_work));
        if(!is_mem_ok)
        {
            return (rocblas_status_memory_error);
        }
    }

    // ---------------------------------------
    // (1)  R1 * R1' = A'*A + sigma * identity
    // ---------------------------------------

    {
        auto const pfree_saved = pfree;
        size_t const size_iinfo = sizeof(INFO) * batch_count;
        INFO* iinfo = (INFO*)pfree;
        pfree += size_iinfo;

        size_t const size_sigma_array = sizeof(S) * batch_count;
        S* const sigma_array = (S*)pfree;
        pfree += size_sigma_array;

        bool const is_memory_ok = (pfree <= (pwork + size_work));
        if(!is_memory_ok)
        {
            std::cout << "memory for cholqr3 "
                      << " pfree = " << pfree << " pwork = " << pwork
                      << " size_work = " << size_work << std::endl;

            return (rocblas_status_memory_error);
        }

        size_t const lwork_remain = (pwork + size_work) - pfree;

        {
            cal_sigma(stream, m, n, batch_count,

                      A, shiftA, lda, strideA,

                      sigma_array, (void*)pfree, lwork_remain);
        }

        // T const sigma = 11 * (m * n * ueps + (n + 1) * (n * ueps)) * gnorm

        auto const istat
            = rocsolver_cholqr1_template<T, I, Istride>(handle,

                                                        m, n,

                                                        Q, shiftQ, ldq, strideQ,

                                                        R1, shiftR1, ldr1, strideR1,

                                                        batch_count, iinfo,

                                                        (void*)pfree, lwork_remain, sigma_array);

        if(istat != rocblas_status_success)
        {
            return (istat);
        }

        pfree = pfree_saved;
    }

    // ----------------
    // (2)   CholQR2(Q)
    // ----------------

    {
        auto const pfree_saved = pfree;

        size_t const size_remain = (pwork + size_work) - pfree;

        auto const istat_cholqr2
            = rocsolver_cholqr2_template<T, I, Istride>(handle, m, n,

                                                        Q, shiftQ, ldq, strideQ,

                                                        R, shiftR, ldr, strideR,

                                                        batch_count, info,

                                                        (void*)pfree, size_remain);

        if(istat_cholqr2 != rocblas_status_success)
        {
            return (istat_cholqr2);
        }

        pfree = pfree_saved;
    }

    // -------------------------------
    // compute R <- R * R1, using TRMM
    // -------------------------------

    {
        // ----------------------------------------------------
        // (i)  set strictly lower triangular part of R be zero
        // ----------------------------------------------------

        char uplo = 'L';
        T alpha = 0;
        set_triangular(stream, uplo, n, n, alpha,

                       R, shiftR, ldr, strideR,

                       batch_count);
    }

    {
        // -----------------
        // (ii) perform TRMM
        // R <- R * R1
        // -----------------

        auto const pfree_saved = pfree;

        rocblas_side const side = rocblas_side_right;
        rocblas_fill const uplo = rocblas_fill_upper;
        rocblas_operation const trans1 = rocblas_operation_none;
        rocblas_diagonal const diag = rocblas_diagonal_non_unit;
        I const mm = n;
        I const nn = n;
        T alpha = 1;

        size_t const size_workArr = sizeof(T*) * batch_count;
        T** const workArr = (T**)pfree;
        pfree += size_workArr;

        bool const is_memory_ok = (pfree <= (pwork + size_work));
        if(!is_memory_ok)
        {
            std::cout << "mem for TRMM: pfree = " << pfree << " pwork = " << pwork
                      << " size_work = " << size_work << " file " << __FILE__ << " line "
                      << __LINE__ << std::endl;

            return (rocblas_status_memory_error);
        }

        Istride const stride_alpha = 0;

        auto const istat = rocblasCall_trmm<T>(

            handle,

            side, uplo, trans1, diag,

            mm, nn,

            &alpha, stride_alpha,

            R1, shiftR1, ldr1, strideR1,

            R, shiftR, ldr, strideR,

            batch_count, workArr);

        if(istat != rocblas_status_success)
        {
            return (istat);
        }

        pfree = pfree_saved;
    }
}

ROCSOLVER_END_NAMESPACE
