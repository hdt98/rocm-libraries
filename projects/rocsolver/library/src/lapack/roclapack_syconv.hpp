/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "rocsolver/rocsolver.h"

#include "rocprim/rocprim.hpp"

ROCSOLVER_BEGIN_NAMESPACE

#ifndef SYCONV_MAX_THDS
#define SYCONV_MAX_THDS 256
#endif

template <typename T, typename I, typename UA, typename Istride>
static __global__
    __launch_bounds__(SYCONV_MAX_THDS) void syconv_restore_from_E(bool const is_upper,

                                                                  I const n,
                                                                  I const batch_count,

                                                                  UA A_arg,
                                                                  Istride const shiftA,
                                                                  I const lda,
                                                                  Istride const strideA,

                                                                  I const* const ipiv_arg,
                                                                  Istride const strideP,

                                                                  T const* const E_arg,
                                                                  Istride const strideE,

                                                                  I const* const icount_arg)
{
    {
        bool const has_work_to_do = (n >= 1) && (batch_count >= 1);
        if(!has_work_to_do)
        {
            return;
        }
    }

    auto idx2F
        = [](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

    auto is_even = [](auto n) -> bool { return ((n % 2) == 0); };

    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const nthreads = (blockDim.x * blockDim.y) * blockDim.z;
    I const tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);

    I const ibx = blockIdx.x;
    I const nbx = gridDim.x;

    I const ij_start = tid + ibx * nthreads;
    I const ij_inc = nthreads * nbx;

    for(I bid = 0 + bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const E_bid = E_arg + bid * strideE;
        T* const A_bid = load_ptr_batch<T>(A_arg, bid, shiftA, strideA);
        I const* const icount_bid = icount_arg + bid * n;
        I const* const ipiv_bid = ipiv_arg + bid * strideP;

        auto A = [=](I i, I j) -> T& { return (A_bid[idx2F(i, j, lda)]); };
        auto E = [=](I i) -> T { return (E_bid[(i - 1)]); };
        auto ipiv = [=](I i) { return (ipiv_bid[(i - 1)]); };

        auto is_second_pivot = [=](I i) -> bool { return (is_even(icount_bid[(i - 1)])); };

        if(is_upper)
        {
            // ----------
            // upper part
            // ----------

            for(I i = 1 + ij_start; i <= n; i += ij_inc)
            {
                bool const is_2x2_block = (ipiv(i) < 0);
                if(is_2x2_block)
                {
                    if(is_second_pivot(i))
                    {
                        A(i - 1, i) = E(i);
                    }
                }
            } // end for i
        }
        else
        {
            // ----------
            // lower part
            // ----------

            for(I i = 1 + ij_start; i <= n; i += ij_inc)
            {
                if(ipiv(i) < 0)
                {
                    if(!is_second_pivot(i))
                    {
                        A(i + 1, i) = E(i);
                    }
                }
            } // end for i
        }

        __syncthreads();
    } // end for bid
}

template <typename T, typename I, typename UA, typename Istride>
static __global__
    __launch_bounds__(SYCONV_MAX_THDS) void syconv_convert_into_E(bool const is_upper,

                                                                  I const n,
                                                                  I const batch_count,

                                                                  UA A_arg,
                                                                  Istride const shiftA,
                                                                  I const lda,
                                                                  Istride const strideA,

                                                                  I const* const ipiv_arg,
                                                                  Istride const strideP,

                                                                  T* const E_arg,
                                                                  Istride const strideE,

                                                                  I const* const icount_arg

    )
{
    {
        bool const has_work_to_do = (n >= 1) && (batch_count >= 1);
        if(!has_work_to_do)
        {
            return;
        }
    }

    auto idx2F
        = [](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

    auto is_even = [](auto n) -> bool { return ((n % 2) == 0); };

    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const nthreads = (blockDim.x * blockDim.y) * blockDim.z;
    I const tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);

    I const ibx = blockIdx.x;
    I const nbx = gridDim.x;

    I const ij_start = tid + ibx * nthreads;
    I const ij_inc = nthreads * nbx;

    T const zero = 0;

    for(I bid = 0 + bid_start; bid < batch_count; bid += bid_inc)
    {
        T* const E_bid = E_arg + bid * strideE;
        T* const A_bid = load_ptr_batch<T>(A_arg, bid, shiftA, strideA);
        I const* const icount_bid = icount_arg + bid * n;
        I const* const ipiv_bid = ipiv_arg + bid * strideP;

        auto A = [=](I i, I j) -> T& { return (A_bid[idx2F(i, j, lda)]); };
        auto E = [=](I i) -> T& { return (E_bid[(i - 1)]); };
        auto ipiv = [=](I i) { return (ipiv_bid[(i - 1)]); };

        auto is_second_pivot = [=](I i) -> bool { return (is_even(icount_bid[(i - 1)])); };

        if(is_upper)
        {
            // ----------
            // upper part
            // ----------

            for(I i = 1 + ij_start; i <= n; i += ij_inc)
            {
                bool const use_2x2_block = (ipiv(i) < 0);
                if(use_2x2_block)
                {
                    if(is_second_pivot(i))
                    {
                        E(i) = A(i - 1, i);
                        E(i - 1) = zero;

                        A(i - 1, i) = zero;
                    }
                }
                else
                {
                    E(i) = zero;
                }
            } // end for i
        }
        else
        {
            // ----------
            // lower part
            // ----------

            for(I i = 1 + ij_start; i <= n; i += ij_inc)
            {
                bool const use_2x2_block = ((i < n) && (ipiv(i) < 0));
                if(use_2x2_block)
                {
                    if(!is_second_pivot(i))
                    {
                        E(i) = A(i + 1, i);
                        E(i + 1) = zero;
                        A(i + 1, i) = zero;
                    }
                }
                else
                {
                    E(i) = zero;
                }
            } // end for i
        }

        __syncthreads();
    } // end for bid
}

template <typename I, typename Istride>
static __global__
    __launch_bounds__(SYCONV_MAX_THDS) void syconv_setup_icount_kernel(I const n,
                                                                       I const batch_count,

                                                                       I* const ipiv_arg,
                                                                       Istride const strideP,

                                                                       I* const icount_arg)
{
    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const nthreads = (blockDim.x * blockDim.y) * blockDim.z;
    I const tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);

    I const ibx = blockIdx.x;
    I const nbx = gridDim.x;
    I const i_start = tid + ibx * nthreads;
    I const i_inc = nbx * nthreads;

    for(I bid = 0 + bid_start; bid < batch_count; bid += bid_inc)
    {
        I* const ipiv = ipiv_arg + static_cast<int64_t>(bid) * strideP;
        I* const icount = icount_arg + static_cast<int64_t>(bid) * n;

        for(I i = 0 + i_start; i < n; i += i_inc)
        {
            icount[i] = (ipiv[i] < 0);
        }
    }
}

template <typename T, typename I, typename UA, typename Istride>
static __global__ __launch_bounds__(SYCONV_MAX_THDS) void syconv_kernel(

    bool const is_upper,
    bool const is_convert,

    I const n,

    UA A_arg,
    Istride const shiftA,
    I const lda,
    Istride const strideA,

    I* const ipiv_arg,
    Istride const strideP,

    I const batch_count

)
{
    {
        bool const has_work = (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return;
        }
    }

    I const bid_start = static_cast<I>(blockIdx.z);
    I const bid_inc = static_cast<I>(gridDim.z);

    I const nbx = static_cast<I>(gridDim.x);
    I const ibx = static_cast<I>(blockIdx.x);

    I const ij_start = static_cast<I>(threadIdx.x + threadIdx.y * blockDim.x
                                      + threadIdx.z * (blockDim.x * blockDim.y));
    I const ij_inc = static_cast<I>((blockDim.x * blockDim.y) * blockDim.z);

    auto ceildiv = [](auto m, auto b) { return ((m <= 0) ? 0 : (m - 1) / b + 1); };

    // ------------------------------------------------
    // note: this thread block handles only columns in
    // [gcol_start,gcol_end] inclusively and
    // gcol_start using is 1-based indexing
    // ------------------------------------------------
    I const nb = ceildiv(n, nbx);
    I const gcol_start = 1 + ibx * nb;
    I const gcol_end = std::min(n, gcol_start + nb - 1);

    // ------------------------
    // Fortran 1-based indexing
    // ------------------------
    auto idx2F
        = [](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

    for(I bid = 0 + bid_start; bid < batch_count; bid += bid_inc)
    {
        T* const A_bid = load_ptr_batch<T>(A_arg, bid, shiftA, strideA);
        I* const ipiv_bid = ipiv_arg + bid * strideP;

        // ----------------------------------------------------
        // use 1-based indexing compatible with  Fortran/matlab
        // ----------------------------------------------------
        auto A = [=](auto i, auto j) -> T& { return (A_bid[idx2F(i, j, lda)]); };

        auto ipiv = [=](auto i) -> I { return (ipiv_bid[(i - 1)]); };

        // ------------------------------------------------------------
        // note __syncthreads() is not needed in syconv_swap_rows()
        // since each column is always consistently handled by the same thread
        //
        //
        // swap( A(row_k, col_start:col_end), A(row_kp, col_start:col_end)
        // ------------------------------------------------------------
        auto syconv_swap_rows
            = [=](I const row_k, I const row_kp, I const col_start, I const col_end) {
                  for(I icol = gcol_start + ij_start; icol <= gcol_end; icol += ij_inc)
                  {
                      bool const is_in_range = (col_start <= icol) && (icol <= col_end);
                      if(is_in_range)
                      {
                          auto const temp = A(row_k, icol);
                          A(row_k, icol) = A(row_kp, icol);
                          A(row_kp, icol) = temp;
                      }
                  }
              };

        if(is_upper)
        {
            // ---------------------
            // A is upper triangular
            // ---------------------
            if(is_convert)
            {
                // -------------------------------
                //           convert A (A is upper triangular)
                //
                //           convert value
                // -------------------------------

                // -------------------------------
                //           convert permutations
                // -------------------------------

                //                 i = n
                // 		do while(i.ge.1)
                // 		   if(ipiv(i).gt.0) then
                // 			ip = ipiv(i)
                // 			if(i.lt.n) then
                // 				do 12 j = i + 1, n
                // 					temp = a(ip, j)
                // 					a(ip, j) = a(i, j)
                // 					a(i, j) = temp
                // 12                              continue
                // 			endif
                //
                // 		   else
                // 			  ip = -ipiv(i)
                // 			  if(i.lt.n) then
                // 				  do 13 j = i + 1, n
                // 					  temp = a(ip, j)
                // 					  a(ip, j) = a(i - 1, j)
                // 					  a(i - 1, j) = temp
                // 13                                continue
                //                           endif
                // 			  i = i - 1
                // 	          endif
                // 		  i = i - 1
                // 		  end do
                {
                    I i = n;
                    while(i >= 1)
                    {
                        if(ipiv(i) > 0)
                        {
                            I const ip = ipiv(i);
                            if(i < n)
                            {
                                auto const col_start = (i + 1);
                                auto const col_end = n;
                                auto const irow_k = i;
                                auto const irow_kp = ip;
                                syconv_swap_rows(irow_k, irow_kp, col_start, col_end);
                            }
                        }
                        else
                        {
                            I const ip = -ipiv(i);
                            if(i < n)
                            {
                                auto const col_start = (i + 1);
                                auto const col_end = n;
                                auto const irow_k = (i - 1);
                                auto const irow_kp = ip;

                                syconv_swap_rows(irow_k, irow_kp, col_start, col_end);
                            }
                            i = i - 1;
                        }
                        i = i - 1;
                    } // end while
                }
            }
            else
            {
                // ------------------------------------------
                //           revert A (A is upper triangular)
                //
                //           revert permutations
                // ------------------------------------------

                //                 i = 1
                // 		do while(i.le.n)
                // 		   if(ipiv(i).gt.0) then
                // 		      ip = ipiv(i)
                // 		      if(i.lt.n) then
                // 			   do j = i + 1, n,
                // 				   temp = a(ip, j)
                // 				   a(ip, j) = a(i, j)
                // 				   a(i, j) = temp
                // 			   end do
                // 		      endif
                //
                // 		    else
                // 		        ip = -ipiv(i)
                // 			i = i + 1
                // 			if(i.lt.n) then
                // 				do j = i + 1, n
                // 					temp = a(ip, j)
                // 					a(ip, j) = a(i - 1, j)
                // 					a(i - 1, j) = temp
                // 				end do
                // 			endif
                // 		  endif
                // 		  i = i + 1
                // 		  end do
                {
                    I i = 1;

                    while(i <= n)
                    {
                        if(ipiv(i) > 0)
                        {
                            I const ip = ipiv(i);
                            if(i < n)
                            {
                                auto const col_start = (i + 1);
                                auto const col_end = n;
                                auto const irow_k = i;
                                auto const irow_kp = ip;

                                syconv_swap_rows(irow_k, irow_kp, col_start, col_end);
                            }
                        }
                        else
                        {
                            I const ip = -ipiv(i);
                            i = i + 1;
                            if(i < n)
                            {
                                auto const col_start = (i + 1);
                                auto const col_end = n;
                                auto const irow_k = (i - 1);
                                auto const irow_kp = ip;

                                syconv_swap_rows(irow_k, irow_kp, col_start, col_end);
                            }
                        }
                        i = i + 1;
                    } // end while
                }

                // ----------------------
                //           revert value
                // ----------------------

            } // end if (is_revert)
        } // end if (is_upper)
        else
        {
            // ------------------------------
            //          A is lower triangular
            // ------------------------------

            if(is_convert)
            {
                // -------------------------------------------
                //           convert A (A is lower triangular)
                //
                //           convert value
                // -------------------------------------------

                // --------------------------------
                //             convert permutations
                // --------------------------------

                //                 i = 1
                // 		do while(i.le.n)
                // 		   if(ipiv(i).gt.0) then
                // 			ip = ipiv(i)
                // 			if(i.gt.1) then
                // 				do 22 j = 1, i - 1
                // 					temp = a(ip, j)
                // 					a(ip, j) = a(i, j)
                // 					a(i, j) = temp
                // 22                              continue
                //                         endif
                // 		   else
                // 		     ip = -ipiv(i)
                // 		     if(i.gt.1) then
                // 		        do 23 j = 1, i - 1
                // 			       temp = a(ip, j)
                // 			       a(ip, j) = a(i + 1, j)
                // 			       a(i + 1, j) = temp
                // 23                      continue
                //                       endif
                //
                // 		      i = i + 1
                // 	           endif
                // 		   i = i + 1
                // 	       end do

                {
                    I i = 1;
                    while(i <= n)
                    {
                        if(ipiv(i) > 0)
                        {
                            I const ip = ipiv(i);
                            if(i > 1)
                            {
                                auto const col_start = 1;
                                auto const col_end = (i - 1);
                                auto const irow_k = i;
                                auto const irow_kp = ip;

                                syconv_swap_rows(irow_k, irow_kp, col_start, col_end);
                            }
                        }
                        else
                        {
                            I const ip = -ipiv(i);
                            if(i > 1)
                            {
                                auto const col_start = 1;
                                auto const col_end = (i - 1);
                                auto const irow_k = i + 1;
                                auto const irow_kp = ip;

                                syconv_swap_rows(irow_k, irow_kp, col_start, col_end);
                            }
                            i = i + 1;
                        }
                        i = i + 1;
                    } // end while
                }
            }
            else
            {
                //  -------------------------------
                //            revert A (A is lower triangular)
                //
                //            revert permutations
                //  -------------------------------

                //                 i = n
                // 		do while(i.ge.1)
                // 		   if(ipiv(i).gt.0) then
                // 			   ip = ipiv(i)
                // 			   if(i.gt.1) then
                // 			      do j = 1, i - 1
                // 				   temp = a(i, j)
                // 				   a(i, j) = a(ip, j)
                // 				   a(ip, j) = temp
                // 			      end do
                // 		            endif
                // 		   else
                // 		      ip = -ipiv(i)
                // 		      i = i - 1
                // 		      if(i.gt.1) then
                // 		         do j = 1, i - 1
                // 				 temp = a(i + 1, j)
                // 				 a(i + 1, j) = a(ip, j)
                // 				 a(ip, j) = temp
                // 			 end do
                // 		      endif
                // 	           endif
                // 		   i = i - 1
                // 	        end do

                {
                    I i = n;
                    while(i >= 1)
                    {
                        if(ipiv(i) > 0)
                        {
                            I const ip = ipiv(i);
                            if(i > 1)
                            {
                                auto const col_start = 1;
                                auto const col_end = (i - 1);
                                auto const irow_k = i;
                                auto const irow_kp = ip;

                                syconv_swap_rows(irow_k, irow_kp, col_start, col_end);
                            }
                        }
                        else
                        {
                            I const ip = -ipiv(i);
                            i = i - 1;
                            if(i > 1)
                            {
                                auto const col_start = 1;
                                auto const col_end = (i - 1);
                                auto const irow_k = (i + 1);
                                auto const irow_kp = ip;

                                syconv_swap_rows(irow_k, irow_kp, col_start, col_end);
                            }
                        }
                        i = i - 1;
                    } // end while
                }

                // ------------------------
                //             revert value
                // ------------------------
            }
        }
        __syncthreads();
    } // end for bid
}

template <typename T, typename I, typename Istride, typename UA>
static inline rocblas_status rocsolver_syconv_argCheck(rocblas_handle handle,
                                                       bool const is_upper,
                                                       bool const is_convert,
                                                       I const n,

                                                       UA A,
                                                       Istride const shiftA,
                                                       I const lda,
                                                       Istride const strideA,

                                                       I* const ipiv,
                                                       Istride const strideP,

                                                       T* const E,
                                                       Istride const strideE,

                                                       I const batch_count,
                                                       void* const work,
                                                       size_t const size_work)
{
    {
        bool const is_valid_handle = (handle != nullptr);
        if(!is_valid_handle)
        {
            return (rocblas_status_invalid_handle);
        }
    }

    // ---------------
    // check arguments
    // ---------------

    {
        bool const is_valid_size = (n >= 0) && (lda >= n) && (batch_count >= 0)
            && (size_work >= sizeof(I) * n * batch_count);

        if(!is_valid_size)
        {
            return (rocblas_status_invalid_size);
        }
    }

    {
        bool const has_work = (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return (rocblas_status_success);
        }
    }

    {
        bool const is_valid_pointers
            = (A != nullptr) && (ipiv != nullptr) && (E != nullptr) && (work != nullptr);

        if(!is_valid_pointers)
        {
            return (rocblas_status_invalid_pointer);
        }
    }

    if(batch_count > 1)
    {
        bool const is_valid_stride
            = (strideA >= (static_cast<Istride>(lda) * n)) && (strideP >= n) && (strideE >= n);

        if(!is_valid_stride)
        {
            return (rocblas_status_invalid_size);
        }
    }

    return (rocblas_status_success);
}

template <typename T, typename I>
static inline void rocsolver_syconv_getMemorySize(

    I const n,
    I const batch_count,

    size_t* p_size_work)
{
    *p_size_work = 0;

    {
        bool const has_work_to_do = (n >= 1) && (batch_count >= 1);
        if(!has_work_to_do)
        {
            return;
        }
    }

    size_t const size_icount = sizeof(I) * n * batch_count;

    size_t size_rocprim = 0;

    {
        // -------------------------------------------------------------
        // This is just an estimate of the upper bound of storage needed
        //
        // Note rocprim storage query requires the actual d_data pointer
        // It is not safe to pass a nullptr as d_data
        //
        // The following estimate is recommended by co-pilot
        // -------------------------------------------------------------
        auto ceildiv = [](auto m, auto b) { return ((m <= 0) ? 0 : (m - 1) / b + 1); };
        size_t constexpr ITEMS_PER_BLOCK = 128;
        size_t constexpr SAFETY = 8;
        size_t const N = static_cast<size_t>(n) * batch_count;
        size_t const num_blocks = ceildiv(N, ITEMS_PER_BLOCK);

        size_t const temp_bytes = (num_blocks * sizeof(int64_t)) * SAFETY;

        size_t const size_rocprim_est1 = temp_bytes;

        // -----------------
        // gross upper bound
        // -----------------
        size_t const size_rocprim_est2 = (n * sizeof(I)) * batch_count;

        size_rocprim = std::max(size_rocprim_est1, size_rocprim_est2);
    }

    *p_size_work = size_icount + size_rocprim;

    return;
}

//  -----------------------------------
// if (is_convert == true) syconv_template converts a given factorization
// produced by SYTRF into L, U factors ready for TRSM.
// NOTE: The content of A will be modified.
//
// if (is_convert == false) syconv_template reverts to the original
// factorization produced by SYTRF
//  -----------------------------------
template <typename T, typename I, typename UA, typename Istride = rocblas_stride>
static rocblas_status rocsolver_syconv_template(rocblas_handle handle,

                                                bool const is_upper,
                                                bool const is_convert,

                                                I const n,

                                                UA A,
                                                Istride const shiftA,
                                                I const lda,
                                                Istride const strideA,

                                                I* const ipiv,
                                                Istride const strideP,

                                                T* const E,
                                                Istride const strideE,

                                                I const batch_count,

                                                void* const work,
                                                size_t const size_work)
{
    // --------------
    // check arguments
    // --------------
    {
        rocblas_status const istat = rocsolver_syconv_argCheck(handle, is_upper, is_convert, n,

                                                               A, shiftA, lda, strideA,

                                                               ipiv, strideP,

                                                               E, strideE,

                                                               batch_count,

                                                               work, size_work);
        if(istat != rocblas_status_success)
        {
            return (istat);
        }
    }

    auto ceildiv = [](auto n, auto b) { return ((n <= 0) ? 0 : (n - 1) / b + 1); };

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I const max_blocks = 64 * 1024 - 3;
    I const nbx = std::max(I(1), std::min(max_blocks, ceildiv(n, I(SYCONV_MAX_THDS))));
    I const nby = 1;
    I const nbz = std::max(I(1), std::min(max_blocks, batch_count));

    I const ln = ceildiv(n, nbx);
    I const nx = std::min(ln, I(SYCONV_MAX_THDS));
    I const ny = 1;
    I const nz = 1;

    std::byte* const pwork = reinterpret_cast<std::byte*>(work);
    std::byte* pfree = pwork;

    I* const icount = reinterpret_cast<I*>(pfree);
    pfree += sizeof(I) * n * batch_count;

    // -------------------------------------------------------------
    // setup icount[] to indicate whether there  is a negative pivot
    // -------------------------------------------------------------
    {
        syconv_setup_icount_kernel<<<dim3(nbx, nby, nbz), dim3(nx, ny, nz), 0, stream>>>(
            n, batch_count, ipiv, strideP, icount);
    }

    // -----------------------------------------------
    // compute segmented prefix sum scan using rocprim
    //
    // there are batch_count number of segments
    // and each segment is  length n
    // -----------------------------------------------
    {
        // Functor to generate offsets
        struct mul_n
        {
            I n;
            __host__ __device__ I operator()(I i) const
            {
                return i * n;
            }
        };

        auto segmented_inclusive_scan_inplace = [=](I* d_data, I const n, I const batch_count) {
            auto counting = rocprim::make_counting_iterator<I>(0);
            auto offsets = rocprim::make_transform_iterator(counting, mul_n{n});

            void* temp = nullptr;
            size_t temp_bytes = 0;

            // ------------------------------
            // size query for scratch storage in in-place prefix sum scan
            //
            // NOTE:  rocprim expect argument temp_bytes as  size_t&
            //        and not as size_t *
            //
            // ------------------------------
            {
                hipError_t const hstat = rocprim::segmented_inclusive_scan(
                    temp, temp_bytes,
                    d_data, // input
                    d_data, // output (same pointer)
                    batch_count, offsets, offsets + 1, rocprim::plus<I>(), stream);

                if(hstat != hipSuccess)
                {
                    return (rocblas_status_internal_error);
                }
            }

            size_t const size_remain = (pwork + size_work) - pfree;
            if(temp_bytes > size_remain)
            {
                return (rocblas_status_memory_error);
            }

            temp = reinterpret_cast<void*>(pfree);

            {
                // --------------------------------
                // actual execution for (in place)
                // prefix sum scan
                // --------------------------------
                hipError_t const hstat = rocprim::segmented_inclusive_scan(
                    temp, temp_bytes, d_data, d_data, batch_count, offsets, offsets + 1,
                    rocprim::plus<I>(), stream);

                if(hstat != hipSuccess)
                {
                    return (rocblas_status_internal_error);
                }
            }

            return (rocblas_status_success);
        };

        rocblas_status const istat = segmented_inclusive_scan_inplace(icount, n, batch_count);
        if(istat != rocblas_status_success)
        {
            return (istat);
        }
    }
    // ----------------------------------------
    // NOTE: icount now contains the prefix sum
    // ----------------------------------------

    {
        if(is_convert)
        {
            syconv_convert_into_E<T, I, UA, Istride>
                <<<dim3(nbx, nby, nbz), dim3(nx, ny, nz), 0, stream>>>(is_upper, n, batch_count,

                                                                       A, shiftA, lda, strideA,

                                                                       ipiv, strideP,

                                                                       E, strideE, icount);
        }
    }

    {
        syconv_kernel<T, I, UA><<<dim3(nbx, nby, nbz), dim3(nx, ny, nz), 0, stream>>>(

            is_upper, is_convert,

            n,

            A, shiftA, lda, strideA,

            ipiv, strideP,

            batch_count);
    }

    {
        // --------------------------------------------------
        // restore back to format initially produced by SYTRF
        // --------------------------------------------------
        if(!is_convert)
        {
            syconv_restore_from_E<T, I, UA, Istride>
                <<<dim3(nbx, nby, nbz), dim3(nx, ny, nz), 0, stream>>>(is_upper, n, batch_count,

                                                                       A, shiftA, lda, strideA,

                                                                       ipiv, strideP,

                                                                       E, strideE, icount);
        }
    }
    return (rocblas_status_success);
}
ROCSOLVER_END_NAMESPACE
