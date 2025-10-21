
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


// ---------------------------------
// kernel to perform B <- B + sigma * identity
// ---------------------------------
template<typename T, typename I, typename Istride, typename UB>
static __device__ void  add_shift_kernel( 
		I const m, I const n, 
		T const sigma, 

		            UB B, Istride const shiftB, I const ldb, Istride const strideB,

			    I const batch_count

			    )
{
   {
	   bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1) && (sigma != 0 );
	   if (!has_work) { return; };
   }

   I const bid_start = blockIdx.z;
   I const bid_inc = gridDim.z;

   I const i_start = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * ( blockDim.x * blockDim.y);
   I const i_inc = (blockDim.x * blockDim.y ) * blockDim.z;

   auto const idx2D(i,j,ld) {
	   return( i + j * static_cast<int64_t>(ld) );
   };

   I const min_mn = std::min( m, n );

   for(I bid = bid_start; bid < batch_count; bid += bid_inc) {
	   auto const Bp = load_ptr_batch( B, bid, shiftB, strideB );
	   for(I i=i_start; i < min_mn; i += i_inc) {

		   // diagonal entry 
		   I const j = i;

		   auto const ij = idx2D( i, j, ldb );
		   Bp[ ij ] += sigma;
	   }
   }

}

// --------------------------------------------
// routine to perform B <- B + sigma * identity
// --------------------------------------------
template<typename T, typename I, typename Istride, typename UB>
static void add_shift( hipStream_t stream,
		I const m, I const n,
		T const sigma,

		UB B, Istride const shiftB, I const ldb, Istride const strideB,

		I const batch_count )
{
	{
		bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1) && (sigma != 0);
		if (!has_work) { return; };
	}

	auto const ceil = [](auto n, auto base) { return(  (n-1)/base + 1 ); };

	I const nx = 64;
	I const ny = 1;
	I const nz = 1;

	I const max_blocks = 1024;
	I const min_mn = std::min( m, n );

	I const nbx = std::min( max_blocks, ceil( min_mn, nx ));
	I const nby = 1;
	I const nbz = std::min( max_blocks, batch_count );

	add_shift_kernel<T,I,Istride,UB><<< dim3(nbx,nby,nbz), dim3(nx,ny,nz), 0, stream >>>(
			m, n, sigma,
			
			UB, shiftB, ldb, strideB,

			batch_count );

}
template <typename T, typename I, typename Istride, typename UA>
static __global__ void set_triangular_kernel(char const uplo, I const m,
                                             I const n, T const alpha,

                                             UA A_arg, Istride shiftA,
                                             I const lda, Istride const strideA,

                                             I const batch_count) {

  {
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if (!has_work) {
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

  auto idx2D = [](auto i, auto j, auto ld) {
    return (i + j * static_cast<int64_t>(ld));
  };

  for (I bid = bid_start; bid < batch_count; bid += bid_inc) {
    T *const A = load_ptr_batch<T>(A_arg, bid, shiftA, strideA);

    for (I j = j_start; j < n; j += j_inc) {
      for (I i = i_start; i < m; i += i_inc) {
        bool const is_strictly_lower = (i > j);
        bool const is_strictly_upper = (j < i);

        bool const do_assign = set_all ||
                               (set_strictly_lower && is_strictly_lower) ||
                               (set_strictly_upper && is_strictly_upper);
        if (do_assign) {
          auto const ij = idx2D(i, j, lda);
          A[ij] = alpha;
        }
      }
    }

  } // end for bid
}

template <typename T, typename I, typename Istride, typename UA>
static void set_triangular(hipStream_t stream, char const uplo, I const m,
                           I const n, T const alpha,

                           UA A_arg, Istride const shiftA, I const lda,
                           Istride const strideA, I const batch_count) {

  {
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if (!has_work) {
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

  set_triangular_kernel<<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream>>>(
      uplo, m, n, alpha,

      A_arg, shiftA, lda, strideA,

      batch_count);
}

template<typename I>
rocblas_status rocsolver_cholqr1_argCheck( rocblas_handle handle, 
		I const m, I const n, 
		I const lda, I const ldr )
{
  bool const isok = (m >= 0) && (n >= 0) && (lda >= m) && (ldr >= n);
  return( isok ? rocblas_status_continue : rocblas_status_invalid_value );
}


template<typename I, typename UA, typename UR>
rocblas_status rocsolver_cholqr1_strided_batched_argCheck(rocblas_handle handle, 
		
		I const m, I const n, I const lda, I const ldr, 
		
		UA A, 
		UR R, 
		
		I const batch_count) 
{

	bool const isok_values = (m >= 0) && (n >= 0) && (batch_count >= 0) &&
		                 (lda >= m) && (ldr >= n);
	if (!isok_values) {
		return( rocblas_status_invalid_value );
	}

	bool const isok_pointer = (A != nullptr) && (R != nullptr);
	if (!isok_pointer) {
		return( rocblas_status_invalid_pointer );
	}

	return( rocblas_status_continue );
}


template<typename I, typename UA, typename UR>
rocblas_status rocsolver_cholqr2_strided_batched_argCheck(rocblas_handle handle, 
		
		I const m, I const n, I const lda, I const ldr, 
		
		UA A, 
		UR R, 
		
		I const batch_count) 
{

	bool const isok_values = (m >= 0) && (n >= 0) && (batch_count >= 0) &&
		                 (lda >= m) && (ldr >= n);
	if (!isok_values) {
		return( rocblas_status_invalid_value );
	}

	bool const isok_pointer = (A != nullptr) && (R != nullptr);
	if (!isok_pointer) {
		return( rocblas_status_invalid_pointer );
	}

	return( rocblas_status_continue );
}





template<typename I>
rocblas_status rocsolver_cholqr2_argCheck( rocblas_handle handle, 
		I const m, I const n, 
		I const lda, I const ldr )
{
  bool const isok = (m >= 0) && (n >= 0) && (lda >= m) && (ldr >= n);
  return( isok ? rocblas_status_continue : rocblas_status_invalid_value );
}

template <bool BATCHED, bool STRIDED, typename T, typename I >
void rocsolver_cholqr1_getMemorySize(I const m, I const n, I const batch_count,
                                     size_t *p_size_work) {
  size_t size_work = 0;
  *p_size_work = 0;

  {
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if (!has_work) {
      return;
    }
  }

  // ----------------------------------
  // storage for computing  B = A' * A
  // ----------------------------------
  size_t const size_syrk_herk = sizeof(T *) * batch_count;

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
    rocsolver_potrf_getMemorySize<BATCHED, STRIDED, T, I>(
        n, uplo, batch_count,

        &size_scalars, &size_work1, &size_work2, &size_work3, &size_work4,
        &size_pivots, &size_iinfo, &optim_mem);

    size_potrf += size_scalars + size_work1 + size_work2 + size_work3 +
                  size_work4 + size_pivots + size_iinfo;
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

    auto const istat_trsm_mem = rocblasCall_trsm_mem<BATCHED, T, I>(
        side, trans1, mm, nn, ld1, ld2, batch_count, &size_x_temp,
        &size_x_temp_arr, &size_invA, &size_invA_arr);

    bool const isok_trsm_mem = (istat_trsm_mem == rocblas_status_success) ||
	                       (istat_trsm_mem == rocblas_status_continue);
    assert(isok_trsm_mem);

    size_trsm += size_x_temp + size_x_temp_arr + size_invA + size_invA_arr;
  }

  size_work = std::max(size_work, size_syrk_herk);
  size_work = std::max(size_work, size_potrf);
  size_work = std::max(size_work, size_trsm);

  *p_size_work = size_work;
}

template <bool BATCHED, bool STRIDED, typename T, typename I >
void rocsolver_cholqr2_getMemorySize(I const m, I const n, I const batch_count,
                                     size_t *p_size_work) {

  using INFO = I;

  size_t size_work = 0;
  *p_size_work = 0;

  {
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if (!has_work) {
      return;
    }
  }


  // --------------------------------
  // storage for computing [Q,R1] = cholqr1(A)
  // --------------------------------
  size_t const size_R1 = sizeof(T) * n * n * batch_count;
  size_work += size_R1;

  // --------------------------------------------
  // storage for iinfo for 2nd call to cholqr1(A)
  // --------------------------------------------
  size_t const size_iinfo = sizeof(INFO) * batch_count;
  size_work += size_iinfo;

  size_t size_cholqr1 = 0;
  { rocsolver_cholqr1_getMemorySize<BATCHED,STRIDED,T,I>(m, n, batch_count, &size_cholqr1); }

  size_t const size_trmm = sizeof(T *) * batch_count;

  size_work += std::max(size_cholqr1, size_trmm);

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
template <bool BATCHED, bool STRIDED, typename T, typename I, typename Istride,
          typename UA, typename UR, typename INFO=I>
rocblas_status rocsolver_cholqr1_template(
		
		rocblas_handle handle, 

		I const m, I const n,

		 UA A, Istride const shiftA, I const lda, Istride strideA, 
		 
		 UR R, Istride const shiftR, I const ldr, Istride strideR,

		 I const batch_count, 

		 I *const info,


		 void *work, size_t const size_work,  
		 
		 T const sigma = 0) {

  bool constexpr is_complex = rocblas_is_complex<T>;

  using S = decltype( std::real(T{}) );

  {
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if (!has_work) {
      return (rocblas_status_success);
    }
  }

  hipStream_t stream;
  rocblas_get_stream(handle, &stream);


  int constexpr idebug = 0;
  auto print_mat = [=](auto name, auto m, auto n, auto A, auto shiftA, auto lda, auto strideA) {
	  std::vector<T> h_A( lda * n );
	  I const bid = 0;
	  T * Ap = nullptr;
	  if constexpr (std::is_pointer_v< decltype(*A) >)  { Ap = *A; }
	  else { Ap = A; };
		  
	  auto const istat = hipMemcpy( &(h_A[0]), Ap, sizeof(T) * lda * n, hipMemcpyDeviceToHost );
	  assert( istat == hipSuccess );

	  for(auto j=0; j < n; j++) {
          for(auto i=0; i < m; i++) {
		  auto const ij = i + j * static_cast<int64_t>(lda);
		  auto const aij = h_A[ij];
		  std::cout << name << "(" << i+1 << "," << j+1 << ") = " << aij << ";" << std::endl;
	  }
	  }
  };



         




  {

    // ----------
    // reset info
    // ----------
    auto const istat =
        hipMemsetAsync(info, 0, sizeof(INFO) * batch_count, stream);
    if (istat != hipSuccess) {
      std::cout << "internal error: FILE " << __FILE__ << " line " << __LINE__ << std::endl;
      return (rocblas_status_internal_error);
    }
  }

  std::byte *const pwork = (std::byte *)work;
  std::byte *pfree = pwork;

  // -------------------------------------
  // form B = A' * A,  reuse storage for R
  // -------------------------------------
  auto const B = R;
  auto const shiftB = shiftR;
  auto const strideB = strideR;
  auto const ldb = ldr;

  {
    auto const pfree_saved = pfree;

    auto const trans1 = (is_complex) ? rocblas_operation_conjugate_transpose
                                     : rocblas_operation_transpose;

    size_t size_work_arr = sizeof(T *) * batch_count;

    void *const work_arr = (void *)pfree;
    pfree += size_work_arr;

    bool is_mem_ok = (pfree <= (pwork + size_work));
    if (!is_mem_ok) {
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


    if (idebug >= 1) {
      std::cout << "before SYRK" << std::endl;
      print_mat("A", m, n, A, shiftA, lda, strideA );
    }

    // ------------------
    // compute B = A' * A
    // ------------------
    auto const istat =
        rocblasCall_syrk_herk<BATCHED, T>(handle, uplo, trans1,

                                          nn, kk,

                                          &alpha,

                                          A, shiftA, lda, strideA,

                                          &beta,

                                          B, shiftB, ldb, strideB,

                                          batch_count);

    if (istat != rocblas_status_success) {
      return (istat);
    }

    if (idebug >= 1) {
      std::cout << "after SYRK" << std::endl;
      print_mat("B", n, n, B, shiftB, ldb, strideB );
    }

    pfree = pfree_saved;
  }

  // -------------------------
  // optional, if sigma != 0
  // B <- B + sigma * identity
  // -------------------------
  if (sigma != 0) {
	  add_shift( stream, m, n, B, shiftB, ldb, strideB, batch_count, sigma );
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
    rocsolver_potrf_getMemorySize<BATCHED, STRIDED, T, I>(
        n, uplo, batch_count,

        &size_scalars, &size_work1, &size_work2, &size_work3, &size_work4,
        &size_pivots, &size_iinfo, &optim_mem);

    // -----------------------------------------------------
    // allocate temporary storage for Cholesky factorization
    // -----------------------------------------------------

    T *const scalars = (T *)pfree;
    pfree += size_scalars;
    T *const pivots = (T *)pfree;
    pfree += size_pivots;
    INFO *const iinfo = (INFO *)pfree;
    pfree += size_iinfo;

    void *const work1 = (void *)pfree;
    pfree += size_work1;
    void *const work2 = (void *)pfree;
    pfree += size_work2;
    void *const work3 = (void *)pfree;
    pfree += size_work3;
    void *const work4 = (void *)pfree;
    pfree += size_work4;

    bool const is_mem_ok = (pfree <= (pwork + size_work));
    if (!is_mem_ok) {
      return (rocblas_status_memory_error);
    }

    auto const istat = rocsolver_potrf_template<BATCHED, STRIDED, T, I, INFO, S>(
        handle, uplo, n,

        B, shiftB, ldb, strideB,

        info, batch_count, 

        scalars, work1, work2, work3, work4, pivots, iinfo, optim_mem);

    if (istat != rocblas_status_success) {
      return (istat);
    }

    if (idebug >= 1) {
      std::cout << "after POTRF" << std::endl;
      print_mat( "B", n, n, B, shiftB, ldb, strideB );
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

    auto const istat_trsm_mem = rocblasCall_trsm_mem<BATCHED, T, I>(
        side, trans1, mm, nn, ld1, ld2, batch_count,

        &size_x_temp, &size_x_temp_arr, &size_invA, &size_invA_arr);

    bool const isok_trsm_mem = (istat_trsm_mem == rocblas_status_success) ||
	                       (istat_trsm_mem == rocblas_status_continue);

    if (idebug >= 1) {
       std::cout << "after TRSM_mem " << " istat_trsm_mem = " << istat_trsm_mem << std::endl;
    }

    if (!isok_trsm_mem) {
      return (istat_trsm_mem);
    }


    // -------------------------------------------------
    // allocate scratch memory for trsm triangular solve
    // -------------------------------------------------

    void *const x_temp = (void *)pfree;
    pfree += size_x_temp;
    void *const x_temp_arr = (void *)pfree;
    pfree += size_x_temp_arr;
    void *const invA = (void *)pfree;
    pfree += size_invA;
    void *const invA_arr = (void *)pfree;
    pfree += size_invA_arr;

    bool is_mem_ok = (pfree <= (pwork + size_work));
    if (!is_mem_ok) {
      return (rocblas_status_memory_error);
    }

    T alpha = 1;
    auto const istat_trsm =
        rocblasCall_trsm(handle, 
			
			side, uplo, trans1, diag, 
			
			mm, nn, &alpha,


                         R, shiftR, ldr, strideR,

                         A, shiftA, lda, strideA,

                         batch_count, optimal_mem,

                         x_temp, x_temp_arr, invA, invA_arr);

    bool const isok_trsm = (istat_trsm == rocblas_status_success) ||
	                   (istat_trsm == rocblas_status_continue);
    if (!isok_trsm) {
      return (istat_trsm);
    }

    if (idebug >= 1) {
      std::cout << "after TRSM " << std::endl;
      print_mat("Q", m, n, A, shiftA, lda, strideA );
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

template <bool BATCHED, bool STRIDED, typename T, typename I, typename Istride,
          typename UA, typename UR, typename INFO=I>
rocblas_status
rocsolver_cholqr2_template(rocblas_handle handle, I const m, I const n,

                  UA A, Istride const shiftA, I const lda, Istride strideA,

                  UR R, Istride const shiftR, I const ldr, Istride strideR,

                  I const batch_count, INFO* const info,

                  void *work, size_t const size_work) {

  {
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if (!has_work) {
      return( rocblas_status_success);
    };
  }

  hipStream_t stream;
  {
  auto const istat = rocblas_get_stream( handle, &stream );
  if (istat != rocblas_status_success) {
	  return( istat );
	  }
  }

  {

    // ----------
    // reset info
    // ----------
    auto const istat =
        hipMemsetAsync(info, 0, sizeof(INFO) * batch_count, stream);
    if (istat != hipSuccess) {
      std::cout << "internal error: FILE " << __FILE__ << " line " << __LINE__ << std::endl;
      return (rocblas_status_internal_error);
    }
  }

  std::byte *const pwork = (std::byte *)work;
  std::byte *pfree = pwork;

  // -------------------------
  // (1) [Q,R1] = cholqr1( A )
  // -------------------------
  I const ldr1 = n;
  size_t const size_R1 = sizeof(T) *  ldr1 * n * batch_count;
  Istride const shiftR1 = 0;
  Istride const strideR1 = static_cast<Istride>(ldr1) * n;

  T *const R1 = (T *)pfree;
  pfree += size_R1;

  {
    auto const pfree_saved = pfree;
    size_t const lwork_remain = (pwork + size_work) - pfree;

    {
     size_t size_cholqr1 = 0;
     rocsolver_cholqr1_getMemorySize<BATCHED,STRIDED,T,I>(m, n, batch_count, &size_cholqr1); 

     bool const is_memory_ok = (lwork_remain >= size_cholqr1);
     if (!is_memory_ok) {
	     std::cout << "memory for 1st cholqr1 " 
		     << " lwork_remain = " << lwork_remain
		     << " size_cholqr1 = " << size_cholqr1
		     << std::endl;
		       
	     return( rocblas_status_memory_error );
     }


    }


    auto const istat = rocsolver_cholqr1_template<BATCHED,STRIDED,T,I,Istride>(
		                        handle, 
					
					m, n,

                                         A, shiftA, lda, strideA,

                                         R1, shiftR1, ldr1, strideR1,

                                         batch_count, info,

                                         (void *) pfree, lwork_remain);

    if (istat != rocblas_status_success) {
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
    INFO *iinfo = (INFO *) pfree;
    pfree += size_iinfo;

    bool const is_memory_ok = ( pfree <= (pwork + size_work));
    if (!is_memory_ok) {
	    std::cout << "memory for 2nd cholqr1 "
		    << " pfree = " << pfree
		    << " pwork = " << pwork
		    << " size_work = " << size_work
		    << std::endl;

	    return( rocblas_status_memory_error );
    }

    size_t const lwork_remain = (pwork + size_work) - pfree;


    auto const istat = rocsolver_cholqr1_template<BATCHED,STRIDED,T,I,Istride>(
		                         handle, 
					 
					 m, n,

					 Q, shiftQ, ldq, strideQ, 

                                         R, shiftR, ldr, strideR,

                                         batch_count, iinfo,

                                         (void *) pfree, lwork_remain);

    if (istat != rocblas_status_success) {
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

    size_t const size_workArr = sizeof(T *) * batch_count;
    T ** const workArr = (T **)pfree;
    pfree += size_workArr;

    bool const is_memory_ok = (pfree <= (pwork + size_work));
    if (!is_memory_ok) {

      std::cout << "mem for TRMM: pfree = " << pfree 
	      << " pwork = " << pwork 
	      << " size_work = " << size_work
	      << std::endl;

      return (rocblas_status_memory_error);
    }

    Istride const stride_alpha = 0;

    auto const istat =
        rocblasCall_trmm<T>(
			
			handle, 
			
			side, uplo, trans1, diag, 
			
			mm, nn, 
			
			&alpha, stride_alpha, 

                         R1, shiftR1, ldr1, strideR1,

                         R, shiftR, ldr, strideR,

                         batch_count, workArr);

    if (istat != rocblas_status_success) {
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
template <bool BATCHED, bool STRIDED, typename T, typename I, typename Istride,
          typename UA, typename UR, typename INFO=I>
rocblas_status
rocsolver_cholqr3_template(rocblas_handle handle, I const m, I const n,

                  UA A, Istride const shiftA, I const lda, Istride strideA,

                  UR R, Istride const shiftR, I const ldr, Istride strideR,

                  I const batch_count, INFO* const info,

                  void *work, size_t const size_work) 
{

	{
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
	if (!has_work) {
		return( rocblas_status_success );
		}
	}


  hipStream_t stream;
  {
  auto const istat = rocblas_get_stream( handle, &stream );
  if (istat != rocblas_status_success) {
	  return( istat );
	  }
  }

  {

    // ----------
    // reset info
    // ----------
    auto const istat =
        hipMemsetAsync(info, 0, sizeof(INFO) * batch_count, stream);
    if (istat != hipSuccess) {
      std::cout << "internal error: FILE " << __FILE__ << " line " << __LINE__ << std::endl;
      return (rocblas_status_internal_error);
    }
  }

  std::byte *const pwork = (std::byte *)work;
  std::byte *pfree = pwork;

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
  T * const R1 = (T *) pfree; pfree += size_R1;

  {
   bool const is_mem_ok = (pfree <= (pwork + size_work));
   if (!is_mem_ok) {
	   return( rocblas_status_memory_error );
	   }
  }

  // ---------------------------------------
  // (1)  R1 * R1' = A'*A + sigma * identity
  // ---------------------------------------

  {
    auto const pfree_saved = pfree;
    size_t const size_iinfo = sizeof(INFO) * batch_count;
    INFO *iinfo = (INFO *) pfree;
    pfree += size_iinfo;

    size_t const size_sigma = sizeof(T) * batch_count;
    T * const sigma = (T *) pfree; pfree += size_sigma;


    bool const is_memory_ok = ( pfree <= (pwork + size_work));
    if (!is_memory_ok) {
	    std::cout << "memory for cholqr3 "
		    << " pfree = " << pfree
		    << " pwork = " << pwork
		    << " size_work = " << size_work
		    << std::endl;

	    return( rocblas_status_memory_error );
    }



    size_t const lwork_remain = (pwork + size_work) - pfree;


    {
    cal_sigma( stream, m, n, batch_count,

		    A, shiftA, lda, strideA,

		    sigma,
	            (void *) pfree,  lwork_remain);
    }


    // T const sigma = 11 * (m * n * ueps + (n + 1) * (n * ueps)) * gnorm

    auto const istat = rocsolver_cholqr1_template<BATCHED,STRIDED,T,I,Istride>(
		                         handle, 
					 
					 m, n,

					 Q, shiftQ, ldq, strideQ, 

                                         R1, shiftR1, ldr1, strideR1,

                                         batch_count, iinfo,

                                         (void *) pfree, lwork_remain, sigma);

    if (istat != rocblas_status_success) {
      return (istat);
    }

    pfree = pfree_saved;
  }

  // (2)   CholQR2(

}

ROCSOLVER_END_NAMESPACE
