/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#include "../testings/testing.hpp"

#include "auto_testing_bad_arg.hpp"
#include "rocsparse_matrix_statistics.hpp"
#include "testing_spmv_dispatch_traits.hpp"

template <rocsparse_format FORMAT,
          typename I,
          typename J,
          typename A,
          typename X,
          typename Y,
          typename T>
struct testing_spmv_dispatch
{
private:
    using traits = testing_spmv_dispatch_traits<FORMAT, I, J, A, X, Y, T>;
    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

public:
    static void testing_spmv_bad_arg(const Arguments& arg)
    {
    }

    static void testing_spmv(const Arguments& arg)
    {
        int M;
	int N;
	rocsparse_operation    trans       = rocsparse_operation_none;
        rocsparse_index_base   base        = rocsparse_index_base_zero;
        rocsparse_datatype     ttype       = rocsparse_datatype_f64_r;

        double alpha = 1.0;
        double beta = 0.0;

	rocsparse_handle     handle;
	rocsparse_spmat_descr matA;
	rocsparse_dnvec_descr vecX;
	rocsparse_dnvec_descr vecY;

	rocsparse_create_handle(&handle);

	host_coo_matrix<double> hA;

	rocsparse_matrix_factory<double> matrix_factory(arg);
    	matrix_factory.init_coo(hA, M, N);


	std::cout << "hA.m: " << hA.m << " hA.n: " << hA.n << " hA.nnz: " << hA.nnz << std::endl;

	/*std::cout << "hA.row_ind" << std::endl;
	for(int i = 0; i < 1000; i++)
	{
	    std::cout << hA.row_ind[i] << " ";
	}
	std::cout << "" << std::endl;

	std::cout << "hA.col_ind" << std::endl;
        for(int i = 0; i < 1000; i++)
        {
            std::cout << hA.col_ind[i] << " ";
        }
        std::cout << "" << std::endl;*/






	host_dense_matrix<double> hx(N, 1);
        host_dense_matrix<double> hy(M, 1);
	for(int i = 0; i < N; i++)
	{
	    hx[i] = 1.0;
	}
	for(int i = 0; i < M; i++)
        {
            hy[i] = 1.0;
        }

    	device_coo_matrix<double>   dA(hA);
    	device_dense_matrix<double> dx(hx), dy(hy);

	hipMemset(dy, 0, sizeof(double) * M);

	// Create sparse matrix A
	rocsparse_create_coo_descr(&matA,
                           dA.m,
                           dA.n,
                           dA.nnz,
                           dA.row_ind,
                           dA.col_ind,
                           dA.val,
                           get_indextype<int>(),
                           base,
                           ttype);

	// Create dense vector X
	rocsparse_create_dnvec_descr(&vecX, N, dx, ttype);

	// Create dense vector Y
	rocsparse_create_dnvec_descr(&vecY, M, dy, ttype);

	// Call spmv to perform computation
	size_t buffer_size = 0;
	void* temp_buffer = nullptr;
	rocsparse_spmv(handle,
               trans,
               &alpha,
               matA,
               vecX,
               &beta,
               vecY,
               ttype,
               rocsparse_spmv_alg_coo_atomic,
               rocsparse_spmv_stage_compute,
               &buffer_size,
               temp_buffer);

	// Clear rocSPARSE
	rocsparse_destroy_spmat_descr(matA);
	rocsparse_destroy_dnvec_descr(vecX);
	rocsparse_destroy_dnvec_descr(vecY);
	rocsparse_destroy_handle(handle);























        /*J                      M           = arg.M;
        J                      N           = arg.N;
        rocsparse_operation    trans       = arg.transA;
        rocsparse_index_base   base        = arg.baseA;
        rocsparse_spmv_alg     alg         = arg.spmv_alg;
        rocsparse_datatype     ttype       = get_datatype<T>();

	T h_alpha = arg.get_alpha<T>();
	T h_beta = arg.get_beta<T>();

        // Create rocsparse handle
        rocsparse_local_handle handle(arg);

        // INITIALIZATE THE SPARSE MATRIX
        host_sparse_matrix<A> hA;
        rocsparse_matrix_factory<A, I, J> matrix_factory(arg);
	traits::sparse_initialization(matrix_factory, hA, M, N, base);

        device_sparse_matrix<A> dA(hA);

        host_dense_matrix<X> hx(N, 1);
        for(int i = 0; i < N; i++)
	{
	    hx[i] = 1.0;
	}
	device_dense_matrix<X> dx(hx);

        host_dense_matrix<Y> hy(M, 1);
        for(int i = 0; i < M; i++)
        {
            hy[i] = 1.0;
        }
	device_dense_matrix<Y> dy(hy);

        rocsparse_local_spmat matA(dA);
        rocsparse_local_dnvec x(dx);
        rocsparse_local_dnvec y(dy);

	CHECK_HIP_ERROR(hipDeviceSynchronize());
        std::cout << "Before host mode compute" << std::endl;

        // Run buffer size
        void*  dbuffer     = nullptr;
        size_t buffer_size = 0;
        CHECK_ROCSPARSE_ERROR(rocsparse_spmv(handle, trans, &h_alpha, matA, x, &h_beta, y, ttype, alg, rocsparse_spmv_stage_compute, &buffer_size, dbuffer));

        CHECK_HIP_ERROR(hipDeviceSynchronize());
        std::cout << "Made it passed host mode compute" << std::endl;*/
    }
};
