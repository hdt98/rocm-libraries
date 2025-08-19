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

#define HIP_CHECK(stat)                                                        \
    {                                                                          \
        if(stat != hipSuccess)                                                 \
        {                                                                      \
            std::cerr << "Error: hip error in line " << __LINE__ << std::endl; \
        }                                                                      \
    }

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
    static void testing_spmv_bad_arg(const Arguments& arg) {}

    static bool load_mtx_file_coo(const std::string&   filename,
                                  std::vector<int>&    row_ind,
                                  std::vector<int>&    col_ind,
                                  std::vector<double>& vals,
                                  int&                 m,
                                  int&                 n,
                                  int&                 nnz)
    {
        std::cout << "filename: " << filename << std::endl;
        std::ifstream file;

        file.open(filename.c_str());

        std::vector<int>    unsorted_row;
        std::vector<int>    unsorted_col;
        std::vector<double> unsorted_val;

        m   = 0;
        n   = 0;
        nnz = 0;

        int index = 0;
        if(file.is_open())
        {
            std::string percent("%");
            std::string space(" ");
            std::string token;

            int currentLine = 0;

            // scan through file
            while(!file.eof())
            {
                std::string line;
                std::getline(file, line);

                if(currentLine > 0 && index == nnz)
                {
                    break;
                }

                // parse the line
                if(line.substr(0, 1).compare(percent) != 0)
                {
                    if(currentLine == 0)
                    {
                        token = line.substr(0, line.find(space));
                        m     = atoi(token.c_str());
                        line.erase(0, line.find(space) + space.length());
                        token = line.substr(0, line.find(space));
                        n     = atoi(token.c_str());
                        line.erase(0, line.find(space) + space.length());
                        token                    = line.substr(0, line.find(space));
                        int lower_triangular_nnz = atoi(token.c_str());

                        nnz = 2 * (lower_triangular_nnz - m) + m;

                        unsorted_row.resize(nnz);
                        unsorted_col.resize(nnz);
                        unsorted_val.resize(nnz);
                    }

                    if(currentLine > 0)
                    {
                        token = line.substr(0, line.find(space));
                        int r = atoi(token.c_str()) - 1;
                        line.erase(0, line.find(space) + space.length());
                        token    = line.substr(0, line.find(space));
                        int    c = atoi(token.c_str()) - 1;
                        double v = 1.0;
                        if(line.find(space)
                           != std::string::
                               npos) // some mtx files do not have any values. In these cases
                        // just use a value of 1
                        {
                            line.erase(0, line.find(space) + space.length());
                            token = line.substr(0, line.find(space));
                            v     = strtod(token.c_str(), NULL);
                        }

                        unsorted_row[index] = r;
                        unsorted_col[index] = c;
                        unsorted_val[index] = v;

                        index++;

                        if(r != c)
                        {
                            unsorted_row[index] = c;
                            unsorted_col[index] = r;
                            unsorted_val[index] = v;

                            index++;
                        }
                    }
                    currentLine++;
                }
            }

            file.close();
        }
        else
        {
            std::cout << "Could not open file: " << filename << std::endl;
            return false;
        }

        // Sort by row and column index
        std::vector<int> perm(nnz);
        for(int i = 0; i < nnz; ++i)
        {
            perm[i] = i;
        }

        std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {
            if(unsorted_row[a] < unsorted_row[b])
            {
                return true;
            }
            else if(unsorted_row[a] == unsorted_row[b])
            {
                return (unsorted_col[a] < unsorted_col[b]);
            }
            else
            {
                return false;
            }
        });

        row_ind.resize(nnz);
        col_ind.resize(nnz);
        vals.resize(nnz);

        for(int i = 0; i < nnz; ++i)
        {
            row_ind[i] = unsorted_row[perm[i]];
        }
        for(int i = 0; i < nnz; ++i)
        {
            col_ind[i] = unsorted_col[perm[i]];
        }
        for(int i = 0; i < nnz; ++i)
        {
            vals[i] = unsorted_val[perm[i]];
        }

        return true;
    }

    static __device__ __forceinline__ double atomic_add(double* ptr, double val)
    {
        return atomicAdd(ptr, val);
    }

    template <uint32_t BLOCKSIZE>
    static __device__ __forceinline__ void
        coomvn_atomic_loops_device(int64_t nnz,
                                   const int* __restrict__ coo_row_ind,
                                   const int* __restrict__ coo_col_ind,
                                   const double* __restrict__ coo_val,
                                   const double* __restrict__ x,
                                   double* __restrict__ y)
    {
        const int tid = hipThreadIdx_x;

        __shared__ int    shared_row[BLOCKSIZE];
        __shared__ double shared_val[BLOCKSIZE];

        int    row;
        int    col;
        double val;

        // Current threads index into COO structure
        int64_t idx = hipBlockIdx_x * BLOCKSIZE + tid;

        if(idx < nnz)
        {
            row = coo_row_ind[idx];
            col = coo_col_ind[idx];
            val = static_cast<double>(coo_val[idx]) * static_cast<double>(x[col]);
        }
        else
        {
            row = -1;
            col = 0;
            val = static_cast<double>(0);
        }

        shared_row[tid] = row;
        shared_val[tid] = val;
        __syncthreads();

        // segmented reduction
        for(uint32_t j = 1; j < BLOCKSIZE; j <<= 1)
        {
            if(tid >= j)
            {
                if(row == shared_row[tid - j])
                {
                    val = val + shared_val[tid - j];
                }
            }
            __syncthreads();
            shared_val[tid] = val;
            __syncthreads();
        }

        if(tid < BLOCKSIZE - 1)
        {
            if(row != shared_row[tid + 1] && row >= 0)
            {
                atomic_add(&y[row], val);
            }
        }

        if(tid == BLOCKSIZE - 1)
        {
            if(row >= 0)
            {
                atomic_add(&y[row], val);
            }
        }
    }

    template <uint32_t BLOCKSIZE>
    __launch_bounds__(BLOCKSIZE) static __global__
        void coomvn_atomic_loops(int64_t nnz,
                                 const int* __restrict__ coo_row_ind,
                                 const int* __restrict__ coo_col_ind,
                                 const double* __restrict__ coo_val,
                                 const double* __restrict__ x,
                                 double* __restrict__ y)
    {
        coomvn_atomic_loops_device<BLOCKSIZE>(nnz, coo_row_ind, coo_col_ind, coo_val, x, y);
    }

    static void testing_spmv(const Arguments& arg)
    {
        int                 m;
        int                 n;
        int                 nnz;
        std::vector<int>    hcoo_row_ind;
        std::vector<int>    hcoo_col_ind;
        std::vector<double> hcoo_val;

        load_mtx_file_coo("../../../../../../../shipsec1/shipsec1.mtx",
                          hcoo_row_ind,
                          hcoo_col_ind,
                          hcoo_val,
                          m,
                          n,
                          nnz);

        std::cout << "m: " << m << " n: " << n << " nnz: " << nnz << std::endl;

        std::vector<double> hx(n, 1.0);
        std::vector<double> hy(m, 1.0);

        int*    dcoo_row_ind = nullptr;
        int*    dcoo_col_ind = nullptr;
        double* dcoo_val     = nullptr;
        double* dx           = nullptr;
        double* dy           = nullptr;
        HIP_CHECK(hipMalloc((void**)&dcoo_row_ind, sizeof(int) * nnz));
        HIP_CHECK(hipMalloc((void**)&dcoo_col_ind, sizeof(int) * nnz));
        HIP_CHECK(hipMalloc((void**)&dcoo_val, sizeof(double) * nnz));
        HIP_CHECK(hipMalloc((void**)&dx, sizeof(double) * n));
        HIP_CHECK(hipMalloc((void**)&dy, sizeof(double) * m));

        HIP_CHECK(
            hipMemcpy(dcoo_row_ind, hcoo_row_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(dcoo_col_ind, hcoo_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(dcoo_val, hcoo_val.data(), sizeof(double) * nnz, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dx, hx.data(), sizeof(double) * n, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dy, hy.data(), sizeof(double) * m, hipMemcpyHostToDevice));

        hipLaunchKernelGGL((coomvn_atomic_loops<256>),
                           dim3((nnz - 1) / (256) + 1),
                           dim3(256),
                           0,
                           0,
                           nnz,
                           dcoo_row_ind,
                           dcoo_col_ind,
                           dcoo_val,
                           dx,
                           dy);

        HIP_CHECK(hipFree(dcoo_row_ind));
        HIP_CHECK(hipFree(dcoo_col_ind));
        HIP_CHECK(hipFree(dcoo_val));
        HIP_CHECK(hipFree(dx));
        HIP_CHECK(hipFree(dy));

        /*int                  M;
        int                  N;
        rocsparse_operation  trans = rocsparse_operation_none;
        rocsparse_index_base base  = rocsparse_index_base_zero;
        rocsparse_datatype   ttype = rocsparse_datatype_f64_r;

        double alpha = 1.0;
        double beta  = 0.0;

        rocsparse_handle      handle;
        rocsparse_spmat_descr matA;
        rocsparse_dnvec_descr vecX;
        rocsparse_dnvec_descr vecY;

        rocsparse_create_handle(&handle);

        host_coo_matrix<double> hA;

        rocsparse_matrix_factory<double> matrix_factory(arg);
        matrix_factory.init_coo(hA, M, N);

        std::cout << "hA.m: " << hA.m << " hA.n: " << hA.n << " hA.nnz: " << hA.nnz << std::endl;

        //std::cout << "hA.row_ind" << std::endl;
        //for(int i = 0; i < 1000; i++)
        //{
        //    std::cout << hA.row_ind[i] << " ";
        //}
        //std::cout << "" << std::endl;

        //std::cout << "hA.col_ind" << std::endl;
        //for(int i = 0; i < 1000; i++)
        //{
        //    std::cout << hA.col_ind[i] << " ";
        //}
        //std::cout << "" << std::endl;

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
        void*  temp_buffer = nullptr;
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
        rocsparse_destroy_handle(handle);*/

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
