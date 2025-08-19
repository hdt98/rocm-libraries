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
        std::vector<int>    coo_row_ind;
        std::vector<int>    coo_col_ind;
        std::vector<double> coo_val;

        load_mtx_file_coo("../../../../../../../shipsec1/shipsec1.mtx",
                          coo_row_ind,
                          coo_col_ind,
                          coo_val,
                          m,
                          n,
                          nnz);

	int* hcoo_row_ind = nullptr;
	int* hcoo_col_ind = nullptr;
	double* hcoo_val = nullptr;
	HIP_CHECK(hipHostMalloc(&hcoo_row_ind, sizeof(int) * nnz));
	HIP_CHECK(hipHostMalloc(&hcoo_col_ind, sizeof(int) * nnz));
	HIP_CHECK(hipHostMalloc(&hcoo_val, sizeof(double) * nnz));

	for(int i = 0; i < nnz; i++)
	{
	    hcoo_row_ind[i] = coo_row_ind[i];
	    hcoo_col_ind[i] = coo_col_ind[i];
	    hcoo_val[i] = coo_val[i];
	}

	int M;
	int N;
	host_coo_matrix<double> hA;
        rocsparse_matrix_factory<double> matrix_factory(arg);
        matrix_factory.init_coo(hA, M, N);
	//bool passed = true;
	//for(int i = 0; i < hA.nnz; i++)
	//{
	//    if(hA.row_ind[i] != hcoo_row_ind[i])
	//    {
	//	passed = false;
	//	break;
	//    }
	//    if(hA.col_ind[i] != hcoo_col_ind[i])
        //    {
        //        passed = false;
        //        break;
        //    }
	//    if(hA.val[i] != hcoo_val[i])
        //    {
        //        passed = false;
        //        break;
        //    }
	//}
	//std::cout << "passed: " << passed << std::endl;

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
            hipMemcpy(dcoo_row_ind, hcoo_row_ind, sizeof(int) * nnz, hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(dcoo_col_ind, hcoo_col_ind, sizeof(int) * nnz, hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(dcoo_val, hcoo_val, sizeof(double) * nnz, hipMemcpyHostToDevice));
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

	HIP_CHECK(hipHostFree(hcoo_row_ind));
	HIP_CHECK(hipHostFree(hcoo_col_ind));
	HIP_CHECK(hipHostFree(hcoo_val));
    }
};
