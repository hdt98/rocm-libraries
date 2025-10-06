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

#include "auto_testing_bad_arg.hpp"
#include "rocsparse_matrix_statistics.hpp"
#include "testing_spmv_dispatch_traits.hpp"

struct col_val
{
    int    col_ind;
    double val;
};

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

    static bool load_mtx_file(const std::string&   filename,
                              std::vector<int>&    csr_row_ptr,
                              std::vector<int>&    csr_col_ind,
                              std::vector<double>& csr_val,
                              int&                 m,
                              int&                 n,
                              int&                 nnz)
    {
        std::cout << "filename: " << filename << std::endl;
        std::ifstream file;

        file.open(filename.c_str());

        std::vector<int>    row_ind;
        std::vector<int>    col_ind;
        std::vector<double> vals;

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
                        std::cout << "line: " << line << std::endl;
                        token = line.substr(0, line.find(space));
                        m     = atoi(token.c_str());
                        line.erase(0, line.find(space) + space.length());
                        token = line.substr(0, line.find(space));
                        n     = atoi(token.c_str());
                        line.erase(0, line.find(space) + space.length());
                        token                    = line.substr(0, line.find(space));
                        int lower_triangular_nnz = atoi(token.c_str());

                        nnz = 2 * (lower_triangular_nnz - m) + m;

                        row_ind.resize(nnz);
                        col_ind.resize(nnz);
                        vals.resize(nnz);

                        std::cout << "m: " << m << " n: " << n << " nnz: " << nnz << std::endl;
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

                        row_ind[index] = r;
                        col_ind[index] = c;
                        vals[index]    = v;

                        index++;

                        if(r != c)
                        {
                            row_ind[index] = c;
                            col_ind[index] = r;
                            vals[index]    = v;

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

        // find number of entries in each row;
        csr_row_ptr.resize(m + 1, 0);
        for(size_t i = 0; i < row_ind.size(); i++)
        {
            csr_row_ptr[row_ind[i] + 1]++;
        }

        for(int i = 0; i < m; i++)
        {
            csr_row_ptr[i + 1] += csr_row_ptr[i];
        }

        // std::cout << "csr_row_ptr" << std::endl;
        // for (size_t i = 0; i < csr_row_ptr.size(); i++)
        //{
        //	std::cout << csr_row_ptr[i] << " ";
        // }
        // std::cout << "" << std::endl;

        csr_col_ind.resize(nnz, -1);
        csr_val.resize(nnz, 0.0);

        for(int i = 0; i < nnz; i++)
        {
            int row_start = csr_row_ptr[row_ind[i]];
            int row_end   = csr_row_ptr[row_ind[i] + 1];

            for(int j = row_start; j < row_end; j++)
            {
                if(csr_col_ind[j] == -1)
                {
                    csr_col_ind[j] = col_ind[i];
                    csr_val[j]     = vals[i];
                    break;
                }
            }
        }

        // Verify no negative 1 found in csr column indices array
        for(size_t i = 0; i < csr_col_ind.size(); i++)
        {
            if(csr_col_ind[i] == -1)
            {
                std::cout << "Error in csr_co_ind array. Negative 1 found" << std::endl;
                return false;
            }
        }

        // Sort columns and values
        for(int i = 0; i < m; i++)
        {
            int row_start = csr_row_ptr[row_ind[i]];
            int row_end   = csr_row_ptr[row_ind[i] + 1];

            std::vector<col_val> unsorted_col_vals(row_end - row_start);
            for(int j = row_start; j < row_end; j++)
            {
                unsorted_col_vals[j - row_start].col_ind = csr_col_ind[j];
                unsorted_col_vals[j - row_start].val     = csr_val[j];
            }

            std::sort(unsorted_col_vals.begin(),
                      unsorted_col_vals.end(),
                      [&](col_val t1, col_val t2) { return t1.col_ind < t2.col_ind; });

            for(int j = row_start; j < row_end; j++)
            {
                csr_col_ind[j] = unsorted_col_vals[j - row_start].col_ind;
                csr_val[j]     = unsorted_col_vals[j - row_start].val;
            }
        }

        // std::cout << "csr_col_ind" << std::endl;
        // for (size_t i = 0; i < csr_col_ind.size(); i++)
        //{
        //	std::cout << csr_col_ind[i] << " ";
        // }
        // std::cout << "" << std::endl;

        // std::cout << "csr_val" << std::endl;
        // for (size_t i = 0; i < csr_val.size(); i++)
        //{
        //	std::cout << csr_val[i] << " ";
        // }
        // std::cout << "" << std::endl;

        return true;
    }

    static void testing_spmv(const Arguments& arg)
    {
        std::cout << "AAAA" << std::endl;
        J                      M           = arg.M;
        J                      N           = arg.N;
        rocsparse_operation    trans       = arg.transA;
        rocsparse_index_base   base        = arg.baseA;
        rocsparse_spmv_alg     alg         = arg.spmv_alg;
        rocsparse_matrix_type  matrix_type = arg.matrix_type;
        rocsparse_fill_mode    uplo        = arg.uplo;
        rocsparse_storage_mode storage     = arg.storage;
        rocsparse_datatype     ttype       = get_datatype<T>();

        const bool call_stage_analysis = arg.call_stage_analysis;

        // Create rocsparse handle
        rocsparse_local_handle handle(arg);

        T h_alpha = static_cast<T>(1);
        T h_beta  = static_cast<T>(0);

        std::vector<int>    hptr;
        std::vector<int>    hind;
        std::vector<double> hval;
        int                 m   = 0;
        int                 n   = 0;
        int                 nnz = 0;
        if(!load_mtx_file("mac_econ_fwd500/mac_econ_fwd500.mtx", hptr, hind, hval, m, n, nnz))
        {
            std::cout << "Error: Failed to load mac_econ_fwd500.mtx file" << std::endl;
        }
        std::cout << "m: " << m << " n: " << n << " nnz: " << nnz << std::endl;

        std::cout << "BBBB" << std::endl;
        host_sparse_matrix<A> hA;
        std::cout << "CCCC" << std::endl;
        rocsparse_matrix_factory<A, I, J> matrix_factory(arg, true, false);
        std::cout << "DDDD" << std::endl;
        traits::sparse_initialization(matrix_factory, hA, M, N, base);
        std::cout << "EEEE" << std::endl;

        device_sparse_matrix<A> dA(hA);
        std::cout << "FFFF" << std::endl;

        host_dense_matrix<X> hx(N, 1);
        for(int i = 0; i < N; i++)
        {
            hx[i] = static_cast<X>(1);
        }
        device_dense_matrix<X> dx(hx);

        std::cout << "GGGG" << std::endl;

        host_dense_matrix<Y> hy(M, 1);
        for(int i = 0; i < M; i++)
        {
            hy[i] = static_cast<Y>(1);
        }
        device_dense_matrix<Y> dy(hy);

        rocsparse_local_spmat matA(dA);
        rocsparse_local_dnvec x(dx);
        rocsparse_local_dnvec y(dy);

        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spmat_set_attribute(
                matA, rocsparse_spmat_matrix_type, &matrix_type, sizeof(matrix_type)),
            rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spmat_set_attribute(matA, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)),
            rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_set_attribute(
                                    matA, rocsparse_spmat_storage_mode, &storage, sizeof(storage)),
                                rocsparse_status_success);

        //CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        std::cout << "HHHH" << std::endl;
        // Run buffer size
        void*  dbuffer     = nullptr;
        size_t buffer_size = 0;
        CHECK_ROCSPARSE_ERROR(rocsparse_spmv(handle,
                                             trans,
                                             &h_alpha,
                                             matA,
                                             x,
                                             &h_beta,
                                             y,
                                             ttype,
                                             alg,
                                             rocsparse_spmv_stage_buffer_size,
                                             &buffer_size,
                                             dbuffer));

        std::cout << "buffer_size: " << buffer_size << std::endl;
        CHECK_HIP_ERROR(hipMalloc((void**)&dbuffer, buffer_size));

        std::cout << "IIII" << std::endl;
        //if(call_stage_analysis)
        //{
        std::cout << "Before analysis" << std::endl;
        // Run preprocess
        CHECK_ROCSPARSE_ERROR(rocsparse_spmv(handle,
                                             trans,
                                             &h_alpha,
                                             matA,
                                             x,
                                             &h_beta,
                                             y,
                                             ttype,
                                             alg,
                                             rocsparse_spmv_stage_preprocess,
                                             &buffer_size,
                                             dbuffer));
        std::cout << "After analysis" << std::endl;
        //}

        std::cout << "JJJJ" << std::endl;

        // Run solve
        CHECK_ROCSPARSE_ERROR(rocsparse_spmv(handle,
                                             trans,
                                             &h_alpha,
                                             matA,
                                             x,
                                             &h_beta,
                                             y,
                                             ttype,
                                             alg,
                                             rocsparse_spmv_stage_compute,
                                             &buffer_size,
                                             dbuffer));

        CHECK_HIP_ERROR(hipFree(dbuffer));
    }

    static void testing_spmv_analysis(const Arguments& arg) {}
};
