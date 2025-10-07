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
    int64_t    col_ind;
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
                              std::vector<int64_t>&    csr_row_ptr,
                              std::vector<int64_t>&    csr_col_ind,
                              std::vector<double>& csr_val,
                              int64_t&                 m,
                              int64_t&                 n,
                              int64_t&                 nnz)
    {
        std::cout << "filename: " << filename << std::endl;
        std::ifstream file;

        file.open(filename.c_str());

        std::vector<int64_t>    row_ind;
        std::vector<int64_t>    col_ind;
        std::vector<double> vals;

        m   = 0;
        n   = 0;
        nnz = 0;

        int64_t index = 0;
        if(file.is_open())
        {

            std::string percent("%");
            std::string space(" ");
            std::string token;

            int64_t currentLine = 0;

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
                        int64_t lower_triangular_nnz = atoi(token.c_str());

                        nnz = 2 * (lower_triangular_nnz - m) + m;

                        row_ind.resize(nnz);
                        col_ind.resize(nnz);
                        vals.resize(nnz);

                        std::cout << "m: " << m << " n: " << n << " nnz: " << nnz << std::endl;
                    }

                    if(currentLine > 0)
                    {
                        token = line.substr(0, line.find(space));
                        int64_t r = atoi(token.c_str()) - 1;
                        line.erase(0, line.find(space) + space.length());
                        token    = line.substr(0, line.find(space));
                        int64_t    c = atoi(token.c_str()) - 1;
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

        for(int64_t i = 0; i < m; i++)
        {
            csr_row_ptr[i + 1] += csr_row_ptr[i];
        }

        csr_col_ind.resize(nnz, -1);
        csr_val.resize(nnz, 0.0);

        for(int64_t i = 0; i < nnz; i++)
        {
            int64_t row_start = csr_row_ptr[row_ind[i]];
            int64_t row_end   = csr_row_ptr[row_ind[i] + 1];

            for(int64_t j = row_start; j < row_end; j++)
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
        for(int64_t i = 0; i < m; i++)
        {
            int64_t row_start = csr_row_ptr[row_ind[i]];
            int64_t row_end   = csr_row_ptr[row_ind[i] + 1];

            std::vector<col_val> unsorted_col_vals(row_end - row_start);
            for(int64_t j = row_start; j < row_end; j++)
            {
                unsorted_col_vals[j - row_start].col_ind = csr_col_ind[j];
                unsorted_col_vals[j - row_start].val     = csr_val[j];
            }

            std::sort(unsorted_col_vals.begin(),
                      unsorted_col_vals.end(),
                      [&](col_val t1, col_val t2) { return t1.col_ind < t2.col_ind; });

            for(int64_t j = row_start; j < row_end; j++)
            {
                csr_col_ind[j] = unsorted_col_vals[j - row_start].col_ind;
                csr_val[j]     = unsorted_col_vals[j - row_start].val;
            }
        }

        return true;
    }

    static void testing_spmv(const Arguments& arg)
    {
        std::cout << "AAAA" << std::endl;
        rocsparse_operation    trans       = arg.transA;
        rocsparse_index_base   base        = arg.baseA;
        rocsparse_spmv_alg     alg         = arg.spmv_alg;
        rocsparse_matrix_type  matrix_type = arg.matrix_type;
        rocsparse_fill_mode    uplo        = arg.uplo;
        rocsparse_storage_mode storage     = arg.storage;
        rocsparse_indextype  row_idx_type = rocsparse_indextype_i64;
        rocsparse_indextype  col_idx_type = rocsparse_indextype_i64;
        rocsparse_datatype   data_type    = rocsparse_datatype_f32_r;
        rocsparse_datatype   compute_type = rocsparse_datatype_f32_r;

        // Create rocsparse handle
        rocsparse_handle handle;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_handle(&handle));

        std::cout << "BBBB" << std::endl;

        // J M;
        // J N;
        // host_csr_matrix<double> hA;
        // {
        //     rocsparse_matrix_factory<double, int64_t, int64_t> matrix_factory(arg, true, false);
        //     sparse_initialization2(matrix_factory, hA, M, N, base);
        // }

        std::vector<int64_t> hcsr_row_ptr;
        std::vector<int64_t> hcsr_col_ind;
        std::vector<double>  hcsr_val;
        int64_t                 m   = 0;
        int64_t                 n   = 0;
        int64_t                 nnz = 0;
        rocsparse_init_csr_rocalution("../matrices/mac_econ_fwd500.csr", hcsr_row_ptr, hcsr_col_ind, hcsr_val, m, n, nnz, base);

        double h_alpha = 1.0;
        double h_beta  = 0.0;

        // std::vector<int64_t>    hcsr_row_ptr;
        // std::vector<int64_t>    hcsr_col_ind;
        // std::vector<double> hcsr_val;
        // int64_t                 m   = 0;
        // int64_t                 n   = 0;
        // int64_t                 nnz = 0;
        // if(!load_mtx_file("../../../../../../../mac_econ_fwd500/mac_econ_fwd500.mtx",
        //                   hcsr_row_ptr,
        //                   hcsr_col_ind,
        //                   hcsr_val,
        //                   m,
        //                   n,
        //                   nnz))
        // {
        //     std::cout << "Error: Failed to load mac_econ_fwd500.mtx file" << std::endl;
        // }



        // int64_t              m   = M;
        // int64_t              n   = N;
        // int64_t              nnz = hA.nnz;
        // std::vector<int64_t> hcsr_row_ptr(m + 1);
        // std::vector<int64_t> hcsr_col_ind(nnz);
        // std::vector<double>  hcsr_val(nnz);

        // for(int i = 0; i < m + 1; i++)
        // {
        //     hcsr_row_ptr[i] = hA.ptr[i];
        // }
        // for(int i = 0; i < nnz; i++)
        // {
        //     hcsr_col_ind[i] = hA.ind[i];
        // }
        // for(int i = 0; i < nnz; i++)
        // {
        //     hcsr_val[i] = hA.val[i];
        // }




        std::cout << "CCCC" << std::endl;
        std::vector<double> hx(n, 1.0);
        std::vector<double> hy(m, 0.0);
        std::cout << "m: " << m << " n: " << n << " nnz: " << nnz << std::endl;

        int64_t* dcsr_row_ptr = nullptr;
        int64_t* dcsr_col_ind = nullptr;
        double* dcsr_val = nullptr;
        double* dx = nullptr;
        double* dy = nullptr;
        CHECK_HIP_ERROR(hipMalloc((void**)&dcsr_row_ptr, sizeof(int64_t) * (m + 1)));
        CHECK_HIP_ERROR(hipMalloc((void**)&dcsr_col_ind, sizeof(int64_t) * nnz));
        CHECK_HIP_ERROR(hipMalloc((void**)&dcsr_val, sizeof(double) * nnz));
        CHECK_HIP_ERROR(hipMalloc((void**)&dx, sizeof(double) * n));
        CHECK_HIP_ERROR(hipMalloc((void**)&dy, sizeof(double) * m));

        std::cout << "DDDD" << std::endl;
        CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int64_t) * (m + 1), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int64_t) * nnz, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(double) * nnz, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(double) * n, hipMemcpyHostToDevice));

        std::cout << "EEEE" << std::endl;
        rocsparse_spmat_descr matA;
        rocsparse_dnvec_descr vecX;
        rocsparse_dnvec_descr vecY;

        // Create sparse matrix A
        CHECK_ROCSPARSE_ERROR(rocsparse_create_csr_descr(&matA,
                                                m,
                                                n,
                                                nnz,
                                                dcsr_row_ptr,
                                                dcsr_col_ind,
                                                dcsr_val,
                                                row_idx_type,
                                                col_idx_type,
                                                base,
                                                data_type));

        // Create dense vector X
        CHECK_ROCSPARSE_ERROR(rocsparse_create_dnvec_descr(&vecX, n, dx, data_type));

        // Create dense vector Y
        CHECK_ROCSPARSE_ERROR(rocsparse_create_dnvec_descr(&vecY, m, dy, data_type));

        CHECK_ROCSPARSE_ERROR(
            rocsparse_spmat_set_attribute(
                matA, rocsparse_spmat_matrix_type, &matrix_type, sizeof(matrix_type)));

        CHECK_ROCSPARSE_ERROR(
            rocsparse_spmat_set_attribute(matA, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)));

        CHECK_ROCSPARSE_ERROR(rocsparse_spmat_set_attribute(
                                    matA, rocsparse_spmat_storage_mode, &storage, sizeof(storage)));

        std::cout << "EEEE" << std::endl;
        // Call spmv to get buffer size
        size_t buffer_size;
        CHECK_ROCSPARSE_ERROR(rocsparse_spmv(handle,
                                    trans,
                                    &h_alpha,
                                    matA,
                                    vecX,
                                    &h_beta,
                                    vecY,
                                    compute_type,
                                    alg,
                                    rocsparse_spmv_stage_buffer_size,
                                    &buffer_size,
                                    nullptr));

        std::cout << "buffer_size: " << buffer_size << std::endl;
        void* temp_buffer;
        CHECK_HIP_ERROR(hipMalloc(&temp_buffer, buffer_size));

        std::cout << "FFFF" << std::endl;
        // Call spmv to perform analysis
        CHECK_ROCSPARSE_ERROR(rocsparse_spmv(handle,
                                    trans,
                                    &h_alpha,
                                    matA,
                                    vecX,
                                    &h_beta,
                                    vecY,
                                    compute_type,
                                    alg,
                                    rocsparse_spmv_stage_preprocess,
                                    &buffer_size,
                                    temp_buffer));

        std::cout << "GGGG" << std::endl;
        // Call spmv to perform computation
        CHECK_ROCSPARSE_ERROR(rocsparse_spmv(handle,
                                    trans,
                                    &h_alpha,
                                    matA,
                                    vecX,
                                    &h_beta,
                                    vecY,
                                    compute_type,
                                    alg,
                                    rocsparse_spmv_stage_compute,
                                    &buffer_size,
                                    temp_buffer));

        std::cout << "HHHH" << std::endl;
        // Copy result back to host
        //CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(double) * m, hipMemcpyDeviceToHost));

        // Clear rocSPARSE
        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spmat_descr(matA));
        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_dnvec_descr(vecX));
        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_dnvec_descr(vecY));
        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_handle(handle));

        std::cout << "IIII" << std::endl;
        // Clear device memory
        CHECK_HIP_ERROR(hipFree(dcsr_row_ptr));
        CHECK_HIP_ERROR(hipFree(dcsr_col_ind));
        CHECK_HIP_ERROR(hipFree(dcsr_val));
        CHECK_HIP_ERROR(hipFree(dx));
        CHECK_HIP_ERROR(hipFree(dy));
        CHECK_HIP_ERROR(hipFree(temp_buffer));
        std::cout << "JJJJ" << std::endl;
    }

    static void testing_spmv_analysis(const Arguments& arg) {}
};
