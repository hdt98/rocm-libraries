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

        // // Run solve
        // CHECK_ROCSPARSE_ERROR(rocsparse_spmv(handle,
        //                                      trans,
        //                                      &h_alpha,
        //                                      matA,
        //                                      x,
        //                                      &h_beta,
        //                                      y,
        //                                      ttype,
        //                                      alg,
        //                                      rocsparse_spmv_stage_compute,
        //                                      &buffer_size,
        //                                      dbuffer));

        CHECK_HIP_ERROR(hipFree(dbuffer));
    }

    static void testing_spmv_analysis(const Arguments& arg) {}
};
