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

        host_scalar<T> h_alpha(arg.get_alpha<T>());
        host_scalar<T> h_beta(arg.get_beta<T>());

        device_scalar<T> d_alpha(h_alpha);
        device_scalar<T> d_beta(h_beta);

        std::cout << "BBBB" << std::endl;
        host_sparse_matrix<A> hA;
        std::cout << "CCCC" << std::endl;
        {
            int dev;
            CHECK_HIP_ERROR(hipGetDevice(&dev));

            std::cout << "dev: " << dev << std::endl;
            hipDeviceProp_t prop;
            CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, dev));

            std::cout << "DDDD" << std::endl;
            const bool has_datafile = rocsparse_arguments_has_datafile(arg);
            bool       to_int       = false;
            to_int |= (prop.warpSize == 32);
            to_int |= (alg != rocsparse_spmv_alg_csr_rowsplit);
            to_int |= (trans != rocsparse_operation_none && has_datafile);
            to_int |= (matrix_type == rocsparse_matrix_type_symmetric && has_datafile);
            static constexpr bool full_rank = false;

            std::cout << "to_int: " << to_int << std::endl;

            rocsparse_matrix_factory<A, I, J> matrix_factory(arg, to_int, full_rank);
            std::cout << "EEEE" << std::endl;
            traits::sparse_initialization(matrix_factory, hA, M, N, base);
            std::cout << "FFFF" << std::endl;
        }

        std::cout << "GGGG" << std::endl;

        device_sparse_matrix<A> dA(hA);

        host_dense_matrix<X> hx((trans == rocsparse_operation_none) ? N : M, 1);
        for(int i = 0; i < ((trans == rocsparse_operation_none) ? N : M); i++)
        {
            hx[i] = static_cast<X>(1);
        }
        device_dense_matrix<X> dx(hx);

        host_dense_matrix<Y> hy((trans == rocsparse_operation_none) ? M : N, 1);
        for(int i = 0; i < ((trans == rocsparse_operation_none) ? M : N); i++)
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

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        std::cout << "HHHH" << std::endl;
        // Run buffer size
        void*  dbuffer     = nullptr;
        size_t buffer_size = 0;
        CHECK_ROCSPARSE_ERROR(rocsparse_spmv(handle,
                                             trans,
                                             h_alpha,
                                             matA,
                                             x,
                                             h_beta,
                                             y,
                                             ttype,
                                             alg,
                                             rocsparse_spmv_stage_buffer_size,
                                             &buffer_size,
                                             dbuffer));

        std::cout << "buffer_size: " << buffer_size << std::endl;
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

        std::cout << "IIII" << std::endl;
        if(call_stage_analysis)
        {
            std::cout << "Before analysis" << std::endl;
            // Run preprocess
            CHECK_ROCSPARSE_ERROR(rocsparse_spmv(handle,
                                                 trans,
                                                 h_alpha,
                                                 matA,
                                                 x,
                                                 h_beta,
                                                 y,
                                                 ttype,
                                                 alg,
                                                 rocsparse_spmv_stage_preprocess,
                                                 &buffer_size,
                                                 dbuffer));
            std::cout << "After analysis" << std::endl;
        }

        std::cout << "JJJJ" << std::endl;

        // Run solve
        CHECK_ROCSPARSE_ERROR(rocsparse_spmv(handle,
                                             trans,
                                             h_alpha,
                                             matA,
                                             x,
                                             h_beta,
                                             y,
                                             ttype,
                                             alg,
                                             rocsparse_spmv_stage_compute,
                                             &buffer_size,
                                             dbuffer));

        // std::cout << "KKKK" << std::endl;
        // host_dense_matrix<Y> hy_copy(hy);
        // traits::host_calculation(trans, h_alpha, hA, hx, h_beta, hy, alg, matrix_type);

        // std::cout << "LLLL" << std::endl;
        // hy.near_check(dy);
        // std::cout << "MMMM" << std::endl;

        // if(ROCSPARSE_REPRODUCIBILITY)
        // {
        //     rocsparse_reproducibility::save("Y_pointer_mode_host", dy);
        // }

        // dy.transfer_from(hy_copy);
        // CHECK_ROCSPARSE_ERROR(
        //     rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        // CHECK_ROCSPARSE_ERROR(testing::rocsparse_spmv(
        //     PARAMS(d_alpha, matA, x, d_beta, y, rocsparse_spmv_stage_compute)));
        // CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // std::cout << "NNNN" << std::endl;
        // hy.near_check(dy);
        // std::cout << "OOOO" << std::endl;
        // if(ROCSPARSE_REPRODUCIBILITY)
        // {
        //     rocsparse_reproducibility::save("Y_pointer_mode_device", dy);
        // }

        CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
    }

    static void testing_spmv_analysis(const Arguments& arg) {}
};
