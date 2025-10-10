/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "rocsparse_enum.hpp"
#include "testing.hpp"

template <typename T>
void testing_csrmv_bad_arg(const Arguments& arg)
{
}

template <typename T>
void testing_csrmv(const Arguments& arg)
{
    rocsparse_int       M                   = arg.M;
    rocsparse_int       N                   = arg.N;
    rocsparse_operation trans               = arg.transA;
    rocsparse_spmv_alg  alg                 = arg.spmv_alg;
    const bool          call_stage_analysis = arg.call_stage_analysis;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create matrix descriptor
    rocsparse_mat_descr descr;
    CHECK_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&descr));
    // rocsparse_local_mat_descr descr;

    // Create matrix info
    rocsparse_mat_info info;
    CHECK_ROCSPARSE_ERROR(rocsparse_create_mat_info(&info));
    // rocsparse_local_mat_info info_ptr;

    //rocsparse_mat_info info = (call_stage_analysis)
    //                              ? ((alg == rocsparse_spmv_alg_csr_adaptive) ? info_ptr : nullptr)
    //                              : nullptr;

    rocsparse_matrix_factory<T> matrix_factory(arg, true, false);

    host_csr_matrix<T> hA;
    matrix_factory.init_csr(hA, M, N);

    // normalize
    //rocsparse_vector_utils<T>::normalize(hA.val);

    device_csr_matrix<T> dA(hA);

    std::vector<T> hx(N, static_cast<T>(1));
    std::vector<T> hy(M, static_cast<T>(1));

    T* dx = nullptr;
    T* dy = nullptr;
    CHECK_HIP_ERROR(hipMalloc((void**)&dx, sizeof(T) * N));
    CHECK_HIP_ERROR(hipMalloc((void**)&dy, sizeof(T) * M));

    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * N, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * M, hipMemcpyHostToDevice));

    // host_dense_matrix<T> hx(trans == rocsparse_operation_none ? N : M, 1);
    // rocsparse_matrix_utils::init_exact(hx);
    // device_dense_matrix<T> dx(hx);

    // host_dense_matrix<T> hy(trans == rocsparse_operation_none ? M : N, 1);
    // rocsparse_matrix_utils::init_exact(hy);
    // device_dense_matrix<T> dy(hy);

    std::cout << "aaaa" << std::endl;

    // If adaptive, run analysis step
    if(call_stage_analysis)
    {
        if(alg == rocsparse_spmv_alg_csr_adaptive)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrmv_analysis<T>(
                handle, trans, dA.m, dA.n, dA.nnz, descr, dA.val, dA.ptr, dA.ind, info));
        }
    }

    std::cout << "bbbb" << std::endl;

    CHECK_ROCSPARSE_ERROR(testing::rocsparse_csrmv<T>(handle,
                                                      trans,
                                                      dA.m,
                                                      dA.n,
                                                      dA.nnz,
                                                      &h_alpha,
                                                      descr,
                                                      dA.val,
                                                      dA.ptr,
                                                      dA.ind,
                                                      info,
                                                      dx,
                                                      &h_beta,
                                                      dy));
    std::cout << "cccc" << std::endl;

    if(info != nullptr)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_csrmv_clear(handle, info));
    }

    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(dy));

    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_mat_descr(descr));
    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_mat_info(info));

    std::cout << "dddd" << std::endl;
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_csrmv_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csrmv<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_csrmv_extra(const Arguments& arg) {}
