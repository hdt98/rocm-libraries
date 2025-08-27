/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

 #include "helper.h"
 #include <hip/hip_runtime.h>
 #include <hipblaslt/hipblaslt.h>
 #include <iostream>
 
 void simpleBasicMatmul4cov(hipblasLtHandle_t  handle,
                          hipblasOperation_t trans_a,
                          hipblasOperation_t trans_b,
                          int64_t            m,
                          int64_t            n,
                          int64_t            k,
                          int64_t            batch_count,
                          float&             alpha,
                          float&             beta,
                          void*              d_a,
                          void*              d_b,
                          void*              d_c,
                          void*              d_d,
                          void*              d_workspace,
                          int64_t            max_workspace_size,
                          hipStream_t        stream);
 
 int main()
 {
     Runner<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, float> runner(
         128, 128, 128, 1, 1.f, 1.f, 32 * 1024 * 1024);
 
     runner.run([&runner] {
        simpleBasicMatmul4cov(runner.handle,
                             HIPBLAS_OP_N,
                             HIPBLAS_OP_N,
                             runner.m,
                             runner.n,
                             runner.k,
                             runner.batch_count,
                             runner.alpha,
                             runner.beta,
                             runner.d_a,
                             runner.d_b,
                             runner.d_c,
                             runner.d_d,
                             runner.d_workspace,
                             runner.max_workspace_size,
                             runner.stream);
     });
 
     return 0;
 }
 
 void simpleBasicMatmul4cov(hipblasLtHandle_t  handle,
                          hipblasOperation_t trans_a,
                          hipblasOperation_t trans_b,
                          int64_t            m,
                          int64_t            n,
                          int64_t            k,
                          int64_t            batch_count,
                          float&             alpha,
                          float&             beta,
                          void*              d_a,
                          void*              d_b,
                          void*              d_c,
                          void*              d_d,
                          void*              d_workspace,
                          int64_t            max_workspace_size,
                          hipStream_t        stream)
 {
    void*              a;
    void*              b;
    void*              c;
    void*              d;

    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
    CHECK_HIP_ERROR(hipMalloc(&d_a, m * k * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipMalloc(&d_b, n * k * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipMalloc(&d_c, m * n * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipMalloc(&d_d, m * n * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipHostMalloc(&a, m * k * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipHostMalloc(&b, n * k * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipHostMalloc(&c, m * n * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipHostMalloc(&d, m * n * batch_count * sizeof(hipblasLtHalf)));

    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_a, a, m * k * batch_count * sizeof(hipblasLtHalf), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_b, b, n * k * batch_count * sizeof(hipblasLtHalf), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_c, c, m * n * batch_count * sizeof(hipblasLtHalf), hipMemcpyHostToDevice, stream));

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA,  HIP_R_16F, m, k, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB,  HIP_R_16F, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC,  HIP_R_16F, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD,  HIP_R_16F, m, n, m));

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                          matmul,
                                          &alpha,
                                          d_a,
                                          matA,
                                          d_b,
                                          matB,
                                          &beta,
                                          d_c,
                                          matC,
                                          d_d,
                                          matD,
                                          nullptr,
                                          nullptr,
                                          0,
                                          0));

    CHECK_HIP_ERROR(hipFree(a));
    CHECK_HIP_ERROR(hipFree(b));
    CHECK_HIP_ERROR(hipFree(c));
    CHECK_HIP_ERROR(hipFree(d));
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
     return;
 }
 