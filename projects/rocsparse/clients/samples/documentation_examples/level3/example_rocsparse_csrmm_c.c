/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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




#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>
#include <stdio.h>
#include <stdlib.h>

#define HIP_CHECK(stat)                                                                       \
    {                                                                                         \
        if(stat != hipSuccess)                                                                \
        {                                                                                     \
            fprintf(stderr, "Error: hip error in line %d\n", __LINE__); \
            return -1;                                                                        \
        }                                                                                     \
    }

#define ROCSPARSE_CHECK(stat)                                                \
    {                                                                        \
        if(stat != rocsparse_status_success)                                 \
        {                                                                    \
            fprintf(stderr, "Error: rocsparse error in line %d\n", __LINE__); \
            return -1;                                                       \
        }                                                                    \
    }

//! [doc example]
int main(int argc, char* argv[])
{
    //     1 2 0 3 0
    // A = 0 4 5 0 0
    //     6 0 0 7 8

    rocsparse_int m   = 3;
    rocsparse_int k   = 5;
    rocsparse_int nnz = 8;

    // alpha and beta
    float alpha = 1.0f;
    float beta  = 0.0f;

    int hcsr_row_ptr[] = {0, 3, 5, 8};
    int hcsr_col_ind[] = {0, 1, 3, 1, 2, 0, 3, 4};
    float hcsr_val[] = {1, 2, 3, 4, 5, 6, 7, 8};

    // Set dimension n of B
    rocsparse_int n = 64;

    // Allocate and generate column-oriented dense matrix B
    float* hB = (float*)malloc(sizeof(float) * (k * n));
    for(rocsparse_int i = 0; i < k * n; ++i)
    {
        hB[i] = (float)(rand()) / (float)(RAND_MAX);
    }
    float* hC = (float*)malloc(sizeof(float) * (m * n));

    int*   dcsr_row_ptr;
    int*   dcsr_col_ind;
    float* dcsr_val;
    float* dB;
    float* dC;
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsr_val, sizeof(float) * nnz));
    HIP_CHECK(hipMalloc((void**)&dB, sizeof(float) * k * n));
    HIP_CHECK(hipMalloc((void**)&dC, sizeof(float) * m * n));

    HIP_CHECK(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr, sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val, sizeof(float) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, sizeof(float) * k * n, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, hC, sizeof(float) * m * n, hipMemcpyHostToDevice));

    // Create rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create matrix descriptor
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    // Perform the matrix multiplication
    ROCSPARSE_CHECK(rocsparse_scsrmm(handle,
                                     rocsparse_operation_none,
                                     rocsparse_operation_none,
                                     m,
                                     n,
                                     k,
                                     nnz,
                                     &alpha,
                                     descr,
                                     dcsr_val,
                                     dcsr_row_ptr,
                                     dcsr_col_ind,
                                     dB,
                                     k,
                                     &beta,
                                     dC,
                                     m));

    HIP_CHECK(hipMemcpy(hC, dC, sizeof(float) * m * n, hipMemcpyDeviceToHost));

    printf("hC\n");
    for(size_t i = 0; i < sizeof(hC)/sizeof(hC[0]); i++)
    {
        printf("%f ", hC[i]);
    }
    printf("\n");

    // Clean up
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear up on device
    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));

    free(hB);
    free(hC);
    return 0;
}
//! [doc example]
