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
    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    float halpha = 1.0f;
    float hbeta  = -1.0f;

    // A, B, and C are mxk, kxn, and mxn
    int m    = 4;
    int k    = 3;
    int n    = 2;
    int nnzC = 5;

    //     2  3  -1
    // A = 0  2   1
    //     0  0   5
    //     0 -2 0.5

    //      0  4
    // B =  1  0
    //     -2  0.5

    //      1 0            1 0
    // C =  2 3   spy(C) = 1 1
    //      0 0            0 0
    //      4 5            1 1

    float hA[] = {2.0f, 3.0f, -1.0f, 0.0, 2.0f, 1.0f, 0.0f, 0.0f, 5.0f, 0.0f, -2.0f, 0.5f};
    float hB[] = {0.0f, 4.0f, 1.0f, 0.0, -2.0f, 0.5f};

    int hcsr_row_ptrC[] = {0, 1, 3, 3, 5};
    int hcsr_col_indC[] = {0, 0, 1, 0, 1};
    float hcsr_valC[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    float* dA;
    float* dB;
    HIP_CHECK(hipMalloc((void**)&dA, sizeof(float) * m * k));
    HIP_CHECK(hipMalloc((void**)&dB, sizeof(float) * k * n));

    int*   dcsr_row_ptrC;
    int*   dcsr_col_indC;
    float* dcsr_valC;
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptrC, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_indC, sizeof(int) * nnzC));
    HIP_CHECK(hipMalloc((void**)&dcsr_valC, sizeof(float) * nnzC));

    HIP_CHECK(hipMemcpy(dA, hA, sizeof(float) * m * k, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, sizeof(float) * k * n, hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(
        dcsr_row_ptrC, hcsr_row_ptrC, sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_col_indC, hcsr_col_indC, sizeof(int) * nnzC, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_valC, hcsr_valC, sizeof(float) * nnzC, hipMemcpyHostToDevice));

    rocsparse_dnmat_descr matA;
    ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(
        &matA, m, k, k, dA, rocsparse_datatype_f32_r, rocsparse_order_row));

    rocsparse_dnmat_descr matB;
    ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(
        &matB, k, n, n, dB, rocsparse_datatype_f32_r, rocsparse_order_row));

    rocsparse_spmat_descr matC;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&matC,
                                               m,
                                               n,
                                               nnzC,
                                               dcsr_row_ptrC,
                                               dcsr_col_indC,
                                               dcsr_valC,
                                               rocsparse_indextype_i32,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f32_r));

    size_t buffer_size = 0;
    ROCSPARSE_CHECK(rocsparse_sddmm_buffer_size(handle,
                                                rocsparse_operation_none,
                                                rocsparse_operation_none,
                                                &halpha,
                                                matA,
                                                matB,
                                                &hbeta,
                                                matC,
                                                rocsparse_datatype_f32_r,
                                                rocsparse_sddmm_alg_default,
                                                &buffer_size));

    void* dbuffer;
    HIP_CHECK(hipMalloc((void**)&dbuffer, buffer_size));

    ROCSPARSE_CHECK(rocsparse_sddmm_preprocess(handle,
                                               rocsparse_operation_none,
                                               rocsparse_operation_none,
                                               &halpha,
                                               matA,
                                               matB,
                                               &hbeta,
                                               matC,
                                               rocsparse_datatype_f32_r,
                                               rocsparse_sddmm_alg_default,
                                               dbuffer));

    ROCSPARSE_CHECK(rocsparse_sddmm(handle,
                                    rocsparse_operation_none,
                                    rocsparse_operation_none,
                                    &halpha,
                                    matA,
                                    matB,
                                    &hbeta,
                                    matC,
                                    rocsparse_datatype_f32_r,
                                    rocsparse_sddmm_alg_default,
                                    dbuffer));

    HIP_CHECK(hipMemcpy(
        hcsr_row_ptrC, dcsr_row_ptrC, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(hcsr_col_indC, dcsr_col_indC, sizeof(int) * nnzC, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcsr_valC, dcsr_valC, sizeof(float) * nnzC, hipMemcpyDeviceToHost));

    printf("hcsr_row_ptrC\n");
    for(size_t i = 0; i < sizeof(hcsr_row_ptrC)/sizeof(hcsr_row_ptrC[0]); i++)
    {
        printf("%f ", hcsr_row_ptrC[i]);
    }
    printf("\n");

    printf("hcsr_col_indC\n");
    for(size_t i = 0; i < sizeof(hcsr_col_indC)/sizeof(hcsr_col_indC[0]); i++)
    {
        printf("%f ", hcsr_col_indC[i]);
    }
    printf("\n");

    printf("hcsr_valC\n");
    for(size_t i = 0; i < sizeof(hcsr_valC)/sizeof(hcsr_valC[0]); i++)
    {
        printf("%f ", hcsr_valC[i]);
    }
    printf("\n");

    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(matA));
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(matB));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matC));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dcsr_row_ptrC));
    HIP_CHECK(hipFree(dcsr_col_indC));
    HIP_CHECK(hipFree(dcsr_valC));
    HIP_CHECK(hipFree(dbuffer));

    return 0;
}
//! [doc example]
