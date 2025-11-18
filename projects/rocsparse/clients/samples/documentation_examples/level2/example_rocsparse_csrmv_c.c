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

#define HIP_CHECK(stat)                                                \
    {                                                                  \
        if(stat != hipSuccess)                                         \
        {                                                              \
            fprintf(stderr, "Error: hip error in line %d\n", __LINE__); \
            return -1;                                                 \
        }                                                              \
    }

#define ROCSPARSE_CHECK(stat)                                                \
    {                                                                        \
        if(stat != rocsparse_status_success)                                 \
        {                                                                    \
            fprintf(stderr, "Error: rocsparse error in line %d\n", __LINE__); \
            return -1;                                                       \
        }                                                                    \
    }

/*! [doc example] */
int main(int argc, char* argv[])
{
    // 1 2 3 0 0
    // 0 0 0 0 3
    // 2 1 0 0 1
    // 0 0 3 4 0
    int m = 4;
    int n = 5;

    float alpha = 1.0f;
    float beta  = -1.0f;

    int   hcsr_row_ptr[] = {0, 3, 4, 7, 9};
    int   hcsr_col_ind[] = {0, 1, 2, 4, 0, 1, 4, 2, 3};
    float hcsr_val[]     = {1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f, 1.0f, 3.0f, 4.0f};

    float hx[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float hy[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    int nnz = hcsr_row_ptr[m] - hcsr_row_ptr[0];

    int*   dcsr_row_ptr;
    int*   dcsr_col_ind;
    float* dcsr_val;
    float* dx;
    float* dy;
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsr_val, sizeof(float) * nnz));
    HIP_CHECK(hipMalloc((void**)&dx, sizeof(float) * n));
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(float) * m));

    HIP_CHECK(hipMemcpy(dcsr_row_ptr, hcsr_row_ptr, sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val, sizeof(float) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx, sizeof(float) * n, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, hy, sizeof(float) * m, hipMemcpyHostToDevice));

    // Create rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create matrix descriptor
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    // Create matrix info structure
    rocsparse_mat_info info;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info));

    // Perform analysis step to obtain meta data
    ROCSPARSE_CHECK(rocsparse_scsrmv_analysis(handle,
                                              rocsparse_operation_none,
                                              m,
                                              n,
                                              nnz,
                                              descr,
                                              dcsr_val,
                                              dcsr_row_ptr,
                                              dcsr_col_ind,
                                              info));

    // Compute y = Ax
    ROCSPARSE_CHECK(rocsparse_scsrmv(handle,
                                     rocsparse_operation_none,
                                     m,
                                     n,
                                     nnz,
                                     &alpha,
                                     descr,
                                     dcsr_val,
                                     dcsr_row_ptr,
                                     dcsr_col_ind,
                                     info,
                                     dx,
                                     &beta,
                                     dy));

    HIP_CHECK(hipMemcpy(hy, dy, sizeof(float) * m, hipMemcpyDeviceToHost));

    printf("hy\n");
    for(int i = 0; i < m; i++)
    {
        printf("%f ", hy[i]);
    }
    printf("\n");

    // Clean up
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear up on device
    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy));

    return 0;
}
/*! [doc example] */

