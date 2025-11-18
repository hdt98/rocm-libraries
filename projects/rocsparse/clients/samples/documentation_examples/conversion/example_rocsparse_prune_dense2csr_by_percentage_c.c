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
    //     1 2 0 7
    // A = 3 0 0 4
    //     5 6 0 4
    //     0 4 2 5
    rocsparse_int m          = 4;
    rocsparse_int n          = 4;
    rocsparse_int lda        = m;
    float         percentage = 50.0f;

    float hdense[] = {1.0f,
                                 3.0f,
                                 5.0f,
                                 0.0f,
                                 2.0f,
                                 0.0f,
                                 6.0f,
                                 4.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 2.0f,
                                 7.0f,
                                 4.0f,
                                 4.0f,
                                 5.0f};

    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    rocsparse_mat_info info;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info));

    float* ddense = NULL;
    HIP_CHECK(hipMalloc((void**)&ddense, sizeof(float) * lda * n));
    HIP_CHECK(hipMemcpy(ddense, hdense, sizeof(float) * lda * n, hipMemcpyHostToDevice));

    rocsparse_int* dcsr_row_ptr = NULL;
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr, sizeof(rocsparse_int) * (m + 1)));

    // Obtain the temporary buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_sprune_dense2csr_by_percentage_buffer_size(handle,
                                                                         m,
                                                                         n,
                                                                         ddense,
                                                                         lda,
                                                                         percentage,
                                                                         descr,
                                                                         NULL,
                                                                         dcsr_row_ptr,
                                                                         NULL,
                                                                         info,
                                                                         &buffer_size));

    // Allocate temporary buffer
    void* temp_buffer;
    HIP_CHECK(hipMalloc((void**)&temp_buffer, buffer_size));

    rocsparse_int nnz;
    ROCSPARSE_CHECK(rocsparse_sprune_dense2csr_nnz_by_percentage(
        handle, m, n, ddense, lda, percentage, descr, dcsr_row_ptr, &nnz, info, temp_buffer));

    rocsparse_int* dcsr_col_ind = NULL;
    float*         dcsr_val     = NULL;
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsr_val, sizeof(float) * nnz));

    ROCSPARSE_CHECK(rocsparse_sprune_dense2csr_by_percentage(handle,
                                                             m,
                                                             n,
                                                             ddense,
                                                             lda,
                                                             percentage,
                                                             descr,
                                                             dcsr_val,
                                                             dcsr_row_ptr,
                                                             dcsr_col_ind,
                                                             info,
                                                             temp_buffer));

    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));

    HIP_CHECK(hipFree(temp_buffer));
    HIP_CHECK(hipFree(ddense));

    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));

    return 0;
}
//! [doc example]
