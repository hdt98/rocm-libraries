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
    int m   = 3;
    int n   = 3;
    int nnz = 7;

    rocsparse_index_base idx_base = rocsparse_index_base_zero;
    rocsparse_datatype   datatype = rocsparse_datatype_f32_r;

    // 2 1 0
    // 1 2 1
    // 0 1 2
    int hcsr_row_ptr[] = {0, 2, 5, 7};
    int hcsr_col_ind[] = {0, 1, 0, 1, 2, 1, 2};
    float hcsr_val[] = {2.0f, 1.0f, 1.0f, 2.0f, 1.0f, 1.0f, 2.0f};

    int*   dcsr_row_ptr;
    int*   dcsr_col_ind;
    float* dcsr_val;
    float* dilu0;
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsr_val, sizeof(float) * nnz));
    HIP_CHECK(hipMalloc((void**)&dilu0, sizeof(float) * nnz));

    HIP_CHECK(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr, sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val, sizeof(float) * nnz, hipMemcpyHostToDevice));

    HIP_CHECK(hipMemset(dilu0, 0, sizeof(float) * nnz));

    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    rocsparse_itilu0_alg alg      = rocsparse_itilu0_alg_async_inplace;
    int                  nmaxiter = 1000;
    int                  option   = 0;
    float                tol      = 1e-7;

    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_csritilu0_buffer_size(handle,
                                                    alg,
                                                    option,
                                                    nmaxiter,
                                                    m,
                                                    nnz,
                                                    dcsr_row_ptr,
                                                    dcsr_col_ind,
                                                    idx_base,
                                                    datatype,
                                                    &buffer_size));

    void* dbuffer;
    HIP_CHECK(hipMalloc((void**)&dbuffer, buffer_size));

    ROCSPARSE_CHECK(rocsparse_csritilu0_preprocess(handle,
                                                   alg,
                                                   option,
                                                   nmaxiter,
                                                   m,
                                                   nnz,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind,
                                                   idx_base,
                                                   datatype,
                                                   buffer_size,
                                                   dbuffer));

    ROCSPARSE_CHECK(rocsparse_scsritilu0_compute(handle,
                                                 alg,
                                                 option,
                                                 &nmaxiter,
                                                 tol,
                                                 m,
                                                 nnz,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 dcsr_val,
                                                 dilu0,
                                                 idx_base,
                                                 buffer_size,
                                                 dbuffer));

    float* hilu0 = (float*)malloc(sizeof(float) * (nnz));
    HIP_CHECK(hipMemcpy(hilu0, dilu0, sizeof(float) * nnz, hipMemcpyDeviceToHost));

    printf("hilu0\n");
    for(size_t i = 0; i < sizeof(hilu0)/sizeof(hilu0[0]); i++)
    {
        printf("%f ", hilu0[i]);
    }
    printf("\n");

    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    HIP_CHECK(hipFree(dbuffer));

    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));
    HIP_CHECK(hipFree(dilu0));

    free(hilu0);
    return 0;
}
//! [doc example]
