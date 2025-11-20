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
    // Number of non-zeros of the sparse vector
    int nnz = 3;

    // Size of sparse and dense vector
    int size = 9;

    // Sparse index vector
    int hx_ind[] = {0, 3, 5};

    // Dense vector
    float hy[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

    // Offload data to device
    int*   dx_ind;
    float* dx_val;
    float* dy;
    HIP_CHECK(hipMalloc((void**)&dx_ind, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dx_val, sizeof(float) * nnz));
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(float) * size));

    HIP_CHECK(hipMemcpy(dx_ind, hx_ind, sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, hy, sizeof(float) * size, hipMemcpyHostToDevice));

    rocsparse_handle      handle;
    rocsparse_spvec_descr vecX;
    rocsparse_dnvec_descr vecY;

    rocsparse_indextype  idx_type  = rocsparse_indextype_i32;
    rocsparse_datatype   data_type = rocsparse_datatype_f32_r;
    rocsparse_index_base idx_base  = rocsparse_index_base_zero;

    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create sparse vector X
    ROCSPARSE_CHECK(rocsparse_create_spvec_descr(
        &vecX, size, nnz, dx_ind, dx_val, idx_type, idx_base, data_type));

    // Create dense vector Y
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vecY, size, dy, data_type));

    // Call axpby to perform gather
    ROCSPARSE_CHECK(rocsparse_gather(handle, vecY, vecX));

    ROCSPARSE_CHECK(rocsparse_spvec_get_values(vecX, (void**)&dx_val));

    // Copy result back to host
    float hx_val[nnz];
    HIP_CHECK(hipMemcpy(hx_val, dx_val, sizeof(float) * nnz, hipMemcpyDeviceToHost));

    printf("x\n");
    for(int i = 0; i < nnz; ++i)
    {
        printf("%f ", hx_val[i]);
    }

    printf("\n");

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_spvec_descr(vecX));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vecY));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dx_ind));
    HIP_CHECK(hipFree(dx_val));
    HIP_CHECK(hipFree(dy));

    return 0;
}
//! [doc example]
