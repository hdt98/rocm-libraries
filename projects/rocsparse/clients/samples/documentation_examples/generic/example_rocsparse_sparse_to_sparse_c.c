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
    // 4 2 0 1 0
    // 2 4 2 0 1
    // 0 2 4 2 0
    // 1 0 2 4 2
    // 0 1 0 2 4
    int m   = 5;
    int n   = 5;
    int nnz = 17;

    int hcsr_row_ptr[] = {0, 3, 7, 10, 14, 17};
    int hcsr_col_ind[] = {0, 1, 3, 0, 1, 2, 4, 1, 2, 3, 0, 2, 3, 4, 1, 3, 4};
    double hcsr_val[] = {4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 4.0};

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    int*    dcsr_row_ptr = NULL;
    int*    dcsr_col_ind = NULL;
    double* dcsr_val     = NULL;
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsr_val, sizeof(double) * nnz));

    HIP_CHECK(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr, sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val, sizeof(double) * nnz, hipMemcpyHostToDevice));

    // It assumes the CSR arrays (ptr, ind, val) have already been allocated and filled.
    // Build Source
    rocsparse_spmat_descr source;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&source,
                                               m,
                                               n,
                                               nnz,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               dcsr_val,
                                               rocsparse_indextype_i32,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f64_r));

    // Build target
    void *                dell_ind, *dell_val;
    int64_t               ell_width = 0;
    rocsparse_spmat_descr target;
    ROCSPARSE_CHECK(rocsparse_create_ell_descr(&target,
                                               m,
                                               n,
                                               dell_ind,
                                               dell_val,
                                               ell_width,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f64_r));

    // Create descriptor
    rocsparse_sparse_to_sparse_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_sparse_to_sparse_descr(
        &descr, source, target, rocsparse_sparse_to_sparse_alg_default));

    // Analysis phase
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_sparse_to_sparse_buffer_size(
        handle, descr, source, target, rocsparse_sparse_to_sparse_stage_analysis, &buffer_size));

    void* buffer = NULL;
    HIP_CHECK(hipMalloc((void**)&buffer, buffer_size));

    ROCSPARSE_CHECK(rocsparse_sparse_to_sparse(handle,
                                               descr,
                                               source,
                                               target,
                                               rocsparse_sparse_to_sparse_stage_analysis,
                                               buffer_size,
                                               buffer));
    HIP_CHECK(hipFree(buffer));

    // the user is responsible to allocate target arrays after the analysis phase.
    int64_t              rows, cols;
    void *               ind, *val;
    rocsparse_indextype  idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    // Get ell_width
    ROCSPARSE_CHECK(rocsparse_ell_get(
        target, &rows, &cols, &ind, &val, &ell_width, &idx_type, &idx_base, &data_type));

    printf("rows: \n");

    // Allocate device arrays for ELL format
    HIP_CHECK(hipMalloc((void**)&dell_ind, sizeof(int) * ell_width * m));
    HIP_CHECK(hipMalloc((void**)&dell_val, sizeof(double) * ell_width * m));

    ROCSPARSE_CHECK(rocsparse_ell_set_pointers(target, dell_ind, dell_val));

    // Calculation phase
    ROCSPARSE_CHECK(rocsparse_sparse_to_sparse_buffer_size(
        handle, descr, source, target, rocsparse_sparse_to_sparse_stage_compute, &buffer_size));
    HIP_CHECK(hipMalloc((void**)&buffer, buffer_size));
    ROCSPARSE_CHECK(rocsparse_sparse_to_sparse(handle,
                                               descr,
                                               source,
                                               target,
                                               rocsparse_sparse_to_sparse_stage_compute,
                                               buffer_size,
                                               buffer));
    HIP_CHECK(hipFree(buffer));

    int* hell_ind = (int*)malloc(sizeof(int) * (ell_width * m));
    double* hell_val = (double*)malloc(sizeof(double) * (ell_width * m));

    HIP_CHECK(
        hipMemcpy(hell_ind, dell_ind, sizeof(int) * ell_width * m, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        hell_val, dell_val, sizeof(double) * ell_width * m, hipMemcpyDeviceToHost));

    printf("hell_ind\n");
    for(size_t i = 0; i < sizeof(hell_ind)/sizeof(hell_ind[0]); i++)
    {
        printf("%f ", hell_ind[i]);
    }
    printf("\n");

    printf("hell_val\n");
    for(size_t i = 0; i < sizeof(hell_val)/sizeof(hell_val[0]); i++)
    {
        printf("%f ", hell_val[i]);
    }
    printf("\n");

    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));

    HIP_CHECK(hipFree(dell_ind));
    HIP_CHECK(hipFree(dell_val));

    free(hell_ind);
    free(hell_val);
    return 0;
}
//! [doc example]
