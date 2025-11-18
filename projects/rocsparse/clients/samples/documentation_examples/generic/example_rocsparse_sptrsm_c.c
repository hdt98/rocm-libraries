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
    //     1 0 0 0
    // A = 4 2 0 0
    //     0 3 7 0
    //     0 0 0 1
    int m = 4;
    int n = 2;

    int hcsr_row_ptr[] = {0, 1, 3, 5, 6};
    int hcsr_col_ind[] = {0, 0, 1, 1, 2, 3};
    float hcsr_val[] = {1, 4, 2, 3, 7, 1};
    float* hB = (float*)malloc(sizeof(float) * (m * n));
    float* hC = (float*)malloc(sizeof(float) * (m * n));

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            hB[m * i + j] = (float)(i + 1);
        }
    }

    // Scalar alpha
    float alpha = 1.0f;

    int nnz = hcsr_row_ptr[m] - hcsr_row_ptr[0];

    // Offload data to device
    int*   dcsr_row_ptr;
    int*   dcsr_col_ind;
    float* dcsr_val;
    float* dB;
    float* dC;
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsr_val, sizeof(float) * nnz));
    HIP_CHECK(hipMalloc((void**)&dB, sizeof(float) * m * n));
    HIP_CHECK(hipMalloc((void**)&dC, sizeof(float) * m * n));

    HIP_CHECK(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr, sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val, sizeof(float) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, sizeof(float) * m * n, hipMemcpyHostToDevice));

    rocsparse_handle      handle;
    rocsparse_spmat_descr matA;
    rocsparse_dnmat_descr matB;
    rocsparse_dnmat_descr matC;

    rocsparse_indextype  row_idx_type = rocsparse_indextype_i32;
    rocsparse_indextype  col_idx_type = rocsparse_indextype_i32;
    rocsparse_datatype   data_type    = rocsparse_datatype_f32_r;
    rocsparse_datatype   compute_type = rocsparse_datatype_f32_r;
    rocsparse_index_base idx_base     = rocsparse_index_base_zero;
    rocsparse_operation  trans_A      = rocsparse_operation_none;
    rocsparse_operation  trans_X      = rocsparse_operation_none;

    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create sparse matrix A
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&matA,
                                               m,
                                               m,
                                               nnz,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               dcsr_val,
                                               row_idx_type,
                                               col_idx_type,
                                               idx_base,
                                               data_type));

    // Create dense matrix B
    ROCSPARSE_CHECK(
        rocsparse_create_dnmat_descr(&matB, m, n, m, dB, data_type, rocsparse_order_column));

    // Create dense matrix C
    ROCSPARSE_CHECK(
        rocsparse_create_dnmat_descr(&matC, m, n, m, dC, data_type, rocsparse_order_column));

    rocsparse_sptrsm_descr sptrsm_descr;
    ROCSPARSE_CHECK(rocsparse_create_sptrsm_descr(&sptrsm_descr));

    const rocsparse_sptrsm_alg sptrsm_alg = rocsparse_sptrsm_alg_default;
    ROCSPARSE_CHECK(rocsparse_sptrsm_set_input(handle,
                                               sptrsm_descr,
                                               rocsparse_sptrsm_input_alg,
                                               &sptrsm_alg,
                                               sizeof(sptrsm_alg),
                                               NULL));

    ROCSPARSE_CHECK(rocsparse_sptrsm_set_input(handle,
                                               sptrsm_descr,
                                               rocsparse_sptrsm_input_operation_A,
                                               &trans_A,
                                               sizeof(trans_A),
                                               NULL));
    ROCSPARSE_CHECK(rocsparse_sptrsm_set_input(handle,
                                               sptrsm_descr,
                                               rocsparse_sptrsm_input_operation_X,
                                               &trans_X,
                                               sizeof(trans_X),
                                               NULL));

    const rocsparse_datatype sptrsm_scalar_datatype = rocsparse_datatype_f32_r;
    ROCSPARSE_CHECK(rocsparse_sptrsm_set_input(handle,
                                               sptrsm_descr,
                                               rocsparse_sptrsm_input_scalar_datatype,
                                               &sptrsm_scalar_datatype,
                                               sizeof(sptrsm_scalar_datatype),
                                               NULL));

    const rocsparse_datatype sptrsm_compute_datatype = rocsparse_datatype_f32_r;
    ROCSPARSE_CHECK(rocsparse_sptrsm_set_input(handle,
                                               sptrsm_descr,
                                               rocsparse_sptrsm_input_compute_datatype,
                                               &sptrsm_compute_datatype,
                                               sizeof(sptrsm_compute_datatype),
                                               NULL));

    const rocsparse_analysis_policy sptrsm_analysis_policy = rocsparse_analysis_policy_reuse;
    ROCSPARSE_CHECK(rocsparse_sptrsm_set_input(handle,
                                               sptrsm_descr,
                                               rocsparse_sptrsm_input_analysis_policy,
                                               &sptrsm_analysis_policy,
                                               sizeof(sptrsm_analysis_policy),
                                               NULL));

    size_t buffer_size;
    void*  temp_buffer;

    // Analysis phase
    ROCSPARSE_CHECK(rocsparse_sptrsm_buffer_size(handle,
                                                 sptrsm_descr,
                                                 matA,
                                                 matB,
                                                 matC,
                                                 rocsparse_sptrsm_stage_analysis,
                                                 &buffer_size,
                                                 NULL));

    HIP_CHECK(hipMalloc((void**)&temp_buffer, buffer_size));

    ROCSPARSE_CHECK(rocsparse_sptrsm(handle,
                                     sptrsm_descr,
                                     matA,
                                     matB,
                                     matC,
                                     rocsparse_sptrsm_stage_analysis,
                                     buffer_size,
                                     temp_buffer,
                                     NULL));

    HIP_CHECK(hipFree(temp_buffer));
    temp_buffer = NULL;

    int64_t zero_pivot;
    ROCSPARSE_CHECK(rocsparse_sptrsm_get_output(handle,
                                                sptrsm_descr,
                                                rocsparse_sptrsm_output_zero_pivot_position,
                                                &zero_pivot,
                                                sizeof(zero_pivot),
                                                NULL));
    if(zero_pivot != -1)
    {
        printf("zero pivot detected during analysis at position \n");
    }
    //
    // Compute phase.
    //
    ROCSPARSE_CHECK(rocsparse_sptrsm_set_input(handle,
                                               sptrsm_descr,
                                               rocsparse_sptrsm_input_scalar_alpha,
                                               &alpha,
                                               sizeof(&alpha),
                                               NULL));

    ROCSPARSE_CHECK(rocsparse_sptrsm_buffer_size(handle,
                                                 sptrsm_descr,
                                                 matA,
                                                 matB,
                                                 matC,
                                                 rocsparse_sptrsm_stage_compute,
                                                 &buffer_size,
                                                 NULL));

    HIP_CHECK(hipMalloc((void**)&temp_buffer, buffer_size));

    ROCSPARSE_CHECK(rocsparse_sptrsm(handle,
                                     sptrsm_descr,
                                     matA,
                                     matB,
                                     matC,
                                     rocsparse_sptrsm_stage_compute,
                                     buffer_size,
                                     temp_buffer,
                                     NULL));

    // Device synchronization
    hipStream_t stream;
    ROCSPARSE_CHECK(rocsparse_get_stream(handle, &stream));

    HIP_CHECK(hipStreamSynchronize(stream));

    ROCSPARSE_CHECK(rocsparse_sptrsm_get_output(handle,
                                                sptrsm_descr,
                                                rocsparse_sptrsm_output_zero_pivot_position,
                                                &zero_pivot,
                                                sizeof(zero_pivot),
                                                NULL));
    if(zero_pivot != -1)
    {
        printf("zero pivot detected during compute phase at position %d\n", zero_pivot);
    }

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matA));
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(matB));
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(matC));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_sptrsm_descr(sptrsm_descr));

    // Copy result back to host
    HIP_CHECK(hipMemcpy(hC, dC, sizeof(float) * m * n, hipMemcpyDeviceToHost));

    printf("hC\n");
    for(size_t i = 0; i < sizeof(hC)/sizeof(hC[0]); ++i)
    {
        printf("%f ", hC[i]);
    }
    printf("\n");

    // Clear device memory
    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    HIP_CHECK(hipFree(temp_buffer));

    free(hB);
    free(hC);

    return 0;
}
//! [doc example]
