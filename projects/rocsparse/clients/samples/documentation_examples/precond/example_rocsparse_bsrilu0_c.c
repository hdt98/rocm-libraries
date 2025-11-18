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

#define max(a,b) ((a) > (b) ? (a) : (b))

//! [doc example]
int main(int argc, char* argv[])
{
    //     2 1 0 0
    // A = 1 2 0 0
    //     0 0 2 1
    //     0 0 1 2

    int mb        = 2;
    int nb        = 2;
    int nnzb      = 2;
    int block_dim = 2;

    double alpha = 1.0;

    int hbsr_row_ptr[] = {0, 1, 2};
    int hbsr_col_ind[] = {0, 1};
    double hbsr_val[] = {2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0};

    double hx[mb * block_dim];

    int*    dbsr_row_ptr;
    int*    dbsr_col_ind;
    double* dbsr_val;
    double* dx;
    double* dy;
    double* dz;
    HIP_CHECK(hipMalloc((void**)&dbsr_row_ptr, sizeof(int) * (mb + 1)));
    HIP_CHECK(hipMalloc((void**)&dbsr_col_ind, sizeof(int) * nnzb));
    HIP_CHECK(hipMalloc((void**)&dbsr_val, sizeof(double) * nnzb * block_dim * block_dim));
    HIP_CHECK(hipMalloc((void**)&dx, sizeof(double) * mb * block_dim));
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(double) * mb * block_dim));
    HIP_CHECK(hipMalloc((void**)&dz, sizeof(double) * mb * block_dim));

    HIP_CHECK(hipMemcpy(
        dbsr_row_ptr, hbsr_row_ptr, sizeof(int) * (mb + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dbsr_col_ind, hbsr_col_ind, sizeof(int) * nnzb, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbsr_val,
                        hbsr_val,
                        sizeof(double) * nnzb * block_dim * block_dim,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx, sizeof(double) * mb * block_dim, hipMemcpyHostToDevice));

    // Create rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create matrix descriptor for M
    rocsparse_mat_descr descr_M;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_M));

    // Create matrix descriptor for L
    rocsparse_mat_descr descr_L;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_L));
    ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descr_L, rocsparse_fill_mode_lower));
    ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descr_L, rocsparse_diag_type_unit));

    // Create matrix descriptor for U
    rocsparse_mat_descr descr_U;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_U));
    ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descr_U, rocsparse_fill_mode_upper));
    ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descr_U, rocsparse_diag_type_non_unit));

    // Create matrix info structure
    rocsparse_mat_info info;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info));

    // Obtain required buffer size
    size_t buffer_size_M;
    size_t buffer_size_L;
    size_t buffer_size_U;
    ROCSPARSE_CHECK(rocsparse_dbsrilu0_buffer_size(handle,
                                                   rocsparse_direction_row,
                                                   mb,
                                                   nnzb,
                                                   descr_M,
                                                   dbsr_val,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   block_dim,
                                                   info,
                                                   &buffer_size_M));
    ROCSPARSE_CHECK(rocsparse_dbsrsv_buffer_size(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_none,
                                                 mb,
                                                 nnzb,
                                                 descr_L,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 block_dim,
                                                 info,
                                                 &buffer_size_L));
    ROCSPARSE_CHECK(rocsparse_dbsrsv_buffer_size(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_none,
                                                 mb,
                                                 nnzb,
                                                 descr_U,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 block_dim,
                                                 info,
                                                 &buffer_size_U));

    size_t buffer_size = max(buffer_size_M, max(buffer_size_L, buffer_size_U));

    // Allocate temporary buffer
    void* temp_buffer;
    HIP_CHECK(hipMalloc((void**)&temp_buffer, buffer_size));

    // Perform analysis steps, using rocsparse_analysis_policy_reuse to improve
    // computation performance
    ROCSPARSE_CHECK(rocsparse_dbsrilu0_analysis(handle,
                                                rocsparse_direction_row,
                                                mb,
                                                nnzb,
                                                descr_M,
                                                dbsr_val,
                                                dbsr_row_ptr,
                                                dbsr_col_ind,
                                                block_dim,
                                                info,
                                                rocsparse_analysis_policy_reuse,
                                                rocsparse_solve_policy_auto,
                                                temp_buffer));
    ROCSPARSE_CHECK(rocsparse_dbsrsv_analysis(handle,
                                              rocsparse_direction_row,
                                              rocsparse_operation_none,
                                              mb,
                                              nnzb,
                                              descr_L,
                                              dbsr_val,
                                              dbsr_row_ptr,
                                              dbsr_col_ind,
                                              block_dim,
                                              info,
                                              rocsparse_analysis_policy_reuse,
                                              rocsparse_solve_policy_auto,
                                              temp_buffer));
    ROCSPARSE_CHECK(rocsparse_dbsrsv_analysis(handle,
                                              rocsparse_direction_row,
                                              rocsparse_operation_none,
                                              mb,
                                              nnzb,
                                              descr_U,
                                              dbsr_val,
                                              dbsr_row_ptr,
                                              dbsr_col_ind,
                                              block_dim,
                                              info,
                                              rocsparse_analysis_policy_reuse,
                                              rocsparse_solve_policy_auto,
                                              temp_buffer));

    // Check for zero pivot
    rocsparse_int position;
    if(rocsparse_status_zero_pivot == rocsparse_bsrilu0_zero_pivot(handle, info, &position))
    {
        printf("A has structural zero at A(%d,%d)\n", position, position);
    }

    // Compute incomplete LU factorization M = LU
    ROCSPARSE_CHECK(rocsparse_dbsrilu0(handle,
                                       rocsparse_direction_row,
                                       mb,
                                       nnzb,
                                       descr_M,
                                       dbsr_val,
                                       dbsr_row_ptr,
                                       dbsr_col_ind,
                                       block_dim,
                                       info,
                                       rocsparse_solve_policy_auto,
                                       temp_buffer));

    // Check for zero pivot
    if(rocsparse_status_zero_pivot == rocsparse_bsrilu0_zero_pivot(handle, info, &position))
    {
        printf("L has structural and/or numerical zero at L(%d,%d)\n", position, position);
    }

    // Solve Lz = x
    ROCSPARSE_CHECK(rocsparse_dbsrsv_solve(handle,
                                           rocsparse_direction_row,
                                           rocsparse_operation_none,
                                           mb,
                                           nnzb,
                                           &alpha,
                                           descr_L,
                                           dbsr_val,
                                           dbsr_row_ptr,
                                           dbsr_col_ind,
                                           block_dim,
                                           info,
                                           dx,
                                           dz,
                                           rocsparse_solve_policy_auto,
                                           temp_buffer));

    // Solve Uy = z
    ROCSPARSE_CHECK(rocsparse_dbsrsv_solve(handle,
                                           rocsparse_direction_row,
                                           rocsparse_operation_none,
                                           mb,
                                           nnzb,
                                           &alpha,
                                           descr_U,
                                           dbsr_val,
                                           dbsr_row_ptr,
                                           dbsr_col_ind,
                                           block_dim,
                                           info,
                                           dz,
                                           dy,
                                           rocsparse_solve_policy_auto,
                                           temp_buffer));

    double* hy = (double*)malloc(sizeof(double) * (mb * block_dim));
    HIP_CHECK(hipMemcpy(hy, dy, sizeof(double) * mb * block_dim, hipMemcpyDeviceToHost));

    printf("hy\n");
    for(size_t i = 0; i < sizeof(hy)/sizeof(hy[0]); i++)
    {
        printf("%f ", hy[i]);
    }
    printf("\n");

    // Clean up
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_M));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_L));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_U));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    HIP_CHECK(hipFree(dbsr_row_ptr));
    HIP_CHECK(hipFree(dbsr_col_ind));
    HIP_CHECK(hipFree(dbsr_val));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy));
    HIP_CHECK(hipFree(dz));
    HIP_CHECK(hipFree(temp_buffer));

    free(hy);
    return 0;
}
//! [doc example]
