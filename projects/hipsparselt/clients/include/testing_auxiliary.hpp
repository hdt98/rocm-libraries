/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
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

#pragma once

#include "flops.hpp"
#include "hipsparselt_datatype2string.hpp"
#include "hipsparselt_init.hpp"
#include "hipsparselt_math.hpp"
#include "hipsparselt_random.hpp"
#include "hipsparselt_test.hpp"
#include "hipsparselt_vector.hpp"
#include "rocsparselt-types.h"
#include "status.h"
#include "tensile_host.hpp"
#include "unit.hpp"
#include "utility.hpp"
#include <hipsparselt/hipsparselt.h>

void testing_aux_get_version(const Arguments& arg)
{
    static int version;
    hipsparselt_local_handle handle;
    hipsparseLtGetVersion(handle, &version);

    int major;
    int minor;
    int patch;
    hipsparseLtGetProperty(HIP_LIBRARY_MAJOR_VERSION, &major);
    hipsparseLtGetProperty(HIP_LIBRARY_MINOR_VERSION, &minor);
    hipsparseLtGetProperty(HIP_LIBRARY_PATCH_LEVEL, &patch);
    int version_ = major * 100000 + minor * 100 + patch;
    ASSERT_EQ(version, version_);

    char *rev = nullptr;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtGetGitRevision(handle, rev), HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_handle_init_bad_arg(const Arguments& arg)
{
    EXPECT_HIPSPARSE_STATUS(hipsparseLtInit(nullptr), HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_handle_destroy_bad_arg(const Arguments& arg)
{
    hipsparseLtHandle_t handle;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtDestroy(&handle), HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtDestroy(nullptr), HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_handle(const Arguments& arg)
{
    hipsparseLtHandle_t handle;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtInit(&handle), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtDestroy(&handle), HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_mat_init_dense_bad_arg(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipsparseLtHandle_t        handle;
    hipsparseLtMatDescriptor_t m_descr;

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtDenseDescriptorInit(
            &handle, &m_descr, row, col, ld, 16, arg.a_type, HIPSPARSE_ORDER_COL),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtDenseDescriptorInit(
            nullptr, &m_descr, row, col, ld, 16, arg.a_type, HIPSPARSE_ORDER_COL),
        HIPSPARSE_STATUS_INVALID_VALUE);
    hipsparselt_local_handle handle_{arg};

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtDenseDescriptorInit(
            handle_, nullptr, row, col, ld, 16, arg.a_type, HIPSPARSE_ORDER_COL),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtDenseDescriptorInit(
                                handle_, &m_descr, 0, col, ld, 16, arg.a_type, HIPSPARSE_ORDER_COL),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtDenseDescriptorInit(
                                handle_, &m_descr, row, 0, ld, 16, arg.a_type, HIPSPARSE_ORDER_COL),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtDenseDescriptorInit(
            handle_, &m_descr, row, col, 0, 16, arg.a_type, HIPSPARSE_ORDER_COL),
        HIPSPARSE_STATUS_INVALID_VALUE);

#ifdef __HIP_PLATFORM_NVIDIA__
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtDenseDescriptorInit(
            handle_, &m_descr, row, col, 129, 16, arg.a_type, HIPSPARSE_ORDER_COL),
        HIPSPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtDenseDescriptorInit(
            handle_, &m_descr, row, col, ld, 17, arg.a_type, HIPSPARSE_ORDER_COL),
        HIPSPARSE_STATUS_NOT_SUPPORTED);
#endif
}

void testing_aux_mat_init_structured_bad_arg(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipsparseLtHandle_t        handle_;
    hipsparseLtMatDescriptor_t m_descr;

    EXPECT_HIPSPARSE_STATUS(hipsparseLtStructuredDescriptorInit(&handle_,
                                                                &m_descr,
                                                                row,
                                                                col,
                                                                ld,
                                                                16,
                                                                arg.a_type,
                                                                HIPSPARSE_ORDER_COL,
                                                                HIPSPARSELT_SPARSITY_50_PERCENT),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtStructuredDescriptorInit(nullptr,
                                                                &m_descr,
                                                                row,
                                                                col,
                                                                ld,
                                                                16,
                                                                arg.a_type,
                                                                HIPSPARSE_ORDER_COL,
                                                                HIPSPARSELT_SPARSITY_50_PERCENT),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    hipsparselt_local_handle handle{arg};
    EXPECT_HIPSPARSE_STATUS(hipsparseLtStructuredDescriptorInit(handle,
                                                                nullptr,
                                                                row,
                                                                col,
                                                                ld,
                                                                16,
                                                                arg.a_type,
                                                                HIPSPARSE_ORDER_COL,
                                                                HIPSPARSELT_SPARSITY_50_PERCENT),
                            HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtStructuredDescriptorInit(handle,
                                                                &m_descr,
                                                                0,
                                                                col,
                                                                ld,
                                                                16,
                                                                arg.a_type,
                                                                HIPSPARSE_ORDER_COL,
                                                                HIPSPARSELT_SPARSITY_50_PERCENT),
                            HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtStructuredDescriptorInit(handle,
                                                                &m_descr,
                                                                6,
                                                                col,
                                                                ld,
                                                                16,
                                                                arg.a_type,
                                                                HIPSPARSE_ORDER_COL,
                                                                HIPSPARSELT_SPARSITY_50_PERCENT),
                            HIPSPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtStructuredDescriptorInit(handle,
                                                                &m_descr,
                                                                row,
                                                                0,
                                                                ld,
                                                                16,
                                                                arg.a_type,
                                                                HIPSPARSE_ORDER_COL,
                                                                HIPSPARSELT_SPARSITY_50_PERCENT),
                            HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtStructuredDescriptorInit(handle,
                                                                &m_descr,
                                                                row,
                                                                6,
                                                                ld,
                                                                16,
                                                                arg.a_type,
                                                                HIPSPARSE_ORDER_COL,
                                                                HIPSPARSELT_SPARSITY_50_PERCENT),
                            HIPSPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtStructuredDescriptorInit(handle,
                                                                &m_descr,
                                                                row,
                                                                col,
                                                                0,
                                                                16,
                                                                arg.a_type,
                                                                HIPSPARSE_ORDER_COL,
                                                                HIPSPARSELT_SPARSITY_50_PERCENT),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtStructuredDescriptorInit(handle,
                                                                &m_descr,
                                                                row,
                                                                col,
                                                                129,
                                                                16,
                                                                arg.a_type,
                                                                HIPSPARSE_ORDER_COL,
                                                                HIPSPARSELT_SPARSITY_50_PERCENT),

#ifdef __HIP_PLATFORM_NVIDIA__
                            HIPSPARSE_STATUS_NOT_SUPPORTED
#else
                            HIPSPARSE_STATUS_SUCCESS
#endif
    );

    EXPECT_HIPSPARSE_STATUS(hipsparseLtStructuredDescriptorInit(handle,
                                                                &m_descr,
                                                                row,
                                                                col,
                                                                127,
                                                                16,
                                                                arg.a_type,
                                                                HIPSPARSE_ORDER_COL,
                                                                HIPSPARSELT_SPARSITY_50_PERCENT),
                            HIPSPARSE_STATUS_INVALID_VALUE);

#ifdef __HIP_PLATFORM_NVIDIA__
    EXPECT_HIPSPARSE_STATUS(hipsparseLtStructuredDescriptorInit(handle,
                                                                &m_descr,
                                                                row,
                                                                col,
                                                                ld,
                                                                17,
                                                                arg.a_type,
                                                                HIPSPARSE_ORDER_COL,
                                                                HIPSPARSELT_SPARSITY_50_PERCENT),
                            HIPSPARSE_STATUS_NOT_SUPPORTED);
#endif

    // The row and column must be mulitples of num_elements.
    int num_elements = 8;
    switch(arg.a_type)
    {
    case HIP_R_8I:
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
    case HIP_R_8F_E5M2:
#endif
        num_elements = 16;
        break;
    default:
        break;
    }

    EXPECT_HIPSPARSE_STATUS(hipsparseLtStructuredDescriptorInit(handle,
                                                                &m_descr,
                                                                num_elements * 4 + 4,
                                                                col,
                                                                ld,
                                                                16,
                                                                arg.a_type,
                                                                HIPSPARSE_ORDER_COL,
                                                                HIPSPARSELT_SPARSITY_50_PERCENT),
                            HIPSPARSE_STATUS_NOT_SUPPORTED);

    // Chcek unsupported datatype
    EXPECT_HIPSPARSE_STATUS(hipsparseLtStructuredDescriptorInit(handle,
                                                                &m_descr,
                                                                row,
                                                                col,
                                                                ld,
                                                                16,
                                                                HIP_R_64F,
                                                                HIPSPARSE_ORDER_COL,
                                                                HIPSPARSELT_SPARSITY_50_PERCENT),
                            HIPSPARSE_STATUS_NOT_SUPPORTED);
}

void testing_aux_mat_dense_init(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_dense, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_mat_structured_init(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_mat_assign(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    int data  = 1;
    int data2 = 0;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescSetAttribute(
                                handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(data)),
                            HIPSPARSE_STATUS_SUCCESS);

    // CHECK mat2 is a copy of mat
    hipsparseLtMatDescriptor_t mat2 = mat;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescGetAttribute(
                                handle, &mat2, HIPSPARSELT_MAT_NUM_BATCHES, &data2, sizeof(data2)),
                            HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(data == data2);

    // CHECK mat2 is not a reference of mat
    data2 = 10;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescSetAttribute(
                                handle, &mat2, HIPSPARSELT_MAT_NUM_BATCHES, &data2, sizeof(data2)),
                            HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescGetAttribute(
                                handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(data)),
                            HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(data != data2);
}

void testing_aux_mat_destroy_bad_arg(const Arguments& arg)
{
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescriptorDestroy(&m_descr), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescriptorDestroy(nullptr),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_set_attr_bad_arg(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);

    int     data;
    int64_t data64;

    hipsparseLtHandle_t        handle_;
    hipsparseLtMatDescriptor_t mat_;

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescSetAttribute(
                                nullptr, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescSetAttribute(
                                &handle_, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescSetAttribute(
                                handle, nullptr, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescSetAttribute(
                                handle, &mat_, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescSetAttribute(
                                handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, nullptr, sizeof(int)),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescSetAttribute(
                                handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, nullptr, sizeof(int)),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    data = 0;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescSetAttribute(
                                handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    data = 1;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(
            handle, mat, HIPSPARSELT_MAT_BATCH_STRIDE, nullptr, sizeof(int64_t)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    data64 = 2;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(
            handle, mat, HIPSPARSELT_MAT_BATCH_STRIDE, &data64, sizeof(int64_t)),
        HIPSPARSE_STATUS_INVALID_VALUE);

    data64 = ld * col;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(handle, mat, HIPSPARSELT_MAT_BATCH_STRIDE, &data64, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_get_attr_bad_arg(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);

    int     data;
    int64_t data64;

    hipsparseLtHandle_t        handle_;
    hipsparseLtMatDescriptor_t mat_;

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescGetAttribute(
                                nullptr, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescGetAttribute(
                                &handle_, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescGetAttribute(
                                handle, nullptr, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescGetAttribute(
                                handle, &mat_, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescGetAttribute(
                                handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, nullptr, sizeof(int)),
                            HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescGetAttribute(handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescGetAttribute(
            handle, mat, HIPSPARSELT_MAT_BATCH_STRIDE, nullptr, sizeof(int64_t)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescGetAttribute(
                                handle, mat, HIPSPARSELT_MAT_BATCH_STRIDE, &data64, sizeof(int)),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_set_get_attr(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);

    int data, data_r;

    data = 2;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescSetAttribute(
                                handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
                            HIPSPARSE_STATUS_SUCCESS);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescGetAttribute(
                                handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data_r, sizeof(int)),
                            HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(data_r == data);

    std::vector<int64_t> data64_v = {0, ld * col};
    int64_t              data64_r = 0;
    for(int64_t data64 : data64_v)
    {
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, mat, HIPSPARSELT_MAT_BATCH_STRIDE, &data64, sizeof(int64_t)),
            HIPSPARSE_STATUS_SUCCESS);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescGetAttribute(
                handle, mat, HIPSPARSELT_MAT_BATCH_STRIDE, &data64_r, sizeof(int64_t)),
            HIPSPARSE_STATUS_SUCCESS);
        ASSERT_TRUE(data64_r == data64);
    }
}

void testing_aux_matmul_init_bad_arg(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, K, M, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparseLtHandle_t           handle_;
    hipsparseLtMatDescriptor_t    mat_;
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            nullptr, &m_descr, opA, opB, matA, matB, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            &handle_, &m_descr, opA, opB, matA, matB, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            handle, nullptr, opA, opB, matA, matB, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulDescriptorInit(handle,
                                                            &m_descr,
                                                            HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                                                            opB,
                                                            matA,
                                                            matB,
                                                            matC,
                                                            matD,
                                                            arg.compute_type),
                            HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulDescriptorInit(handle,
                                                            &m_descr,
                                                            opA,
                                                            HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                                                            matA,
                                                            matB,
                                                            matC,
                                                            matD,
                                                            arg.compute_type),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            handle, &m_descr, opA, opB, &mat_, matB, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            handle, &m_descr, opA, opB, nullptr, matB, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            handle, &m_descr, opA, opB, matA, &mat_, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            handle, &m_descr, opA, opB, matA, nullptr, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            handle, &m_descr, opA, opB, matA, matB, &mat_, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            handle, &m_descr, opA, opB, matA, matB, nullptr, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            handle, &m_descr, opA, opB, matA, matB, matC, &mat_, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            handle, &m_descr, opA, opB, matA, matB, matC, nullptr, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);

#ifdef __HIP_PLATFORM_NVIDIA__
    if(arg.a_type == HIP_R_8I)
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescriptorInit(
                handle, &m_descr, opA, opA, matA, matB, matC, matD, arg.compute_type),
            HIPSPARSE_STATUS_NOT_SUPPORTED);
#endif

    hipsparseLtComputetype_t tmpComputeType;
    switch(arg.a_type)
    {
    case HIP_R_16F:
    case HIP_R_16BF:
        tmpComputeType = HIPSPARSELT_COMPUTE_32I;
        break;
    default:
#ifdef __HIP_PLATFORM_AMD__
        tmpComputeType = HIPSPARSELT_COMPUTE_32F;
#else
        tmpComputeType = HIPSPARSELT_COMPUTE_16F;
#endif
        break;
    }
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulDescriptorInit(
                                handle, &m_descr, opA, opB, matA, matB, matC, matD, tmpComputeType),
                            HIPSPARSE_STATUS_NOT_SUPPORTED);

    hipsparselt_local_mat_descr mat_128_112(
        hipsparselt_matrix_type_dense, handle, 128, 112, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat_128_112.status(), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            handle, &m_descr, opA, opB, matA, mat_128_112, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);

    hipsparselt_local_mat_descr mat_112_112(
        hipsparselt_matrix_type_dense, handle, 112, 112, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat_112_112.status(), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            handle, &m_descr, opA, opB, matA, mat_112_112, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);

    //Singal abort at CUDA backend
#ifdef __HIP_PLATFORM_AMD__
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            handle, &m_descr, opA, opB, matA, matB, mat_112_112, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            handle, &m_descr, opA, opB, matA, matB, matC, mat_112_112, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
#endif

    hipsparselt_local_mat_descr matA_(
        hipsparselt_matrix_type_structured, handle, K, N, ldb, HIP_R_32F, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA_.status(), HIPSPARSE_STATUS_SUCCESS);

    // HIP_R_32F is unsupported.
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            handle, &m_descr, opA, opB, matA_, matB, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_NOT_SUPPORTED);

    // C or D must be dense martrix
    hipsparselt_local_mat_descr matCS_(
        hipsparselt_matrix_type_structured, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matCS_.status(), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            handle, &m_descr, opA, opB, matA, matB, matCS_, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);

    // Only one of the A or B can be structured sparsity matrix
    hipsparselt_local_mat_descr matBS(
        hipsparselt_matrix_type_structured, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matBS.status(), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            handle, &m_descr, opA, opB, matA, matBS, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_NOT_SUPPORTED);

    //
    hipDataType tmpDataType;
    auto        get_diff_datatype = [&](hipDataType type) {
        switch(type)
        {
        case HIP_R_16BF:
            return HIP_R_16F;
        default:
            return HIP_R_16BF;
        }
    };

    tmpDataType = get_diff_datatype(arg.b_type);
    hipsparselt_local_mat_descr matB_(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, tmpDataType, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB_.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC_(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, tmpDataType, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC_.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD_(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, tmpDataType, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD_.status(), HIPSPARSE_STATUS_SUCCESS);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            handle, &m_descr, opA, opB, matA, matB_, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_NOT_SUPPORTED);

#ifdef __HIP_PLATFORM_AMD__
    if(arg.a_type != HIP_R_8I)
    {
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescriptorInit(
                handle, &m_descr, opA, opB, matA, matB, matC_, matD, arg.compute_type),
            HIPSPARSE_STATUS_NOT_SUPPORTED);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescriptorInit(
                handle, &m_descr, opA, opB, matA, matB, matC, matD_, arg.compute_type),
            HIPSPARSE_STATUS_NOT_SUPPORTED);
    }
#endif

    // check the C and D matrices has the same memory order.
    hipsparselt_local_mat_descr matDR_(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_ROW);
    EXPECT_HIPSPARSE_STATUS(matDR_.status(), HIPSPARSE_STATUS_SUCCESS);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            handle, &m_descr, opA, opB, matA, matB, matC, matDR_, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, M, K, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, opA, opB, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_matmul_set_attr_bad_arg(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, K, M, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, opA, opB, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparseLtHandle_t           handle_;
    hipsparseLtMatmulDescriptor_t matmul_;

    int data   = 0;
    int data64 = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            nullptr, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            &handle_, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, nullptr, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, &matmul_, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, nullptr, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulDescSetAttribute(
                                handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, 1),
                            HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data64, sizeof(data64)),
        HIPSPARSE_STATUS_SUCCESS);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, nullptr, sizeof(data64)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND, &data64, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);

#ifdef __HIP_PLATFORM_AMD__
    if(arg.d_type == HIP_R_8I)
    {
        int dataSigmoid = 1;
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescSetAttribute(
                handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID, &dataSigmoid, sizeof(dataSigmoid)),
            HIPSPARSE_STATUS_NOT_SUPPORTED);
    }
#endif

    void* dBias;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_BIAS_POINTER, &dBias, sizeof(dBias) - 1),
        HIPSPARSE_STATUS_INVALID_VALUE);

    int64_t bias_stride = M - 1;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_BIAS_STRIDE, &bias_stride, sizeof(bias_stride)),
        HIPSPARSE_STATUS_INVALID_VALUE);

#ifdef __HIP_PLATFORM_AMD__
        char bias_type;
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescSetAttribute(
                handle, matmul, HIPSPARSELT_MATMUL_BIAS_TYPE, &bias_type, sizeof(char)),
            HIPSPARSE_STATUS_INVALID_VALUE);
#endif
}

void testing_aux_matmul_get_attr_bad_arg(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, K, M, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, opA, opB, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparseLtHandle_t           handle_;
    hipsparseLtMatmulDescriptor_t matmul_;

    int data   = 0;
    int data64 = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            nullptr, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            &handle_, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, nullptr, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, &matmul_, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, nullptr, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulDescGetAttribute(
                                handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, 1),
                            HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data64, sizeof(data64)),
        HIPSPARSE_STATUS_SUCCESS);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND, nullptr, sizeof(data64)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND, &data64, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD, nullptr, sizeof(data64)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD, &data64, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU_ALPHA, nullptr, sizeof(data64)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU_ALPHA, &data64, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_TANH_ALPHA, nullptr, sizeof(data64)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_TANH_ALPHA, &data64, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_TANH_BETA, nullptr, sizeof(data64)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_TANH_BETA, &data64, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);

    size_t bad_ptr_size = sizeof(void*) - 1;
    void* dBias;
    CHECK_HIP_ERROR(hipMalloc((void**)&dBias, (M) * sizeof(float)));
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulDescGetAttribute(
                                handle, matmul, HIPSPARSELT_MATMUL_BIAS_POINTER, &dBias, bad_ptr_size),
                            HIPSPARSE_STATUS_INVALID_VALUE);
    CHECK_HIP_ERROR(hipFree(dBias));

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_BIAS_STRIDE, nullptr, sizeof(data64)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_BIAS_STRIDE, &data64, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);

#ifdef __HIP_PLATFORM_AMD__
    hipDataType biasType;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_BIAS_STRIDE, nullptr, sizeof(biasType)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_BIAS_STRIDE, &biasType, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
#endif

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, nullptr, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &data, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);

#ifdef __HIP_PLATFORM_AMD__
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_BETA_VECTOR_SCALING, &data, sizeof(data)),
        HIPSPARSE_STATUS_NOT_SUPPORTED);
#endif
}

void testing_aux_matmul_set_get_bias_vector(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, K, M, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, opA, opB, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    device_vector<float> dBias(M, 1);
    CHECK_DEVICE_ALLOCATION(dBias.memcheck());
    host_vector<float> hBias_gold(M);
    host_vector<float> hBias(M);

    hipsparselt_seedrand();
    hipsparselt_init<float>(hBias_gold, M, 1, M, M, 1);
    CHECK_HIP_ERROR(dBias.transfer_from(hBias_gold));

    void* _dBias = dBias;

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_BIAS_POINTER, &_dBias, sizeof(void*)),
        HIPSPARSE_STATUS_SUCCESS);

    void* dBias_r;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_BIAS_POINTER, &dBias_r, sizeof(void*)),
        HIPSPARSE_STATUS_SUCCESS);

    CHECK_HIP_ERROR(hipMemcpy(hBias, dBias_r, sizeof(float) * M, hipMemcpyDeviceToHost));

    unit_check_general<float>(M, 1, M, M, hBias_gold, hBias, 1);
}

void testing_aux_matmul_set_get_attr(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, K, M, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, opA, opB, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparseLtHandle_t           handle_;
    hipsparseLtMatmulDescriptor_t matmul_;

    int   data = 1, data_r = 0;
    float dataf = 1.0f, dataf_r = 0.0f;

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data_r, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(data == data_r);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND, &dataf, sizeof(dataf)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND, &dataf_r, sizeof(dataf)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(dataf == dataf_r);
}

void testing_aux_matmul_assign(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, M, K, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparseLtMatmulDescriptor_t matmul;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(
            handle, &matmul, opA, opB, matA, matB, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_SUCCESS);

    int data = 1, data_r = 0;

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, &matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);

    {
        hipsparseLtMatmulDescriptor_t lMatmul = matmul;
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescGetAttribute(
                handle, &lMatmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data_r, sizeof(data)),
            HIPSPARSE_STATUS_SUCCESS);
        ASSERT_TRUE(data == data_r);
        data_r = 100;
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescSetAttribute(
                handle, &lMatmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data_r, sizeof(data_r)),
            HIPSPARSE_STATUS_SUCCESS);
    }

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, &matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(data != data_r);
}

void testing_aux_matmul_alg_init_bad_arg(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparseLtHandle_t             handle_;
    hipsparseLtMatmulDescriptor_t   matmul_;
    hipsparseLtMatmulAlgSelection_t alg_sel;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, K, M, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, opA, opB, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulAlgSelectionInit(
                                nullptr, &alg_sel, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulAlgSelectionInit(
                                &handle_, &alg_sel, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSelectionInit(handle, nullptr, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulAlgSelectionInit(
                                handle, &alg_sel, nullptr, HIPSPARSELT_MATMUL_ALG_DEFAULT),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulAlgSelectionInit(
                                handle, &alg_sel, &matmul_, HIPSPARSELT_MATMUL_ALG_DEFAULT),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_init(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, K, M, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, opA, opB, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    EXPECT_HIPSPARSE_STATUS(alg_sel.status(), HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_matmul_alg_assign(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, K, M, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, opA, opB, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    EXPECT_HIPSPARSE_STATUS(alg_sel.status(), HIPSPARSE_STATUS_SUCCESS);

    int data  = 20;
    int data2 = 0;

    // CHECK alg_sel2 is copy from alg_sel
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(
            handle, alg_sel, HIPSPARSELT_MATMUL_SEARCH_ITERATIONS, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);

    hipsparseLtMatmulAlgSelection_t alg_sel2 = alg_sel;

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgGetAttribute(
            handle, &alg_sel2, HIPSPARSELT_MATMUL_SEARCH_ITERATIONS, &data2, sizeof(data2)),
        HIPSPARSE_STATUS_SUCCESS);

    ASSERT_TRUE(data2 == data);

    // CHECK alg_sel2 is not a referenece of alg_sel
    data2 = 100;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(
            handle, &alg_sel2, HIPSPARSELT_MATMUL_SEARCH_ITERATIONS, &data2, sizeof(data2)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgGetAttribute(
            handle, alg_sel, HIPSPARSELT_MATMUL_SEARCH_ITERATIONS, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(data2 != data);
}

void testing_aux_matmul_alg_set_attr_bad_arg(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, K, M, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, opA, opB, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    EXPECT_HIPSPARSE_STATUS(alg_sel.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparseLtHandle_t             handle_;
    hipsparseLtMatmulAlgSelection_t alg_sel_;

    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(
            nullptr, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(
            &handle_, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(
            handle, nullptr, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(
            handle, &alg_sel_, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(
            handle, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);

#ifdef __HIP_PLATFORM_AMD__
    //TODO hip backend not support split k yet. Remove this test once hip backend support splitk
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulAlgSetAttribute(
                                handle, alg_sel, HIPSPARSELT_MATMUL_SPLIT_K, &data, sizeof(data)),
                            HIPSPARSE_STATUS_NOT_SUPPORTED);
#endif

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(
            handle, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, nullptr, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulAlgSetAttribute(
                                handle, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, 1),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    data = 100;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(
            handle, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(
            handle, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(
            handle, alg_sel, HIPSPARSELT_MATMUL_SEARCH_ITERATIONS, &data, 1),
            HIPSPARSE_STATUS_INVALID_VALUE);
    data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(
            handle, alg_sel, HIPSPARSELT_MATMUL_SEARCH_ITERATIONS, &data, sizeof(data)),
            HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_get_attr_bad_arg(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, K, M, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, opA, opB, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    EXPECT_HIPSPARSE_STATUS(alg_sel.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparseLtHandle_t             handle_;
    hipsparseLtMatmulAlgSelection_t alg_sel_;

    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgGetAttribute(
            nullptr, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgGetAttribute(
            &handle_, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgGetAttribute(
            handle, nullptr, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgGetAttribute(
            handle, &alg_sel_, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgGetAttribute(
            handle, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, nullptr, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulAlgGetAttribute(
                                handle, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, 1),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_plan_init_bad_arg(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, K, M, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, opA, opB, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    EXPECT_HIPSPARSE_STATUS(alg_sel.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparseLtHandle_t             handle_;
    hipsparseLtMatmulDescriptor_t   matmul_;
    hipsparseLtMatmulAlgSelection_t alg_sel_;
    hipsparseLtMatmulPlan_t         plan;

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanInit(nullptr, &plan, matmul, alg_sel),
                            HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanInit(&handle_, &plan, matmul, alg_sel),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanInit(handle, nullptr, matmul, alg_sel),
                            HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanInit(handle, &plan, &matmul_, alg_sel),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanInit(handle, &plan, nullptr, alg_sel),
                            HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanInit(handle, &plan, &matmul_, alg_sel),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanInit(handle, &plan, matmul, nullptr),
                            HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanInit(handle, &plan, matmul, &alg_sel_),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    // check the A and B matrices has the same value of num_batches.
    int num_batches_a = 2;
    int num_batches_b = 3;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescSetAttribute(
            handle, matA, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches_a, sizeof(num_batches_a)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescSetAttribute(
            handle, matB, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches_b, sizeof(num_batches_b)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanInit(handle, &plan, matmul, alg_sel),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_plan_destroy_bad_arg(const Arguments& arg)
{
    hipsparseLtMatmulPlan_t plan;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanDestroy(nullptr), HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanDestroy(&plan), HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_plan_init(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, K, M, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, opA, opB, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    EXPECT_HIPSPARSE_STATUS(alg_sel.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_plan plan(handle, matmul, alg_sel);
    EXPECT_HIPSPARSE_STATUS(plan.status(), HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_get_workspace_size_bad_arg(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, K, M, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, opA, opB, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    EXPECT_HIPSPARSE_STATUS(alg_sel.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_plan plan(handle, matmul, alg_sel);
    EXPECT_HIPSPARSE_STATUS(plan.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparseLtHandle_t     handle_;
    hipsparseLtMatmulPlan_t plan_;
    size_t                  workspace_size = 0;

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulGetWorkspace(nullptr, plan, &workspace_size),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulGetWorkspace(&handle_, plan, &workspace_size),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulGetWorkspace(handle, nullptr, &workspace_size),
                            HIPSPARSE_STATUS_INVALID_VALUE);

#ifdef __HIP_PLATFORM_AMD__

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulGetWorkspace(handle, &plan_, &workspace_size),
                            HIPSPARSE_STATUS_INVALID_VALUE);

#endif

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulGetWorkspace(handle, plan, nullptr),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_get_workspace_size(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, K, M, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, opA, opB, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    EXPECT_HIPSPARSE_STATUS(alg_sel.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_plan plan(handle, matmul, alg_sel);
    EXPECT_HIPSPARSE_STATUS(plan.status(), HIPSPARSE_STATUS_SUCCESS);

    size_t workspace_size = 0;

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulGetWorkspace(handle, plan, &workspace_size),
                            HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_string_helper(const Arguments& arg)
{
    constexpr int invalid_enum = -1;

    const std::pair<const char*, int> hip_datatype_cases[] = {
        {"f32_r", HIP_R_32F},
        {"s", HIP_R_32F},
        {"f16_r", HIP_R_16F},
        {"h", HIP_R_16F},
        {"bf16_r", HIP_R_16BF},
        {"i8_r", HIP_R_8I},
#if defined(HIP_FP8_TYPE_OCP) || defined(__HIP_PLATFORM_NVIDIA__)
        {"f8_r", HIP_R_8F_E4M3},
        {"bf8_r", HIP_R_8F_E5M2},
#endif
        {"invalid", invalid_enum},
        {"", invalid_enum},
        {"f64_r", invalid_enum},
        {"unknown", invalid_enum},
        {"F32_R", invalid_enum},
    };

    for(const auto& test : hip_datatype_cases)
    {
        EXPECT_EQ(string_to_hip_datatype(test.first), test.second);
    }

    const std::pair<const char*, int> str_to_compute_type_cases[] = {
        {"f32_r", HIPSPARSELT_COMPUTE_32F},
        {"s", HIPSPARSELT_COMPUTE_32F},
        {"i32_r", HIPSPARSELT_COMPUTE_32I},
        {"f16_r", HIPSPARSELT_COMPUTE_16F},
        {"h", HIPSPARSELT_COMPUTE_16F},
        {"tf32_r", HIPSPARSELT_COMPUTE_TF32},
        {"tf32f_r", HIPSPARSELT_COMPUTE_TF32_FAST},
        {"invalid", invalid_enum},
        {"", invalid_enum},
        {"unknown", invalid_enum},
    };

    for(const auto& test : str_to_compute_type_cases)
    {
        EXPECT_EQ(string_to_hipsparselt_computetype(test.first), test.second);
    }

    const std::pair<const char*, hipsparselt_activation_type> activation_type_cases[] = {
        {"abs", hipsparselt_activation_type::abs},
        {"clippedrelu", hipsparselt_activation_type::clippedrelu},
        {"exp", hipsparselt_activation_type::exp},
        {"gelu", hipsparselt_activation_type::gelu},
        {"leakyrelu", hipsparselt_activation_type::leakyrelu},
        {"relu", hipsparselt_activation_type::relu},
        {"sigmoid", hipsparselt_activation_type::sigmoid},
        {"tanh", hipsparselt_activation_type::tanh},
        {"all", hipsparselt_activation_type::all},
        {"none", hipsparselt_activation_type::none},
        {"invalid", static_cast<hipsparselt_activation_type>(invalid_enum)},
    };

    for(const auto& test : activation_type_cases)
    {
        EXPECT_EQ(string_to_hipsparselt_activation_type(test.first), test.second);
        EXPECT_STREQ(hipsparselt_activation_type_to_string(test.second), test.first);
    }

    const std::pair<rocsparselt_operation, const char*> roc_operation_cases[] = {
        {rocsparselt_operation_conjugate_transpose, "conjugate_transpose"},
    };

    for(const auto& test : roc_operation_cases)
    {
        EXPECT_STREQ(rocsparselt_operation_to_string(test.first), test.second);
    }

    const std::pair<rocsparselt_matrix_type, const char*> mat_type_cases[] = {
        {rocsparselt_matrix_type_unknown, "unknown"},
    };
    for(const auto& test : mat_type_cases)
    {
        EXPECT_STREQ(rocsparselt_matrix_type_to_string(test.first), test.second);
    }

    const std::pair<rocsparselt_layer_mode , const char*> layer_mode_cases[] = {
        {rocsparselt_layer_mode_none, "None"},
        {rocsparselt_layer_mode_log_error, "Error"},
        {rocsparselt_layer_mode_log_hints, "Hints"},
        {rocsparselt_layer_mode_log_info, "Info"},
        {static_cast<rocsparselt_layer_mode>(-1), "Invalid"},
    };

    for(const auto& test : layer_mode_cases)
    {
        EXPECT_STREQ(rocsparselt_layer_mode2string(test.first), test.second);
    }

    const std::pair<rocsparselt_matmul_descr_attribute, const char*> mat_attr_cases[] = {
        {rocsparselt_matmul_activation_abs, "abs"},
        {rocsparselt_matmul_activation_gelu, "gelu"},
        {rocsparselt_matmul_activation_leakyrelu, "leakyrelu"},
        {rocsparselt_matmul_activation_tanh, "tanh"},
    };

    for(const auto& test : mat_attr_cases)
    {
        EXPECT_STREQ(rocsparselt_activation_type_to_string(test.first), test.second);
    }

    const std::pair<hipsparseStatus_t, const char*> status_cases[] = {
        {HIPSPARSE_STATUS_SUCCESS, "HIPSPARSE_STATUS_SUCCESS"},
        {HIPSPARSE_STATUS_NOT_INITIALIZED, "HIPSPARSE_STATUS_NOT_INITIALIZED"},
        {HIPSPARSE_STATUS_ALLOC_FAILED, "HIPSPARSE_STATUS_ALLOC_FAILED"},
        {HIPSPARSE_STATUS_INVALID_VALUE, "HIPSPARSE_STATUS_INVALID_VALUE"},
        {HIPSPARSE_STATUS_ARCH_MISMATCH, "HIPSPARSE_STATUS_ARCH_MISMATCH"},
        {HIPSPARSE_STATUS_MAPPING_ERROR, "HIPSPARSE_STATUS_MAPPING_ERROR"},
        {HIPSPARSE_STATUS_EXECUTION_FAILED, "HIPSPARSE_STATUS_EXECUTION_FAILED"},
        {HIPSPARSE_STATUS_INTERNAL_ERROR, "HIPSPARSE_STATUS_INTERNAL_ERROR"},
        {HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED, "HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED"},
        {HIPSPARSE_STATUS_ZERO_PIVOT, "HIPSPARSE_STATUS_ZERO_PIVOT"},
        {HIPSPARSE_STATUS_NOT_SUPPORTED, "HIPSPARSE_STATUS_NOT_SUPPORTED"},
    };

    for(const auto& test : status_cases)
    {
        EXPECT_STREQ(hipsparse_status_to_string(test.first), test.second);
    }

    const std::pair<hipsparseOperation_t, const char*> operation_cases[] = {
        {HIPSPARSE_OPERATION_NON_TRANSPOSE, "N"},
        {HIPSPARSE_OPERATION_TRANSPOSE, "T"},
        {HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE, "C"},
        {static_cast<hipsparseOperation_t>(invalid_enum), "invalid"},
    };

    for(const auto& test : operation_cases)
    {
        EXPECT_STREQ(hipsparselt_operation_to_string(test.first), test.second);
    }

    const std::pair<char, hipsparseOperation_t> operation_char_cases[] = {
        {'N', HIPSPARSE_OPERATION_NON_TRANSPOSE},
        {'n', HIPSPARSE_OPERATION_NON_TRANSPOSE},
        {'T', HIPSPARSE_OPERATION_TRANSPOSE},
        {'t', HIPSPARSE_OPERATION_TRANSPOSE},
        {'C', HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE},
        {'c', HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE},
        {'X', static_cast<hipsparseOperation_t>(invalid_enum)},
    };

    for(const auto& test : operation_char_cases)
    {
        EXPECT_EQ(char_to_hipsparselt_operation(test.first), test.second);
    }

    const std::pair<char, hipsparseOrder_t> order_cases[] = {
        {'C', HIPSPARSE_ORDER_COL},
        {'c', HIPSPARSE_ORDER_COL},
        {'R', HIPSPARSE_ORDER_ROW},
        {'r', HIPSPARSE_ORDER_ROW},
        {'X', static_cast<hipsparseOrder_t>(invalid_enum)},
    };

    for(const auto& test : order_cases)
    {
        EXPECT_EQ(char_to_hipsparselt_order(test.first), test.second);
    }

    const std::pair<hipsparseLtComputetype_t, const char*> compute_type_cases[] = {
        {HIPSPARSELT_COMPUTE_16F, "f16_r"},
        {HIPSPARSELT_COMPUTE_32I, "i32_r"},
        {HIPSPARSELT_COMPUTE_32F, "f32_r"},
        {static_cast<hipsparseLtComputetype_t>(invalid_enum), "invalid"},
    };

    for(const auto& test : compute_type_cases)
    {
        EXPECT_STREQ(hipsparselt_computetype_to_string(test.first), test.second);
    }
}

void testing_aux_value_mapping(const Arguments& arg)
{
    constexpr int invalid_enum = -1;

    const std::pair<rocsparselt_status, hipError_t> status_mapping[] = {
        {rocsparselt_status_success, hipSuccess},
        {rocsparselt_status_memory_error, hipErrorMemoryAllocation},
        {rocsparselt_status_memory_error, hipErrorLaunchOutOfResources},
        {rocsparselt_status_invalid_pointer, hipErrorInvalidDevicePointer},
        {rocsparselt_status_invalid_handle, hipErrorInvalidDevice},
        {rocsparselt_status_invalid_handle, hipErrorInvalidResourceHandle},
        {rocsparselt_status_internal_error, hipErrorInvalidValue},
        {rocsparselt_status_internal_error, hipErrorNoDevice},
        {rocsparselt_status_internal_error, hipErrorUnknown},
    };

    for(const auto& pair : status_mapping)
    {
        EXPECT_EQ(get_rocsparselt_status_for_hip_status(pair.second), pair.first);
    }

    EXPECT_EQ(get_rocsparselt_status_for_hip_status(static_cast<hipError_t>(invalid_enum)),
              rocsparselt_status_internal_error);
}

void testing_aux_math(const Arguments& arg)
{
    __half_raw v_raw;
    v_raw.x = 0x7C01; // NaN
    EXPECT_TRUE(hipsparselt_isnan(__half(v_raw)));

    v_raw.x = 0x3C00;
    EXPECT_FALSE(hipsparselt_isnan(__half(v_raw)));

    v_raw.x = 0x7C00; // Inf
    EXPECT_TRUE(hipsparselt_isinf(__half(v_raw))); // Inf

    __hip_fp8_e4m3 v_fp8_e4m3;
    v_fp8_e4m3.__x = 0x7E;
    EXPECT_TRUE(hipsparselt_isnan(v_fp8_e4m3)); // NaN

    v_fp8_e4m3.__x = 0x82;
    EXPECT_FALSE(hipsparselt_isnan(v_fp8_e4m3));

    __hip_fp8_e5m2 v_fp8_e5m2;
    v_fp8_e5m2.__x = 0x82;
    EXPECT_FALSE(hipsparselt_isnan(v_fp8_e5m2));
}

void testing_aux_misc(const Arguments& arg)
{
    std::string arch = "gfx950";
    std::string_view empty;
    EXPECT_TRUE(gpu_arch_match(arch, empty));
}

template <typename Ti, typename To, typename Tc>
std::ostringstream& operator<<(std::ostringstream& os, const RocsparseltContractionProblem<Ti, To, Tc>& prob)
{
    hipsparselt_internal_ostream hos;
    hos << prob;
    os << hos.str();
    return os;
}

template <typename Ti, typename To, typename Tc>
std::string testing_aux_rocsparselt_ostream_helper(const Arguments& arg)
{
    _rocsparselt_handle handle;

    // Test data - use different values to ensure all branches are covered
    constexpr size_t m = 2, n = 4, k = 1;
    constexpr size_t batch_count = 2;

    // Create test arrays
    std::vector<Ti> A_data(m * k * batch_count, Ti(1.5));
    std::vector<Ti> B_data(k * n * batch_count, Ti(2.5));
    std::vector<To> C_data(m * n * batch_count, To(3.5));
    std::vector<To> D_data(m * n * batch_count, To(0.0));
    std::vector<unsigned char> metadata(m * k / 2, 0xFF);
    std::vector<float> bias_data(m, 1.0f);

    Tc alpha = Tc(1.0);
    Tc beta = Tc(0.5);

    RocsparseltContractionProblem<Ti, To, Tc> prob(
        &handle,
        rocsparselt_operation_transpose, // trans_a
        rocsparselt_operation_none,      // trans_b
        rocsparselt_order_row,           // order
        m, n, k,                         // dimensions
        &alpha,                          // alpha
        A_data.data(), nullptr,          // A, batch_A
        k, 0, 0,                         // ld_a, batch_stride_a, offset_a
        B_data.data(), nullptr,          // B, batch_B
        n, 0, 0,                         // ld_b, batch_stride_b, offset_b
        &beta,                           // beta
        C_data.data(), nullptr,          // C, batch_C
        n, 0, 0,                         // ld_c, batch_stride_c, offset_c
        D_data.data(), nullptr,          // D, batch_D
        n, 0, 0,                         // ld_d, batch_stride_d, offset_d
        batch_count,                     // batch_count
        true,                            // strided_batch
        false,                           // sparseA
        nullptr,                         // metadata
        hipsparselt_activation_type::relu, // act_type
        0.1f, 6.0f,                      // act_arg0, act_arg1
        bias_data.data(),                // bias_vector (not nullptr)
        1,                               // bias_stride
        HIP_R_32F,                       // bias_type
        true,                            // alpha_vector_scaling
        nullptr,                         // workspace
        1024,                            // workspaceSize
        nullptr,                         // streams
        1                                // numStreams
    );

    hipsparselt_internal_ostream hos;

    hos << prob;
    return hos.str();
}

void testing_aux_ostream(const Arguments& arg)
{
    const char* expected =
        "{ a_type: \"f16_r\","
        " b_type: \"f16_r\","
        " c_type: \"f16_r\","
        " d_type: \"f16_r\","
        " compute_type: \"f32_r\","
        " transA: \"T\","
        " transB: \"N\","
        " M: 2,"
        " N: 4,"
        " K: 1,"
        " alpha: 1,"
        " row_stride_a: 1,"
        " col_stride_a: 1,"
        " row_stride_b: 1,"
        " col_stride_b: 4,"
        " row_stride_c: 1,"
        " col_stride_c: 4,"
        " row_stride_d: 1,"
        " col_stride_d: 4,"
        " beta: 0.5,"
        " batch_count: 2,"
        " strided_batch: true,"
        " stride_a: 0,"
        " stride_b: 0,"
        " stride_c: 0,"
        " stride_d: 0,"
        " activation: \"relu\","
        " activation_argument_0: 0.1,"
        " activation_argument_1: 6,"
        " has_bias: true,"
        " bias_stride: 1,"
        " bias_type: \"f32_r\","
        " alpha_vector_scaling: true }\n";

    std::string output = testing_aux_rocsparselt_ostream_helper<__half, __half, float>(arg);

    EXPECT_STREQ(output.c_str(), expected);
}
