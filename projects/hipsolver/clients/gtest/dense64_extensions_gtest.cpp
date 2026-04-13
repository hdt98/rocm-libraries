/* ************************************************************************
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

/*
 * Tests for the hipSOLVER 64-bit DnX solver extensions:
 *   hipsolverDnXsyevd       — symmetric eigenvalue decomposition
 *   hipsolverDnXsyevBatched — batched symmetric eigenvalue decomposition
 *   hipsolverDnXgeev         — non-symmetric eigenvalue decomposition
 *   hipsolverDnXsytrs        — symmetric triangular solve (BK pivoting)
 */

#include "clientcommon.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using ::testing::Matcher;
using ::testing::UnitTest;

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class checkin_misc_DENSE64_EXT : public ::testing::Test
{
protected:
    hipsolverHandle_t handle;

    void SetUp() override
    {
        ASSERT_EQ(hipsolverCreate(&handle), HIPSOLVER_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        ASSERT_EQ(hipsolverDestroy(handle), HIPSOLVER_STATUS_SUCCESS);
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }
};

// =========================================================================
//  hipsolverDnXsyevd
// =========================================================================

TEST_F(checkin_misc_DENSE64_EXT, syevd_bufferSize_null_handle)
{
    size_t dw = 0, hw = 0;
    EXPECT_ROCBLAS_STATUS(
        hipsolverDnXsyevd_bufferSize(
            nullptr, nullptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_LOWER,
            3, HIP_R_64F, nullptr, 3, HIP_R_64F, nullptr, HIP_R_64F, &dw, &hw),
        HIPSOLVER_STATUS_NOT_INITIALIZED);
}

TEST_F(checkin_misc_DENSE64_EXT, syevd_identity_double)
{
    // Symmetric eigenvalue decomposition of 3x3 identity → eigenvalues = [1,1,1]
    const int64_t n = 3;

    // Host identity matrix (column-major)
    std::vector<double> hA(n * n, 0.0);
    for(int64_t i = 0; i < n; i++)
        hA[i + i * n] = 1.0;

    // Device allocations
    double* dA = nullptr;
    double* dW = nullptr;
    int*    dInfo = nullptr;
    hipMalloc(&dA, n * n * sizeof(double));
    hipMalloc(&dW, n * sizeof(double));
    hipMalloc(&dInfo, sizeof(int));
    hipMemcpy(dA, hA.data(), n * n * sizeof(double), hipMemcpyHostToDevice);

    // Query workspace
    size_t dw = 0, hw = 0;
    ASSERT_EQ(hipsolverDnXsyevd_bufferSize(
                  handle, nullptr, HIPSOLVER_EIG_MODE_VECTOR, HIPSOLVER_FILL_MODE_LOWER,
                  n, HIP_R_64F, dA, n, HIP_R_64F, dW, HIP_R_64F, &dw, &hw),
              HIPSOLVER_STATUS_SUCCESS);

    void* dWork = nullptr;
    void* hWork = nullptr;
    if(dw > 0) hipMalloc(&dWork, dw);
    if(hw > 0) hWork = malloc(hw);

    // Execute
    ASSERT_EQ(hipsolverDnXsyevd(
                  handle, nullptr, HIPSOLVER_EIG_MODE_VECTOR, HIPSOLVER_FILL_MODE_LOWER,
                  n, HIP_R_64F, dA, n, HIP_R_64F, dW, HIP_R_64F,
                  dWork, dw, hWork, hw, dInfo),
              HIPSOLVER_STATUS_SUCCESS);
    hipDeviceSynchronize();

    // Verify eigenvalues
    std::vector<double> hW(n);
    hipMemcpy(hW.data(), dW, n * sizeof(double), hipMemcpyDeviceToHost);
    for(int64_t i = 0; i < n; i++)
        EXPECT_NEAR(hW[i], 1.0, 1e-12) << "eigenvalue[" << i << "]";

    // Verify info
    int hInfo = -1;
    hipMemcpy(&hInfo, dInfo, sizeof(int), hipMemcpyDeviceToHost);
    EXPECT_EQ(hInfo, 0);

    hipFree(dA);
    hipFree(dW);
    hipFree(dInfo);
    if(dWork) hipFree(dWork);
    if(hWork) free(hWork);
}

TEST_F(checkin_misc_DENSE64_EXT, syevd_diagonal_float)
{
    // Eigenvalues of diag(1, 2, 3, 4) should be [1, 2, 3, 4]
    const int64_t n = 4;

    std::vector<float> hA(n * n, 0.0f);
    for(int64_t i = 0; i < n; i++)
        hA[i + i * n] = (float)(i + 1);

    float* dA = nullptr;
    float* dW = nullptr;
    int*   dInfo = nullptr;
    hipMalloc(&dA, n * n * sizeof(float));
    hipMalloc(&dW, n * sizeof(float));
    hipMalloc(&dInfo, sizeof(int));
    hipMemcpy(dA, hA.data(), n * n * sizeof(float), hipMemcpyHostToDevice);

    size_t dw = 0, hw = 0;
    ASSERT_EQ(hipsolverDnXsyevd_bufferSize(
                  handle, nullptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_LOWER,
                  n, HIP_R_32F, dA, n, HIP_R_32F, dW, HIP_R_32F, &dw, &hw),
              HIPSOLVER_STATUS_SUCCESS);

    void* dWork = nullptr;
    void* hWork = nullptr;
    if(dw > 0) hipMalloc(&dWork, dw);
    if(hw > 0) hWork = malloc(hw);

    ASSERT_EQ(hipsolverDnXsyevd(
                  handle, nullptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_LOWER,
                  n, HIP_R_32F, dA, n, HIP_R_32F, dW, HIP_R_32F,
                  dWork, dw, hWork, hw, dInfo),
              HIPSOLVER_STATUS_SUCCESS);
    hipDeviceSynchronize();

    std::vector<float> hW(n);
    hipMemcpy(hW.data(), dW, n * sizeof(float), hipMemcpyDeviceToHost);

    // rocSOLVER syevd returns eigenvalues in ascending order
    std::sort(hW.begin(), hW.end());
    for(int64_t i = 0; i < n; i++)
        EXPECT_NEAR(hW[i], (float)(i + 1), 1e-5f) << "eigenvalue[" << i << "]";

    hipFree(dA);
    hipFree(dW);
    hipFree(dInfo);
    if(dWork) hipFree(dWork);
    if(hWork) free(hWork);
}

TEST_F(checkin_misc_DENSE64_EXT, syevd_zero_size)
{
    size_t dw = 0, hw = 0;
    EXPECT_ROCBLAS_STATUS(
        hipsolverDnXsyevd_bufferSize(
            handle, nullptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_LOWER,
            0, HIP_R_64F, nullptr, 1, HIP_R_64F, nullptr, HIP_R_64F, &dw, &hw),
        HIPSOLVER_STATUS_SUCCESS);
}

// =========================================================================
//  hipsolverDnXsyevBatched
// =========================================================================

TEST_F(checkin_misc_DENSE64_EXT, syevBatched_bufferSize_null_handle)
{
    size_t dw = 0, hw = 0;
    EXPECT_ROCBLAS_STATUS(
        hipsolverDnXsyevBatched_bufferSize(
            nullptr, nullptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_LOWER,
            3, HIP_R_64F, nullptr, 3, HIP_R_64F, nullptr, HIP_R_64F, &dw, &hw, 1),
        HIPSOLVER_STATUS_NOT_INITIALIZED);
}

TEST_F(checkin_misc_DENSE64_EXT, syevBatched_two_identities_double)
{
    // Batch of 2 identity matrices → all eigenvalues = 1
    const int64_t n = 3;
    const int64_t batch = 2;
    const int64_t strideA = n * n;
    const int64_t strideW = n;

    std::vector<double> hA(strideA * batch, 0.0);
    for(int64_t b = 0; b < batch; b++)
        for(int64_t i = 0; i < n; i++)
            hA[b * strideA + i + i * n] = 1.0;

    double* dA = nullptr;
    double* dW = nullptr;
    int*    dInfo = nullptr;
    hipMalloc(&dA, strideA * batch * sizeof(double));
    hipMalloc(&dW, strideW * batch * sizeof(double));
    hipMalloc(&dInfo, batch * sizeof(int));
    hipMemcpy(dA, hA.data(), strideA * batch * sizeof(double), hipMemcpyHostToDevice);

    size_t dw = 0, hw = 0;
    ASSERT_EQ(hipsolverDnXsyevBatched_bufferSize(
                  handle, nullptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_LOWER,
                  n, HIP_R_64F, dA, n, HIP_R_64F, dW, HIP_R_64F, &dw, &hw, batch),
              HIPSOLVER_STATUS_SUCCESS);

    void* dWork = nullptr;
    void* hWork = nullptr;
    if(dw > 0) hipMalloc(&dWork, dw);
    if(hw > 0) hWork = malloc(hw);

    ASSERT_EQ(hipsolverDnXsyevBatched(
                  handle, nullptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_LOWER,
                  n, HIP_R_64F, dA, n, HIP_R_64F, dW, HIP_R_64F,
                  dWork, dw, hWork, hw, dInfo, batch),
              HIPSOLVER_STATUS_SUCCESS);
    hipDeviceSynchronize();

    std::vector<double> hW(strideW * batch);
    hipMemcpy(hW.data(), dW, strideW * batch * sizeof(double), hipMemcpyDeviceToHost);
    for(int64_t b = 0; b < batch; b++)
        for(int64_t i = 0; i < n; i++)
            EXPECT_NEAR(hW[b * strideW + i], 1.0, 1e-12)
                << "batch " << b << " eigenvalue[" << i << "]";

    hipFree(dA);
    hipFree(dW);
    hipFree(dInfo);
    if(dWork) hipFree(dWork);
    if(hWork) free(hWork);
}

// =========================================================================
//  hipsolverDnXgeev
// =========================================================================

TEST_F(checkin_misc_DENSE64_EXT, geev_bufferSize_null_handle)
{
    size_t dw = 0, hw = 0;
    EXPECT_ROCBLAS_STATUS(
        hipsolverDnXgeev_bufferSize(
            nullptr, nullptr,
            HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_MODE_VECTOR,
            3, HIP_R_64F, nullptr, 3, HIP_R_64F, nullptr,
            HIP_R_64F, nullptr, 1, HIP_R_64F, nullptr, 3,
            HIP_R_64F, &dw, &hw),
        HIPSOLVER_STATUS_NOT_INITIALIZED);
}

TEST_F(checkin_misc_DENSE64_EXT, geev_identity_double)
{
    // Non-symmetric eigenvalue of identity → eigenvalues all (1,0), eigenvectors = identity
    const int64_t n = 3;

    // Host: identity matrix (column-major)
    std::vector<double> hA(n * n, 0.0);
    for(int64_t i = 0; i < n; i++)
        hA[i + i * n] = 1.0;

    // W stores [wr[0]..wr[n-1], wi[0]..wi[n-1]] for real types
    double *dA = nullptr, *dW = nullptr, *dVR = nullptr;
    int*    dInfo = nullptr;
    hipMalloc(&dA, n * n * sizeof(double));
    hipMalloc(&dW, 2 * n * sizeof(double));
    hipMalloc(&dVR, n * n * sizeof(double));
    hipMalloc(&dInfo, sizeof(int));
    hipMemcpy(dA, hA.data(), n * n * sizeof(double), hipMemcpyHostToDevice);

    size_t dw = 0, hw = 0;
    ASSERT_EQ(hipsolverDnXgeev_bufferSize(
                  handle, nullptr,
                  HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_MODE_VECTOR,
                  n, HIP_R_64F, dA, n, HIP_R_64F, dW,
                  HIP_R_64F, nullptr, 1, HIP_R_64F, dVR, n,
                  HIP_R_64F, &dw, &hw),
              HIPSOLVER_STATUS_SUCCESS);

    void* dWork = nullptr;
    void* hWork = nullptr;
    if(dw > 0) hipMalloc(&dWork, dw);
    if(hw > 0) hWork = malloc(hw);

    ASSERT_EQ(hipsolverDnXgeev(
                  handle, nullptr,
                  HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_MODE_VECTOR,
                  n, HIP_R_64F, dA, n, HIP_R_64F, dW,
                  HIP_R_64F, nullptr, 1, HIP_R_64F, dVR, n,
                  HIP_R_64F, dWork, dw, hWork, hw, dInfo),
              HIPSOLVER_STATUS_SUCCESS);
    hipDeviceSynchronize();

    // Verify: W = [wr0..wrn, wi0..win] = [1,1,1, 0,0,0]
    std::vector<double> hW(2 * n);
    hipMemcpy(hW.data(), dW, 2 * n * sizeof(double), hipMemcpyDeviceToHost);
    for(int64_t i = 0; i < n; i++)
    {
        EXPECT_NEAR(hW[i], 1.0, 1e-12) << "wr[" << i << "]";
        EXPECT_NEAR(hW[n + i], 0.0, 1e-12) << "wi[" << i << "]";
    }

    int hInfo = -1;
    hipMemcpy(&hInfo, dInfo, sizeof(int), hipMemcpyDeviceToHost);
    EXPECT_EQ(hInfo, 0);

    hipFree(dA);
    hipFree(dW);
    hipFree(dVR);
    hipFree(dInfo);
    if(dWork) hipFree(dWork);
    if(hWork) free(hWork);
}

TEST_F(checkin_misc_DENSE64_EXT, geev_identity_complex128)
{
    // Complex identity → eigenvalues = (1+0i, 1+0i, 1+0i)
    const int64_t n = 3;
    using T = hipDoubleComplex;

    std::vector<T> hA(n * n, {0.0, 0.0});
    for(int64_t i = 0; i < n; i++)
        hA[i + i * n] = {1.0, 0.0};

    T*   dA = nullptr;
    T*   dW = nullptr;
    T*   dVR = nullptr;
    int* dInfo = nullptr;
    hipMalloc(&dA, n * n * sizeof(T));
    hipMalloc(&dW, n * sizeof(T));
    hipMalloc(&dVR, n * n * sizeof(T));
    hipMalloc(&dInfo, sizeof(int));
    hipMemcpy(dA, hA.data(), n * n * sizeof(T), hipMemcpyHostToDevice);

    size_t dw = 0, hw = 0;
    ASSERT_EQ(hipsolverDnXgeev_bufferSize(
                  handle, nullptr,
                  HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_MODE_VECTOR,
                  n, HIP_C_64F, dA, n, HIP_C_64F, dW,
                  HIP_C_64F, nullptr, 1, HIP_C_64F, dVR, n,
                  HIP_C_64F, &dw, &hw),
              HIPSOLVER_STATUS_SUCCESS);

    void* dWork = nullptr;
    void* hWork = nullptr;
    if(dw > 0) hipMalloc(&dWork, dw);
    if(hw > 0) hWork = malloc(hw);

    ASSERT_EQ(hipsolverDnXgeev(
                  handle, nullptr,
                  HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_MODE_VECTOR,
                  n, HIP_C_64F, dA, n, HIP_C_64F, dW,
                  HIP_C_64F, nullptr, 1, HIP_C_64F, dVR, n,
                  HIP_C_64F, dWork, dw, hWork, hw, dInfo),
              HIPSOLVER_STATUS_SUCCESS);
    hipDeviceSynchronize();

    std::vector<T> hW(n);
    hipMemcpy(hW.data(), dW, n * sizeof(T), hipMemcpyDeviceToHost);
    for(int64_t i = 0; i < n; i++)
    {
        EXPECT_NEAR(hW[i].x, 1.0, 1e-12) << "w[" << i << "].real";
        EXPECT_NEAR(hW[i].y, 0.0, 1e-12) << "w[" << i << "].imag";
    }

    hipFree(dA);
    hipFree(dW);
    hipFree(dVR);
    hipFree(dInfo);
    if(dWork) hipFree(dWork);
    if(hWork) free(hWork);
}

TEST_F(checkin_misc_DENSE64_EXT, geev_zero_size)
{
    size_t dw = 0, hw = 0;
    EXPECT_ROCBLAS_STATUS(
        hipsolverDnXgeev_bufferSize(
            handle, nullptr,
            HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_MODE_NOVECTOR,
            0, HIP_R_64F, nullptr, 1, HIP_R_64F, nullptr,
            HIP_R_64F, nullptr, 1, HIP_R_64F, nullptr, 1,
            HIP_R_64F, &dw, &hw),
        HIPSOLVER_STATUS_SUCCESS);
}

// =========================================================================
//  hipsolverDnXsytrs
// =========================================================================

TEST_F(checkin_misc_DENSE64_EXT, sytrs_bufferSize_null_handle)
{
    size_t dw = 0, hw = 0;
    EXPECT_ROCBLAS_STATUS(
        hipsolverDnXsytrs_bufferSize(
            nullptr, HIPSOLVER_FILL_MODE_LOWER,
            3, 1, HIP_R_64F, nullptr, 3, nullptr, HIP_R_64F, nullptr, 3, &dw, &hw),
        HIPSOLVER_STATUS_NOT_INITIALIZED);
}

TEST_F(checkin_misc_DENSE64_EXT, sytrs_identity_double)
{
    // LDL factorization of identity is itself: L=I, D=I, ipiv=[1,2,3]
    // Solving I*X = B gives X = B
    const int64_t n = 3;
    const int64_t nrhs = 2;

    // Identity matrix (already in LDL form)
    std::vector<double> hA(n * n, 0.0);
    for(int64_t i = 0; i < n; i++)
        hA[i + i * n] = 1.0;

    // B = [1,2,3; 4,5,6] (column-major, n x nrhs)
    std::vector<double> hB(n * nrhs);
    for(int64_t j = 0; j < nrhs; j++)
        for(int64_t i = 0; i < n; i++)
            hB[i + j * n] = (double)(i + 1 + j * n);

    // Trivial pivots (no swaps): ipiv = [1, 2, 3] (1-indexed, positive = 1x1 block)
    std::vector<int64_t> hIpiv(n);
    for(int64_t i = 0; i < n; i++)
        hIpiv[i] = i + 1;

    double*  dA = nullptr;
    double*  dB = nullptr;
    int64_t* dIpiv = nullptr;
    int*     dInfo = nullptr;
    hipMalloc(&dA, n * n * sizeof(double));
    hipMalloc(&dB, n * nrhs * sizeof(double));
    hipMalloc(&dIpiv, n * sizeof(int64_t));
    hipMalloc(&dInfo, sizeof(int));
    hipMemcpy(dA, hA.data(), n * n * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB.data(), n * nrhs * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(dIpiv, hIpiv.data(), n * sizeof(int64_t), hipMemcpyHostToDevice);

    size_t dw = 0, hw = 0;
    ASSERT_EQ(hipsolverDnXsytrs_bufferSize(
                  handle, HIPSOLVER_FILL_MODE_LOWER,
                  n, nrhs, HIP_R_64F, dA, n, dIpiv, HIP_R_64F, dB, n, &dw, &hw),
              HIPSOLVER_STATUS_SUCCESS);

    void* dWork = nullptr;
    void* hWork = nullptr;
    if(dw > 0) hipMalloc(&dWork, dw);
    if(hw > 0) hWork = malloc(hw);

    ASSERT_EQ(hipsolverDnXsytrs(
                  handle, HIPSOLVER_FILL_MODE_LOWER,
                  n, nrhs, HIP_R_64F, dA, n, dIpiv, HIP_R_64F, dB, n,
                  dWork, dw, hWork, hw, dInfo),
              HIPSOLVER_STATUS_SUCCESS);
    hipDeviceSynchronize();

    // X should equal original B
    std::vector<double> hX(n * nrhs);
    hipMemcpy(hX.data(), dB, n * nrhs * sizeof(double), hipMemcpyDeviceToHost);
    for(int64_t j = 0; j < nrhs; j++)
        for(int64_t i = 0; i < n; i++)
            EXPECT_NEAR(hX[i + j * n], hB[i + j * n], 1e-12)
                << "X[" << i << "," << j << "]";

    int hInfo = -1;
    hipMemcpy(&hInfo, dInfo, sizeof(int), hipMemcpyDeviceToHost);
    EXPECT_EQ(hInfo, 0);

    hipFree(dA);
    hipFree(dB);
    hipFree(dIpiv);
    hipFree(dInfo);
    if(dWork) hipFree(dWork);
    if(hWork) free(hWork);
}

TEST_F(checkin_misc_DENSE64_EXT, sytrs_diagonal_float)
{
    // LDL of diag(2, 3, 5) with no pivoting: L=I, D=diag(2,3,5), ipiv=[1,2,3]
    // Solve diag(2,3,5) * X = [4,9,25] → X = [2,3,5]
    const int64_t n = 3;
    const int64_t nrhs = 1;

    std::vector<float> hA(n * n, 0.0f);
    hA[0] = 2.0f; hA[4] = 3.0f; hA[8] = 5.0f;

    std::vector<float> hB = {4.0f, 9.0f, 25.0f};
    std::vector<int64_t> hIpiv = {1, 2, 3};

    float*   dA = nullptr;
    float*   dB = nullptr;
    int64_t* dIpiv = nullptr;
    int*     dInfo = nullptr;
    hipMalloc(&dA, n * n * sizeof(float));
    hipMalloc(&dB, n * sizeof(float));
    hipMalloc(&dIpiv, n * sizeof(int64_t));
    hipMalloc(&dInfo, sizeof(int));
    hipMemcpy(dA, hA.data(), n * n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB.data(), n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dIpiv, hIpiv.data(), n * sizeof(int64_t), hipMemcpyHostToDevice);

    size_t dw = 0, hw = 0;
    ASSERT_EQ(hipsolverDnXsytrs_bufferSize(
                  handle, HIPSOLVER_FILL_MODE_LOWER,
                  n, nrhs, HIP_R_32F, dA, n, dIpiv, HIP_R_32F, dB, n, &dw, &hw),
              HIPSOLVER_STATUS_SUCCESS);

    void* dWork = nullptr;
    if(dw > 0) hipMalloc(&dWork, dw);

    ASSERT_EQ(hipsolverDnXsytrs(
                  handle, HIPSOLVER_FILL_MODE_LOWER,
                  n, nrhs, HIP_R_32F, dA, n, dIpiv, HIP_R_32F, dB, n,
                  dWork, dw, nullptr, 0, dInfo),
              HIPSOLVER_STATUS_SUCCESS);
    hipDeviceSynchronize();

    std::vector<float> hX(n);
    hipMemcpy(hX.data(), dB, n * sizeof(float), hipMemcpyDeviceToHost);
    EXPECT_NEAR(hX[0], 2.0f, 1e-5f);
    EXPECT_NEAR(hX[1], 3.0f, 1e-5f);
    EXPECT_NEAR(hX[2], 5.0f, 1e-5f);

    hipFree(dA);
    hipFree(dB);
    hipFree(dIpiv);
    hipFree(dInfo);
    if(dWork) hipFree(dWork);
}
