/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************
 * Minimal reproducer for ROCM-9250: fp16 matmul accuracy (softmax A @ B).
 * Compares fp64 reference vs fp16 hipBLASlt result; large diff suggests fp16
 * accumulation on some backends (e.g. MI200).
 *
 * Second test: same fp64 -> fp16 cast, then H2D2H round-trip so host tensors
 * match device memory exactly; CPU reference uses cblas_gemm with fp16 inputs
 * (float compute), matching hipblaslt-test style — compare to hipBLASlt on
 * those same tensors.
 *******************************************************************************/

#include "cblas_interface.hpp"
#include "datatype_interface.hpp"
#include "hipblaslt_test.hpp"
#include "hipblaslt_ostream.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

namespace
{

constexpr int64_t M = 1024;
constexpr int64_t N = 64;
constexpr int64_t K = 1024;
constexpr unsigned SEED = 73;

// Stable softmax per row of A. For column-major A (M rows, K cols, lda = M),
// row i has elements at A[i], A[i+M], A[i+2*M], ...
void softmax_rows_colmajor(double* A, int64_t rows, int64_t cols, int64_t lda)
{
    for(int64_t i = 0; i < rows; i++)
    {
        double max_val = A[i];
        for(int64_t j = 1; j < cols; j++)
            max_val = std::max(max_val, A[i + j * lda]);
        double sum = 0;
        for(int64_t j = 0; j < cols; j++)
        {
            A[i + j * lda] = std::exp(A[i + j * lda] - max_val);
            sum += A[i + j * lda];
        }
        for(int64_t j = 0; j < cols; j++)
            A[i + j * lda] /= sum;
    }
}

// Run NN fp16 GEMM (COMPUTE_32F) on device; C_h receives D matrix (M x N, ldc = M).
void hipblasltNnFp16Gemm(const std::vector<hipblasLtHalf>& A_h,
                         const std::vector<hipblasLtHalf>& B_h,
                         std::vector<hipblasLtHalf>&       C_h)
{
    C_h.resize(M * N);
    CHECK_HIP_ERROR(hipSetDevice(0));

    hipblasLtHandle_t handle;
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));

    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;
    void* dD = nullptr;
    void* dWorkspace = nullptr;
    size_t workspaceSize = 32 * 1024 * 1024;

    CHECK_HIP_ERROR(hipMalloc(&dA, A_h.size() * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipMalloc(&dB, B_h.size() * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipMalloc(&dC, M * N * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipMalloc(&dD, M * N * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipMalloc(&dWorkspace, workspaceSize));

    CHECK_HIP_ERROR(hipMemset(dC, 0, M * N * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(
        hipMemcpy(dA, A_h.data(), A_h.size() * sizeof(hipblasLtHalf), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dB, B_h.data(), B_h.size() * sizeof(hipblasLtHalf), hipMemcpyHostToDevice));

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, M, K, M));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, K, N, K));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, M, N, M));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, M, N, M));

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t transA = HIPBLAS_OP_N, transB = HIPBLAS_OP_N;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(int32_t)));
    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(pref,
                                                                HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                                &workspaceSize,
                                                                sizeof(workspaceSize)));

    const int                        request_solutions = 1;
    hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int                              returnedAlgoCount = 0;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                          matmul,
                                                          matA,
                                                          matB,
                                                          matC,
                                                          matD,
                                                          pref,
                                                          request_solutions,
                                                          heuristicResult,
                                                          &returnedAlgoCount));
    CHECK_SOLUTION_FOUND(returnedAlgoCount);

    uint64_t algoWorkspaceSize = 0;
    for(int i = 0; i < returnedAlgoCount; i++)
        algoWorkspaceSize = std::max(algoWorkspaceSize, heuristicResult[i].workspaceSize);

    // Tensile / solution metadata (same helpers as hipblaslt-test print_kernel_info path).
    {
        hipblasLtMatmulAlgo_t algoMeta = heuristicResult[0].algo;
        const int             algoIdx  = hipblaslt_ext::getIndexFromAlgo(algoMeta);
        const std::string     solName  = hipblaslt_ext::getSolutionNameFromAlgo(handle, algoMeta);
        const std::string     kernName = hipblaslt_ext::getKernelNameFromAlgo(handle, algoMeta);
        hipblaslt_cout << "[fp16_accumulation_gtest] hipBLASLt selected heuristic[0]: algo_index="
                       << algoIdx << "\n  solution_name: " << solName
                       << "\n  kernel_name:   " << kernName
                       << "\n  workspace_bytes=" << heuristicResult[0].workspaceSize
                       << " waves_count=" << heuristicResult[0].wavesCount << std::endl;
    }

    float h_alpha = 1.0f, h_beta = 0.0f;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                          matmul,
                                          &h_alpha,
                                          dA,
                                          matA,
                                          dB,
                                          matB,
                                          &h_beta,
                                          dC,
                                          matC,
                                          dD,
                                          matD,
                                          &heuristicResult[0].algo,
                                          dWorkspace,
                                          algoWorkspaceSize,
                                          stream));
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    CHECK_HIP_ERROR(
        hipMemcpy(C_h.data(), dD, C_h.size() * sizeof(hipblasLtHalf), hipMemcpyDeviceToHost));

    hipblasLtMatrixLayoutDestroy(matA);
    hipblasLtMatrixLayoutDestroy(matB);
    hipblasLtMatrixLayoutDestroy(matC);
    hipblasLtMatrixLayoutDestroy(matD);
    hipblasLtMatmulDescDestroy(matmul);
    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtDestroy(handle);
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));
    CHECK_HIP_ERROR(hipFree(dC));
    CHECK_HIP_ERROR(hipFree(dD));
    CHECK_HIP_ERROR(hipFree(dWorkspace));
}

TEST(Fp16Accumulation, ROCM9250_SoftmaxMatmulFp64VsFp16)
{
    // 1. Create attention-like data: A = softmax(random), B = random (fp64, seed 73)
    //    Column-major layout to match BLAS: A is M*K lda=M, B is K*N ldb=K.
    std::mt19937                     gen(SEED);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> A_fp64(M * K);
    std::vector<double> B_fp64(K * N);
    for(int64_t j = 0; j < K; j++)
        for(int64_t i = 0; i < M; i++)
            A_fp64[i + j * M] = dist(gen);
    for(int64_t j = 0; j < N; j++)
        for(int64_t i = 0; i < K; i++)
            B_fp64[i + j * K] = dist(gen);

    softmax_rows_colmajor(A_fp64.data(), M, K, M);

    // 2. Reference: C_fp64 = A_fp64 @ B_fp64 on CPU (fp64), column-major
    std::vector<double> C_fp64(M * N, 0.0);
    computeTypeInterface alpha, beta;
    alpha.f64 = 1.0;
    beta.f64  = 0.0;
    double scaleVal = 1.0; // cblas_gemm dereferences scaleA/scaleB even when not vector; pass valid ptr
    cblas_gemm(HIPBLAS_OP_N,
               HIPBLAS_OP_N,
               M,
               N,
               K,
               alpha,
               A_fp64.data(),
               M,
               B_fp64.data(),
               K,
               beta,
               C_fp64.data(),
               M,
               nullptr,
               &scaleVal,
               &scaleVal,
               &scaleVal,
               false,
               false,
               HIP_R_64F,
               HIP_R_64F,
               HIP_R_64F,
               HIP_R_64F,
               HIP_R_64F,
               HIP_R_64F,
               false,
               false,
               false);

    // 3. Cast A, B to fp16 (same column-major layout for hipBLASlt)
    std::vector<hipblasLtHalf> A_fp16(M * K);
    std::vector<hipblasLtHalf> B_fp16(K * N);
    for(int64_t i = 0; i < M * K; i++)
        A_fp16[i] = __float2half(static_cast<float>(A_fp64[i]));
    for(int64_t i = 0; i < K * N; i++)
        B_fp16[i] = __float2half(static_cast<float>(B_fp64[i]));

    // 4. hipBLASlt fp16 matmul on device (same path factored in hipblasltNnFp16Gemm)
    int deviceCount = 0;
    CHECK_HIP_ERROR(hipGetDeviceCount(&deviceCount));
    if(deviceCount == 0)
    {
        GTEST_SKIP() << "No HIP device found.";
    }
    CHECK_HIP_ERROR(hipSetDevice(0));

    std::vector<hipblasLtHalf> C_fp16;
    hipblasltNnFp16Gemm(A_fp16, B_fp16, C_fp16);

    // 5. Compare: diff_64_vs_16 = |C_fp64 - C_fp16|, report max_abs_diff and max_rel_error
    double max_abs_diff = 0.0;
    double max_rel_error = 0.0;
    for(int64_t i = 0; i < M * N; i++)
    {
        double ref_val = C_fp64[i];
        double got_val = __half2float(C_fp16[i]);
        double diff    = std::fabs(ref_val - got_val);
        double denom   = std::fabs(ref_val) + 1e-10;
        max_abs_diff   = std::max(max_abs_diff, diff);
        max_rel_error  = std::max(max_rel_error, diff / denom);
    }

    // Diagnostics: device info and ROCM-9250 comparison
    hipDeviceProp_t prop;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, 0));
    hipblaslt_cout << "\n========== Fp16 accumulation (ROCM-9250) diagnostics ==========\n"
                   << "  Device     : " << prop.name << " (" << prop.gcnArchName << ")\n"
                   << "  Problem    : M=" << M << " N=" << N << " K=" << K << " (softmax A @ B, seed=" << SEED
                   << ")\n"
                   << "  max_abs_diff  = " << max_abs_diff << "\n"
                   << "  max_rel_error = " << max_rel_error << "\n"
                   << "  ROCM-9250 ref : CPU/MI300 ~1.24e-4 (OK), MI200 ~7.8e-4 (~6x higher, fp16 accum)\n"
                   << "  If max_abs_diff >> 2e-4 on MI200/MI210, likely reproduces ROCM-9250.\n"
                   << "================================================================\n"
                   << std::endl;

    RecordProperty("max_abs_diff", std::to_string(max_abs_diff));
    RecordProperty("max_rel_error", std::to_string(max_rel_error));
    RecordProperty("M", static_cast<int>(M));
    RecordProperty("N", static_cast<int>(N));
    RecordProperty("K", static_cast<int>(K));
    RecordProperty("seed", static_cast<int>(SEED));

    // ROCM-9250: MI200 shows ~6x higher max_abs_diff than CPU/MI300 when backend uses fp16 accumulation.
    // Optional: uncomment to fail on severe degradation (e.g. > 5e-4).
    // EXPECT_LE(max_abs_diff, 5e-4) << "max_abs_diff=" << max_abs_diff << " max_rel_error=" << max_rel_error;
}

TEST(Fp16Accumulation, ROCM9250_SoftmaxMatmulFp64VsFp16_UniformInput)
{
    // Same as ROCM9250_SoftmaxMatmulFp64VsFp16, but A and B are filled with U(-1,1) before softmax on A.
    std::mt19937                          gen(SEED);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    std::vector<double> A_fp64(M * K);
    std::vector<double> B_fp64(K * N);
    for(int64_t j = 0; j < K; j++)
        for(int64_t i = 0; i < M; i++)
            A_fp64[i + j * M] = uni(gen);
    for(int64_t j = 0; j < N; j++)
        for(int64_t i = 0; i < K; i++)
            B_fp64[i + j * K] = uni(gen);

    softmax_rows_colmajor(A_fp64.data(), M, K, M);

    std::vector<double> C_fp64(M * N, 0.0);
    computeTypeInterface alpha, beta;
    alpha.f64 = 1.0;
    beta.f64  = 0.0;
    double scaleVal = 1.0;
    cblas_gemm(HIPBLAS_OP_N,
               HIPBLAS_OP_N,
               M,
               N,
               K,
               alpha,
               A_fp64.data(),
               M,
               B_fp64.data(),
               K,
               beta,
               C_fp64.data(),
               M,
               nullptr,
               &scaleVal,
               &scaleVal,
               &scaleVal,
               false,
               false,
               HIP_R_64F,
               HIP_R_64F,
               HIP_R_64F,
               HIP_R_64F,
               HIP_R_64F,
               HIP_R_64F,
               false,
               false,
               false);

    std::vector<hipblasLtHalf> A_fp16(M * K);
    std::vector<hipblasLtHalf> B_fp16(K * N);
    for(int64_t i = 0; i < M * K; i++)
        A_fp16[i] = __float2half(static_cast<float>(A_fp64[i]));
    for(int64_t i = 0; i < K * N; i++)
        B_fp16[i] = __float2half(static_cast<float>(B_fp64[i]));

    int deviceCount = 0;
    CHECK_HIP_ERROR(hipGetDeviceCount(&deviceCount));
    if(deviceCount == 0)
    {
        GTEST_SKIP() << "No HIP device found.";
    }
    CHECK_HIP_ERROR(hipSetDevice(0));

    std::vector<hipblasLtHalf> C_fp16;
    hipblasltNnFp16Gemm(A_fp16, B_fp16, C_fp16);

    double max_abs_diff = 0.0;
    double max_rel_error = 0.0;
    for(int64_t i = 0; i < M * N; i++)
    {
        double ref_val = C_fp64[i];
        double got_val = __half2float(C_fp16[i]);
        double diff    = std::fabs(ref_val - got_val);
        double denom   = std::fabs(ref_val) + 1e-10;
        max_abs_diff   = std::max(max_abs_diff, diff);
        max_rel_error  = std::max(max_rel_error, diff / denom);
    }

    hipDeviceProp_t prop;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, 0));
    hipblaslt_cout << "\n========== Fp16 accumulation (ROCM-9250) uniform input ==========\n"
                   << "  Device     : " << prop.name << " (" << prop.gcnArchName << ")\n"
                   << "  Problem    : M=" << M << " N=" << N << " K=" << K << " (softmax A @ B, seed=" << SEED
                   << ")\n"
                   << "  A,B prior  : uniform U(-1,1), then softmax on rows of A\n"
                   << "  max_abs_diff  = " << max_abs_diff << "\n"
                   << "  max_rel_error = " << max_rel_error << "\n"
                   << "==================================================================\n"
                   << std::endl;

    RecordProperty("max_abs_diff", std::to_string(max_abs_diff));
    RecordProperty("max_rel_error", std::to_string(max_rel_error));
    RecordProperty("input_distribution", "uniform_U(-1,1)");
    RecordProperty("M", static_cast<int>(M));
    RecordProperty("N", static_cast<int>(N));
    RecordProperty("K", static_cast<int>(K));
    RecordProperty("seed", static_cast<int>(SEED));
}

TEST(Fp16Accumulation, ROCM9250_SoftmaxMatmulSyncedFp16_CpuRefVsHipblaslt)
{
    // Same data as ROCM9250_SoftmaxMatmulFp64VsFp16, then H2D2H on A,B so host fp16 matches device.
    // CPU reference: cblas_gemm with fp16 A/B and float compute (hipblaslt-test style).
    // GPU: hipBLASlt on the same synced tensors — apples-to-apples inputs; diff isolates kernel vs ref.
    std::mt19937                     gen(SEED);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> A_fp64(M * K);
    std::vector<double> B_fp64(K * N);
    for(int64_t j = 0; j < K; j++)
        for(int64_t i = 0; i < M; i++)
            A_fp64[i + j * M] = dist(gen);
    for(int64_t j = 0; j < N; j++)
        for(int64_t i = 0; i < K; i++)
            B_fp64[i + j * K] = dist(gen);

    softmax_rows_colmajor(A_fp64.data(), M, K, M);

    std::vector<hipblasLtHalf> A_fp16(M * K);
    std::vector<hipblasLtHalf> B_fp16(K * N);
    for(int64_t i = 0; i < M * K; i++)
        A_fp16[i] = __float2half(static_cast<float>(A_fp64[i]));
    for(int64_t i = 0; i < K * N; i++)
        B_fp16[i] = __float2half(static_cast<float>(B_fp64[i]));

    int deviceCount = 0;
    CHECK_HIP_ERROR(hipGetDeviceCount(&deviceCount));
    if(deviceCount == 0)
    {
        GTEST_SKIP() << "No HIP device found.";
    }
    CHECK_HIP_ERROR(hipSetDevice(0));

    void* dRoundA = nullptr;
    void* dRoundB = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&dRoundA, A_fp16.size() * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipMalloc(&dRoundB, B_fp16.size() * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipMemcpy(dRoundA, A_fp16.data(), A_fp16.size() * sizeof(hipblasLtHalf), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dRoundB, B_fp16.data(), B_fp16.size() * sizeof(hipblasLtHalf), hipMemcpyHostToDevice));

    std::vector<hipblasLtHalf> A_sync(M * K);
    std::vector<hipblasLtHalf> B_sync(K * N);
    CHECK_HIP_ERROR(hipMemcpy(A_sync.data(), dRoundA, A_sync.size() * sizeof(hipblasLtHalf), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(B_sync.data(), dRoundB, B_sync.size() * sizeof(hipblasLtHalf), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipFree(dRoundA));
    CHECK_HIP_ERROR(hipFree(dRoundB));

    std::vector<hipblasLtHalf> C_cpu(M * N);

    computeTypeInterface alpha, beta;
    alpha.f32 = 1.0f;
    beta.f32  = 0.0f;
    float scaleVal = 1.0f;
    cblas_gemm(HIPBLAS_OP_N,
               HIPBLAS_OP_N,
               M,
               N,
               K,
               alpha,
               A_sync.data(),
               M,
               B_sync.data(),
               K,
               beta,
               C_cpu.data(),
               M,
               nullptr,
               &scaleVal,
               &scaleVal,
               &scaleVal,
               false,
               false,
               HIP_R_16F,
               HIP_R_16F,
               HIP_R_16F,
               HIP_R_32F,
               HIP_R_32F,
               HIP_R_32F,
               false,
               false,
               false);

    std::vector<hipblasLtHalf> C_gpu;
    hipblasltNnFp16Gemm(A_sync, B_sync, C_gpu);

    double max_abs_cpu_vs_gpu = 0.0;
    double max_rel_cpu_vs_gpu = 0.0;
    for(int64_t i = 0; i < M * N; i++)
    {
        double c_cpu = __half2float(C_cpu[i]);
        double c_gpu = __half2float(C_gpu[i]);
        double diff  = std::fabs(c_cpu - c_gpu);
        double denom = std::fabs(c_cpu) + 1e-10;
        max_abs_cpu_vs_gpu  = std::max(max_abs_cpu_vs_gpu, diff);
        max_rel_cpu_vs_gpu  = std::max(max_rel_cpu_vs_gpu, diff / denom);
    }

    hipDeviceProp_t prop;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, 0));
    hipblaslt_cout << "\n========== ROCM-9250: synced fp16 inputs, CPU cblas vs hipBLASlt ==========\n"
                   << "  Device     : " << prop.name << " (" << prop.gcnArchName << ")\n"
                   << "  Problem    : M=" << M << " N=" << N << " K=" << K << " (softmax A @ B, seed=" << SEED
                   << ")\n"
                   << "  A,B        : fp64 gen -> fp16 cast -> H2D2H (same bytes host & GPU)\n"
                   << "  CPU ref    : cblas_gemm fp16 I/O, float compute (client harness style)\n"
                   << "  max_abs_diff (CPU ref vs GPU) = " << max_abs_cpu_vs_gpu << "\n"
                   << "  max_rel_error (CPU ref vs GPU) = " << max_rel_cpu_vs_gpu << "\n"
                   << "  (Near zero => GPU matches CPU ref on identical fp16 tensors.)\n"
                   << "==========================================================================\n"
                   << std::endl;

    RecordProperty("max_abs_diff_cpu_ref_vs_gpu", std::to_string(max_abs_cpu_vs_gpu));
    RecordProperty("max_rel_error_cpu_ref_vs_gpu", std::to_string(max_rel_cpu_vs_gpu));
}

} // namespace
