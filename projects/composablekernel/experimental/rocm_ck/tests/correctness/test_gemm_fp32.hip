// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// FP32 GEMM correctness tests.
//
// OnesTimesZeros:      A = 1, B = 0  =>  C = 0
// IdentityTimesMatrix: A = I, B = f(k,n)  =>  C = B
// RademacherGemm:      A,B random +/-1  =>  C vs CPU reference (exact)
// NormalGemmTriple:     A,B ~ N(0,1)  =>  CK vs CPU / multiword / FP64 refs

#include <hip/hip_runtime.h>

#include <rocm_ck/args.hpp>
#include <rocm_ck/gemm_spec.hpp>
#include <rocm_ck/hip_check.hpp>
#include <rocm_ck/layout.hpp>
#include <rocm_ck/typed_buffer.hpp>
#include <rocm_ck/verify.hpp>

#ifdef __HIP_DEVICE_COMPILE__
#include <rocm_ck/gemm_dev.hpp>
#endif

#include "gemm_ref_gpu.hip"
#include "gemm_test_data.hpp"
#include "gemm_test_utils.hpp"

#include <gtest/gtest.h>
#include <vector>

// ============================================================================
// Spec + kernel
// ============================================================================

static constexpr auto kSpec = rocm_ck::makeSpec(
    rocm_ck::Signature{.dtype = rocm_ck::DataType::FP32,
                       .ops   = {rocm_ck::GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    rocm_ck::GemmAlgorithm{.block_tile  = {128, 128, 32},
                           .block_waves = {2, 2, 1},
                           .wave_tile   = {16, 16, 16}},
    rocm_ck::TargetSet::cdna());

__global__ void gemm_kernel(rocm_ck::Args args)
{
#ifdef __HIP_DEVICE_COMPILE__
    rocm_ck::run<kSpec>(args);
#endif
}

// ============================================================================
// Test fixture
// ============================================================================

class GemmFP32 : public ::testing::Test
{
protected:
    void SetUp() override { rocm_ck::test::skipIfNoGpu(); }
};

// ============================================================================
// Test: A = all 1s, B = all 0s => C = all 0s
// ============================================================================

TEST_F(GemmFP32, OnesTimesZeros)
{
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 32;

    rocm_ck::TypedBuffer buf_a(kSpec.lhs().dtype, M * K);
    rocm_ck::TypedBuffer buf_b(kSpec.rhs().dtype, K * N);
    rocm_ck::TypedBuffer buf_c(kSpec.output().dtype, M * N);

    auto args = rocm_ck::test::buildGemmArgs(
        kSpec, buf_a.ptr(), buf_b.ptr(), buf_c.ptr(), M, N, K);

    rocm_ck::test::fillConstant2D(args, kSpec.lhs().args_slot, kSpec.lhs().dtype, 1.0f);
    rocm_ck::test::fillConstant2D(args, kSpec.rhs().args_slot, kSpec.rhs().dtype, 0.0f);
    rocm_ck::test::fillConstant2D(args, kSpec.output().args_slot, kSpec.output().dtype, 0.0f);
    HIP_CHECK(hipDeviceSynchronize());

    dim3 grid  = rocm_ck::test::gemmGrid(kSpec, M, N);
    dim3 block(kSpec.workgroup_size);
    gemm_kernel<<<grid, block>>>(args);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> result(M * N);
    buf_c.download(result.data());

    std::vector<float> ref(M * N, 0.0f);

    auto vr = rocm_ck::verify(result.data(), ref.data(), M * N, kSpec.output().dtype);
    EXPECT_TRUE(vr.passed) << "Mismatch at [" << vr.first_mismatch
                           << "] got=" << vr.got << " expected=" << vr.expected;
}

// ============================================================================
// Test: A = identity, B = sequential pattern => C = B
// ============================================================================

TEST_F(GemmFP32, IdentityTimesMatrix)
{
    constexpr int M    = 128;
    constexpr int N    = 128;
    constexpr int K    = 128; // M == K so A is a square identity
    constexpr int kMod = 8;   // small ints, exact in any FP format

    rocm_ck::TypedBuffer buf_a(kSpec.lhs().dtype, M * K);
    rocm_ck::TypedBuffer buf_b(kSpec.rhs().dtype, K * N);
    rocm_ck::TypedBuffer buf_c(kSpec.output().dtype, M * N);

    auto args = rocm_ck::test::buildGemmArgs(
        kSpec, buf_a.ptr(), buf_b.ptr(), buf_c.ptr(), M, N, K);

    rocm_ck::test::fill2D(
        args, kSpec.lhs().args_slot, kSpec.lhs().dtype, rocm_ck::test::IdentityGen{});
    rocm_ck::test::fill2D(
        args, kSpec.rhs().args_slot, kSpec.rhs().dtype, rocm_ck::test::SequentialGen{kMod});
    rocm_ck::test::fillConstant2D(
        args, kSpec.output().args_slot, kSpec.output().dtype, 0.0f);
    HIP_CHECK(hipDeviceSynchronize());

    dim3 grid  = rocm_ck::test::gemmGrid(kSpec, M, N);
    dim3 block(kSpec.workgroup_size);
    gemm_kernel<<<grid, block>>>(args);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> result(M * N);
    buf_c.download(result.data());

    auto [c_sm, c_sn] = rocm_ck::layoutStrides(kSpec.output().layout, M, N);
    std::vector<float> ref(M * N);
    for(int m = 0; m < M; ++m)
        for(int n = 0; n < N; ++n)
            ref[m * c_sm + n * c_sn] = static_cast<float>((m * N + n) % kMod);

    auto vr = rocm_ck::verify(result.data(), ref.data(), M * N, kSpec.output().dtype);
    EXPECT_TRUE(vr.passed) << "Mismatch at [" << vr.first_mismatch
                           << "] got=" << vr.got << " expected=" << vr.expected;
}

// ============================================================================
// Test: A,B = Rademacher (+/-1) => C vs CPU reference (exact in FP32)
// ============================================================================

TEST_F(GemmFP32, RademacherGemm)
{
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 128;

    rocm_ck::TypedBuffer buf_a(kSpec.lhs().dtype, M * K);
    rocm_ck::TypedBuffer buf_b(kSpec.rhs().dtype, K * N);
    rocm_ck::TypedBuffer buf_c(kSpec.output().dtype, M * N);

    auto args = rocm_ck::test::buildGemmArgs(
        kSpec, buf_a.ptr(), buf_b.ptr(), buf_c.ptr(), M, N, K);

    rocm_ck::test::fill2D(
        args, kSpec.lhs().args_slot, kSpec.lhs().dtype, rocm_ck::test::RademacherGen{37});
    rocm_ck::test::fill2D(
        args, kSpec.rhs().args_slot, kSpec.rhs().dtype, rocm_ck::test::RademacherGen{73});
    rocm_ck::test::fillConstant2D(
        args, kSpec.output().args_slot, kSpec.output().dtype, 0.0f);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> host_a(M * K);
    std::vector<float> host_b(K * N);
    buf_a.download(host_a.data());
    buf_b.download(host_b.data());

    std::vector<float> ref(M * N, 0.0f);
    rocm_ck::test::cpuGemm(kSpec, host_a.data(), host_b.data(), ref.data(),
                            M, N, K, args);

    dim3 grid  = rocm_ck::test::gemmGrid(kSpec, M, N);
    dim3 block(kSpec.workgroup_size);
    gemm_kernel<<<grid, block>>>(args);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> result(M * N);
    buf_c.download(result.data());

    auto vr = rocm_ck::verify(result.data(), ref.data(), M * N, kSpec.output().dtype);
    EXPECT_TRUE(vr.passed) << "Mismatch at [" << vr.first_mismatch
                           << "] got=" << vr.got << " expected=" << vr.expected;
}

// ============================================================================
// Test: A,B ~ N(0,1) => CK vs CPU reference, GPU multiword, GPU FP64
//
// Compares the CK kernel result against all three reference implementations.
// Random normal data exercises general FP32 accumulation (not exact).
// ============================================================================

TEST_F(GemmFP32, NormalGemmTriple)
{
    constexpr int M = 256;
    constexpr int N = 256;
    constexpr int K = 128;

    rocm_ck::TypedBuffer buf_a(kSpec.lhs().dtype, M * K);
    rocm_ck::TypedBuffer buf_b(kSpec.rhs().dtype, K * N);
    rocm_ck::TypedBuffer buf_c(kSpec.output().dtype, M * N);

    auto args = rocm_ck::test::buildGemmArgs(
        kSpec, buf_a.ptr(), buf_b.ptr(), buf_c.ptr(), M, N, K);

    // --- Fill A, B with standard normal N(0, 1) ---
    rocm_ck::test::fill2D(
        args, kSpec.lhs().args_slot, kSpec.lhs().dtype,
        rocm_ck::test::NormalGen{0.0f, 1.0f, 42});
    rocm_ck::test::fill2D(
        args, kSpec.rhs().args_slot, kSpec.rhs().dtype,
        rocm_ck::test::NormalGen{0.0f, 1.0f, 137});
    rocm_ck::test::fillConstant2D(
        args, kSpec.output().args_slot, kSpec.output().dtype, 0.0f);
    HIP_CHECK(hipDeviceSynchronize());

    // --- Download A, B for CPU reference ---
    std::vector<float> host_a(M * K);
    std::vector<float> host_b(K * N);
    buf_a.download(host_a.data());
    buf_b.download(host_b.data());

    // --- CPU reference (FP64 accumulation) ---
    std::vector<float> ref_cpu(M * N, 0.0f);
    rocm_ck::test::cpuGemm(kSpec, host_a.data(), host_b.data(), ref_cpu.data(),
                            M, N, K, args);

    // --- CK kernel ---
    dim3 grid  = rocm_ck::test::gemmGrid(kSpec, M, N);
    dim3 block(kSpec.workgroup_size);
    gemm_kernel<<<grid, block>>>(args);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> result_ck(M * N);
    buf_c.download(result_ck.data());

    // --- GPU references (operate on device float arrays) ---
    rocm_ck::test::GemmBackend backend;
    auto* dev_a = static_cast<float*>(buf_a.ptr());
    auto* dev_b = static_cast<float*>(buf_b.ptr());

    // Multiword reference
    float* dev_ref_mw = nullptr;
    HIP_CHECK(hipMalloc(&dev_ref_mw, M * N * sizeof(float)));
    rocm_ck::test::gpuGemmMultiword(kSpec, dev_ref_mw, dev_a, dev_b,
                                    M, N, K, args, backend);

    std::vector<float> ref_mw(M * N);
    HIP_CHECK(hipMemcpy(ref_mw.data(), dev_ref_mw,
                         M * N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(dev_ref_mw));

    // FP64 reference
    float* dev_ref_f64 = nullptr;
    HIP_CHECK(hipMalloc(&dev_ref_f64, M * N * sizeof(float)));
    rocm_ck::test::gpuGemmFP64(kSpec, dev_ref_f64, dev_a, dev_b,
                                M, N, K, args, backend);

    std::vector<float> ref_f64(M * N);
    HIP_CHECK(hipMemcpy(ref_f64.data(), dev_ref_f64,
                         M * N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(dev_ref_f64));

    // --- Compare CK vs each reference ---
    auto vr_cpu = rocm_ck::verify(
        result_ck.data(), ref_cpu.data(), M * N, kSpec.output().dtype);
    EXPECT_TRUE(vr_cpu.passed) << "CK vs CPU: mismatch at [" << vr_cpu.first_mismatch
                               << "] got=" << vr_cpu.got << " expected=" << vr_cpu.expected;

    auto vr_mw = rocm_ck::verify(
        result_ck.data(), ref_mw.data(), M * N, kSpec.output().dtype);
    EXPECT_TRUE(vr_mw.passed) << "CK vs Multiword: mismatch at [" << vr_mw.first_mismatch
                              << "] got=" << vr_mw.got << " expected=" << vr_mw.expected;

    auto vr_f64 = rocm_ck::verify(
        result_ck.data(), ref_f64.data(), M * N, kSpec.output().dtype);
    EXPECT_TRUE(vr_f64.passed) << "CK vs FP64: mismatch at [" << vr_f64.first_mismatch
                               << "] got=" << vr_f64.got << " expected=" << vr_f64.expected;

    // --- Cross-check: multiword vs FP64 (should be very close) ---
    auto vr_cross = rocm_ck::verify(
        ref_mw.data(), ref_f64.data(), M * N, kSpec.output().dtype);
    EXPECT_TRUE(vr_cross.passed) << "Multiword vs FP64: mismatch at ["
                                 << vr_cross.first_mismatch
                                 << "] got=" << vr_cross.got
                                 << " expected=" << vr_cross.expected;
}
