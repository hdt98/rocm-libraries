// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// GEMM correctness tests: GPU output vs known reference.
//
// OnesTimesZeros:      A = 1, B = 0  =>  C = 0
// IdentityTimesMatrix: A = I, B = f(k,n)  =>  C = B
// RademacherGemm:      A,B random +/-1  =>  C vs CPU reference (exact)

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

class GemmCorrectness : public ::testing::Test
{
protected:
    void SetUp() override { rocm_ck::test::skipIfNoGpu(); }
};

// ============================================================================
// Test: A = all 1s, B = all 0s => C = all 0s
// ============================================================================

TEST_F(GemmCorrectness, OnesTimesZeros)
{
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 32;

    // Allocate device buffers
    rocm_ck::TypedBuffer buf_a(kSpec.lhs().dtype, M * K);
    rocm_ck::TypedBuffer buf_b(kSpec.rhs().dtype, K * N);
    rocm_ck::TypedBuffer buf_c(kSpec.output().dtype, M * N);

    // Build Args — strides computed from spec layout, stored in tensor slots
    auto args = rocm_ck::test::buildGemmArgs(
        kSpec, buf_a.ptr(), buf_b.ptr(), buf_c.ptr(), M, N, K);

    // Fill A = 1.0, B = 0.0, C = 0.0 on GPU using Args tensor slots
    rocm_ck::test::fillConstant2D(args, kSpec.lhs().args_slot, kSpec.lhs().dtype, 1.0f);
    rocm_ck::test::fillConstant2D(args, kSpec.rhs().args_slot, kSpec.rhs().dtype, 0.0f);
    rocm_ck::test::fillConstant2D(args, kSpec.output().args_slot, kSpec.output().dtype, 0.0f);
    HIP_CHECK(hipDeviceSynchronize());

    // Launch GEMM
    dim3 grid  = rocm_ck::test::gemmGrid(kSpec, M, N);
    dim3 block(kSpec.workgroup_size);
    gemm_kernel<<<grid, block>>>(args);
    HIP_CHECK(hipDeviceSynchronize());

    // Download result
    std::vector<float> result(M * N);
    buf_c.download(result.data());

    // Reference: all zeros
    std::vector<float> ref(M * N, 0.0f);

    // Verify
    auto vr = rocm_ck::verify(result.data(), ref.data(), M * N, kSpec.output().dtype);
    EXPECT_TRUE(vr.passed) << "Mismatch at [" << vr.first_mismatch
                           << "] got=" << vr.got << " expected=" << vr.expected;
}

// ============================================================================
// Test: A = identity, B = sequential pattern => C = B
// ============================================================================

TEST_F(GemmCorrectness, IdentityTimesMatrix)
{
    constexpr int M   = 128;
    constexpr int N   = 128;
    constexpr int K   = 128; // M == K so A is a square identity
    constexpr int kMod = 8;  // small ints, exact in any FP format

    // Allocate device buffers
    rocm_ck::TypedBuffer buf_a(kSpec.lhs().dtype, M * K);
    rocm_ck::TypedBuffer buf_b(kSpec.rhs().dtype, K * N);
    rocm_ck::TypedBuffer buf_c(kSpec.output().dtype, M * N);

    // Build Args
    auto args = rocm_ck::test::buildGemmArgs(
        kSpec, buf_a.ptr(), buf_b.ptr(), buf_c.ptr(), M, N, K);

    // Fill: A = identity, B = sequential pattern, C = zero
    rocm_ck::test::fill2D(
        args, kSpec.lhs().args_slot, kSpec.lhs().dtype, rocm_ck::test::IdentityGen{});
    rocm_ck::test::fill2D(
        args, kSpec.rhs().args_slot, kSpec.rhs().dtype, rocm_ck::test::SequentialGen{kMod});
    rocm_ck::test::fillConstant2D(
        args, kSpec.output().args_slot, kSpec.output().dtype, 0.0f);
    HIP_CHECK(hipDeviceSynchronize());

    // Launch GEMM
    dim3 grid  = rocm_ck::test::gemmGrid(kSpec, M, N);
    dim3 block(kSpec.workgroup_size);
    gemm_kernel<<<grid, block>>>(args);
    HIP_CHECK(hipDeviceSynchronize());

    // Download result
    std::vector<float> result(M * N);
    buf_c.download(result.data());

    // Reference: C(m,n) = I * B = B(m,n) = (m * N + n) % kMod
    // Build in C's flat layout using output strides.
    auto [c_sm, c_sn] = rocm_ck::layoutStrides(kSpec.output().layout, M, N);
    std::vector<float> ref(M * N);
    for(int m = 0; m < M; ++m)
        for(int n = 0; n < N; ++n)
            ref[m * c_sm + n * c_sn] = static_cast<float>((m * N + n) % kMod);

    // Verify
    auto vr = rocm_ck::verify(result.data(), ref.data(), M * N, kSpec.output().dtype);
    EXPECT_TRUE(vr.passed) << "Mismatch at [" << vr.first_mismatch
                           << "] got=" << vr.got << " expected=" << vr.expected;
}

// ============================================================================
// Test: A,B = Rademacher (+/-1) => C vs CPU reference (exact in FP32)
// ============================================================================

TEST_F(GemmCorrectness, RademacherGemm)
{
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 128;

    // Allocate device buffers
    rocm_ck::TypedBuffer buf_a(kSpec.lhs().dtype, M * K);
    rocm_ck::TypedBuffer buf_b(kSpec.rhs().dtype, K * N);
    rocm_ck::TypedBuffer buf_c(kSpec.output().dtype, M * N);

    // Build Args
    auto args = rocm_ck::test::buildGemmArgs(
        kSpec, buf_a.ptr(), buf_b.ptr(), buf_c.ptr(), M, N, K);

    // Fill: A,B = Rademacher with different seeds, C = zero
    rocm_ck::test::fill2D(
        args, kSpec.lhs().args_slot, kSpec.lhs().dtype, rocm_ck::test::RademacherGen{37});
    rocm_ck::test::fill2D(
        args, kSpec.rhs().args_slot, kSpec.rhs().dtype, rocm_ck::test::RademacherGen{73});
    rocm_ck::test::fillConstant2D(
        args, kSpec.output().args_slot, kSpec.output().dtype, 0.0f);
    HIP_CHECK(hipDeviceSynchronize());

    // Download A and B for CPU reference
    std::vector<float> host_a(M * K);
    std::vector<float> host_b(K * N);
    buf_a.download(host_a.data());
    buf_b.download(host_b.data());

    // CPU reference: C_ref = A * B (exact — all values are +/-1, sums are integers)
    std::vector<float> ref(M * N, 0.0f);
    rocm_ck::test::cpuGemm(host_a.data(), host_b.data(), ref.data(),
                            M, N, K, args,
                            kSpec.lhs().args_slot,
                            kSpec.rhs().args_slot,
                            kSpec.output().args_slot);

    // Launch GEMM on GPU
    dim3 grid  = rocm_ck::test::gemmGrid(kSpec, M, N);
    dim3 block(kSpec.workgroup_size);
    gemm_kernel<<<grid, block>>>(args);
    HIP_CHECK(hipDeviceSynchronize());

    // Download GPU result
    std::vector<float> result(M * N);
    buf_c.download(result.data());

    // Verify — exact match since all arithmetic is integer-valued in FP32
    auto vr = rocm_ck::verify(result.data(), ref.data(), M * N, kSpec.output().dtype);
    EXPECT_TRUE(vr.passed) << "Mismatch at [" << vr.first_mismatch
                           << "] got=" << vr.got << " expected=" << vr.expected;
}
