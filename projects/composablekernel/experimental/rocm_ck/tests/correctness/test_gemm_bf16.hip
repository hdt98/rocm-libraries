// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// BF16 GEMM correctness tests.
//
// BF16 has 7 mantissa bits (8-bit significand including implicit 1).
// Key precision boundaries:
//   [0, 256)    ULP = 1  — all integers representable
//   [256, 512)  ULP = 2  — only even integers
//   [512, 1024) ULP = 4  — only multiples of 4
//
// Accumulation is in FP32, so the precision concern is the store
// from FP32 accumulator to BF16 output.
//
// OnesCross256:  A = B = 1, K = 512  =>  C = 512 (BF16 roundtrip)

#include <hip/hip_runtime.h>

#include <rocm_ck/args.hpp>
#include <rocm_ck/datatype_convert.hpp>
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
    rocm_ck::Signature{.dtype = rocm_ck::DataType::BF16,
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

class GemmBF16 : public ::testing::Test
{
protected:
    void SetUp() override { rocm_ck::test::skipIfNoGpu(); }
};

// ============================================================================
// Test: A = B = all 1s, K = 512 => C = 512 (crosses 256 precision boundary)
//
// The reference uses BF16 roundtrip conversion so the test catches
// incorrect rounding at any K value.
// ============================================================================

TEST_F(GemmBF16, OnesCross256)
{
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 512; // result > 256, tests BF16 precision boundary

    rocm_ck::TypedBuffer buf_a(kSpec.lhs().dtype, M * K);
    rocm_ck::TypedBuffer buf_b(kSpec.rhs().dtype, K * N);
    rocm_ck::TypedBuffer buf_c(kSpec.output().dtype, M * N);

    auto args = rocm_ck::test::buildGemmArgs(
        kSpec, buf_a.ptr(), buf_b.ptr(), buf_c.ptr(), M, N, K);

    rocm_ck::test::fillConstant2D(
        args, kSpec.lhs().args_slot, kSpec.lhs().dtype, 1.0f);
    rocm_ck::test::fillConstant2D(
        args, kSpec.rhs().args_slot, kSpec.rhs().dtype, 1.0f);
    rocm_ck::test::fillConstant2D(
        args, kSpec.output().args_slot, kSpec.output().dtype, 0.0f);
    HIP_CHECK(hipDeviceSynchronize());

    dim3 grid  = rocm_ck::test::gemmGrid(kSpec, M, N);
    dim3 block(kSpec.workgroup_size);
    gemm_kernel<<<grid, block>>>(args);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> result(M * N);
    buf_c.download(result.data());

    // Reference: each element = K, rounded through BF16
    float expected = rocm_ck::bf16BitsToFloat(
        rocm_ck::floatToBf16Bits(static_cast<float>(K)));
    std::vector<float> ref(M * N, expected);

    auto vr = rocm_ck::verify(result.data(), ref.data(), M * N, kSpec.output().dtype);
    EXPECT_TRUE(vr.passed) << "Mismatch at [" << vr.first_mismatch
                           << "] got=" << vr.got << " expected=" << vr.expected
                           << " (expected BF16 repr of " << K << " = " << expected << ")";
}
