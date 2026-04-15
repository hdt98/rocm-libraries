// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// GPU integration test: bare CK Tile vs rocm_ck for vector addition.
// All data stays on device — no host transfers, no CPU reference.

#include <hip/hip_runtime.h>

#include <rocm_ck/args.hpp>
#include <rocm_ck/elementwise_spec.hpp>
#include <rocm_ck/hip_check.hpp>

#ifdef __HIP_DEVICE_COMPILE__
#include "ck_tile/core.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include <rocm_ck/elementwise_dev.hpp>
#endif

#include "gpu_test_utils.hpp"

#include <cstdio>

// ============================================================================
// Specs
// ============================================================================

static constexpr auto kSpecFP32 = rocm_ck::makeSpec(
    rocm_ck::Signature{.dtype = rocm_ck::DataType::FP32,
                       .ops   = {rocm_ck::AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    rocm_ck::ElementwiseAlgorithm{1024, 1, 1024, true},
    rocm_ck::TargetSet::cdna());

static constexpr auto kSpecFP16 = rocm_ck::makeSpec(
    rocm_ck::Signature{.dtype = rocm_ck::DataType::FP16,
                       .ops   = {rocm_ck::AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    rocm_ck::ElementwiseAlgorithm{1024, 1, 1024, true},
    rocm_ck::TargetSet::cdna());

// ============================================================================
// Kernels
// ============================================================================

__global__ void bare_vectoradd_fp32(int32_t n, const float* a, const float* b, float* c)
{
#ifdef __HIP_DEVICE_COMPILE__
    using Shape = ck_tile::ElementWiseShape<ck_tile::sequence<1>,
                                            ck_tile::sequence<1024>,
                                            ck_tile::sequence<1024>,
                                            float>;
    using Problem =
        ck_tile::ElementWisePipelineProblem<float, float, float, Shape, ck_tile::element_wise::Add>;
    using Kernel = ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy>;

    auto lens    = ck_tile::make_tuple(static_cast<ck_tile::index_t>(n));
    auto strides = ck_tile::make_tuple(ck_tile::index_t{1});
    auto inputs  = ck_tile::make_tuple(a, b);
    Kernel{}(lens, strides, strides, inputs, c);
#endif
}

__global__ void rocmck_vectoradd_fp32(rocm_ck::Args args)
{
#ifdef __HIP_DEVICE_COMPILE__
    rocm_ck::run<kSpecFP32>(args);
#endif
}

__global__ void bare_vectoradd_fp16(int32_t n, const void* a, const void* b, void* c)
{
#ifdef __HIP_DEVICE_COMPILE__
    using X = ck_tile::half_t;
    using Y = ck_tile::half_t;
    using Shape = ck_tile::ElementWiseShape<ck_tile::sequence<1>,
                                            ck_tile::sequence<1024>,
                                            ck_tile::sequence<1024>,
                                            X>;
    using Problem =
        ck_tile::ElementWisePipelineProblem<X, float, Y, Shape, ck_tile::element_wise::Add>;
    using Kernel = ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy>;

    auto lens    = ck_tile::make_tuple(static_cast<ck_tile::index_t>(n));
    auto strides = ck_tile::make_tuple(ck_tile::index_t{1});
    auto inputs  = ck_tile::make_tuple(static_cast<const X*>(a), static_cast<const X*>(b));
    Kernel{}(lens, strides, strides, inputs, static_cast<Y*>(c));
#endif
}

__global__ void rocmck_vectoradd_fp16(rocm_ck::Args args)
{
#ifdef __HIP_DEVICE_COMPILE__
    rocm_ck::run<kSpecFP16>(args);
#endif
}

// ============================================================================
// Test fixtures
// ============================================================================

class VectorAddFP32 : public ::testing::TestWithParam<int>
{
    protected:
    void SetUp() override { rocm_ck::test::skipIfNoGpu(); }
};

class VectorAddFP16 : public ::testing::TestWithParam<int>
{
    protected:
    void SetUp() override { rocm_ck::test::skipIfNoGpu(); }
};

// ============================================================================
// FP32 tests
// ============================================================================

TEST_P(VectorAddFP32, Correctness)
{
    const int n      = GetParam();
    const int ebytes = 4;
    size_t bytes     = n * ebytes;

    void *d_a, *d_b, *d_bare, *d_rocmck;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_bare, bytes));
    HIP_CHECK(hipMalloc(&d_rocmck, bytes));

    rocm_ck::test::deviceFill(d_a, n, ebytes, 8);
    rocm_ck::test::deviceFill(d_b, n, ebytes, 5);

    int grid  = (n + kSpecFP32.block_tile - 1) / kSpecFP32.block_tile;
    int block = kSpecFP32.workgroup_size;

    bare_vectoradd_fp32<<<dim3(grid), dim3(block)>>>(
        n, static_cast<float*>(d_a), static_cast<float*>(d_b), static_cast<float*>(d_bare));

    rocm_ck::Args args{};
    args.tensors[kSpecFP32.lhs().args_slot]    = {d_a, rocm_ck::makeShape(n), rocm_ck::makeStrides(1)};
    args.tensors[kSpecFP32.rhs().args_slot]    = {d_b, rocm_ck::makeShape(n), rocm_ck::makeStrides(1)};
    args.tensors[kSpecFP32.output().args_slot] = {d_rocmck, rocm_ck::makeShape(n), rocm_ck::makeStrides(1)};
    args.scalars[0].f32 = 1.0f;
    args.scalars[1].f32 = 1.0f;

    rocmck_vectoradd_fp32<<<dim3(grid), dim3(block)>>>(args);

    auto mismatch = rocm_ck::test::deviceBitwiseCompare(d_bare, d_rocmck, bytes);
    EXPECT_EQ(mismatch, -1) << "FP32 bare vs rocm_ck mismatch at byte " << mismatch;

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_bare));
    HIP_CHECK(hipFree(d_rocmck));
}

TEST_P(VectorAddFP32, Performance)
{
    const int n = GetParam();
    if(n < 1000000)
        GTEST_SKIP() << "Performance only measured for large sizes";

    const int ebytes = 4;
    size_t bytes     = n * ebytes;

    void *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));
    rocm_ck::test::deviceFill(d_a, n, ebytes);
    rocm_ck::test::deviceFill(d_b, n, ebytes);
    rocm_ck::test::deviceFill(d_c, n, ebytes);
    HIP_CHECK(hipDeviceSynchronize());

    int grid  = (n + kSpecFP32.block_tile - 1) / kSpecFP32.block_tile;
    int block = kSpecFP32.workgroup_size;

    float bare_ms = rocm_ck::test::benchmark("bare_fp32", 10, 50, [&]() {
        bare_vectoradd_fp32<<<dim3(grid), dim3(block)>>>(
            n, static_cast<float*>(d_a), static_cast<float*>(d_b), static_cast<float*>(d_c));
    });

    rocm_ck::Args args{};
    args.tensors[kSpecFP32.lhs().args_slot]    = {d_a, rocm_ck::makeShape(n), rocm_ck::makeStrides(1)};
    args.tensors[kSpecFP32.rhs().args_slot]    = {d_b, rocm_ck::makeShape(n), rocm_ck::makeStrides(1)};
    args.tensors[kSpecFP32.output().args_slot] = {d_c, rocm_ck::makeShape(n), rocm_ck::makeStrides(1)};
    args.scalars[0].f32 = 1.0f;
    args.scalars[1].f32 = 1.0f;

    float rocmck_ms = rocm_ck::test::benchmark("rocm_ck_fp32", 10, 50, [&]() {
        rocmck_vectoradd_fp32<<<dim3(grid), dim3(block)>>>(args);
    });

    double gbps = 3.0 * n * ebytes / (bare_ms * 1e6);
    std::printf("  [PERF] bare GB/s: %.1f\n", gbps);
    rocm_ck::test::reportOverhead("vectoradd_fp32", bare_ms, rocmck_ms);

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
}

INSTANTIATE_TEST_SUITE_P(Sizes,
                         VectorAddFP32,
                         ::testing::Values(1024, 1048576, 67108864),
                         [](const auto& info) {
                             return "n" + std::to_string(info.param);
                         });

// ============================================================================
// FP16 tests
// ============================================================================

TEST_P(VectorAddFP16, Correctness)
{
    const int n      = GetParam();
    const int ebytes = 2;
    size_t bytes     = n * ebytes;

    void *d_a, *d_b, *d_bare, *d_rocmck;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_bare, bytes));
    HIP_CHECK(hipMalloc(&d_rocmck, bytes));

    rocm_ck::test::deviceFill(d_a, n, ebytes, 8);
    rocm_ck::test::deviceFill(d_b, n, ebytes, 5);

    int grid  = (n + kSpecFP16.block_tile - 1) / kSpecFP16.block_tile;
    int block = kSpecFP16.workgroup_size;

    bare_vectoradd_fp16<<<dim3(grid), dim3(block)>>>(n, d_a, d_b, d_bare);

    rocm_ck::Args args{};
    args.tensors[kSpecFP16.lhs().args_slot]    = {d_a, rocm_ck::makeShape(n), rocm_ck::makeStrides(1)};
    args.tensors[kSpecFP16.rhs().args_slot]    = {d_b, rocm_ck::makeShape(n), rocm_ck::makeStrides(1)};
    args.tensors[kSpecFP16.output().args_slot] = {d_rocmck, rocm_ck::makeShape(n), rocm_ck::makeStrides(1)};
    args.scalars[0].f32 = 1.0f;
    args.scalars[1].f32 = 1.0f;

    rocmck_vectoradd_fp16<<<dim3(grid), dim3(block)>>>(args);

    auto mismatch = rocm_ck::test::deviceBitwiseCompare(d_bare, d_rocmck, bytes);
    EXPECT_EQ(mismatch, -1) << "FP16 bare vs rocm_ck mismatch at byte " << mismatch;

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_bare));
    HIP_CHECK(hipFree(d_rocmck));
}

TEST_P(VectorAddFP16, Performance)
{
    const int n = GetParam();
    if(n < 1000000)
        GTEST_SKIP() << "Performance only measured for large sizes";

    const int ebytes = 2;
    size_t bytes     = n * ebytes;

    void *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));
    rocm_ck::test::deviceFill(d_a, n, ebytes);
    rocm_ck::test::deviceFill(d_b, n, ebytes);
    rocm_ck::test::deviceFill(d_c, n, ebytes);
    HIP_CHECK(hipDeviceSynchronize());

    int grid  = (n + kSpecFP16.block_tile - 1) / kSpecFP16.block_tile;
    int block = kSpecFP16.workgroup_size;

    float bare_ms = rocm_ck::test::benchmark("bare_fp16", 10, 50, [&]() {
        bare_vectoradd_fp16<<<dim3(grid), dim3(block)>>>(n, d_a, d_b, d_c);
    });

    rocm_ck::Args args{};
    args.tensors[kSpecFP16.lhs().args_slot]    = {d_a, rocm_ck::makeShape(n), rocm_ck::makeStrides(1)};
    args.tensors[kSpecFP16.rhs().args_slot]    = {d_b, rocm_ck::makeShape(n), rocm_ck::makeStrides(1)};
    args.tensors[kSpecFP16.output().args_slot] = {d_c, rocm_ck::makeShape(n), rocm_ck::makeStrides(1)};
    args.scalars[0].f32 = 1.0f;
    args.scalars[1].f32 = 1.0f;

    float rocmck_ms = rocm_ck::test::benchmark("rocm_ck_fp16", 10, 50, [&]() {
        rocmck_vectoradd_fp16<<<dim3(grid), dim3(block)>>>(args);
    });

    double gbps = 3.0 * n * ebytes / (bare_ms * 1e6);
    std::printf("  [PERF] bare GB/s: %.1f\n", gbps);
    rocm_ck::test::reportOverhead("vectoradd_fp16", bare_ms, rocmck_ms);

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
}

INSTANTIATE_TEST_SUITE_P(Sizes,
                         VectorAddFP16,
                         ::testing::Values(1024, 1048576, 67108864),
                         [](const auto& info) {
                             return "n" + std::to_string(info.param);
                         });
