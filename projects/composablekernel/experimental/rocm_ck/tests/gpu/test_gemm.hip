// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// GPU integration test: bare CK Tile vs rocm_ck for plain FP16 GEMM.
// All data stays on device — no host transfers, no CPU reference.

#include <hip/hip_runtime.h>

#include <rocm_ck/args.hpp>
#include <rocm_ck/gemm_spec.hpp>
#include <rocm_ck/hip_check.hpp>
#include <rocm_ck/layout.hpp>

#ifdef __HIP_DEVICE_COMPILE__
#include <rocm_ck/gemm_dev.hpp>
#endif

#include "gpu_test_utils.hpp"

#include <cstdio>

// ============================================================================
// Spec
// ============================================================================

static constexpr auto kSpec = rocm_ck::makeSpec(
    rocm_ck::Signature{.dtype = rocm_ck::DataType::FP16,
                       .ops   = {rocm_ck::GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    rocm_ck::GemmAlgorithm{.block_tile  = {128, 128, 32},
                           .block_waves = {2, 2, 1},
                           .wave_tile   = {16, 16, 16}},
    rocm_ck::TargetSet::cdna());

// ============================================================================
// BareGemmArgs
// ============================================================================

struct BareGemmArgs
{
    const void* a_ptr;
    const void* b_ptr;
    void* c_ptr;
    int32_t M, N, K;
    int32_t stride_a, stride_b, stride_c;
};

// ============================================================================
// Kernels
// ============================================================================

__global__ void bare_gemm_fp16(BareGemmArgs bargs)
{
#ifdef __HIP_DEVICE_COMPILE__
    using AType   = ck_tile::half_t;
    using BType   = ck_tile::half_t;
    using AccType = float;
    using OType   = ck_tile::half_t;

    using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
    using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<128, 128, 32>,
                               ck_tile::sequence<2, 2, 1>,
                               ck_tile::sequence<16, 16, 16>>;

    using Traits = ck_tile::TileGemmTraits<false, false, false, ALayout, BLayout, CLayout>;

    using Pipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<
        ck_tile::GemmPipelineProblem<AType, BType, AccType, GemmShape, Traits>>;

    using Partitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;

    using Epilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<AType,
                                         BType,
                                         ck_tile::tuple<>,
                                         AccType,
                                         OType,
                                         ck_tile::tuple<>,
                                         CLayout,
                                         ck_tile::element_wise::PassThrough,
                                         Partitioner::MPerBlock,
                                         Partitioner::NPerBlock,
                                         2, 2, 16, 16, 16,
                                         false>>;

    using CkKernel   = ck_tile::GemmKernel<Partitioner, Pipeline, Epilogue>;
    using KernelArgs = typename CkKernel::UniversalGemmKernel::KernelArgs;

    const KernelArgs kargs{
        {static_cast<const AType*>(bargs.a_ptr)},
        {static_cast<const BType*>(bargs.b_ptr)},
        {},
        bargs.c_ptr,
        bargs.M, bargs.N, bargs.K,
        {bargs.stride_a}, {bargs.stride_b}, {},
        bargs.stride_c, 1};
    CkKernel{}(kargs);
#endif
}

__global__ void rocmck_gemm_fp16(rocm_ck::Args args)
{
#ifdef __HIP_DEVICE_COMPILE__
    rocm_ck::run<kSpec>(args);
#endif
}

// ============================================================================
// Helpers
// ============================================================================

static BareGemmArgs makeBareArgs(void* a, void* b, void* c, int M, int N, int K)
{
    auto [a_sm, a_sk] = rocm_ck::layoutStrides(kSpec.lhs().layout, M, K);
    auto [b_sk, b_sn] = rocm_ck::layoutStrides(kSpec.rhs().layout, K, N);
    auto [c_sm, c_sn] = rocm_ck::layoutStrides(kSpec.output().layout, M, N);

    BareGemmArgs bargs{};
    bargs.a_ptr    = a;
    bargs.b_ptr    = b;
    bargs.c_ptr    = c;
    bargs.M        = M;
    bargs.N        = N;
    bargs.K        = K;
    bargs.stride_a = static_cast<int32_t>(rocm_ck::leadingDimStride(kSpec.lhs().layout, rocm_ck::makeStrides(a_sm, a_sk)));
    bargs.stride_b = static_cast<int32_t>(rocm_ck::leadingDimStride(kSpec.rhs().layout, rocm_ck::makeStrides(b_sk, b_sn)));
    bargs.stride_c = static_cast<int32_t>(rocm_ck::leadingDimStride(kSpec.output().layout, rocm_ck::makeStrides(c_sm, c_sn)));
    return bargs;
}

static rocm_ck::Args makeRocmCkArgs(void* a, void* b, void* c, int M, int N, int K)
{
    auto [a_sm, a_sk] = rocm_ck::layoutStrides(kSpec.lhs().layout, M, K);
    auto [b_sk, b_sn] = rocm_ck::layoutStrides(kSpec.rhs().layout, K, N);
    auto [c_sm, c_sn] = rocm_ck::layoutStrides(kSpec.output().layout, M, N);

    rocm_ck::Args args{};
    args.tensors[kSpec.lhs().args_slot]    = {a, rocm_ck::makeShape(M, K), rocm_ck::makeStrides(a_sm, a_sk)};
    args.tensors[kSpec.rhs().args_slot]    = {b, rocm_ck::makeShape(K, N), rocm_ck::makeStrides(b_sk, b_sn)};
    args.tensors[kSpec.output().args_slot] = {c, rocm_ck::makeShape(M, N), rocm_ck::makeStrides(c_sm, c_sn)};
    return args;
}

// ============================================================================
// Test fixture
// ============================================================================

struct GemmSize { int M, N, K; };

class GemmFP16 : public ::testing::TestWithParam<GemmSize>
{
    protected:
    void SetUp() override { rocm_ck::test::skipIfNoGpu(); }
};

static std::string gemmSizeName(const ::testing::TestParamInfo<GemmSize>& info)
{
    auto s = info.param;
    return std::to_string(s.M) + "x" + std::to_string(s.N) + "x" + std::to_string(s.K);
}

// ============================================================================
// Tests
// ============================================================================

TEST_P(GemmFP16, Correctness)
{
    auto [M, N, K] = GetParam();
    constexpr int ebytes = 2;

    void *d_a, *d_b, *d_bare, *d_rocmck;
    HIP_CHECK(hipMalloc(&d_a, M * K * ebytes));
    HIP_CHECK(hipMalloc(&d_b, K * N * ebytes));
    HIP_CHECK(hipMalloc(&d_bare, M * N * ebytes));
    HIP_CHECK(hipMalloc(&d_rocmck, M * N * ebytes));

    rocm_ck::test::deviceFill(d_a, M * K, ebytes, 4);
    rocm_ck::test::deviceFill(d_b, K * N, ebytes, 4);
    HIP_CHECK(hipMemset(d_bare, 0, M * N * ebytes));
    HIP_CHECK(hipMemset(d_rocmck, 0, M * N * ebytes));

    int grid = ((M + 127) / 128) * ((N + 127) / 128);

    auto bargs = makeBareArgs(d_a, d_b, d_bare, M, N, K);
    bare_gemm_fp16<<<dim3(grid), dim3(kSpec.workgroup_size)>>>(bargs);

    auto args = makeRocmCkArgs(d_a, d_b, d_rocmck, M, N, K);
    rocmck_gemm_fp16<<<dim3(grid), dim3(kSpec.workgroup_size)>>>(args);

    auto mismatch = rocm_ck::test::deviceBitwiseCompare(d_bare, d_rocmck, M * N * ebytes);
    EXPECT_EQ(mismatch, -1) << "GEMM bare vs rocm_ck mismatch at byte " << mismatch;

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_bare));
    HIP_CHECK(hipFree(d_rocmck));
}

TEST_P(GemmFP16, Performance)
{
    auto [M, N, K] = GetParam();
    if(M < 1024)
        GTEST_SKIP() << "Performance only measured for larger sizes";

    constexpr int ebytes = 2;

    void *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, M * K * ebytes));
    HIP_CHECK(hipMalloc(&d_b, K * N * ebytes));
    HIP_CHECK(hipMalloc(&d_c, M * N * ebytes));
    rocm_ck::test::deviceFill(d_a, M * K, ebytes, 4);
    rocm_ck::test::deviceFill(d_b, K * N, ebytes, 4);
    rocm_ck::test::deviceFill(d_c, M * N, ebytes);
    HIP_CHECK(hipDeviceSynchronize());

    int grid = ((M + 127) / 128) * ((N + 127) / 128);

    auto bargs = makeBareArgs(d_a, d_b, d_c, M, N, K);
    float bare_ms = rocm_ck::test::benchmark("bare_gemm_fp16", 5, 20, [&]() {
        bare_gemm_fp16<<<dim3(grid), dim3(kSpec.workgroup_size)>>>(bargs);
    });

    auto args = makeRocmCkArgs(d_a, d_b, d_c, M, N, K);
    float rocmck_ms = rocm_ck::test::benchmark("rocm_ck_gemm_fp16", 5, 20, [&]() {
        rocmck_gemm_fp16<<<dim3(grid), dim3(kSpec.workgroup_size)>>>(args);
    });

    double tflops = 2.0 * M * N * K / (bare_ms * 1e9);
    std::printf("  [PERF] bare TFLOPS: %.2f\n", tflops);
    rocm_ck::test::reportOverhead("gemm_fp16", bare_ms, rocmck_ms);

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
}

INSTANTIATE_TEST_SUITE_P(Sizes,
                         GemmFP16,
                         ::testing::Values(GemmSize{256, 256, 256},
                                           GemmSize{1024, 1024, 256},
                                           GemmSize{2048, 2048, 256}),
                         gemmSizeName);
