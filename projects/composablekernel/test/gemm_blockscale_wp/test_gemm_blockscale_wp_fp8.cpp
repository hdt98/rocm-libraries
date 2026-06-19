// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <cstring>
#include <tuple>

#include "gtest/gtest.h"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "test_gemm_common.hpp"

using F8   = ck::f8_t;
using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

namespace {

template <typename X, typename Y>
struct tuple_concat;

template <typename... Xs, typename... Ys>
struct tuple_concat<std::tuple<Xs...>, std::tuple<Ys...>>
{
    using type = std::tuple<Xs..., Ys...>;
};

bool run_long_gemm_regression_tests()
{
    const char* value = std::getenv("CK_RUN_LONG_GEMM_REGRESSION_TESTS");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

} // namespace

template <typename Tuple>
class TestGemmBlockScaleWP_FP8_MK_NK : public ck::test::TestGemmBlockscaleWPCommon<
                                           typename tuple_concat<std::tuple<Row, Col>, Tuple>::type>
{
};

// clang-format off
using KernelTypes_MK_NK = ::testing::Types<
#if defined(CK_ENABLE_FP8)
    std::tuple< F8, F32, F8, F32, F8, BF16>
#endif
    >;
// clang-format on

TYPED_TEST_SUITE(TestGemmBlockScaleWP_FP8_MK_NK, KernelTypes_MK_NK);

TYPED_TEST(TestGemmBlockScaleWP_FP8_MK_NK, Regular0)
{
    std::vector<int> Ms{128, 256, 512};
    constexpr int N = 512;
    constexpr int K = 512;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestGemmBlockScaleWP_FP8_MK_NK, ReportedWkvDeterminism)
{
    if(!run_long_gemm_regression_tests())
    {
        GTEST_SKIP() << "Set CK_RUN_LONG_GEMM_REGRESSION_TESTS=1 to run the ROCm/aiter#3261 "
                        "8192x512xK determinism repro shapes.";
    }

    // ROCm/aiter#3261 repro sweep from the original CK stabilizer investigation.
    // Skip the host reference because these are large determinism regression
    // cases, not practical CPU GEMM unit-test sizes.
    constexpr int M    = 8192;
    constexpr int N    = 512;
    constexpr int Ks[] = {384, 640, 3968, 4096, 4224};

    for(const int K : Ks)
        this->Run(M, N, K, 0, 1, 8, false);
}

TYPED_TEST(TestGemmBlockScaleWP_FP8_MK_NK, Glm52OutOfAllowlistAccuracyAndDeterminism)
{
    if(!run_long_gemm_regression_tests())
    {
        GTEST_SKIP() << "Set CK_RUN_LONG_GEMM_REGRESSION_TESTS=1 to run the GLM-5.2-FP8 "
                        "out-of-allowlist bpreshuffle accuracy and determinism regression shapes.";
    }

    constexpr std::tuple<int, int, int> shapes[] = {
        // M, N, K. GLM-5.2-FP8 block-FP8 linear shapes from
        // sgl-project/sglang#28685: concrete projection repros plus the
        // kv_b_proj row-boundary sweep where bad rows clustered near 16-row
        // tile edges.
        {8, 2048, 6144},   // q_a_proj: hidden_size -> q_lora_rank
        {8, 16384, 2048},  // q_b_proj short prefill
        {64, 16384, 2048}, // q_b_proj reporter repro
        {12, 6144, 12288}, // mlp.down_proj short prefill
        {64, 6144, 12288}, // mlp.down_proj reporter repro
        {8, 28672, 512},   // kv_b_proj short smoke
        {12, 28672, 512},  // kv_b_proj short prefill
        {32, 28672, 512},  // kv_b_proj row-boundary sweep
        {48, 28672, 512},  // kv_b_proj row-boundary sweep
        {56, 28672, 512},  // kv_b_proj row-boundary sweep
        {64, 28672, 512},  // kv_b_proj reporter repro
        {72, 28672, 512},  // kv_b_proj row-boundary sweep
        {96, 28672, 512},  // kv_b_proj row-boundary sweep
        {128, 28672, 512}, // kv_b_proj row-boundary sweep
    };

    for(const auto& shape : shapes)
    {
        // do_verification=true compares against the profiler's dequantized FP32
        // host GEMM oracle; determinism_check=4 reruns the same device problem.
        this->Run(std::get<0>(shape), std::get<1>(shape), std::get<2>(shape), 0, 1, 4, true);
    }
}
