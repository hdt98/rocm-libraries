// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <string>
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
    return value != nullptr && std::string{value} != "0";
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
                        "8192x512x4096 determinism repro.";
    }

    // ROCm/aiter#3261 repro shape. Skip the host reference because this is a large
    // determinism regression case, not a practical CPU GEMM unit-test size.
    this->Run(8192, 512, 4096, 0, 1, 8, false);
}

TYPED_TEST(TestGemmBlockScaleWP_FP8_MK_NK, Glm52OutOfAllowlistAccuracyAndDeterminism)
{
    if(!run_long_gemm_regression_tests())
    {
        GTEST_SKIP() << "Set CK_RUN_LONG_GEMM_REGRESSION_TESTS=1 to run the GLM-5.2-FP8 "
                        "out-of-allowlist bpreshuffle accuracy and determinism regression shapes.";
    }

    constexpr std::tuple<int, int, int> shapes[] = {
        // M, N, K. Representative GLM-5.2-FP8 block-FP8 linear projections
        // reported to route to bpreshuffle on gfx950 because their (N, K) pairs
        // are outside SGLang's tuned Triton allowlist.
        {8, 2048, 6144},   // q_a_proj: hidden_size -> q_lora_rank
        {8, 16384, 2048},  // q_b_proj: q_lora_rank -> n_head * qk_head_dim
        {8, 28672, 512},   // kv_b_proj: kv_lora_rank -> n_head * (qk_nope + v)
    };

    for(const auto& shape : shapes)
    {
        // do_verification=true compares against the profiler's dequantized FP32
        // host GEMM oracle; determinism_check=2 reruns the same device problem.
        this->Run(std::get<0>(shape), std::get<1>(shape), std::get<2>(shape), 0, 1, 2, true);
    }
}
