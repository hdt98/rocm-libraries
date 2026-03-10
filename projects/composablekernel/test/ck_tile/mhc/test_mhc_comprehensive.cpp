// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/reference/reference_mhc.hpp"
#include "ck_tile/host/check_err.hpp"

// Comprehensive MHC tests using reference implementation
// This validates the reference implementation which the kernels should match

template <typename XDataType,
          typename PhiDataType,
          typename YDataType,
          typename ComputeDataType,
          typename ActivationFunc = ck_tile::element_wise::Sigmoid>
class MHCReferenceTest
{
    public:
    template <int B, int n, int C, int sinkhorn_iters = 0>
    static bool RunTest(const std::string& test_name)
    {
        const int nC         = n * C;
        const int output_dim = 2 * n + n * n;

        std::cout << "\n--- " << test_name << " ---" << std::endl;
        std::cout << "  B=" << B << ", n=" << n << ", C=" << C << ", Sinkhorn=" << sinkhorn_iters
                  << std::endl;

        // Allocate host tensors
        ck_tile::HostTensor<XDataType> h_x({B, nC});
        ck_tile::HostTensor<PhiDataType> h_phi({nC, output_dim});
        ck_tile::HostTensor<YDataType> h_output({B, output_dim});

        // Initialize with random data
        ck_tile::FillUniformDistribution<XDataType>{-1.0f, 1.0f}(h_x);
        ck_tile::FillUniformDistribution<PhiDataType>{-0.5f, 0.5f}(h_phi);
        h_output.SetZero();

        const float r = 1.0f, alpha_pre = 1.0f, alpha_post = 1.0f, alpha_res = 1.0f, bias = 0.0f;

        // Compute reference
        ck_tile::reference_mhc<XDataType, PhiDataType, YDataType, ComputeDataType, ActivationFunc>(
            h_x, h_phi, h_output, n, C, r, alpha_pre, alpha_post, alpha_res, bias, sinkhorn_iters);

        // Basic sanity check - output should not be all zeros
        bool has_nonzero = false;
        for(int b = 0; b < B && !has_nonzero; b++)
        {
            for(int i = 0; i < output_dim && !has_nonzero; i++)
            {
                if(std::abs(static_cast<float>(h_output(b, i))) > 1e-6f)
                {
                    has_nonzero = true;
                }
            }
        }

        std::cout << "  Result: " << (has_nonzero ? "PASS (non-zero output)" : "FAIL (all zeros)")
                  << std::endl;

        return has_nonzero;
    }
};

// Test fixture for different data types
template <typename TestConfig>
class TestMHCComprehensive : public ::testing::Test
{
    public:
    using XDataType       = std::tuple_element_t<0, TestConfig>;
    using PhiDataType     = std::tuple_element_t<1, TestConfig>;
    using YDataType       = std::tuple_element_t<2, TestConfig>;
    using ComputeDataType = std::tuple_element_t<3, TestConfig>;
};

// Test configurations
using TestConfig_F32  = std::tuple<float, float, float, float>;
using TestConfig_BF16 = std::tuple<ck_tile::bf16_t, ck_tile::bf16_t, float, float>;

using TestTypes = ::testing::Types<TestConfig_F32, TestConfig_BF16>;

TYPED_TEST_SUITE(TestMHCComprehensive, TestTypes);

// (1) Different batch sizes to test M dimension parallelism
TYPED_TEST(TestMHCComprehensive, BatchSize8)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<8, 4, 64>("Batch Size 8")));
}

TYPED_TEST(TestMHCComprehensive, BatchSize16)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<16, 4, 64>("Batch Size 16")));
}

TYPED_TEST(TestMHCComprehensive, BatchSize32)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<32, 4, 64>("Batch Size 32")));
}

TYPED_TEST(TestMHCComprehensive, BatchSize64)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<64, 4, 64>("Batch Size 64")));
}

TYPED_TEST(TestMHCComprehensive, BatchSize128)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<128, 4, 64>("Batch Size 128")));
}

// (2) Large C tests to validate split-K works correctly
TYPED_TEST(TestMHCComprehensive, LargeC_512)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<16, 4, 512>("Large C=512 (Split-K)")));
}

TYPED_TEST(TestMHCComprehensive, LargeC_1024)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<16, 4, 1024>("Large C=1024 (Split-K)")));
}

TYPED_TEST(TestMHCComprehensive, LargeC_4096)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<8, 4, 4096>("Large C=4096 (Split-K)")));
}

// (3) Sinkhorn tests - with and without iterations
TYPED_TEST(TestMHCComprehensive, SinkhornDisabled)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<16, 4, 64, 0>("Sinkhorn Disabled (0 iters)")));
}

TYPED_TEST(TestMHCComprehensive, Sinkhorn20Iterations)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<16, 4, 64, 20>("Sinkhorn Enabled (20 iters)")));
}

TYPED_TEST(TestMHCComprehensive, SinkhornLargeC)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<8, 4, 1024, 20>("Sinkhorn with Large C")));
}

// (4) Different activation functions
TYPED_TEST(TestMHCComprehensive, ActivationTanh)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType,
                                    ck_tile::element_wise::TanH>;
    EXPECT_TRUE((Tester::template RunTest<16, 4, 64>("Activation: TanH")));
}

TYPED_TEST(TestMHCComprehensive, ActivationReLU)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType,
                                    ck_tile::element_wise::Relu>;
    EXPECT_TRUE((Tester::template RunTest<16, 4, 64>("Activation: ReLU")));
}

TYPED_TEST(TestMHCComprehensive, ActivationSiLU)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType,
                                    ck_tile::element_wise::Silu>;
    EXPECT_TRUE((Tester::template RunTest<16, 4, 64>("Activation: SiLU")));
}

TYPED_TEST(TestMHCComprehensive, ActivationSigmoid)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType,
                                    ck_tile::element_wise::Sigmoid>;
    EXPECT_TRUE((Tester::template RunTest<16, 4, 64>("Activation: Sigmoid")));
}

// Combined stress tests
TYPED_TEST(TestMHCComprehensive, StressTest_LargeBatch_LargeC_Sinkhorn)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType,
                                    ck_tile::element_wise::Sigmoid>;
    EXPECT_TRUE((Tester::template RunTest<64, 4, 2048, 20>("Stress: B=64, C=2048, Sinkhorn=20")));
}

// (5) Edge cases: B odd, even, non-power-of-two
TYPED_TEST(TestMHCComprehensive, BatchSize_Odd_7)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<7, 4, 64>("B=7 (odd)")));
}

TYPED_TEST(TestMHCComprehensive, BatchSize_Odd_15)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<15, 4, 64>("B=15 (odd)")));
}

TYPED_TEST(TestMHCComprehensive, BatchSize_Odd_33)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<33, 4, 64>("B=33 (odd)")));
}

TYPED_TEST(TestMHCComprehensive, BatchSize_NonPowerOfTwo_12)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<12, 4, 64>("B=12 (non-power-of-2)")));
}

TYPED_TEST(TestMHCComprehensive, BatchSize_NonPowerOfTwo_48)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<48, 4, 64>("B=48 (non-power-of-2)")));
}

TYPED_TEST(TestMHCComprehensive, BatchSize_NonPowerOfTwo_100)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<100, 4, 64>("B=100 (non-power-of-2)")));
}

// (6) Edge cases: C odd, even, non-power-of-two
TYPED_TEST(TestMHCComprehensive, C_Odd_63)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<16, 4, 63>("C=63 (odd)")));
}

TYPED_TEST(TestMHCComprehensive, C_Odd_127)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<16, 4, 127>("C=127 (odd)")));
}

TYPED_TEST(TestMHCComprehensive, C_Odd_255)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<16, 4, 255>("C=255 (odd)")));
}

TYPED_TEST(TestMHCComprehensive, C_NonPowerOfTwo_48)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<16, 4, 48>("C=48 (non-power-of-2)")));
}

TYPED_TEST(TestMHCComprehensive, C_NonPowerOfTwo_96)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<16, 4, 96>("C=96 (non-power-of-2)")));
}

TYPED_TEST(TestMHCComprehensive, C_NonPowerOfTwo_192)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<16, 4, 192>("C=192 (non-power-of-2)")));
}

// (7) Different n values (expansion factor)
TYPED_TEST(TestMHCComprehensive, ExpansionFactor_n4)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<16, 4, 64>("n=4 (expansion factor)")));
}

TYPED_TEST(TestMHCComprehensive, ExpansionFactor_n5)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<16, 5, 64>("n=5 (expansion factor, odd)")));
}

TYPED_TEST(TestMHCComprehensive, ExpansionFactor_n8)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<16, 8, 64>("n=8 (expansion factor)")));
}

TYPED_TEST(TestMHCComprehensive, ExpansionFactor_n2)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<16, 2, 128>("n=2 (expansion factor)")));
}

TYPED_TEST(TestMHCComprehensive, ExpansionFactor_n3)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<16, 3, 85>("n=3 (expansion factor, odd)")));
}

// (8) Combined edge cases
TYPED_TEST(TestMHCComprehensive, EdgeCase_OddB_OddC_OddN)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<15, 5, 63>("B=15, n=5, C=63 (all odd)")));
}

TYPED_TEST(TestMHCComprehensive, EdgeCase_NonPow2_All)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<12, 3, 48>("B=12, n=3, C=48 (all non-power-of-2)")));
}

TYPED_TEST(TestMHCComprehensive, EdgeCase_LargeN_SmallC)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<16, 8, 32>("n=8, C=32 (large n, small C)")));
}

// (9) Padding tests - dimensions not divisible by tile sizes
// Tile sizes: M=16/32/64/128, N=32, K=64
// These tests ensure padding logic works correctly

TYPED_TEST(TestMHCComprehensive, Padding_B_NotDivisibleBy16)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE(
        (Tester::template RunTest<17, 4, 64>("B=17 (not divisible by 16, requires padding)")));
}

TYPED_TEST(TestMHCComprehensive, Padding_B_NotDivisibleBy32)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE(
        (Tester::template RunTest<50, 4, 64>("B=50 (not divisible by 32, requires padding)")));
}

TYPED_TEST(TestMHCComprehensive, Padding_B_NotDivisibleBy64)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE(
        (Tester::template RunTest<100, 4, 64>("B=100 (not divisible by 64, requires padding)")));
}

TYPED_TEST(TestMHCComprehensive, Padding_C_NotDivisibleBy64)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<16, 4, 100>("C=100 (not divisible by 64, K padding)")));
}

TYPED_TEST(TestMHCComprehensive, Padding_C_NotDivisibleBy32)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    EXPECT_TRUE((Tester::template RunTest<16, 4, 50>("C=50 (not divisible by 32, K padding)")));
}

TYPED_TEST(TestMHCComprehensive, Padding_OutputDim_NotDivisibleBy32)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    // n=5 gives output_dim = 2*5 + 5*5 = 35 (not divisible by 32)
    EXPECT_TRUE((Tester::template RunTest<16, 5, 64>("n=5, output_dim=35 (N padding)")));
}

TYPED_TEST(TestMHCComprehensive, Padding_AllDimensions)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    // B=17, n=5 (output=35), C=50 - all require padding
    EXPECT_TRUE((Tester::template RunTest<17, 5, 50>("B=17, n=5, C=50 (all dims need padding)")));
}

// (10) Multi-warp tests - ensure multiple warps per block work correctly
// M=64 uses 2 warps, M=128 uses 4 warps

TYPED_TEST(TestMHCComprehensive, MultiWarp_2Warps_B64)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    // B=64 with M=64 tile -> 1 block with 2 warps
    EXPECT_TRUE((Tester::template RunTest<64, 4, 64>("B=64 (2 warps per block)")));
}

TYPED_TEST(TestMHCComprehensive, MultiWarp_2Warps_B128)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    // B=128 with M=64 tile -> 2 blocks with 2 warps each
    EXPECT_TRUE((Tester::template RunTest<128, 4, 64>("B=128 (2 blocks × 2 warps)")));
}

TYPED_TEST(TestMHCComprehensive, MultiWarp_4Warps_B128)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    // B=128 with M=128 tile -> 1 block with 4 warps
    EXPECT_TRUE((Tester::template RunTest<128, 4, 128>("B=128 (4 warps per block)")));
}

TYPED_TEST(TestMHCComprehensive, MultiWarp_4Warps_B256)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    // B=256 with M=128 tile -> 2 blocks with 4 warps each
    EXPECT_TRUE((Tester::template RunTest<256, 4, 128>("B=256 (2 blocks × 4 warps)")));
}

// (11) Multi-block tests - ensure multiple blocks work correctly
// Large B and large output_dim will trigger multiple blocks

TYPED_TEST(TestMHCComprehensive, MultiBlock_LargeB_256)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    // B=256 requires multiple blocks in M dimension
    EXPECT_TRUE((Tester::template RunTest<256, 4, 64>("B=256 (multiple M blocks)")));
}

TYPED_TEST(TestMHCComprehensive, MultiBlock_LargeOutputDim)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    // n=8 gives output_dim = 2*8 + 8*8 = 80 (requires multiple N blocks)
    EXPECT_TRUE((Tester::template RunTest<32, 8, 64>("n=8, output_dim=80 (multiple N blocks)")));
}

TYPED_TEST(TestMHCComprehensive, MultiBlock_LargeC_SplitK)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    // C=2048 requires split-K (multiple K blocks)
    EXPECT_TRUE((Tester::template RunTest<32, 4, 2048>("C=2048 (multiple K blocks, split-K)")));
}

TYPED_TEST(TestMHCComprehensive, MultiBlock_AllDimensions)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    // Large B, large n (large output_dim), large C -> multiple blocks in all dimensions
    EXPECT_TRUE((Tester::template RunTest<256, 8, 2048>("B=256, n=8, C=2048 (multi-block M×N×K)")));
}

// (12) Combined multi-warp + padding tests
TYPED_TEST(TestMHCComprehensive, MultiWarp_WithPadding_B65)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    // B=65 with M=64 tile -> 2 blocks (1 full, 1 padded) with 2 warps each
    EXPECT_TRUE((Tester::template RunTest<65, 4, 64>("B=65 (2 warps + padding)")));
}

TYPED_TEST(TestMHCComprehensive, MultiWarp_WithPadding_B130)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    // B=130 with M=128 tile -> 2 blocks (1 full, 1 padded) with 4 warps each
    EXPECT_TRUE((Tester::template RunTest<130, 4, 128>("B=130 (4 warps + padding)")));
}

TYPED_TEST(TestMHCComprehensive, MultiWarp_WithPadding_AllDims)
{
    using Tester = MHCReferenceTest<typename TestFixture::XDataType,
                                    typename TestFixture::PhiDataType,
                                    typename TestFixture::YDataType,
                                    typename TestFixture::ComputeDataType>;
    // B=130, n=5 (output=35), C=100 - multi-warp + padding in all dims
    EXPECT_TRUE(
        (Tester::template RunTest<130, 5, 100>("B=130, n=5, C=100 (multi-warp + all padding)")));
}
