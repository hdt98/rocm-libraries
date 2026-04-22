// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_mx_gemm_config.hpp"
#include "test_mx_gemm_util.hpp"

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

using MxFp8Types =
    ::testing::Types<std::tuple<ck_tile::fp8_t, ck_tile::fp8_t, MXfp8_GemmConfig16, Row, Col, Row>>;

using MxFp8TypesPadMN = ::testing::Types<
    std::tuple<ck_tile::fp8_t, ck_tile::fp8_t, MXfp8_GemmConfig16_PadMN, Row, Col, Row>>;

template <typename TypeParam>
class TestMxGemmFp8 : public TestMxGemmUtil<std::tuple_element_t<0, TypeParam>,
                                            std::tuple_element_t<1, TypeParam>,
                                            std::tuple_element_t<2, TypeParam>,
                                            std::tuple_element_t<3, TypeParam>,
                                            std::tuple_element_t<4, TypeParam>,
                                            std::tuple_element_t<5, TypeParam>>
{
};

template <typename TypeParam>
class TestMxGemmFp8PadMN : public TestMxGemmUtil<std::tuple_element_t<0, TypeParam>,
                                                 std::tuple_element_t<1, TypeParam>,
                                                 std::tuple_element_t<2, TypeParam>,
                                                 std::tuple_element_t<3, TypeParam>,
                                                 std::tuple_element_t<4, TypeParam>,
                                                 std::tuple_element_t<5, TypeParam>>
{
};

TYPED_TEST_SUITE(TestMxGemmFp8, MxFp8Types);
TYPED_TEST_SUITE(TestMxGemmFp8PadMN, MxFp8TypesPadMN);

TYPED_TEST(TestMxGemmFp8, BasicSizes)
{
    this->Run(64, 64, 256);
    this->Run(128, 128, 256);
    this->Run(64, 128, 512);
}

// Regression for the hot-loop / tail dispatch bug: num_loop == 3 must not enter the hot loop.
// K = 768 with K_Tile = 256 gives num_loop = 3 and was computing 5 gemms instead of 3.
TYPED_TEST(TestMxGemmFp8, HotLoopTailNumLoopThree)
{
    this->Run(64, 64, 768);
    this->Run(128, 128, 768);
    this->Run(256, 256, 768);
}

// Regression for split-K: with the MX kernel not threading blockIdx.z and not offsetting its
// scale windows, split_k > 1 produced wildly wrong results. These shapes exercise both the
// full_k_read and partial_k_read paths of SplitKBatchOffset.
TYPED_TEST(TestMxGemmFp8, SplitK)
{
    this->Run(128, 128, 512, /*k_batch=*/2);
    this->Run(128, 128, 1024, /*k_batch=*/2);
    this->Run(128, 128, 1024, /*k_batch=*/4);
    this->Run(256, 256, 2048, /*k_batch=*/4);
}

// M/N padding: M and/or N not multiples of M_Tile = N_Tile = 64. Requires kPadM/kPadN = true
// on the pipeline (so the Underlying base builds pad views / relies on buffer OOB) and on
// the MX kernel (so scale windows and the C output view are padded). K stays aligned: the
// MX async pipeline does not currently support K padding (vector loads can straddle the K
// pad boundary). M and N must each be >= their block tile (CShuffleEpilogue cannot safely
// run with a single partial tile along either dimension).
TYPED_TEST(TestMxGemmFp8PadMN, MNPaddingAligned)
{
    // Sanity: kPadM/kPadN = true with already-aligned M, N must not regress the normal path.
    this->Run(64, 64, 256);
}

TYPED_TEST(TestMxGemmFp8PadMN, MPadding)
{
    // M has a full tile + partial trailing tile.
    this->Run(96, 128, 256);
    this->Run(192, 128, 256);
}

TYPED_TEST(TestMxGemmFp8PadMN, NPadding)
{
    // N has a full tile + partial trailing tile.
    this->Run(128, 96, 256);
    this->Run(128, 192, 256);
}

TYPED_TEST(TestMxGemmFp8PadMN, MNPadding)
{
    // Both M and N unaligned (full + partial trailing tiles).
    this->Run(96, 96, 256);
    this->Run(128, 96, 512);
    this->Run(192, 192, 256);
}
