// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_mx_gemm_config.hpp"
#include "test_mx_gemm_util.hpp"

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;
using F8  = ck_tile::fp8_t;

using MxFP8Types =
    ::testing::Types<std::tuple<F8, F8, MXfp8_GemmConfig16, Row, Col, Row>,
                     std::tuple<F8, F8, MXfp8_GemmConfig16_Preshuffle, Row, Col, Row>,
                     std::tuple<F8, F8, MXfp8_GemmConfig16_PermuteN, Row, Col, Row>>;

template <typename TypeParam>
class TestMxGemmFp8 : public TestMxGemmUtil<TypeParam>
{
};

TYPED_TEST_SUITE(TestMxGemmFp8, MxFP8Types);

TYPED_TEST(TestMxGemmFp8, BasicSizes)
{
    this->Run(128, 256, 256);
    this->Run(256, 256, 256);
    this->Run(256, 512, 512);
    this->Run(256, 512, 1024);
}
