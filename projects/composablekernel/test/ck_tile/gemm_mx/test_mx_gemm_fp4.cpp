// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_mx_gemm_config.hpp"
#include "test_mx_gemm_util.hpp"

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;
using FP4 = ck_tile::pk_fp4_t;

using MxFP4Types =
    ::testing::Types<std::tuple<FP4, FP4, MXfp4_GemmConfig16, Row, Col, Row>,
                     std::tuple<FP4, FP4, MXfp4_GemmConfig16_Preshuffle, Row, Col, Row>,
                     std::tuple<FP4, FP4, MXfp4_GemmConfig16_PermuteN, Row, Col, Row>>;

template <typename TypeParam>
class TestMxGemmFp4 : public TestMxGemmUtil<TypeParam>
{
};

TYPED_TEST_SUITE(TestMxGemmFp4, MxFP4Types);

TYPED_TEST(TestMxGemmFp4, BasicSizes)
{
    this->Run(128, 512, 256);
    this->Run(256, 512, 256);
    this->Run(256, 1024, 512);
    this->Run(256, 1024, 1024);
}
