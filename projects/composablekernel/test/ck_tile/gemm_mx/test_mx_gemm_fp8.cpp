// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_mx_gemm_config.hpp"
#include "test_mx_gemm_util.hpp"

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// 16x16x128 warp tile configs
// Only A=Row,B=Col is supported: KWarpTile=128 exceeds ds_read_tr limit, disabling transpose loads
using MxFp8Types16 =
    ::testing::Types<std::tuple<ck_tile::fp8_t, ck_tile::fp8_t, MXfp8_GemmConfig16, Row, Col, Row>>;

template <typename TypeParam>
class TestMxGemmFp8_16 : public TestMxGemmUtil<std::tuple_element_t<0, TypeParam>,
                                               std::tuple_element_t<1, TypeParam>,
                                               std::tuple_element_t<2, TypeParam>,
                                               std::tuple_element_t<3, TypeParam>,
                                               std::tuple_element_t<4, TypeParam>,
                                               std::tuple_element_t<5, TypeParam>>
{
};

TYPED_TEST_SUITE(TestMxGemmFp8_16, MxFp8Types16);

TYPED_TEST(TestMxGemmFp8_16, BasicSizes)
{
    this->Run(64, 64, 256);
    this->Run(128, 128, 256);
    this->Run(64, 128, 512);
}

// 32x32x64 warp tile configs (enables ds_read_tr for transpose loads)
using MxFp8Types32 =
    ::testing::Types<std::tuple<ck_tile::fp8_t, ck_tile::fp8_t, MXfp8_GemmConfig32, Row, Col, Row>,
                     std::tuple<ck_tile::fp8_t, ck_tile::fp8_t, MXfp8_GemmConfig32, Row, Row, Row>,
                     std::tuple<ck_tile::fp8_t, ck_tile::fp8_t, MXfp8_GemmConfig32, Col, Col, Row>,
                     std::tuple<ck_tile::fp8_t, ck_tile::fp8_t, MXfp8_GemmConfig32, Col, Row, Row>>;

template <typename TypeParam>
class TestMxGemmFp8_32 : public TestMxGemmUtil<std::tuple_element_t<0, TypeParam>,
                                               std::tuple_element_t<1, TypeParam>,
                                               std::tuple_element_t<2, TypeParam>,
                                               std::tuple_element_t<3, TypeParam>,
                                               std::tuple_element_t<4, TypeParam>,
                                               std::tuple_element_t<5, TypeParam>>
{
};

TYPED_TEST_SUITE(TestMxGemmFp8_32, MxFp8Types32);

TYPED_TEST(TestMxGemmFp8_32, BasicSizes)
{
    this->Run(128, 128, 256);
    this->Run(128, 128, 512);
}

TYPED_TEST(TestMxGemmFp8_32, MultiBlockMN)
{
    this->Run(256, 128, 256);
    this->Run(128, 256, 256);
    this->Run(256, 256, 256);
}
