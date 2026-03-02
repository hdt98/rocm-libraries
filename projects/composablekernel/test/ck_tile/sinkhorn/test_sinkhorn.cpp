// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <tuple>
#include <iostream>
#include <cstring>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/sinkhorn_knopp.hpp"
#include "ck_tile/host/kernel_launch.hpp"

#include "test_sinkhorn_impl.hpp"

// Shape parameters for different test configurations
using BatchSize32 = ck_tile::sequence<32>;
using N4          = ck_tile::sequence<4>;

// Test configurations for different data types and input size
using TestConfig_B1_N4_F16 = std::tuple<ck_tile::half_t, // XDataType
                                        float,           // ComputeDataType
                                        ck_tile::half_t, // YDataType
                                        BatchSize32,     // Batch size (number of N x N matrices)
                                        N4>;             // Size N of the N x N matrix

using TestConfig_B1_N4_F32 = std::tuple<float,       // XDataType
                                        float,       // ComputeDataType
                                        float,       // YDataType
                                        BatchSize32, // Batch size (number of N x N matrices)
                                        N4>;         // Size N of the N x N matrix

using TestTypes = ::testing::Types<TestConfig_B1_N4_F16, TestConfig_B1_N4_F32>;

TYPED_TEST_SUITE(TestCkTileSinkHorn, TestTypes);

TYPED_TEST(TestCkTileSinkHorn, Test_32x4x4) { this->RunGenericTest({32, 4, 4}, 20); }
