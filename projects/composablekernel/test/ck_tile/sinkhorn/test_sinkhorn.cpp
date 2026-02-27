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
using Shape1_BatchSize = ck_tile::sequence<2>;
using Shape1_N         = ck_tile::sequence<4>;

// Test configurations for different data types and input size
using TestConfig_F16 = std::tuple<float,            // XDataType
                                  float,            // ComputeDataType
                                  float,            // YDataType
                                  Shape1_BatchSize, // Batch size (number of N x N matrices)
                                  Shape1_N>;        // Size N of the N x N matrix

using TestTypes = ::testing::Types<TestConfig_F16>;

TYPED_TEST_SUITE(TestCkTileSinkHorn, TestTypes);

TYPED_TEST(TestCkTileSinkHorn, Test_4x4) { this->RunGenericTest({2, 4, 4}, 20); }
