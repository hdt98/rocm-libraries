// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"

#include "mx_wmma_coexecution.hpp"

using ck::e8m0_bexp_t;
using ck::f4_t;
using ck::f4x2_pk_t;
using ck::type_convert;

/**
 * @brief Run the test for the given WMMA scale instruction
 *
 * @param init - selects initialization algorithm for A and B tensors
 */
template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AType,
          typename BType,
          typename CType,
          ck::WMMA_SCALE wmma,
          ck::index_t num_steps>
bool run_mx_wmma_coexecution_test(ck::index_t init)
{
    static_assert((wmma == ck::WMMA_SCALE::SCALE_F32_16x16x128) ||
                      (wmma == ck::WMMA_SCALE::SCALE_F32_32x16x128),
                  "Only SCALE_F32_16x16x128 and SCALE_F32_32x16x128 are supported");

    using AccType   = float;           // only F32 instructions supported
    using ScaleType = ck::e8m0_bexp_t; // biased exponent type

    // WMMA scale instruction parameters
    ck::mfma_type<static_cast<ck::MfmaInstr>(wmma)> wmma_instr;
    constexpr auto BLOCK_M = wmma_instr.m_per_blk;
    constexpr auto BLOCK_N = wmma_instr.n_per_blk;
    constexpr auto BLOCK_K = wmma_instr.num_input_blks * wmma_instr.k_per_blk;
    constexpr auto BLOCK_X = 32; // scaling vector size

    const auto mx_wmma_kernel = ck::matmul<AType,
                                           BType,
                                           ScaleType,
                                           CType,
                                           AccType,
                                           BLOCK_M,
                                           BLOCK_N,
                                           BLOCK_K,
                                           BLOCK_X,
                                           ALayout,
                                           BLayout,
                                           CLayout,
                                           num_steps>;

    bool pass = true;

    pass = ck::mx_wmma_test::TestMXWMMA<decltype(mx_wmma_kernel),
                                        AType,
                                        BType,
                                        ScaleType,
                                        CType,
                                        ALayout,
                                        BLayout,
                                        CLayout,
                                        BLOCK_M,
                                        BLOCK_N,
                                        BLOCK_K,
                                        BLOCK_X>{}(mx_wmma_kernel, init);

    return pass;
}

const ck::index_t common_init = -1;

TEST(MXWMMA, MXFP4WMMA16x16x128)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 1;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             ck::WMMA_SCALE::SCALE_F32_16x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA, MXFP4WMMA32x16x128)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 2;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             ck::WMMA_SCALE::SCALE_F32_32x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}
