// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// CK Tile vector addition kernel, compiled device-only into .hsaco code objects.
//
// This demonstrates the "bridge pattern" for using CK Tile kernels with kpack:
// an extern "C" __global__ wrapper takes flat arguments compatible with
// hipModuleLaunchKernel and delegates to CK Tile's template-based kernel.

#include "ck_tile/core.hpp"
#include "ck_tile/ops/elementwise.hpp"

using XDataType       = float;
using ComputeDataType = float;
using YDataType       = float;

using BlockTile  = ck_tile::sequence<256>;
using BlockWarps = ck_tile::sequence<1>;
using WarpTile   = ck_tile::sequence<256>;
using Shape      = ck_tile::ElementWiseShape<BlockWarps, BlockTile, WarpTile, ComputeDataType>;

using Problem = ck_tile::ElementWisePipelineProblem<
    XDataType, ComputeDataType, YDataType, Shape, ck_tile::element_wise::Add>;
using Kernel = ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy>;

extern "C" __global__ void ck_tile_vector_add(
    ck_tile::index_t n,
    const float* a,
    const float* b,
    float* c)
{
    auto lens    = ck_tile::make_tuple(n);
    auto strides = ck_tile::make_tuple(ck_tile::index_t{1});
    auto inputs  = ck_tile::make_tuple(a, b);
    Kernel{}(lens, strides, strides, inputs, c);
}