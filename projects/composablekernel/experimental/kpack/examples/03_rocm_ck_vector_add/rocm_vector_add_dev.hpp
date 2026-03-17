// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Device-side interface for rocm_ck vector add. Wraps CK Tile behind a
// clean config-driven API. This is the only header that .hip files need
// to include.
//
// Uses C++20 struct NTTPs: template <vector_add_config Config>.

#pragma once

#include "rocm_vector_add_api.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/elementwise.hpp"

namespace rocm_ck {

/// Maps a vector_add_config to the CK Tile type machinery.
template <vector_add_config Config>
struct VectorAddTypes
{
    using XDataType       = float;
    using ComputeDataType = float;
    using YDataType       = float;

    using BlockTile  = ck_tile::sequence<Config.block_size>;
    using BlockWarps = ck_tile::sequence<1>;
    using WarpTile   = ck_tile::sequence<Config.block_size>;
    using Shape      = ck_tile::ElementWiseShape<BlockWarps, BlockTile, WarpTile, ComputeDataType>;

    using Problem = ck_tile::ElementWisePipelineProblem<XDataType,
                                                        ComputeDataType,
                                                        YDataType,
                                                        Shape,
                                                        ck_tile::element_wise::Add>;
    using Kernel  = ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy>;
};

/// Device function that bridges VectorAddArgs to CK Tile's kernel.
/// Call this from an extern "C" __global__ wrapper.
template <vector_add_config Config>
__device__ void run_vector_add(VectorAddArgs args)
{
    using Types = VectorAddTypes<Config>;

    auto lens = ck_tile::make_tuple(static_cast<ck_tile::index_t>(args.n));
    static_assert(sizeof(rocm_ck::index_t) == sizeof(ck_tile::index_t),
                  "rocm_ck::index_t and ck_tile::index_t must match");
    auto strides = ck_tile::make_tuple(ck_tile::index_t{1});
    auto inputs  = ck_tile::make_tuple(args.a, args.b);
    typename Types::Kernel{}(lens, strides, strides, inputs, args.c);
}

} // namespace rocm_ck