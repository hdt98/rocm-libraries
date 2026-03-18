// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Device-side interface for rocm_ck vector add. Wraps CK Tile behind a
// clean config-driven API. This is the only header that .hip files need
// to include.
//
// Uses C++20 struct NTTPs: template <VectorAddKernel K>.

#pragma once

#include "rocm_vector_add_api.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/elementwise.hpp"

namespace rocm_ck {

/// Maps a DataType enum value to the corresponding CK Tile numeric type.
/// Primary template is intentionally undefined — only valid specializations compile.
template <DataType>
struct CkTypeMap;

template <>
struct CkTypeMap<DataType::FP32>
{
    using type = float;
};
template <>
struct CkTypeMap<DataType::FP16>
{
    using type = ck_tile::half_t;
};
template <>
struct CkTypeMap<DataType::BF16>
{
    using type = ck_tile::bf16_t;
};
template <>
struct CkTypeMap<DataType::FP8>
{
    using type = ck_tile::fp8_t;
};

/// Maps a VectorAddKernel to the CK Tile type machinery.
template <VectorAddKernel K>
struct VectorAddTypes
{
    using XDataType       = typename CkTypeMap<K.compute_type>::type;
    using ComputeDataType = float;
    using YDataType       = typename CkTypeMap<K.compute_type>::type;

    using BlockTile  = ck_tile::sequence<K.block_tile>;
    using BlockWarps = ck_tile::sequence<K.block_warps>;
    using WarpTile   = ck_tile::sequence<K.warp_tile>;
    using Shape      = ck_tile::ElementWiseShape<BlockWarps, BlockTile, WarpTile, XDataType>;

    using Problem = ck_tile::ElementWisePipelineProblem<XDataType,
                                                        ComputeDataType,
                                                        YDataType,
                                                        Shape,
                                                        ck_tile::element_wise::Add>;
    using Kernel  = ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy>;
};

/// Device function that bridges VectorAddArgs to CK Tile's kernel.
/// Call this from an extern "C" __global__ wrapper.
template <VectorAddKernel K>
__device__ void runVectorAdd(VectorAddArgs args)
{
    using Types = VectorAddTypes<K>;
    using X     = typename Types::XDataType;
    using Y     = typename Types::YDataType;

    auto lens = ck_tile::make_tuple(static_cast<ck_tile::index_t>(args.n));
    static_assert(sizeof(rocm_ck::index_t) == sizeof(ck_tile::index_t),
                  "rocm_ck::index_t and ck_tile::index_t must match");
    auto strides = ck_tile::make_tuple(ck_tile::index_t{1});
    auto inputs = ck_tile::make_tuple(static_cast<const X*>(args.a), static_cast<const X*>(args.b));
    typename Types::Kernel{}(lens, strides, strides, inputs, static_cast<Y*>(args.c));
}

} // namespace rocm_ck