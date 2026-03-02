// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

namespace ck_tile {

// TODO: Generalize to Warps > 1, Batchsize != BlockSize
template <typename _BatchSize, // Number of N x N matrices
          typename _N>         // N x N matrix size
struct SinkhornKnoppShape
{
    static constexpr index_t N         = _N::at(number<0>{});
    static constexpr index_t BatchSize = _BatchSize::at(number<0>{});
    static constexpr index_t BlockSize = ck_tile::get_warp_size();
};

template <typename _InDataType,
          typename _OutDataType,
          typename _BlockShape,
          typename _ComputeDataType = float>
struct SinkhornKnoppProblem
{
    using InDataType      = remove_cvref_t<_InDataType>;
    using ComputeDataType = remove_cvref_t<_ComputeDataType>;
    using OutDataType     = remove_cvref_t<_OutDataType>;

    using BlockShape = remove_cvref_t<_BlockShape>;
};

} // namespace ck_tile
