// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Naive Sinkhorn-Knopp iteration with the whole NxN matrix loaded within the thread
// FIXME: N should be deduced from the input (or compute) tile or otherwise generalized out
template <index_t N,
          typename InDistributedTensor,
          typename OutDistributedTensor,
          typename ComputeDataType = float>
CK_TILE_DEVICE void sinkhorn_knopp_naive_full(const InDistributedTensor& input_tile,
                                              OutDistributedTensor out_tile,
                                              index_t iterations)
{

    using InDataType  = InDistributedTensor::DataType;
    using OutDataType = OutDistributedTensor::DataType;

    // Exponentiate the input to make it strictly positive
    auto exp_func = [](InDataType x) -> ComputeDataType {
        return ck_tile::exp(type_convert<ComputeDataType>(x));
    };

    auto compute_tile = tile_elementwise_in(exp_func, input_tile);

    // Control iteration over rows and columns with a passed indexer function
    // FIXME: This is specific to MakeFullMatrixBlockTileDistribution; there should be a more
    // general way to loop over the "inner" indices of the thread-specific NxN matrix and disregard
    // the batch dimension
    constexpr auto mkRowIdx = [](const auto i, const auto j) {
        return make_tuple(tile_distributed_index<>{}, tile_distributed_index<i, j>{});
    };

    constexpr auto mkColIdx = [](const auto i, const auto j) {
        return make_tuple(tile_distributed_index<>{}, tile_distributed_index<j, i>{});
    };

    auto normalize = [&compute_tile](const auto& mkIdx) {
        // Loop over first dim
        static_for<0, N, 1>{}([&](const auto i) {
            // 1. Sum over second dim
            ComputeDataType sum = 0.0;
            static_for<0, N, 1>{}([&](const auto j) { sum += compute_tile(mkIdx(i, j)); });
            // 2. Divide values by the sum
            static_for<0, N, 1>{}([&](const auto j) { compute_tile(mkIdx(i, j)) /= sum; });
        });
    };

    // Iterate normalization for rows and columns
    for(int it = 0; it < iterations; it++)
    {
        normalize(mkRowIdx);
        normalize(mkColIdx);
    }

    store_tile(out_tile, cast_tile<OutDataType>(compute_tile));
}

} // namespace ck_tile
