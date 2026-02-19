#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Naive Sinkhorn-Knopp iteration with the whole matrix loaded within the thread
template <typename InDistributedTensor,
          typename OutDistributedTensor,
          typename ComputeDataType = float>
CK_TILE_DEVICE void sinkhorn_knopp_naive_full(const InDistributedTensor& input_tile,
                                              OutDistributedTensor out_tile,
                                              index_t iterations)
{
    using InDataType  = InDistributedTensor::DataType;
    using OutDataType = OutDistributedTensor::DataType;

    // Run the first steps iteration of the Sinkhorn-Knopp algorithm
    // Exponentiate the input to make it strictly positive
    auto exp_func = [](InDataType x) -> ComputeDataType {
        return ck_tile::exp(type_convert<ComputeDataType>(x));
    };

    auto compute_tile    = tile_elementwise_in(exp_func, input_tile);
    constexpr auto spans = compute_tile.get_distributed_spans();

    // Loop the Sinkhorn-Knopp normalization for rows and columns
    for(int i = 0; i < iterations; i++)
    {
        sweep_tile_span(spans[number<0>{}], [&](const auto idx0) {
            // 1. Compute row sums
            ComputeDataType row_sum = 0.0;
            sweep_tile_span(spans[number<1>{}], [&](const auto idx1) {
                constexpr auto idx = make_tuple(idx0, idx1);
                row_sum += compute_tile(idx);
            });

            // 2. Divide values in the row by the sum
            sweep_tile_span(spans[number<1>{}], [&](const auto idx1) {
                constexpr auto c_idx = make_tuple(idx0, idx1);
                compute_tile(c_idx)  = compute_tile(c_idx) / row_sum;
            });
        });

        // Repeat for columns
        sweep_tile_span(spans[number<1>{}], [&](const auto idx1) {
            ComputeDataType col_sum = 0.0;
            sweep_tile_span(spans[number<0>{}], [&](const auto idx0) {
                constexpr auto idx = make_tuple(idx0, idx1);
                col_sum += compute_tile(idx);
            });

            sweep_tile_span(spans[number<0>{}], [&](const auto idx0) {
                constexpr auto idx = make_tuple(idx0, idx1);
                compute_tile(idx)  = compute_tile(idx) / col_sum;
            });
        });
    }

    store_tile(out_tile, cast_tile<OutDataType>(compute_tile));
}

} // namespace ck_tile
