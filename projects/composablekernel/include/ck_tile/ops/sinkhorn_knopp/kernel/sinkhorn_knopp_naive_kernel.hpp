// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/sinkhorn_knopp/block/block_sinkhorn_naive.hpp"

namespace ck_tile {

struct SinkhornKnoppArgs
{
    void* p_out;
    const void* p_in;
    const std::vector<index_t> input_shape; // Only used for checking input size compatibility
    int iterations;
};

template <typename Problem, typename Policy>
struct SinkhornKnoppNaiveKernel
{
    static constexpr index_t kBlockSize = Problem::BlockShape::BlockSize;

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return is_wave32() ? kBlockSize / 2 : kBlockSize;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const SinkhornKnoppArgs& args)
    {
        bool supported = true;
        // Allow zero iterations, because why not; Might be helpful for debugging
        supported &= args.iterations >= 0;
        // supported &= input_shape.get_length(0) == input_shape.get_length(1);
        return supported;
    }

    CK_TILE_DEVICE void operator()(const SinkhornKnoppArgs& args) const
    {
        using S = Problem::BlockShape;

        auto* p_in  = static_cast<const Problem::InDataType*>(args.p_in);
        auto* p_out = static_cast<Problem::OutDataType*>(args.p_out);

        // Input and output match
        const auto in_out_desc = make_naive_tensor_descriptor(make_tuple(S::BatchSize, S::N * S::N),
                                                              make_tuple(S::N * S::N, 1),
                                                              number<S::N>{},
                                                              number<1>{});

        const auto input_window = [&]() {
            auto in_tensor = make_tensor_view<address_space_enum::global>(p_in, in_out_desc);
            return make_tile_window(
                in_tensor,
                make_tuple(number<S::BatchSize>{}, number<S::N * S::N>{}),
                {0, 0},
                Policy::template MakeFullMatrixBlockTileDistribution<Problem>());
        }();

        auto out_window = [&]() {
            auto out_tensor = make_tensor_view<address_space_enum::global>(p_out, in_out_desc);
            return make_tile_window(
                out_tensor,
                make_tuple(number<S::BatchSize>{}, number<S::N * S::N>{}),
                {0, 0},
                Policy::template MakeFullMatrixBlockTileDistribution<Problem>());
        }();

        auto input_tile = load_tile(input_window);

        sinkhorn_knopp_naive_full<S::N>(input_tile, out_window, args.iterations);
    }
};

} // namespace ck_tile
