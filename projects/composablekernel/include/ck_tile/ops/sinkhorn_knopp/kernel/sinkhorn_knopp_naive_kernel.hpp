// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/sinkhorn_knopp/block/block_sinkhorn_reduce.hpp"

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
    // TODO: Resolve why `double exp(double)` returns all zeroes and remove this
    static_assert(!std::is_same_v<typename Problem::ComputeDataType, double>,
                  "ComputeDataType == double is not supported");

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
        using S               = Problem::BlockShape;
        using InDataType      = typename Problem::InDataType;
        using ComputeDataType = typename Problem::ComputeDataType;
        using OutDataType     = typename Problem::OutDataType;

        auto* p_in  = static_cast<const Problem::InDataType*>(args.p_in);
        auto* p_out = static_cast<Problem::OutDataType*>(args.p_out);

        // Input and output match
        const auto in_out_desc = make_naive_tensor_descriptor(make_tuple(S::BatchSize, S::N * S::N),
                                                              make_tuple(S::N * S::N, 1),
                                                              number<S::N>{},
                                                              number<1>{});

        const auto input_window = [&]() {
            // We require exp(input) > 0, and exp(padding) == 0
            const InDataType input_padding_value = -ck_tile::numeric<InDataType>::infinity();

            auto buffer_view = make_buffer_view<address_space_enum::global>(
                p_in, in_out_desc.get_element_space_size(), input_padding_value);

            const auto in_tensor =
                tensor_view<decltype(buffer_view), decltype(in_out_desc)>{buffer_view, in_out_desc};

            return make_tile_window(
                in_tensor,
                make_tuple(number<S::BatchSize>{}, number<S::N * S::N>{}),
                {0, 0},
                Policy::template MakeFullMatrixBlockTileDistribution<Problem>());
        }();

        auto out_window = [&]() {
            const OutDataType out_padding_value = static_cast<ComputeDataType>(0.0);
            auto out_buffer_view                = make_buffer_view<address_space_enum::global>(
                p_out, in_out_desc.get_element_space_size(), out_padding_value);

            auto out_tensor = tensor_view<decltype(out_buffer_view), decltype(in_out_desc)>{
                out_buffer_view, in_out_desc};

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
