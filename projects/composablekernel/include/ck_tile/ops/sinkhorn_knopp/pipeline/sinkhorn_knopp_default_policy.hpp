// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct SinkhornKnoppDefaultPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeFullMatrixBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<S::BatchSize, S::N>, sequence<1, S::N>>,
                                       tuple<sequence<1>>,
                                       tuple<sequence<0>>,
                                       sequence<1, 2>,
                                       sequence<1, 1>>{});
    }
};

} // namespace ck_tile
