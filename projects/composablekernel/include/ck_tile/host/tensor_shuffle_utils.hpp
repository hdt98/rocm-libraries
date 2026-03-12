// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "device_prop.hpp"
#include <stdexcept>

namespace ck_tile {
template <typename T>
auto shuffle_aq(const ck_tile::HostTensor<T>* t, int block_aq_k)
{
    if(t->get_lengths().size() != 2)
    {
        throw std::runtime_error("Host tensor is not rank 2 tensor.");
    }
    int m_   = t->get_lengths()[0];
    int aqk_ = t->get_lengths()[1];

    if(aqk_ % block_aq_k != 0)
    {
        throw std::runtime_error("shuffle_aq needs a aqk of multiple times of block_aq_k.");
    }
    ck_tile::HostTensor<T> t_view({m_, aqk_ / block_aq_k, block_aq_k});
    std::copy(t->begin(), t->end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {1, 0, 2});
}

template <typename T>
auto shuffle_bq(const ck_tile::HostTensor<T>* t, int block_bq_k)
{
    const auto& lengths = t->get_lengths();
    const size_t rank   = lengths.size();

    // Validate block_bq_k divisibility based on rank
    int bqk_dim = (rank == 5) ? lengths[4] : (rank == 2) ? lengths[0] : -1;

    if(bqk_dim < 0)
    {
        throw std::runtime_error("shuffle_bq expects either rank-2 or rank-5 tensor, got rank " +
                                 std::to_string(rank));
    }

    if(bqk_dim % block_bq_k != 0)
    {
        throw std::runtime_error("shuffle_bq needs bqk dimension to be a multiple of block_bq_k.");
    }

    // For TilePermuteN
    if(rank == 5)
    {
        // Handle 5D tensor: [n, nrepeat, nwarp, n_warp_tile, bqk]
        ck_tile::HostTensor<T> t_view({static_cast<int>(lengths[0]),
                                       static_cast<int>(lengths[1]),
                                       static_cast<int>(lengths[2]),
                                       static_cast<int>(lengths[3]),
                                       bqk_dim / block_bq_k,
                                       block_bq_k});
        std::copy(t->begin(), t->end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {4, 0, 1, 2, 3, 5});
    }
    else // rank == 2
    {
        // Handle 2D tensor: [bqk, n]
        int n_ = lengths[1];
        ck_tile::HostTensor<T> t_view({n_, bqk_dim / block_bq_k, block_bq_k});
        std::copy(t->begin(), t->end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {1, 0, 2});
    }
}

template <typename GemmConfig, typename T>
auto shuffle_b(const ck_tile::HostTensor<T>& t, GemmConfig)
{
    assert(t.get_lengths().size() == 2);
    int n_ = t.get_lengths()[1];
    int k_ = t.get_lengths()[0];

    if(ck_tile::is_gfx12_supported())
    {
        constexpr int divisor      = 2;
        constexpr int kABK1PerLane = 8;
        int kABK0PerLane           = GemmConfig::K_Warp_Tile / divisor / kABK1PerLane;
        ck_tile::HostTensor<T> t_view({n_ / GemmConfig::N_Warp_Tile,
                                       GemmConfig::N_Warp_Tile,
                                       k_ / GemmConfig::K_Warp_Tile,
                                       kABK0PerLane,
                                       divisor,
                                       kABK1PerLane});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 4, 1, 3, 5});
    }
    else if(ck_tile::is_gfx11_supported())
    {
        int divisor = 1;
        ck_tile::HostTensor<T> t_view({n_ / GemmConfig::N_Warp_Tile,
                                       GemmConfig::N_Warp_Tile,
                                       k_ / GemmConfig::K_Warp_Tile,
                                       divisor,
                                       GemmConfig::K_Warp_Tile / divisor});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
    }
    else
    {
        constexpr int KLane = ck_tile::get_warp_size() / GemmConfig::N_Warp_Tile;
        constexpr int ItemsPerAccess =
            std::min(16 / static_cast<int>(sizeof(T)), GemmConfig::K_Warp_Tile / KLane);

        ck_tile::HostTensor<T> t_view({n_ / GemmConfig::N_Warp_Tile,
                                       GemmConfig::N_Warp_Tile,
                                       k_ / ItemsPerAccess,
                                       ItemsPerAccess});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 1, 3});
    }
}

template <typename GemmConfig, typename T>
auto shuffle_b(const ck_tile::HostTensor<T>& t)
{
    return shuffle_b(t, GemmConfig{});
}

/*
For the permuteN feature, the BQ (scale) tensor must be shuffled so its layout matches the MFMA
result. The required shuffle depends on group_N (the N dimension of the quant group). The BQ tensor
is treated as an N×K matrix with N = 128 / group_N: 1×8×128 → N = 16 (BQ 16×1) 1×16×128 → N = 8
1×32×128 → N = 4
1×64×128 → N = 2
1×128×128 → N = 1 (one value per full block tile)
The shuffle is given as a 4-tuple whose meaning is tied to n, N_warp, N_warp_tile, and NRepeat:
group_N = 8: Shuffle (1, 4, 2, 2) — i.e. (1, N_warp, N_warp_tile/group_N, NRepeat) with
N_warp_tile/8 = 2 (e.g. 16/8). group_N = 16: Shuffle (1, 4, 1, 2) — 1 value per N_warp_tile
(N_warp_tile/16 = 1). group_N = 32: Shuffle (1, 2, 1, 2) — (1, N_warp/2, 1, NRepeat); 2 N_warp_tiles
share the same scale. group_N = 64: Shuffle (1, 1, 1, 2) — (1, N_warp/4, 1, NRepeat); 4 N_warp_tiles
share the same value. group_N = 128: Shuffle (1, 1, 1, 1) — 1 value for the full block tile.

The alignment problem:
When the BQ tensor is shuffled according to these rules (the 4-tuples above), its layout no longer
matches what the block pipeline expects after block-level MFMA. So even with the “correct” shuffle
for each group_N, BQ is misaligned with the MFMA result at the block level. That’s why the code
today only enables TiledPermuteN for BQuantGroupSize::kN equal to 1 or 128.

Options to fix alignment
1) Update tile_distribution_encoding for permuteN
Adjust the BQ tile distribution encoding so the encoded distribution matches the shuffle layout and
aligns with how the block consumes C and BQ after MFMA. 2) Update the tile window when reading from
DRAM Keep the shuffle as defined above and change how the BQ tile window is built when reading from
device memory so that the window layout matches the post-MFMA layout. 3) Update indexes for BQ reads
Keep the shuffle and tile window; change the indexing used when reading BQ in the block so that each
thread/warp loads the scale that corresponds to its part of the MFMA result.
*/
template <typename GemmConfig, typename T>
auto bq_permuteN(const ck_tile::HostTensor<T>& t, index_t group_n)
{
    assert(t.get_lengths().size() == 2);

    int n_                = t.get_lengths()[1];
    int bqk_              = t.get_lengths()[0];
    constexpr int NRepeat = GemmConfig::N_Tile / GemmConfig::N_Warp_Tile / GemmConfig::N_Warp;
    int dim               = (group_n == 1) ? n_ / GemmConfig::N_Tile : n_;

    ck_tile::HostTensor<T> t_view =
        (group_n == 1) ? ck_tile::HostTensor<T>(
                             {dim, GemmConfig::N_Warp, GemmConfig::N_Warp_Tile, NRepeat, bqk_})
                       : ck_tile::HostTensor<T>({dim, 1, 1, 1, bqk_});

    std::copy(t.begin(), t.end(), t_view.begin());

    return ck_tile::reference_permute(t_view, {0, 3, 1, 2, 4});
}

template <typename GemmConfig, typename T>
auto shuffle_b_permuteN(const ck_tile::HostTensor<T>& t, const GemmConfig& gemmConfig)
{
    assert(t.get_lengths().size() == 2);
    int n_      = t.get_lengths()[1];
    int k_      = t.get_lengths()[0];
    int NRepeat = gemmConfig.N_Tile / gemmConfig.N_Warp_Tile / gemmConfig.N_Warp;
    if(ck_tile::is_gfx12_supported())
    {
        constexpr int divisor      = 2;
        constexpr int kABK1PerLane = 8;
        int kABK0PerLane           = gemmConfig.K_Warp_Tile / divisor / kABK1PerLane;
        ck_tile::HostTensor<T> t_view({n_ / gemmConfig.N_Tile,
                                       gemmConfig.N_Warp,
                                       gemmConfig.N_Warp_Tile,
                                       NRepeat,
                                       k_ / gemmConfig.K_Warp_Tile,
                                       kABK0PerLane,
                                       divisor,
                                       kABK1PerLane});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 3, 1, 4, 6, 5, 2, 7});
    }
    else
    {
        constexpr int KLane = ck_tile::get_warp_size() / GemmConfig::N_Warp_Tile;
        constexpr int ItemsPerAccess =
            std::min(16 / static_cast<int>(sizeof(T)), GemmConfig::K_Warp_Tile / KLane);
        ck_tile::HostTensor<T> t_view({n_ / gemmConfig.N_Tile,
                                       gemmConfig.N_Warp,
                                       gemmConfig.N_Warp_Tile,
                                       NRepeat,
                                       k_ / ItemsPerAccess,
                                       ItemsPerAccess});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 3, 1, 4, 2, 5});
    }
}

template <typename GemmConfig, typename T>
auto shuffle_b_permuteN(const ck_tile::HostTensor<T>& t)
{
    return shuffle_b_permuteN(t, GemmConfig{});
}
} // namespace ck_tile
