// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../config.hpp"

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
#include "ck_tile/core.hpp"
#endif

#ifdef UNIFIED_TILE_BACKEND_MINT
#include "mint/mint.h"
#endif

namespace unified_tile {
namespace distribution {

// ============================================================================
// Shared: block_copy_2d_config
// ============================================================================
// Backend-independent decomposition of a 2D tile across threads.
// Both CK_TILE and MINT read from this config to build their native types.
//
// Model: A 2D tile of size (Dim0 x Dim1) is distributed across BlockSize
// threads. Dim1 is the "inner" (contiguous) dimension, vectorized by VecSize.
//
//   threads_inner = Dim1 / VecSize     (thread groups along inner dim)
//   threads_outer = BlockSize / threads_inner
//   elems_outer   = Dim0 / threads_outer  (per-thread iteration count)
//   elems_inner   = VecSize               (per-thread vector width)
//
// Thread ID decomposed as: tid = tid_outer * threads_inner + tid_inner
// ============================================================================

template <int BlockSize, int Dim0, int Dim1, int VecSize>
struct block_copy_2d_config
{
    static constexpr int kBlockSize = BlockSize;
    static constexpr int kDim0 = Dim0;        // outer (non-contiguous)
    static constexpr int kDim1 = Dim1;        // inner (contiguous)
    static constexpr int kVecSize = VecSize;

    // Thread decomposition
    static constexpr int kThreadsInner = Dim1 / VecSize;
    static constexpr int kThreadsOuter = BlockSize / kThreadsInner;

    // Per-thread element counts
    static constexpr int kElemsOuter = Dim0 / kThreadsOuter;
    static constexpr int kElemsInner = VecSize;
    static constexpr int kElemsPerThread = kElemsOuter * kElemsInner;

    // Compile-time validation
    static_assert(Dim1 % VecSize == 0,
                  "Inner dim must be divisible by VecSize");
    static_assert(BlockSize % kThreadsInner == 0,
                  "BlockSize must be divisible by threads along inner dim");
    static_assert(Dim0 % kThreadsOuter == 0,
                  "Outer dim must be divisible by threads along outer dim");
    static_assert(kElemsPerThread * BlockSize == Dim0 * Dim1,
                  "Total elements must equal tile area");
};

// ============================================================================
// Backend-specific: distribution construction from config
// ============================================================================

#ifdef UNIFIED_TILE_BACKEND_CK_TILE

namespace detail {

/// @brief Build CK_TILE tile_distribution from a block_copy_2d_config.
/// Maps config → tile_distribution_encoding_pattern_2d → tile_distribution.
template <typename Config>
CK_TILE_DEVICE constexpr auto make_distribution_from_config()
{
    // CK_TILE convention: Y = outer (non-contiguous), X = inner (contiguous)
    using Pattern = ck_tile::tile_distribution_encoding_pattern_2d<
        Config::kBlockSize,
        Config::kDim0,
        Config::kDim1,
        Config::kVecSize,
        ck_tile::tile_distribution_pattern::thread_raked>;
    return Pattern::make_2d_static_tile_distribution();
}

} // namespace detail

#else // UNIFIED_TILE_BACKEND_MINT

namespace detail {

/// @brief Build MINT distributed_tensor_descriptor from a block_copy_2d_config.
/// Maps config -> morphers (merge P, split Dim0, split Dim1) -> polymorpher.
/// @tparam Config         The block_copy_2d_config type
/// @tparam kDim0Alias     Alias for outer dim (e.g., "M")
/// @tparam kDim1Alias     Alias for inner dim (e.g., "K")
/// @tparam kDim0ElemAlias Alias for outer element sub-dim (e.g., "M_0")
/// @tparam kDim0ThrAlias  Alias for outer thread sub-dim (e.g., "M_1")
/// @tparam kDim1GrpAlias  Alias for inner group sub-dim (e.g., "K_0")
/// @tparam kDim1ElemAlias Alias for inner element sub-dim (e.g., "K_1")
template <typename Config,
          mint::alias_t kDim0Alias,
          mint::alias_t kDim1Alias,
          mint::alias_t kDim0ElemAlias,
          mint::alias_t kDim0ThrAlias,
          mint::alias_t kDim1GrpAlias,
          mint::alias_t kDim1ElemAlias>
MINT_DEVICE constexpr auto make_distribution_from_config()
{
    using namespace mint;
    using namespace mint::poly;
    using namespace mint::tensor;

    // Read decomposition from shared config
    constexpr auto threads_outer = static_cast<index_t>(Config::kThreadsOuter);
    constexpr auto threads_inner = static_cast<index_t>(Config::kThreadsInner);
    constexpr auto elems_outer = static_cast<index_t>(Config::kElemsOuter);
    constexpr auto vec_size = static_cast<index_t>(Config::kVecSize);
    constexpr auto block_size = static_cast<index_t>(Config::kBlockSize);
    constexpr auto dim0 = static_cast<index_t>(Config::kDim0);
    constexpr auto dim1 = static_cast<index_t>(Config::kDim1);

    // 3 morphers: merge(P), split(Dim0), split(Dim1)
    constexpr auto p_merge =
        merge<nd_index<2>>{{threads_outer, threads_inner}};
    constexpr auto d0_split =
        split<nd_index<2>>{{elems_outer, threads_outer}};
    constexpr auto d1_split =
        split<nd_index<2>>{{threads_inner, vec_size}};

    // Graph edges: P.sub1 === Dim0.sub1, P.sub2 === Dim1.sub0
    constexpr auto dim_pairs =
        nd_array<index_t, 2, 2, 2>{{{0, 1}, {1, 1}}, {{0, 2}, {2, 0}}};
    constexpr auto morphers = mint::make_tuple(p_merge, d0_split, d1_split);

    constexpr auto alias_to_morpher = []() {
        static_map<alias_t, index_t, 3> ret;
        ret["P"] = 0;
        ret[kDim0Alias] = 1;
        ret[kDim1Alias] = 2;
        return ret;
    }();

    constexpr auto alias_to_dim = []() {
        static_map<alias_t, nd_index<2>, 7> ret;
        ret["P"] = {0, 0};
        ret[kDim0Alias] = {1, 2};
        ret[kDim0ElemAlias] = {1, 0};
        ret[kDim0ThrAlias] = {1, 1};
        ret[kDim1Alias] = {2, 2};
        ret[kDim1GrpAlias] = {2, 0};
        ret[kDim1ElemAlias] = {2, 1};
        return ret;
    }();

    constexpr auto lengths = nd_index<7>{
        block_size, dim0, elems_outer, threads_outer,
        dim1, threads_inner, vec_size};

    constexpr auto poly_result =
        make_polymorpher<dim_pairs, alias_to_morpher, alias_to_dim>(
            morphers, lengths);

    constexpr auto top_dim_aliases = array<alias_t, 2>{kDim0Alias, kDim1Alias};
    constexpr auto partition_dim_aliases = array<alias_t, 1>{"P"};
    constexpr auto element_dim_aliases =
        array<alias_t, 2>{kDim0ElemAlias, kDim1ElemAlias};

    return distributed_tensor_descriptor<
        poly_result,
        top_dim_aliases,
        partition_dim_aliases,
        element_dim_aliases>{};
}

} // namespace detail

#endif // UNIFIED_TILE_BACKEND

// ============================================================================
// Public API: generic 2D distribution
// ============================================================================

/// @brief Create a block-level 2D copy distribution.
/// Distributes a (Dim0 x Dim1) tile across BlockSize threads.
/// Dim0 is the outer (non-contiguous) dimension.
/// Dim1 is the inner (contiguous) dimension, vectorized by VecSize.
///
/// Usage:
///   A RowMajor (M x K):  make_block_copy_2d_distribution<BS, M, K, KPack>()
///   B RowMajor (K x N):  make_block_copy_2d_distribution<BS, K, N, VecN>()
template <int BlockSize, int Dim0, int Dim1, int VecSize>
UNIFIED_TILE_DEVICE constexpr auto make_block_copy_2d_distribution()
{
    using Config = block_copy_2d_config<BlockSize, Dim0, Dim1, VecSize>;

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    return detail::make_distribution_from_config<Config>();
#else
    return detail::make_distribution_from_config<
        Config, "D0", "D1", "D0_0", "D0_1", "D1_0", "D1_1">();
#endif
}

// ============================================================================
// Public API: named A/B distributions
// ============================================================================
// These provide proper dimension aliases for MINT downstream ops
// (load/store freeze dims reference "M", "K", "N" by name).

/// @brief Block copy distribution for A matrix (RowMajor: M x K).
/// Outer=M (non-contiguous), Inner=K (contiguous), VecSize along K.
template <int BlockSize, int MPerBlock, int KPerBlock, int VecSize>
UNIFIED_TILE_DEVICE constexpr auto make_block_copy_a_distribution()
{
    using Config = block_copy_2d_config<BlockSize, MPerBlock, KPerBlock, VecSize>;

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    return detail::make_distribution_from_config<Config>();
#else
    return detail::make_distribution_from_config<
        Config, "M", "K", "M_0", "M_1", "K_0", "K_1">();
#endif
}

/// @brief Block copy distribution for B matrix (RowMajor: K x N).
/// Outer=K (non-contiguous), Inner=N (contiguous), VecSize along N.
template <int BlockSize, int KPerBlock, int NPerBlock, int VecSize>
UNIFIED_TILE_DEVICE constexpr auto make_block_copy_b_distribution()
{
    using Config = block_copy_2d_config<BlockSize, KPerBlock, NPerBlock, VecSize>;

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    return detail::make_distribution_from_config<Config>();
#else
    return detail::make_distribution_from_config<
        Config, "K", "N", "K_0", "K_1", "N_0", "N_1">();
#endif
}

// ============================================================================
// Query functions (backend-agnostic via #ifdef)
// ============================================================================

/// @brief Get per-thread element count from a distribution.
/// Both backends guarantee: elements_per_thread == Dim0 * Dim1 / BlockSize.
template <typename Distribution>
UNIFIED_TILE_DEVICE constexpr auto get_elements_per_thread(
    const Distribution& dstr)
{
#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    return dstr.get_ys_to_d_descriptor().get_element_space_size();
#else
    (void)dstr;
    return Distribution::element_size();
#endif
}

/// @brief Get number of tile (output) dimensions from a distribution.
template <typename Distribution>
UNIFIED_TILE_DEVICE constexpr int get_num_tile_dims(const Distribution&)
{
#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    return Distribution::get_num_of_dimension_x();
#else
    return Distribution::top_ndim();
#endif
}

/// @brief Get per-thread element count directly from config (no distribution).
template <typename Config>
constexpr int get_elements_per_thread_from_config()
{
    return Config::kElemsPerThread;
}

} // namespace distribution
} // namespace unified_tile
