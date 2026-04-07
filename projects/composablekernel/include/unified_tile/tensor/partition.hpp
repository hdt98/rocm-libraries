// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../config.hpp"

#ifdef UNIFIED_TILE_BACKEND_MINT
#include "mint/tile/simt/partition.h"
#endif

namespace unified_tile {

// ============================================================================
// Partition types: identify the current thread within its scope
// ============================================================================
// CK_TILE handles partitioning internally (lane_id/warp_id from distribution).
// MINT requires explicit partition objects passed to make_distributed_window.
//
// These types are used internally by make_tile_window; users rarely need them.
// ============================================================================

#ifdef UNIFIED_TILE_BACKEND_MINT

/// @brief Block-scope partition: thread_id = threadIdx.x
template <int BlockSize>
using block_partition = mint::thread_in_this_block<static_cast<mint::index_t>(BlockSize)>;

/// @brief Warp-scope partition: thread_id = threadIdx.x % warp_size
using warp_partition = mint::thread_in_this_warp;

#endif

} // namespace unified_tile
