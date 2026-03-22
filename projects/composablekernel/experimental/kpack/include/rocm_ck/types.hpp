// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Shared type definitions for kpack API.

#pragma once

#include <cstdint>

namespace rocm_ck {

/// Fixed-width integer type for kernel index/size arguments.
/// Matches ck_tile::index_t but avoids pulling in CK Tile headers.
using index_t = std::int32_t;

/// AMD GCN/CDNA wavefront size.
constexpr int warp_size = 64;

} // namespace rocm_ck
