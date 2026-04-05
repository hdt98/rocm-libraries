// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: types — index_t, wavefront_size. No runtime, no CK deps.
//
// Shared type definitions for rocm_ck.

#pragma once

#include <cstdint>

namespace rocm_ck {

/// Fixed-width integer type for kernel index/size arguments.
/// Matches ck_tile::index_t but avoids pulling in CK Tile headers.
using index_t = std::int32_t;

/// 64-bit index type for large strides (batch_stride * nhead can exceed int32).
/// Matches ck_tile::long_index_t.
using long_index_t = std::int64_t;

/// CDNA wavefront size (64 work-items per wavefront).
constexpr int wavefront_size = 64;

/// GPU ISA target for architecture-specific validation.
///
/// Uses ISA target identifiers (matching -mcpu flags), not marketing names.
/// Any = portable across all targets (intersection of valid configurations).
enum class GpuTarget
{
    Any,     // Portable — only configs valid on all targets
    gfx90a,  // CDNA 2
    gfx942,  // CDNA 3
    gfx950,  // CDNA 4
    gfx1151, // RDNA 3.5
};

/// Wavefront size for a given GPU target.
consteval int targetWavefrontSize(GpuTarget target)
{
    switch(target)
    {
    case GpuTarget::gfx1151: return 32;
    default: return 64;
    }
}

} // namespace rocm_ck
