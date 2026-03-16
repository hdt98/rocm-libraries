// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

// ═══════════════════════════════════════════════════════════════════════
// BlockImageToIm2winProblem
//
// Compile-time problem description for the im2win transform kernel.
//
// The kernel reads from I[G, N, C, Hi, Wi] (GNCHW, channels-first) and
// writes to I'[G, N, C, Ho, Wi_pad, Y] (packed, Y innermost).
//
// InDataType  — element type of the input tensor I
// OutDataType — element type of the output tensor I' (same or cast)
// BlockShape_ — a TileImageToIm2winShape specialisation
// AlignIn_    — vector alignment of input reads  (elements)
// AlignOut_   — vector alignment of output writes (elements)
// ═══════════════════════════════════════════════════════════════════════
template <typename InDataType_,
          typename OutDataType_,
          typename BlockShape_,
          index_t AlignIn_,
          index_t AlignOut_>
struct BlockImageToIm2winProblem
{
    using InDataType  = remove_cvref_t<InDataType_>;
    using OutDataType = remove_cvref_t<OutDataType_>;
    using BlockShape  = remove_cvref_t<BlockShape_>;

    static constexpr index_t AlignIn  = AlignIn_;
    static constexpr index_t AlignOut = AlignOut_;
};

} // namespace ck_tile
