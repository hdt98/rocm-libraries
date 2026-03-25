// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/tensor/tiled_im2col_coordinate.hpp"
#include "ck_tile/core/tensor/tensor_descriptor_tiled.hpp"

namespace ck_tile {

// ============================================================================
// TiledIm2ColDescriptor
// ============================================================================
// Wraps an existing tensor_descriptor_tiled and attaches TiledIm2ColMetadata.
//
// The presence of get_im2col_meta() is detected at compile time via the
// has_im2col_meta_v<T> trait in tiled_im2col_coordinate.hpp.  When this trait
// is true, make_tensor_coordinate and move_tensor_coordinate use the fast
// tiled path instead of the generic transform-chain path.
//
// Note (Issue 1 investigation): a fully standalone descriptor (without BaseDesc
// inheritance) was explored but found to regress performance.  The root cause
// is that BaseDesc carries GuaranteedVectorLengths propagated through the full
// transform chain (NHWGC → embed → merge → M×K).  Reproducing those values
// correctly in a lightweight stub (without running the same propagation logic)
// resulted in different get_top_dimension_safe_vector_length_strides() output
// which changed ScalarPerVector and degraded the kernel.  We retain the
// BaseDesc inheritance for correctness; the metadata payload is the only
// runtime addition to the kernel args struct.

template <typename BaseDesc>
struct TiledIm2ColDescriptor : public BaseDesc
{
    using Base = BaseDesc;

    CK_TILE_HOST_DEVICE TiledIm2ColDescriptor() = default;

    CK_TILE_HOST_DEVICE TiledIm2ColDescriptor(const BaseDesc& base,
                                               const TiledIm2ColMetadata& meta)
        : Base(base), meta_(meta)
    {
    }

    CK_TILE_HOST_DEVICE const TiledIm2ColMetadata& get_im2col_meta() const { return meta_; }

    // Inherit is_tiled() from tensor_descriptor_tiled via BaseDesc
    using Base::is_tiled;

    private:
    TiledIm2ColMetadata meta_;
};

// Factory function: wrap a tensor_descriptor_tiled with im2col metadata.
template <typename BaseDesc>
CK_TILE_HOST_DEVICE constexpr auto
make_tiled_im2col_descriptor(const BaseDesc& base_desc, const TiledIm2ColMetadata& meta)
{
    return TiledIm2ColDescriptor<BaseDesc>{base_desc, meta};
}

} // namespace ck_tile
