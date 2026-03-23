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
// Usage:
//   auto base_desc = transform_tensor_descriptor_tiled(...);   // existing call
//   auto meta      = transformer.MakeATileMetadata<NHWGC>();    // new call
//   auto tiled_desc = make_tiled_im2col_descriptor(base_desc, meta);
//
// The resulting descriptor is drop-in compatible with tensor_descriptor_tiled:
// it exposes the same interface (get_length, calculate_offset, etc.) via
// inheritance, plus the additional get_im2col_meta() for fast-path dispatch.

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
