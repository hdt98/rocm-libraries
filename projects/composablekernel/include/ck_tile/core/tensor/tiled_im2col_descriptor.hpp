// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/tensor/im2col_coordinate.hpp"
#include "ck_tile/core/tensor/tensor_descriptor_tiled.hpp"

namespace ck_tile {

// ============================================================================
// TiledIm2ColDescriptor
// ============================================================================
// Wraps an existing tensor_descriptor_tiled and attaches Im2ColMetadata.
//
// The presence of get_im2col_meta() is detected at compile time via the
// has_im2col_meta_v<T> trait in im2col_coordinate.hpp.  When this trait
// is true, make_tensor_coordinate and move_tensor_coordinate use the fast
// tiled path instead of the generic transform-chain path.
//
// The im2col_tensor_kind static member signals which Im2ColCoordinate
// specialization to use (FwdInput or FwdOutput).  It is read by
// im2col_tensor_kind_of_v<Desc> in tensor_coordinate.hpp.
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

// WaveUniformM = true signals that all 64 threads in a warp share the same
// m_gemm value (guaranteed when KPerBlock >= warp_size * VecSizeA).
// prepare_coords uses this flag to apply amd_wave_read_first_lane() on m_gemm
// before calling init_m(), promoting the 3 M divmods to the scalar unit (SALU).
template <typename BaseDesc,
          bool WaveUniformM          = false,
          Im2ColTensor TensorKind    = Im2ColTensor::FwdInput>
struct TiledIm2ColDescriptor : public BaseDesc
{
    using Base = BaseDesc;

    static constexpr bool         wave_uniform_m    = WaveUniformM;
    static constexpr Im2ColTensor im2col_tensor_kind = TensorKind;

    CK_TILE_HOST_DEVICE TiledIm2ColDescriptor() = default;

    CK_TILE_HOST_DEVICE TiledIm2ColDescriptor(const BaseDesc& base, const Im2ColMetadata& meta)
        : Base(base), meta_(meta)
    {
    }

    CK_TILE_HOST_DEVICE const Im2ColMetadata& get_im2col_meta() const { return meta_; }

    // Inherit is_tiled() from tensor_descriptor_tiled via BaseDesc
    using Base::is_tiled;

    private:
    Im2ColMetadata meta_;
};

// Factory function: wrap a tensor_descriptor_tiled with im2col metadata.
// WaveUniformM should be true when KPerBlock >= warp_size * VecSizeA.
// TensorKind selects the Im2ColCoordinate specialization (default: FwdInput).
template <bool WaveUniformM       = false,
          Im2ColTensor TensorKind = Im2ColTensor::FwdInput,
          typename BaseDesc>
CK_TILE_HOST_DEVICE constexpr auto
make_tiled_im2col_descriptor(const BaseDesc& base_desc, const Im2ColMetadata& meta)
{
    return TiledIm2ColDescriptor<BaseDesc, WaveUniformM, TensorKind>{base_desc, meta};
}

} // namespace ck_tile
