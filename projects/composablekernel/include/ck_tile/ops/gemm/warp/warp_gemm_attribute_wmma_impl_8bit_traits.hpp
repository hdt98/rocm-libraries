// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "warp_gemm_attribute_wmma_impl_base_traits.hpp"
#include "warp_gemm_params.hpp"
namespace ck_tile {
// int8 specialization - GFX11
template <>
struct WmmaTraits<gfx11_t, int8_t, int8_t, int32_t, 16, 16, 16>
    : WmmaTraitsBase<gfx11_t, int8_t, int8_t, int32_t, 16>
{
    using ArchType = gfx11_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx11__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, // neg_a
                                                          bit_cast<int32x4_t>(a_vec),
                                                          true, // neg_b
                                                          bit_cast<int32x4_t>(b_vec),
                                                          bit_cast<int32x8_t>(c_vec),
                                                          P::clamp);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

// int8 specialization - GFX12
template <>
struct WmmaTraits<gfx120_t, int8_t, int8_t, int32_t, 16, 16, 16>
    : WmmaTraitsBase<gfx12_t, int8_t, int8_t, int32_t, 16>
{
    using ArchType = gfx120_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx120__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, // neg_a
                                                                bit_cast<int32x2_t>(a_vec),
                                                                true, // neg_b
                                                                bit_cast<int32x2_t>(b_vec),
                                                                bit_cast<int32x8_t>(c_vec),
                                                                P::clamp);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

// fp8/bf8 specialization - GFX12
template <>
struct WmmaTraits<gfx120_t, fp8_t, fp8_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx12_t, fp8_t, fp8_t, float, 16>
{
    using ArchType = gfx120_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx120__
        return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
            bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), bit_cast<fp32x8_t>(c_vec));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0.f};
#endif
    }
};

template <>
struct WmmaTraits<gfx120_t, bf8_t, bf8_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx12_t, bf8_t, bf8_t, float, 16>
{
    using ArchType = gfx120_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx120__
        return __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(
            bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), bit_cast<fp32x8_t>(c_vec));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx120_t, fp8_t, bf8_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx12_t, fp8_t, bf8_t, float, 16>
{
    using ArchType = gfx120_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx120__
        return __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(
            bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), bit_cast<fp32x8_t>(c_vec));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx120_t, bf8_t, fp8_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx12_t, bf8_t, fp8_t, float, 16>
{
    using ArchType = gfx120_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx120__
        return __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(
            bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), bit_cast<fp32x8_t>(c_vec));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0.f};
#endif
    }
};

// iu8 specialization - GFX125
template <>
struct WmmaTraits<gfx125_t, int8_t, int8_t, int32_t, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, int8_t, int8_t, int32_t, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_i32_16x16x64_iu8(true, // neg_a
                                                      bit_cast<int32x8_t>(a_vec),
                                                      true, // neg_b
                                                      bit_cast<int32x8_t>(b_vec),
                                                      bit_cast<int32x8_t>(c_vec),
                                                      P::reuse_a,
                                                      P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, uint8_t, uint8_t, uint32_t, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, uint8_t, uint8_t, uint32_t, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_i32_16x16x64_iu8(false, // neg_a
                                                      bit_cast<int32x8_t>(a_vec),
                                                      false, // neg_b
                                                      bit_cast<int32x8_t>(b_vec),
                                                      bit_cast<int32x8_t>(c_vec),
                                                      P::reuse_a,
                                                      P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

// fp8/bf8 specialization - GFX125
template <>
struct WmmaTraits<gfx125_t, fp8_t, fp8_t, float, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, fp8_t, fp8_t, float, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(bit_cast<int32x8_t>(a_vec),
                                                          bit_cast<int32x8_t>(b_vec),
                                                          0,
                                                          bit_cast<fp32x8_t>(c_vec),
                                                          P::reuse_a,
                                                          P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, bf8_t, bf8_t, float, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, bf8_t, bf8_t, float, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8(bit_cast<int32x8_t>(a_vec),
                                                          bit_cast<int32x8_t>(b_vec),
                                                          0,
                                                          bit_cast<fp32x8_t>(c_vec),
                                                          P::reuse_a,
                                                          P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, fp8_t, bf8_t, float, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, fp8_t, bf8_t, float, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8(bit_cast<int32x8_t>(a_vec),
                                                          bit_cast<int32x8_t>(b_vec),
                                                          0,
                                                          bit_cast<fp32x8_t>(c_vec),
                                                          P::reuse_a,
                                                          P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, bf8_t, fp8_t, float, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, bf8_t, fp8_t, float, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8(bit_cast<int32x8_t>(a_vec),
                                                          bit_cast<int32x8_t>(b_vec),
                                                          0,
                                                          bit_cast<fp32x8_t>(c_vec),
                                                          P::reuse_a, // matrix_a_reuse
                                                          P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, fp8_t, fp8_t, fp16_t, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, fp8_t, fp8_t, fp16_t, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8(bit_cast<int32x8_t>(a_vec),
                                                          bit_cast<int32x8_t>(b_vec),
                                                          0,
                                                          bit_cast<fp16x8_t>(c_vec),
                                                          P::reuse_a,
                                                          P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, fp8_t, bf8_t, fp16_t, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, fp8_t, bf8_t, fp16_t, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8(bit_cast<int32x8_t>(a_vec),
                                                          bit_cast<int32x8_t>(b_vec),
                                                          0,
                                                          bit_cast<fp16x8_t>(c_vec),
                                                          P::reuse_a,
                                                          P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};
template <>
struct WmmaTraits<gfx125_t, bf8_t, fp8_t, fp16_t, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, bf8_t, fp8_t, fp16_t, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8(bit_cast<int32x8_t>(a_vec),
                                                          bit_cast<int32x8_t>(b_vec),
                                                          0,
                                                          bit_cast<fp16x8_t>(c_vec),
                                                          P::reuse_a,
                                                          P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, bf8_t, bf8_t, fp16_t, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, bf8_t, bf8_t, fp16_t, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8(bit_cast<int32x8_t>(a_vec),
                                                          bit_cast<int32x8_t>(b_vec),
                                                          0,
                                                          bit_cast<fp16x8_t>(c_vec),
                                                          P::reuse_a,
                                                          P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, fp8_t, fp8_t, float, 16, 16, 128>
    : WmmaTraitsBase<gfx12_t, fp8_t, fp8_t, float, 128, false>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(bit_cast<int32x16_t>(a_vec),
                                                           bit_cast<int32x16_t>(b_vec),
                                                           0,
                                                           bit_cast<fp32x8_t>(c_vec),
                                                           P::reuse_a,  // matrix_a_reuse
                                                           P::reuse_b); // matrix_b_reuse
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, bf8_t, bf8_t, float, 16, 16, 128>
    : WmmaTraitsBase<gfx12_t, bf8_t, bf8_t, float, 128, false>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8(bit_cast<int32x16_t>(a_vec),
                                                           bit_cast<int32x16_t>(b_vec),
                                                           0,
                                                           bit_cast<fp32x8_t>(c_vec),
                                                           P::reuse_a,  // matrix_a_reuse
                                                           P::reuse_b); // matrix_b_reuse
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, fp8_t, bf8_t, float, 16, 16, 128>
    : WmmaTraitsBase<gfx12_t, fp8_t, bf8_t, float, 128, false>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8(bit_cast<int32x16_t>(a_vec),
                                                           bit_cast<int32x16_t>(b_vec),
                                                           0,
                                                           bit_cast<fp32x8_t>(c_vec),
                                                           P::reuse_a,  // matrix_a_reuse
                                                           P::reuse_b); // matrix_b_reuse
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, bf8_t, fp8_t, float, 16, 16, 128>
    : WmmaTraitsBase<gfx12_t, bf8_t, fp8_t, float, 128, false>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8(bit_cast<int32x16_t>(a_vec),
                                                           bit_cast<int32x16_t>(b_vec),
                                                           0,
                                                           bit_cast<fp32x8_t>(c_vec),
                                                           P::reuse_a,  // matrix_a_reuse
                                                           P::reuse_b); // matrix_b_reuse
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

// f8f6f4 specialization - GFX125
enum F8F6F4OpSelEnum
{
    E4M3, // 0x0
    E5M2, // 0x1
    E2M3, // 0x2
    E3M2, // 0x3
    E2M1, // 0x4
};

template <>
struct WmmaTraits<gfx125_t, fp8_t, pk_fp4_t, float, 16, 16, 128>
    : WmmaTraitsBase<gfx12_t, fp8_t, pk_fp4_t, float, 128, true>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        int32x8_t arg_b = bit_cast<int32x8_t>(b_vec);
        return __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(F8F6F4OpSelEnum::E4M3,
                                                          bit_cast<int32x16_t>(a_vec),
                                                          F8F6F4OpSelEnum::E2M1,
                                                          int32x16_t{arg_b[0],
                                                                     arg_b[1],
                                                                     arg_b[2],
                                                                     arg_b[3],
                                                                     arg_b[4],
                                                                     arg_b[5],
                                                                     arg_b[6],
                                                                     arg_b[7],
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0},
                                                          0,
                                                          bit_cast<fp32x8_t>(c_vec));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, pk_fp4_t, fp8_t, float, 16, 16, 128>
    : WmmaTraitsBase<gfx12_t, pk_fp4_t, fp8_t, float, 128, true>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        int32x8_t arg_a = bit_cast<int32x8_t>(a_vec);
        return __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(F8F6F4OpSelEnum::E2M1,
                                                          int32x16_t{arg_a[0],
                                                                     arg_a[1],
                                                                     arg_a[2],
                                                                     arg_a[3],
                                                                     arg_a[4],
                                                                     arg_a[5],
                                                                     arg_a[6],
                                                                     arg_a[7],
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0},
                                                          F8F6F4OpSelEnum::E4M3,
                                                          bit_cast<int32x16_t>(b_vec),
                                                          0,
                                                          bit_cast<fp32x8_t>(c_vec));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, pk_fp4_t, pk_fp4_t, float, 16, 16, 128>
    : WmmaTraitsBase<gfx12_t, pk_fp4_t, pk_fp4_t, float, 128, true>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        int32x8_t arg_a = bit_cast<int32x8_t>(a_vec);
        int32x8_t arg_b = bit_cast<int32x8_t>(b_vec);
        return __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(F8F6F4OpSelEnum::E2M1,
                                                          int32x16_t{arg_a[0],
                                                                     arg_a[1],
                                                                     arg_a[2],
                                                                     arg_a[3],
                                                                     arg_a[4],
                                                                     arg_a[5],
                                                                     arg_a[6],
                                                                     arg_a[7],
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0},
                                                          F8F6F4OpSelEnum::E2M1,
                                                          int32x16_t{arg_b[0],
                                                                     arg_b[1],
                                                                     arg_b[2],
                                                                     arg_b[3],
                                                                     arg_b[4],
                                                                     arg_b[5],
                                                                     arg_b[6],
                                                                     arg_b[7],
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0},
                                                          0,
                                                          bit_cast<fp32x8_t>(c_vec));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

} // namespace ck_tile
