// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_smfmac_impl.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_traits.hpp"

namespace ck_tile::core::arch::mma {

// TODO: c++20 template <CtrlFlagsSparseWmmaI CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        static constexpr index_t ABVecN            = vector_traits<AVecType>::vector_size;
        static constexpr index_t kCompressionRatio = 2;
        static constexpr index_t CompressedSize    = ABVecN / kCompressionRatio;
        using AVecCompressed                       = ext_vector_t<fp16_t, CompressedSize>;

        static_assert(CompressedSize == 8);
        // TODO: Compressing A on-the-fly should be OK for now, but we need to validate
        // and evaluate changing this to a transform at a higher level.
        // aVec not being const can cause problems when running multiple intrinsics.
        const uint32_t idx = ck_tile::compress_a_impl<fp16_t, CompressedSize>(aVec);

        const AVecCompressed a_vec_pruned = {
            aVec[0], aVec[1], aVec[2], aVec[3], aVec[4], aVec[5], aVec[6], aVec[7]};

        return {__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a_vec_pruned, bVec, cVec, idx)};
    }
};

// __builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32
// signature: V8fV8sV16sV8fi
// flags: n/a
// layouts: A=KP{2, 4} L{KP1M} V{KP2KP0} B=K{2, 8} L{K1N} V{K2K0} C/D=M{8} L{M1N} V{M0} Idx=KP{2, 4}
// L{KP1M} V{SKP2KP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        static constexpr index_t ABVecN            = vector_traits<AVecType>::vector_size;
        static constexpr index_t kCompressionRatio = 2;
        static constexpr index_t CompressedSize    = ABVecN / kCompressionRatio;
        using AVecCompressed                       = ext_vector_t<ADataType, CompressedSize>;

        static_assert(CompressedSize == 8);
        const uint32_t idx = ck_tile::compress_a_impl<ADataType, CompressedSize>(aVec);

        const AVecCompressed a_vec_pruned = {
            aVec[0], aVec[1], aVec[2], aVec[3], aVec[4], aVec[5], aVec[6], aVec[7]};
        return {__builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32(a_vec_pruned, bVec, cVec, idx)};
    }
};

// __builtin_amdgcn_swmmac_f16_16x16x32_f16_w32
// signature: V8hV8hV16hV8hi
// flags: n/a
// layouts: A=KP{2, 4} L{KP1M} V{KP2KP0} B=K{2, 8} L{K1N} V{K2K0} C/D=M{8} L{M1N} V{M0} Idx=KP{2, 4}
// L{KP1M} V{SKP2KP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp16_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp16_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        static constexpr index_t ABVecN            = vector_traits<AVecType>::vector_size;
        static constexpr index_t kCompressionRatio = 2;
        static constexpr index_t CompressedSize    = ABVecN / kCompressionRatio;
        using AVecCompressed                       = ext_vector_t<ADataType, CompressedSize>;

        static_assert(CompressedSize == 8);
        const uint32_t idx = ck_tile::compress_a_impl<ADataType, CompressedSize>(aVec);

        const AVecCompressed a_vec_pruned = {
            aVec[0], aVec[1], aVec[2], aVec[3], aVec[4], aVec[5], aVec[6], aVec[7]};
        return {__builtin_amdgcn_swmmac_f16_16x16x32_f16_w32(a_vec_pruned, bVec, cVec, idx)};
    }
};

// __builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w32
// signature: V8sV8sV16sV8si
// flags: n/a
// layouts: A=KP{2, 4} L{KP1M} V{KP2KP0} B=K{2, 8} L{K1N} V{K2K0} C/D=M{8} L{M1N} V{M0} Idx=KP{2, 4}
// L{KP1M} V{SKP2KP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, bf16_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, bf16_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        static constexpr index_t ABVecN            = vector_traits<AVecType>::vector_size;
        static constexpr index_t kCompressionRatio = 2;
        static constexpr index_t CompressedSize    = ABVecN / kCompressionRatio;
        using AVecCompressed                       = ext_vector_t<ADataType, CompressedSize>;

        static_assert(CompressedSize == 8);
        const uint32_t idx = ck_tile::compress_a_impl<ADataType, CompressedSize>(aVec);

        const AVecCompressed a_vec_pruned = {
            aVec[0], aVec[1], aVec[2], aVec[3], aVec[4], aVec[5], aVec[6], aVec[7]};
        return {__builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w32(a_vec_pruned, bVec, cVec, idx)};
    }
};

// __builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32
// signature: V8iIbV2iIbV4iV8iiIb
// flags: sign, sign, clamp
// layouts: A=KP{8} L{KP1M} V{KP0} B=K{16} L{K1N} V{K0} C/D=M{8} L{M1N} V{M0} Idx=KP{8} L{KP1M}
// V{SKP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<uint8_t, uint8_t, int32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
: amdgcn_mma_base<uint8_t, uint8_t, int32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        static constexpr index_t ABVecN            = vector_traits<AVecType>::vector_size;
        static constexpr index_t kCompressionRatio = 2;
        static constexpr index_t CompressedSize    = ABVecN / kCompressionRatio;
        using AVecCompressed                       = ext_vector_t<ADataType, CompressedSize>;

        static_assert(CompressedSize == 8);
        const uint32_t idx = ck_tile::compress_a_impl<ADataType, CompressedSize>(aVec);

        const AVecCompressed a_vec_pruned = {
            aVec[0], aVec[1], aVec[2], aVec[3], aVec[4], aVec[5], aVec[6], aVec[7]};
        return {__builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32(
            0,
            bit_cast<ext_vector_t<int32_t, 2>>(a_vec_pruned),
            0,
            bit_cast<ext_vector_t<int32_t, 4>>(bVec),
            cVec,
            idx,
            0)};
    }
};

// // __builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32
// // signature: V8iIbiIbV2iV8iiIb
// // flags: sign, sign, clamp
// // layouts: A=KP{8} L{KP1M} V{KP0} B=K{16} L{K1N} V{K0} C/D=M{8} L{M1N} V{M0} Idx=KP{8} L{KP1M}
// V{SKP0} template <typename CtrlFlags, typename CompilerTarget>
// // clang-format off
// //               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
// struct amdgcn_mma<pk_int4_t, pk_int4_t, int32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget,
// MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>> :
// amdgcn_mma_base<pk_int4_t, pk_int4_t, int32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp,
// MmaOpFamily::SPARSE>
// // clang-format on
// {
//     CK_TILE_DEVICE static auto
//     exec(AVecType& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
//     {
//         static constexpr index_t ABVecN            = vector_traits<AVecType>::vector_size;
//         static constexpr index_t kCompressionRatio = 2;
//         static constexpr index_t CompressedSize    = ABVecN / kCompressionRatio;
//         using AVecCompressed                       = ext_vector_t<ADataType, CompressedSize>;

//         static_assert(CompressedSize == 8);
//         const uint32_t idx = ck_tile::compress_a_impl<ADataType, CompressedSize>(aVec);

//         const AVecCompressed a_vec_pruned = {aVec[0], aVec[1], aVec[2], aVec[3], aVec[4],
//         aVec[5], aVec[6], aVec[7]}; return {__builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32(
//             0, bit_cast<int32_t>(a_vec_pruned), 0, bit_cast<ext_vector_t<int32_t, 2>>(bVec),
//             cVec, idx, 0)};
//     }
// };

// // __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32
// // signature: V8iIbV2iIbV4iV8iiIb
// // flags: sign, sign, clamp
// // layouts: A=KP{16} L{KP1M} V{KP0} B=K{32} L{K1N} V{K0} C/D=M{8} L{M1N} V{M0} Idx=KP{16} L{KP1M}
// V{KP0} template <typename CtrlFlags, typename CompilerTarget>
// // clang-format off
// //               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
// struct amdgcn_mma<pk_int4_t, pk_int4_t, int32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget,
// MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>> :
// amdgcn_mma_base<pk_int4_t, pk_int4_t, int32_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp,
// MmaOpFamily::SPARSE>
// // clang-format on
// {
//     CK_TILE_DEVICE static auto
//     exec(AVecType& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
//     {
//         static constexpr index_t ABVecN            = vector_traits<AVecType>::vector_size;
//         static constexpr index_t kCompressionRatio = 2;
//         static constexpr index_t CompressedSize    = ABVecN / kCompressionRatio;
//         using AVecCompressed                       = ext_vector_t<ADataType, CompressedSize>;

//         static_assert(CompressedSize == 16);
//         const uint32_t idx = ck_tile::compress_a_impl<ADataType, CompressedSize>(aVec);

//         const AVecCompressed a_vec_pruned = {aVec[0], aVec[1], aVec[2], aVec[3], aVec[4],
//         aVec[5], aVec[6], aVec[7], aVec[8], aVec[9], aVec[10], aVec[11], aVec[12], aVec[13],
//         aVec[14], aVec[15]}; return {__builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(
//             0, bit_cast<ext_vector_t<int32_t, 2>>(a_vec_pruned), 0,
//             bit_cast<ext_vector_t<int32_t, 4>>(bVec), cVec, idx, 0)};
//     }
// };

// __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32
// signature: V8fV2iV4iV8fi
// flags: n/a
// layouts: A=KP{8} L{KP1M} V{KP0} B=K{16} L{K1N} V{K0} C/D=M{8} L{M1N} V{M0} Idx=KP{8} L{KP1M}
// V{SKP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        static constexpr index_t ABVecN            = vector_traits<AVecType>::vector_size;
        static constexpr index_t kCompressionRatio = 2;
        static constexpr index_t CompressedSize    = ABVecN / kCompressionRatio;
        using AVecCompressed                       = ext_vector_t<ADataType, CompressedSize>;

        static_assert(CompressedSize == 8);
        const uint32_t idx = ck_tile::compress_a_impl<ADataType, CompressedSize>(aVec);

        const AVecCompressed a_vec_pruned = {
            aVec[0], aVec[1], aVec[2], aVec[3], aVec[4], aVec[5], aVec[6], aVec[7]};
        return {__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(
            bit_cast<ext_vector_t<int32_t, 2>>(a_vec_pruned),
            bit_cast<ext_vector_t<int32_t, 4>>(bVec),
            cVec,
            idx)};
    }
};

// __builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32
// signature: V8fV2iV4iV8fi
// flags: n/a
// layouts: A=KP{8} L{KP1M} V{KP0} B=K{16} L{K1N} V{K0} C/D=M{8} L{M1N} V{M0} Idx=KP{8} L{KP1M}
// V{SKP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        static constexpr index_t ABVecN            = vector_traits<AVecType>::vector_size;
        static constexpr index_t kCompressionRatio = 2;
        static constexpr index_t CompressedSize    = ABVecN / kCompressionRatio;
        using AVecCompressed                       = ext_vector_t<ADataType, CompressedSize>;

        static_assert(CompressedSize == 8);
        const uint32_t idx = ck_tile::compress_a_impl<ADataType, CompressedSize>(aVec);

        const AVecCompressed a_vec_pruned = {
            aVec[0], aVec[1], aVec[2], aVec[3], aVec[4], aVec[5], aVec[6], aVec[7]};
        return {__builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32(
            bit_cast<ext_vector_t<int32_t, 2>>(a_vec_pruned),
            bit_cast<ext_vector_t<int32_t, 4>>(bVec),
            cVec,
            idx)};
    }
};

// __builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32
// signature: V8fV2iV4iV8fi
// flags: n/a
// layouts: A=KP{8} L{KP1M} V{KP0} B=K{16} L{K1N} V{K0} C/D=M{8} L{M1N} V{M0} Idx=KP{8} L{KP1M}
// V{SKP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        static constexpr index_t ABVecN            = vector_traits<AVecType>::vector_size;
        static constexpr index_t kCompressionRatio = 2;
        static constexpr index_t CompressedSize    = ABVecN / kCompressionRatio;
        using AVecCompressed                       = ext_vector_t<ADataType, CompressedSize>;

        static_assert(CompressedSize == 8);
        const uint32_t idx = ck_tile::compress_a_impl<ADataType, CompressedSize>(aVec);

        const AVecCompressed a_vec_pruned = {
            aVec[0], aVec[1], aVec[2], aVec[3], aVec[4], aVec[5], aVec[6], aVec[7]};
        return {__builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32(
            bit_cast<ext_vector_t<int32_t, 2>>(a_vec_pruned),
            bit_cast<ext_vector_t<int32_t, 4>>(bVec),
            cVec,
            idx)};
    }
};

// __builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32
// signature: V8fV2iV4iV8fi
// flags: n/a
// layouts: A=KP{8} L{KP1M} V{KP0} B=K{16} L{K1N} V{K0} C/D=M{8} L{M1N} V{M0} Idx=KP{8} L{KP1M}
// V{SKP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        static constexpr index_t ABVecN            = vector_traits<AVecType>::vector_size;
        static constexpr index_t kCompressionRatio = 2;
        static constexpr index_t CompressedSize    = ABVecN / kCompressionRatio;
        using AVecCompressed                       = ext_vector_t<ADataType, CompressedSize>;

        static_assert(CompressedSize == 8);
        const uint32_t idx = ck_tile::compress_a_impl<ADataType, CompressedSize>(aVec);

        const AVecCompressed a_vec_pruned = {
            aVec[0], aVec[1], aVec[2], aVec[3], aVec[4], aVec[5], aVec[6], aVec[7]};
        return {__builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32(
            bit_cast<ext_vector_t<int32_t, 2>>(a_vec_pruned),
            bit_cast<ext_vector_t<int32_t, 4>>(bVec),
            cVec,
            idx)};
    }
};

} // namespace ck_tile::core::arch::mma
