// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_smfmac_impl.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_traits.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for Sparse MFMA (SMFMA) on GFX942, GFX950 targets
 *
 * This specialization implements the SMFMA instruction for fp16_t A and B
 * matrices with structured sparsity, fp32_t accumulator, with 16x16x32 fragment sizes.
 *
 * @tparam CtrlFlags Control flags for the Sparse MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsSparseMfmaI CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, std::enable_if_t<is_any_value_of(CompilerTarget::TARGET_ID, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950)>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        static constexpr index_t ABVecN            = vector_traits<AVecType>::vector_size;
        static constexpr index_t kCompressionRatio = 2;
        static constexpr index_t CompressedSize    = ABVecN / kCompressionRatio;
        using AVecCompressed                       = ext_vector_t<fp16_t, CompressedSize>;

        static_assert(CompressedSize == 4);
        // TODO: Compressing A on-the-fly should be OK for now, but we need to validate
        // and evaluate changing this to a transform at a higher level.
        // aVec not being const can cause problems when running multiple intrinsics.
        const uint32_t idx = ck_tile::compress_a_impl<fp16_t, CompressedSize>(aVec);

        const AVecCompressed a_vec_pruned = {aVec[0], aVec[1], aVec[2], aVec[3]};

        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x32_f16(
            a_vec_pruned, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

// __builtin_amdgcn_smfmac_f32_32x32x16_f16
// signature: V16fV4hV8hV16fiIiIi
// flags: CBSZ, ABID
// layouts: A=KP{4} L{KP1M} V{KP0} B=K{8} L{K1N} V{K0} C/D=M{2, 4} L{M1N} V{M2M0} Idx=KP{4} L{KP1M}
// V{SKP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 32u, 32u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        static constexpr index_t ABVecN            = vector_traits<AVecType>::vector_size;
        static constexpr index_t kCompressionRatio = 2;
        static constexpr index_t CompressedSize    = ABVecN / kCompressionRatio;
        using AVecCompressed                       = ext_vector_t<ADataType, CompressedSize>;

        static_assert(CompressedSize == 4);
        const uint32_t idx = ck_tile::compress_a_impl<ADataType, CompressedSize>(aVec);

        const AVecCompressed a_vec_pruned = {aVec[0], aVec[1], aVec[2], aVec[3]};

        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_32x32x16_f16(
            a_vec_pruned, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

// __builtin_amdgcn_smfmac_f32_16x16x32_bf16
// signature: V4fV4sV8sV4fiIiIi
// flags: CBSZ, ABID
// layouts: A=KP{4} L{KP1M} V{KP0} B=K{8} L{K1N} V{K0} C/D=M{4} L{M1N} V{BM0} Idx=KP{4} L{KP1M}
// V{SKP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        static constexpr index_t ABVecN            = vector_traits<AVecType>::vector_size;
        static constexpr index_t kCompressionRatio = 2;
        static constexpr index_t CompressedSize    = ABVecN / kCompressionRatio;
        using AVecCompressed                       = ext_vector_t<ADataType, CompressedSize>;

        static_assert(CompressedSize == 4);
        const uint32_t idx = ck_tile::compress_a_impl<ADataType, CompressedSize>(aVec);

        const AVecCompressed a_vec_pruned = {aVec[0], aVec[1], aVec[2], aVec[3]};

        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x32_bf16(
            a_vec_pruned, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

// __builtin_amdgcn_smfmac_f32_32x32x16_bf16
// signature: V16fV4sV8sV16fiIiIi
// flags: CBSZ, ABID
// layouts: A=KP{4} L{KP1M} V{KP0} B=K{8} L{K1N} V{K0} C/D=M{2, 4} L{M1N} V{M2M0} Idx=KP{4} L{KP1M}
// V{SKP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 32u, 32u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        static constexpr index_t ABVecN            = vector_traits<AVecType>::vector_size;
        static constexpr index_t kCompressionRatio = 2;
        static constexpr index_t CompressedSize    = ABVecN / kCompressionRatio;
        using AVecCompressed                       = ext_vector_t<ADataType, CompressedSize>;

        static_assert(CompressedSize == 4);
        const uint32_t idx = ck_tile::compress_a_impl<ADataType, CompressedSize>(aVec);

        const AVecCompressed a_vec_pruned = {aVec[0], aVec[1], aVec[2], aVec[3]};

        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_32x32x16_bf16(
            a_vec_pruned, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

// __builtin_amdgcn_smfmac_i32_16x16x64_i8
// signature: V4iV2iV4iV4iiIiIi
// flags: CBSZ, ABID
// layouts: A=KP{8} L{KP1M} V{KP0} B=K{16} L{K1N} V{K0} C/D=M{4} L{M1N} V{BM0} Idx=KP{8} L{KP1M}
// V{SKP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
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

        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_i32_16x16x64_i8(
            bit_cast<ext_vector_t<int32_t, 2>>(a_vec_pruned),
            bit_cast<ext_vector_t<int32_t, 4>>(bVec),
            cVec,
            idx,
            PARAMS.UseFirstIndex,
            PARAMS.ByteIndexToOverride)};
    }
};

// __builtin_amdgcn_smfmac_i32_32x32x32_i8
// signature: V16iV2iV4iV16iiIiIi
// flags: CBSZ, ABID
// layouts: A=KP{8} L{KP1M} V{KP0} B=K{16} L{K1N} V{K0} C/D=M{2, 4} L{M1N} V{M2M0} Idx=KP{8} L{KP1M}
// V{SKP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 32u, 32u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
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

        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_i32_32x32x32_i8(
            bit_cast<ext_vector_t<int32_t, 2>>(a_vec_pruned),
            bit_cast<ext_vector_t<int32_t, 4>>(bVec),
            cVec,
            idx,
            PARAMS.UseFirstIndex,
            PARAMS.ByteIndexToOverride)};
    }
};

// __builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8
// signature: V4fV2iV4iV4fiIiIi
// flags: CBSZ, ABID
// layouts: A=KP{8} L{KP1M} V{KP0} B=K{16} L{K1N} V{K0} C/D=M{4} L{M1N} V{BM0} Idx=KP{8} L{KP1M}
// V{SKP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
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

        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8(
            bit_cast<ext_vector_t<int32_t, 2>>(a_vec_pruned),
            bit_cast<ext_vector_t<int32_t, 4>>(bVec),
            cVec,
            idx,
            PARAMS.UseFirstIndex,
            PARAMS.ByteIndexToOverride)};
    }
};

// __builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8
// signature: V4fV2iV4iV4fiIiIi
// flags: CBSZ, ABID
// layouts: A=KP{8} L{KP1M} V{KP0} B=K{16} L{K1N} V{K0} C/D=M{4} L{M1N} V{BM0} Idx=KP{8} L{KP1M}
// V{SKP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
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

        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8(
            bit_cast<ext_vector_t<int32_t, 2>>(a_vec_pruned),
            bit_cast<ext_vector_t<int32_t, 4>>(bVec),
            cVec,
            idx,
            PARAMS.UseFirstIndex,
            PARAMS.ByteIndexToOverride)};
    }
};

// __builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8
// signature: V4fV2iV4iV4fiIiIi
// flags: CBSZ, ABID
// layouts: A=KP{8} L{KP1M} V{KP0} B=K{16} L{K1N} V{K0} C/D=M{4} L{M1N} V{BM0} Idx=KP{8} L{KP1M}
// V{SKP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
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

        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8(
            bit_cast<ext_vector_t<int32_t, 2>>(a_vec_pruned),
            bit_cast<ext_vector_t<int32_t, 4>>(bVec),
            cVec,
            idx,
            PARAMS.UseFirstIndex,
            PARAMS.ByteIndexToOverride)};
    }
};

// __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8
// signature: V4fV2iV4iV4fiIiIi
// flags: CBSZ, ABID
// layouts: A=KP{8} L{KP1M} V{KP0} B=K{16} L{K1N} V{K0} C/D=M{4} L{M1N} V{BM0} Idx=KP{8} L{KP1M}
// V{SKP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
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

        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(
            bit_cast<ext_vector_t<int32_t, 2>>(a_vec_pruned),
            bit_cast<ext_vector_t<int32_t, 4>>(bVec),
            cVec,
            idx,
            PARAMS.UseFirstIndex,
            PARAMS.ByteIndexToOverride)};
    }
};

// __builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8
// signature: V16fV2iV4iV16fiIiIi
// flags: CBSZ, ABID
// layouts: A=KP{8} L{KP1M} V{KP0} B=K{16} L{K1N} V{K0} C/D=M{2, 4} L{M1N} V{M2M0} Idx=KP{8} L{KP1M}
// V{SKP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 32u, 32u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
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

        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8(
            bit_cast<ext_vector_t<int32_t, 2>>(a_vec_pruned),
            bit_cast<ext_vector_t<int32_t, 4>>(bVec),
            cVec,
            idx,
            PARAMS.UseFirstIndex,
            PARAMS.ByteIndexToOverride)};
    }
};

// __builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8
// signature: V16fV2iV4iV16fiIiIi
// flags: CBSZ, ABID
// layouts: A=KP{8} L{KP1M} V{KP0} B=K{16} L{K1N} V{K0} C/D=M{2, 4} L{M1N} V{M2M0} Idx=KP{8} L{KP1M}
// V{SKP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 32u, 32u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
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

        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8(
            bit_cast<ext_vector_t<int32_t, 2>>(a_vec_pruned),
            bit_cast<ext_vector_t<int32_t, 4>>(bVec),
            cVec,
            idx,
            PARAMS.UseFirstIndex,
            PARAMS.ByteIndexToOverride)};
    }
};

// __builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8
// signature: V16fV2iV4iV16fiIiIi
// flags: CBSZ, ABID
// layouts: A=KP{8} L{KP1M} V{KP0} B=K{16} L{K1N} V{K0} C/D=M{2, 4} L{M1N} V{M2M0} Idx=KP{8} L{KP1M}
// V{SKP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 32u, 32u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
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

        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8(
            bit_cast<ext_vector_t<int32_t, 2>>(a_vec_pruned),
            bit_cast<ext_vector_t<int32_t, 4>>(bVec),
            cVec,
            idx,
            PARAMS.UseFirstIndex,
            PARAMS.ByteIndexToOverride)};
    }
};

// __builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8
// signature: V16fV2iV4iV16fiIiIi
// flags: CBSZ, ABID
// layouts: A=KP{8} L{KP1M} V{KP0} B=K{16} L{K1N} V{K0} C/D=M{2, 4} L{M1N} V{M2M0} Idx=KP{8} L{KP1M}
// V{SKP0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 32u, 32u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
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

        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8(
            bit_cast<ext_vector_t<int32_t, 2>>(a_vec_pruned),
            bit_cast<ext_vector_t<int32_t, 4>>(bVec),
            cVec,
            idx,
            PARAMS.UseFirstIndex,
            PARAMS.ByteIndexToOverride)};
    }
};

} // namespace ck_tile::core::arch::mma
