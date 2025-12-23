// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
namespace ck_tile {
template <typename DataType, index_t K, bool MixPrec>
struct LayoutFromDataType;

template <typename DataType, index_t K>
struct LayoutFromDataType<DataType, K, false>
{
    static constexpr index_t kKLane     = 2;
    static constexpr index_t kK1PerLane = 8;
    static constexpr index_t kK0PerLane = K / (kK1PerLane * kKLane);
};

template <>
struct LayoutFromDataType<fp8_t, 128, true>
{
    static constexpr index_t kK1PerLane = 16;
    static constexpr index_t kK0PerLane = 4;
};

template <>
struct LayoutFromDataType<pk_fp4_t, 128, true>
{
    static constexpr index_t kK1PerLane = 32;
    static constexpr index_t kK0PerLane = 2;
};

template <typename Arch,
          typename ADType,
          typename BDType,
          typename CDType,
          index_t K,
          bool MixPrec = false>
struct WmmaTraitsBase;

// GFX11 specialization
template <typename ADType, typename BDType, typename CDType, index_t K, bool MixPrec>
struct WmmaTraitsBase<gfx11_t, ADType, BDType, CDType, K, MixPrec>
{
    using ArchType = gfx11_t;

    using ADataType = ADType;
    using BDataType = BDType;
    using CDataType = CDType;

    using AVecType = ext_vector_t<ADataType, 16>;
    using BVecType = ext_vector_t<BDataType, 16>;
    using CVecType = ext_vector_t<CDataType, 8>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = K;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kRepeat     = 2;
    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 1;
    static constexpr index_t kAK0PerLane = 1;
    static constexpr index_t kAK1PerLane = K / (kAK0PerLane * kABKLane);
    static constexpr index_t kBK0PerLane = 1;
    static constexpr index_t kBK1PerLane = K / (kBK0PerLane * kABKLane);

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 8;
    static constexpr index_t kCM1PerLane = 1;

    using kABPs2RHssMajor = sequence<0, 2, 1>;
    using kABPs2RHssMinor = sequence<0, 1, 0>;
    using kABYs2RHsMajor  = sequence<2, 2>;
    using kABYs2RHsMinor  = sequence<0, 2>;

    using kCPs2RHssMajor = sequence<1, 2>;
    using kCPs2RHssMinor = sequence<1, 0>;
    using kCYs2RHsMajor  = sequence<1, 1>;
    using kCYs2RHsMinor  = sequence<0, 2>;

    using kCTPs2RHssMajor = sequence<2, 1>;
    using kCTPs2RHssMinor = sequence<1, 0>;
    using kCTYs2RHsMajor  = sequence<2, 2>;
    using kCTYs2RHsMinor  = sequence<0, 2>;
};

// GFX12 specialization
template <typename ADType, typename BDType, typename CDType, index_t K, bool MixPrec>
struct WmmaTraitsBase<gfx12_t, ADType, BDType, CDType, K, MixPrec>
{
    using ArchType = gfx12_t;

    using ADataType = ADType;
    using BDataType = BDType;
    using CDataType = CDType;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = K;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kRepeat = 1;
    static constexpr index_t kAMLane = 16;
    static constexpr index_t kBNLane = 16;

    static constexpr index_t kAK1PerLane = LayoutFromDataType<ADType, K, MixPrec>::kK1PerLane;
    static constexpr index_t kAK0PerLane = LayoutFromDataType<ADType, K, MixPrec>::kK0PerLane;
    static constexpr index_t kBK1PerLane = LayoutFromDataType<BDType, K, MixPrec>::kK1PerLane;
    static constexpr index_t kBK0PerLane = LayoutFromDataType<BDType, K, MixPrec>::kK0PerLane;
    static constexpr index_t kABKLane    = 2;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 8;

    using kABPs2RHssMajor = sequence<2, 1>;
    using kABPs2RHssMinor = sequence<1, 0>;
    using kABYs2RHsMajor  = sequence<2, 2>;
    using kABYs2RHsMinor  = sequence<0, 2>;

    using kCPs2RHssMajor = sequence<1, 2>;
    using kCPs2RHssMinor = sequence<1, 0>;
    using kCYs2RHsMajor  = sequence<1, 1>;
    using kCYs2RHsMinor  = sequence<0, 2>;

    using kCTPs2RHssMajor = sequence<2, 1>;
    using kCTPs2RHssMinor = sequence<1, 0>;
    using kCTYs2RHsMajor  = sequence<2, 2>;
    using kCTYs2RHsMinor  = sequence<0, 2>;

    static constexpr index_t kAPackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
    static constexpr index_t kBPackedSize = ck_tile::numeric_traits<BDataType>::PackedSize;
    static constexpr index_t kAInputSize  = kK / (kABKLane * kAPackedSize);
    static constexpr index_t kBInputSize  = kK / (kABKLane * kBPackedSize);
    static constexpr index_t kCOutputSize = kM / kCMLane;
    using AVecType                        = ext_vector_t<ADataType, kAInputSize>;
    using BVecType                        = ext_vector_t<BDataType, kBInputSize>;
    using CVecType                        = ext_vector_t<CDataType, kCOutputSize>;
};
} // namespace ck_tile
