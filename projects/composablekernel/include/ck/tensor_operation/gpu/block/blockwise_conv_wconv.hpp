// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/warp/wconv_conv.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

template <typename T>
struct Debug;

namespace ck {

template <index_t BlockSize,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t HRepeat,
          index_t WRepeat,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t WaveSize>
__device__ static auto GetWconvWaveIdx()
{
    using ThisThreadBlock    = ThisThreadBlock<BlockSize>;
    constexpr index_t HWaves = HPerBlock / (HRepeat * HPerWconv);
    constexpr index_t WWaves = WPerBlock / (WRepeat * WPerWconv);
    constexpr index_t KWaves = BlockSize / WaveSize / HWaves / WWaves;

    const index_t thread_id = ThisThreadBlock::GetThreadId();

    constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
        make_tuple(make_merge_transform(make_tuple(HWaves, WWaves, KWaves, WaveSize))),
        make_tuple(Sequence<0, 1, 2, 3>{}),
        make_tuple(Sequence<0>{}));

    return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
}

template <typename WeiDataType,
          typename InDataType,
          typename AccDataType,
          index_t CPerBlock,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t FilterSize>
static constexpr index_t GetFilterIters()
{
    if constexpr(FilterSize != 1)
    {
        return 1;
    }
    else
    {
        // Enable iters if CPerBlock is mulitple of CPerWconv
        constexpr index_t CPerWconv = WconvConv<WeiDataType,
                                                InDataType,
                                                AccDataType,
                                                HPerWconv,
                                                WPerWconv,
                                                FilterSize,
                                                1,
                                                1>::GetNumInputChannels();
        constexpr index_t CRepeat   = CPerBlock / CPerWconv;
        if constexpr(CRepeat % 4 == 0)
        {
            return 4;
        }
        else if constexpr(CRepeat % 2 == 0)
        {
            return 2;
        }
        else
        {
            return 1;
        }
    }
}

template <index_t BlockSize,
          typename WeiDataType,
          typename InDataType,
          typename AccDataType,
          typename WeiDataBlockDesc,
          typename InDataBlockDesc,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t CPerBlock,
          index_t KPerBlock,
          index_t HRepeat,
          index_t WRepeat,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          bool WeiDataEnableLds = false,
          bool InDataEnableLds  = false,
          bool Aco              = false>
/* Option: Read from LDS, big buffer hold all threads required data
 * Source
 * Weight: NumYX x KPerBlock x  CPerBlock (YXKC)
 * InData: HPerBlock x WPerBlock x CPerBlock (HWC)
 * Destination
 * Accum
 * thread level: HRepeat x WRepeat xKRepeat x AccVgprs
 * block  level: HWave x WWave x KWave x HRepeat x WRepeat x KRepeat x AccVgprs
 *
 * Option: Read from VGPR, small buffer hold each thread own required data (Skip LDS)
 * Source:
 * Weight(if skip LDS): KRepeat x CRepeat x WeightVgprs
 * InData(if skip LDS): HRepeat x WRepeat x CRepeat x InDataVgprs
 * Destination
 * Accum
 * block level: HWave x WWave x KWave x HRepeat x WRepeat x KRepeat x AccVgprs
 */
struct BlockwiseConvWconv
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    // Hardcode of WaveSize, since GFX13 conv only support wave32 mode
    static constexpr index_t WaveSize = 32;

    // Wave properties
    static constexpr index_t Iters   = GetFilterIters<WeiDataType,
                                                    InDataType,
                                                    AccDataType,
                                                    CPerBlock,
                                                    HPerWconv,
                                                    WPerWconv,
                                                    FilterSize>();
    static constexpr auto wconv_conv = WconvConv<WeiDataType,
                                                 InDataType,
                                                 AccDataType,
                                                 HPerWconv,
                                                 WPerWconv,
                                                 FilterSize,
                                                 DilationX,
                                                 DilationY,
                                                 Iters,
                                                 Aco>{};

    static constexpr index_t CPerWconv = wconv_conv.GetNumInputChannels();
    static constexpr index_t KPerWconv = wconv_conv.GetNumOutputChannels();

    // Output Size Per Wave: (HRepeat * HPerWconv) * (WRepeat * WPerWconv) * (KRepeat * KPerWconv)
    static constexpr index_t HWaves = HPerBlock / (HRepeat * HPerWconv);
    static constexpr index_t WWaves = WPerBlock / (WRepeat * WPerWconv);
    static constexpr index_t KWaves = BlockSize / WaveSize / HWaves / WWaves;

    static constexpr index_t KRepeat = KPerBlock / KWaves / KPerWconv;
    static constexpr index_t CRepeat = CPerBlock / CPerWconv;

    static_assert(HPerBlock % (HRepeat * HPerWconv) == 0, "");
    static_assert(WPerBlock % (WRepeat * WPerWconv) == 0, "");
    static_assert(KWaves > 0, "");
    static_assert(BlockSize == (WaveSize * HWaves * WWaves * KWaves), "");
    static_assert(KRepeat >= 1, "");
    static_assert(CRepeat >= 1, "");

    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                              AccDataType,
                              HRepeat * WRepeat * KRepeat,
                              wconv_conv.GetNumAccumComponents(),
                              true>
        accum_thread_buf_;

    __host__ __device__ constexpr auto& GetAccumThreadBuffer() { return accum_thread_buf_; }

    __device__ static auto GetWaveIdx()
    {
        return GetWconvWaveIdx<BlockSize,
                               HPerBlock,
                               WPerBlock,
                               HRepeat,
                               WRepeat,
                               HPerWconv,
                               WPerWconv,
                               WaveSize>();
    }

    // Default, Block buffer in LDS, thread level offset enabled
    __device__ static auto CalculateWeiDataThreadOriginDataIndex()
    {
        if constexpr(WeiDataEnableLds)
        {
            const auto wave_idx      = GetWaveIdx();
            const auto waveId_k      = wave_idx[I2];
            const auto wconv_wei_idx = wconv_conv.CalculateWeiDataThreadOriginDataIndex();

            if constexpr(Iters > 1)
            {
                return make_tuple(waveId_k * KRepeat,
                                  wconv_wei_idx[I0],
                                  0,
                                  wconv_wei_idx[I1],
                                  wconv_wei_idx[I2],
                                  wconv_wei_idx[I3]);
            }
            else
            {
                return make_tuple(waveId_k * KRepeat,
                                  0,
                                  wconv_wei_idx[I0],
                                  wconv_wei_idx[I1],
                                  wconv_wei_idx[I2],
                                  wconv_wei_idx[I3]);
            }
        }
        else
        {
            return tensor_operation::element_wise::PassThrough{};
        }
    }

    __device__ static auto CalculateInDataThreadOriginDataIndex()
    {
        if constexpr(InDataEnableLds)
        {
            const auto wave_idx          = GetWaveIdx();
            const auto waveId_h          = wave_idx[I0];
            const auto waveId_w          = wave_idx[I1];
            const auto wconv_in_data_idx = wconv_conv.CalculateInDataThreadOriginDataIndex();

            return make_tuple(waveId_h * HRepeat,
                              waveId_w * WRepeat,
                              0,
                              0,
                              wconv_in_data_idx[I0],
                              wconv_in_data_idx[I1],
                              wconv_in_data_idx[I2]);
        }
        else
        {
            // The constructor of ThreadwiseTensorSliceTransfer_StaticToStatic is different with
            // ThreadwiseTensorSliceTransfer_v4, it only needs an element operation.
            return tensor_operation::element_wise::PassThrough{};
        }
    }

    // Accum descriptor info, used by grid level classes
    static constexpr auto NumAccComp          = wconv_conv.GetNumAccumComponents();
    static constexpr auto NumAccImageSubTiles = wconv_conv.GetNumSubTilesPerImageTile();
    static constexpr auto NumAccCompPerTile   = NumAccComp / NumAccImageSubTiles;
    static constexpr auto NumAccSwizzleComp   = Aco ? 2 : 4;
    static constexpr auto NumAccCompSubTile =
        NumAccCompPerTile > NumAccSwizzleComp ? NumAccCompPerTile / NumAccSwizzleComp : 1;
    // Thread level, register descriptor. Vector-write
    __host__ __device__ static constexpr auto GetAccThreadDescriptor()
    {
        static_assert(NumAccComp == 4 || NumAccComp == 8, "");
        static_assert(NumAccCompPerTile % NumAccCompSubTile == 0, "");

        // HRepeat x WRepeat x KRepeat x H1 x H2 x W1 x K1 x K2
        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<HRepeat>{},
                       Number<WRepeat>{},
                       Number<KRepeat>{},
                       Number<NumAccImageSubTiles>{},
                       I1,
                       I1,
                       Number<NumAccCompSubTile>{},
                       Number<NumAccCompPerTile / NumAccCompSubTile>{}));
    }

    __host__ __device__ static constexpr auto GetAccThreadDescLength()
    {
        return Sequence<HRepeat,
                        WRepeat,
                        KRepeat,
                        NumAccImageSubTiles,
                        I1,
                        I1,
                        NumAccCompSubTile,
                        NumAccCompPerTile / NumAccCompSubTile>{};
    }

    template <typename AccBlockDesc_>
    __host__ __device__ static constexpr auto GetAccBlockWaveDescriptor(const AccBlockDesc_&)
    {
        return transform_tensor_descriptor(
            AccBlockDesc_{},
            make_tuple(
                make_unmerge_transform(make_tuple(Number<HPerBlock / HPerWconv>{},
                                                  Number<NumAccImageSubTiles>{},
                                                  Number<HPerWconv / NumAccImageSubTiles>{})),
                make_unmerge_transform(
                    make_tuple(Number<WPerBlock / WPerWconv>{}, Number<WPerWconv>{})),
                make_unmerge_transform(make_tuple(Number<KPerBlock / KPerWconv>{},
                                                  Number<NumAccCompSubTile>{},
                                                  Number<KPerWconv / NumAccCompSubTile>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0, 3, 4>{}, Sequence<1, 5>{}, Sequence<2, 6, 7>{}));
    }

    __device__ static auto CalculateAccThreadOriginDataIndex()
    {
        const auto wave_idx          = GetWaveIdx();
        const auto wconv_in_data_idx = wconv_conv.CalculateAccThreadOriginDataIndex();
        const auto waveId_h          = wave_idx[I0];
        const auto waveId_w          = wave_idx[I1];
        const auto waveId_k          = wave_idx[I2];

        return make_tuple(waveId_h * HRepeat,
                          waveId_w * WRepeat,
                          waveId_k * KRepeat,
                          wconv_in_data_idx[I0],
                          wconv_in_data_idx[I1],
                          wconv_in_data_idx[I2],
                          wconv_in_data_idx[I3],
                          wconv_in_data_idx[I4]);
    }

    using TupleWeiData = decltype(CalculateWeiDataThreadOriginDataIndex());
    using TupleInData  = decltype(CalculateInDataThreadOriginDataIndex());
    __host__ __device__
    BlockwiseConvWconv(TupleWeiData weight_origin = CalculateWeiDataThreadOriginDataIndex(),
                       TupleInData indata_origin  = CalculateInDataThreadOriginDataIndex())
        : weight_thread_copy_(weight_origin), indata_thread_copy_(indata_origin)
    {
        static_assert(WeiDataBlockDesc::IsKnownAtCompileTime() &&
                          InDataBlockDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ThisThreadBlock::GetNumOfThread() == HWaves * WWaves * KWaves * WaveSize,
                      "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize\n");

        static_assert(HPerBlock % (HPerWconv * HRepeat) == 0 &&
                          WPerBlock % (WPerWconv * WRepeat) == 0 &&
                          KPerBlock % (KPerWconv * KRepeat) == 0,
                      "wrong!");
    }

    __host__ __device__ static constexpr auto GetWeightRemapTable()
    {
        return wconv_conv.GetWeightRemapTable();
    }

    __host__ __device__ static constexpr auto GetWeightSecondTapMapTable()
    {
        return wconv_conv.GetWeightSecondTapMapTable();
    }
    // Describe how data allocated in thread copy src buffer
    static constexpr WeiDataBlockDesc weight_block_desc_;
    static constexpr InDataBlockDesc indata_block_desc_;

    template <typename WeightBlockBuffer, typename InDataBlockBuffer, typename AccumThreadBuffer>
    __device__ void Run(const WeightBlockBuffer& weight_block_buf,
                        const InDataBlockBuffer& indata_block_buf,
                        AccumThreadBuffer& accum_thread_buf) const
    {
        auto weight_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, WeiDataType>(
            weight_thread_desc_.GetElementSpaceSize());
        auto indata_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, InDataType>(
            indata_thread_desc_.GetElementSpaceSize());

        using WeiDataVec    = typename decltype(wconv_conv)::WeiDataVec::type;
        using WeiDataTapVec = typename decltype(wconv_conv)::WeiDataTapVec::type;
        using InDataVec     = typename decltype(wconv_conv)::InDataVec::type;

        // Change the loop order to K -> C -> H -> W to avoid load weight data repeatly.
        // Now, it is blocked with LLVM issues.
        static_for<0, KRepeat, 1>{}([&](auto k0) {
            static_for<0, HRepeat, 1>{}([&](auto h0) {
                static_for<0, WRepeat, 1>{}([&](auto w0) {
                    static_for<0, CRepeat, Iters>{}([&](auto c0) {
                        typename decltype(wconv_conv)::WeiDataVec weight_thread_vec;
                        // Load weights
                        if constexpr(FilterSize == 1)
                        {
                            constexpr auto WeightCPerIter = wconv_conv.GetNumWeightTapPerWave();
                            static_for<0, Iters, WeightCPerIter>{}([&](auto iter_idx) {
                                weight_thread_copy_.Run(
                                    weight_block_desc_,
                                    make_tuple(k0, c0 + iter_idx, I0, I0, I0, I0),
                                    weight_block_buf,
                                    weight_thread_desc_,
                                    make_tuple(I0, I0, I0, I0, I0, I0),
                                    weight_thread_buf);
                                weight_thread_vec.template AsType<WeiDataTapVec>()(
                                    iter_idx / Number<WeightCPerIter>{}) =
                                    *reinterpret_cast<const WeiDataTapVec*>(
                                        &(weight_thread_buf[I0]));
                            });
                        }
                        else if constexpr(FilterSize == 3)
                        {
                            static_assert(Iters == 1, "");
                            constexpr index_t WeightTapPerWave =
                                WeiDataEnableLds ? wconv_conv.GetNumWeightTapPerWave() : 1;
                            static_for<0, wconv_conv.GetNumWeightTap(), 1>{}([&](auto tape_idx) {
                                weight_thread_copy_.Run(
                                    weight_block_desc_,
                                    make_tuple(
                                        k0, c0, Number<WeightTapPerWave * tape_idx>{}, I0, I0, I0),
                                    weight_block_buf,
                                    weight_thread_desc_,
                                    make_tuple(I0, I0, I0, I0, I0, I0),
                                    weight_thread_buf);

                                weight_thread_vec.template AsType<WeiDataTapVec>()(
                                    Number<tape_idx>{}) =
                                    *reinterpret_cast<const WeiDataTapVec*>(&weight_thread_buf[I0]);
                            });
                        }
                        constexpr index_t accum_offset =
                            accum_thread_desc_.CalculateOffset(make_tuple(h0, w0, k0, I0));

                        // Load input tensor data
                        if constexpr(FilterSize == 1)
                        {

                            if constexpr(Iters == 1)
                            {
                                indata_thread_copy_.Run(indata_block_desc_,
                                                        make_tuple(h0, w0, c0, I0, I0, I0, I0),
                                                        indata_block_buf,
                                                        indata_thread_desc_,
                                                        make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                                        indata_thread_buf);

                                InDataVec indata_thread_vec;

                                indata_thread_vec =
                                    *reinterpret_cast<const InDataVec*>(&(indata_thread_buf[I0]));

                                wconv_conv.wconv_instr.Run(
                                    weight_thread_vec.template AsType<WeiDataVec>()(I0),
                                    indata_thread_vec,
                                    accum_thread_buf.GetVectorTypeReference(
                                        Number<accum_offset>{}));
                            }
                            else
                            {
                                InDataVec indata_thread_vec[Iters];

                                static_for<0, Iters, 1>{}([&](auto iter_idx) {
                                    indata_thread_copy_.Run(
                                        indata_block_desc_,
                                        make_tuple(h0, w0, c0 + iter_idx, I0, I0, I0, I0),
                                        indata_block_buf,
                                        indata_thread_desc_,
                                        make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                        indata_thread_buf);
                                    indata_thread_vec[iter_idx] =
                                        *reinterpret_cast<const InDataVec*>(
                                            &(indata_thread_buf[I0]));
                                });

                                wconv_conv.wconv_instr.Run(
                                    weight_thread_vec.template AsType<WeiDataVec>()(I0),
                                    indata_thread_vec,
                                    accum_thread_buf.GetVectorTypeReference(
                                        Number<accum_offset>{}));
                            }
                        }
                        else if constexpr(FilterSize == 3)
                        {
                            InDataVec indata_thread_vec[FilterSize];

                            //  read tensor
                            indata_thread_copy_.Run(indata_block_desc_,
                                                    make_tuple(h0, w0, c0, I0, I0, I0, I0),
                                                    indata_block_buf,
                                                    indata_thread_desc_,
                                                    make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                                    indata_thread_buf);

                            typename decltype(wconv_conv)::InDataVec indata_tmp;
                            constexpr index_t NumDataCompPerTile =
                                wconv_conv.GetNumDataComponents() / NumDataTiles;

                            // 0: Center buffer, 1: Left buffer, 2: Right buffer
                            constexpr index_t InDataRemap[FilterSize] = {1, 0, 2};
                            static_for<0, FilterSize, 1>{}([&](auto array_idx) {
                                using InDataTileVec =
                                    typename vector_type<InDataType, NumDataCompPerTile>::type;
                                static_for<0, NumDataTiles, 1>{}([&](auto tile_idx) {
                                    indata_tmp.template AsType<InDataTileVec>()(
                                        Number<tile_idx>{}) =
                                        *reinterpret_cast<const InDataTileVec*>(
                                            &indata_thread_buf
                                                [Number<tile_idx * FilterSize * NumDataCompPerTile +
                                                        array_idx * NumDataCompPerTile>{}]);
                                });
                                indata_thread_vec[InDataRemap[array_idx]] =
                                    indata_tmp.template AsType<InDataVec>()(I0);
                            });

                            wconv_conv.wconv_instr.Run(
                                weight_thread_vec.template AsType<WeiDataVec>()(I0),
                                indata_thread_vec,
                                accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{}));
                        }
                    });
                });
            });
        });
    }

    protected:
    // Thread descriptor for weight and input data
    static constexpr auto NumWeightSubTiles    = wconv_conv.GetNumSubTilesPerWeightTap();
    static constexpr auto NumWeightCompPerTile = wconv_conv.GetNumWeightCompPerTile();
    static constexpr auto weight_thread_desc_  = make_naive_tensor_descriptor_packed(
        make_tuple(I1, I1, I1, I1, Number<NumWeightSubTiles>{}, Number<NumWeightCompPerTile>{}));

    static constexpr auto NumDataSubTiles    = wconv_conv.GetNumSubTilesPerImageTile();
    static constexpr auto NumDataCompPerTile = wconv_conv.GetNumDataCompPerTile();
    static constexpr auto NumDataTiles       = wconv_conv.GetNumImageTilesInVertical();
    static constexpr auto indata_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(Number<NumDataTiles>{},
                                                       Number<FilterSize>{},
                                                       I1,
                                                       Number<NumDataSubTiles>{},
                                                       I1,
                                                       I1,
                                                       Number<NumDataCompPerTile>{}));

    // C[H, W, K, NumAccComp]
    static constexpr auto accum_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<HRepeat>{},
                   Number<WRepeat>{},
                   Number<KRepeat>{},
                   Number<wconv_conv.GetNumAccumComponents()>{}));

    // Initialize thread copy classes
    template <bool EnableLds>
    struct WeightThreadCopySelector;

    template <>
    struct WeightThreadCopySelector<true>
    {
        using type = ThreadwiseTensorSliceTransfer_v4<
            WeiDataType,
            WeiDataType,
            decltype(weight_block_desc_),
            decltype(weight_thread_desc_),
            Sequence<1, 1, 1, 1, NumWeightSubTiles, NumWeightCompPerTile>,
            Sequence<0, 1, 2, 3, 4, 5>,
            5,
            NumWeightCompPerTile,
            NumWeightCompPerTile>;
    };

    template <>
    struct WeightThreadCopySelector<false>
    {
        using type = ThreadwiseTensorSliceTransfer_StaticToStatic<
            WeiDataType,
            WeiDataType,
            decltype(weight_block_desc_),
            decltype(weight_thread_desc_),
            tensor_operation::element_wise::PassThrough,
            Sequence<1, 1, 1, 1, NumWeightSubTiles, NumWeightCompPerTile>,
            Sequence<0, 1, 2, 3, 4, 5>,
            5,
            NumWeightCompPerTile>;
    };

    template <bool EnableLds>
    struct InDataThreadCopySelector;

    template <>
    struct InDataThreadCopySelector<true>
    {
        using type = ThreadwiseTensorSliceTransfer_v4<
            InDataType,
            InDataType,
            decltype(indata_block_desc_),
            decltype(indata_thread_desc_),
            Sequence<NumDataTiles, FilterSize, 1, NumDataSubTiles, 1, 1, NumDataCompPerTile>,
            Sequence<0, 1, 2, 3, 4, 5, 6>,
            6,
            NumDataCompPerTile,
            NumDataCompPerTile>;
    };

    template <>
    struct InDataThreadCopySelector<false>
    {
        using type = ThreadwiseTensorSliceTransfer_StaticToStatic<
            InDataType,
            InDataType,
            decltype(indata_block_desc_),
            decltype(indata_thread_desc_),
            tensor_operation::element_wise::PassThrough,
            Sequence<NumDataTiles, FilterSize, 1, NumDataSubTiles, 1, 1, NumDataCompPerTile>,
            Sequence<0, 1, 2, 3, 4, 5, 6>,
            6,
            NumDataCompPerTile>;
    };

    typename WeightThreadCopySelector<WeiDataEnableLds>::type weight_thread_copy_;
    typename InDataThreadCopySelector<InDataEnableLds>::type indata_thread_copy_;
};
} // namespace ck
