// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/warp/wconv_conv.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/warp/acc_sba.hpp"

template <typename T>
struct Debug;

namespace ck {

template <typename ThisThreadBlock,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t HRepeat,
          index_t WRepeat,
          index_t HPerWconv,
          index_t WPerWconv>
__device__ static auto GetWconvWaveIdx()
{
    static constexpr index_t WaveSize = 32;
    constexpr index_t HWaves          = HPerBlock / (HRepeat * HPerWconv);
    constexpr index_t WWaves          = WPerBlock / (WRepeat * WPerWconv);
    constexpr index_t KWaves = ThisThreadBlock::GetNumOfThread() / WaveSize / HWaves / WWaves;

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
        // Enable Iters if CPerBlock is multiple of CPerWconv
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

template <typename ThisThreadBlock,
          typename WeiDataType,
          typename InDataType,
          typename DsDataType,
          typename AccDataType,
          typename WeiDataBlockDesc,
          typename InDataBlockDesc,
          typename DsBlockDesc,
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
          index_t activeFun,
          bool scaleBiasPacked  = false,
          bool uniformScale     = false,
          bool WeiDataEnableLds = false,
          bool InDataEnableLds  = false,
          bool DsEnableLds      = false,
          bool ConvertToTensor  = false,
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
 * InData(if skip LDS): WRepeat x CRepeat x HRepeat x InDataVgprs
 * Destination
 * Accum
 * block level: HWave x WWave x KWave x HRepeat x WRepeat x KRepeat x AccVgprs
 */
struct BlockwiseSubaConvWconv
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};

    // Hardcode of WaveSize, since GFX13 conv only support wave32 mode
    static constexpr index_t WaveSize = 32;

    static constexpr auto NumOfThread = ThisThreadBlock::GetNumOfThread();
    // Debug<ThisThreadBlock> bbb;
    // static_assert(NumOfThread == 32, "");
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
                                                 ThisThreadBlock::InWaveGroup(),
                                                 Aco>{};

    static constexpr auto acc_sba =
        AccSba<AccDataType, HPerWconv, WPerWconv, activeFun, scaleBiasPacked, uniformScale>{};

    static constexpr index_t CPerWconv = wconv_conv.GetNumInputChannels();
    static constexpr index_t KPerWconv = wconv_conv.GetNumOutputChannels();

    // Output Size Per Wave: (HRepeat * HPerWconv) * (WRepeat * WPerWconv) * (KRepeat * KPerWconv)
    static constexpr index_t HWaves = HPerBlock / (HRepeat * HPerWconv);
    static constexpr index_t WWaves = WPerBlock / (WRepeat * WPerWconv);
    static constexpr index_t KWaves = NumOfThread / WaveSize / HWaves / WWaves;

    static constexpr index_t KRepeat = KPerBlock / KWaves / KPerWconv;
    static constexpr index_t CRepeat = CPerBlock / CPerWconv;

    static_assert(HPerBlock % (HRepeat * HPerWconv) == 0, "");
    static_assert(WPerBlock % (WRepeat * WPerWconv) == 0, "");
    static_assert(KWaves > 0, "");
    static_assert(NumOfThread == (WaveSize * HWaves * WWaves * KWaves), "");
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
        return GetWconvWaveIdx<ThisThreadBlock,
                               HPerBlock,
                               WPerBlock,
                               HRepeat,
                               WRepeat,
                               HPerWconv,
                               WPerWconv>();
    }

    // Default, Block buffer in LDS, thread level offset enabled
    __device__ __host__ static auto CalculateWeiDataThreadOriginDataIndex()
    {
#ifdef __HIP_DEVICE_COMPILE__
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
#else
        return make_tuple(0, 0, 0, 0, 0, 0);
#endif
    }

    __device__ __host__ static auto CalculateInDataThreadOriginDataIndex()
    {
#ifdef __HIP_DEVICE_COMPILE__
        const auto wave_idx          = GetWaveIdx();
        const auto waveId_h          = wave_idx[I0];
        const auto waveId_w          = wave_idx[I1];
        const auto wconv_in_data_idx = wconv_conv.CalculateInDataThreadOriginDataIndex();
        // W0 x C0 x H0 x H1 x H2 x W1 x C1
        return make_tuple(waveId_w * WRepeat,
                          0,
                          waveId_h * HRepeat,
                          0,
                          wconv_in_data_idx[I0],
                          wconv_in_data_idx[I1],
                          wconv_in_data_idx[I2]);
#else
        return make_tuple(0, 0, 0, 0, 0, 0, 0);
#endif
    }

    __device__ static auto CalculateDsThreadOriginDataIndex()
    {
#ifdef __HIP_DEVICE_COMPILE__
        const auto wave_idx     = GetWaveIdx();
        const auto waveId_k     = wave_idx[I2];
        const auto wconv_ds_idx = acc_sba.CalculateDsThreadOriginDataIndex(KPerWconv);

        return make_tuple(waveId_k * KRepeat, wconv_ds_idx[I0], wconv_ds_idx[I1]);
#else
        return make_tuple(0, 0, 0);
#endif
    }

    // Accum descriptor info, used by grid level classes
    static constexpr auto NumAccComp          = wconv_conv.GetNumAccumComponents();
    static constexpr auto NumAccImageSubTiles = wconv_conv.GetNumSubTilesPerImageTile();
    static constexpr auto NumAccCompPerTile   = NumAccComp / NumAccImageSubTiles;
    static constexpr auto NumAccSwizzleComp   = Aco ? 2 : 4;
    static constexpr auto NumAccCompSubTile =
        NumAccCompPerTile > NumAccSwizzleComp ? NumAccCompPerTile / NumAccSwizzleComp : 1;

    static constexpr auto NumDataCompPerTile = wconv_conv.GetNumDataCompPerTile();
    // static constexpr index_t NumSubTilePerImage = wconv_conv.GetNumSubTilesPerImageTile();
    // Thread level, register descriptor. Vector-write
    __host__ __device__ static constexpr auto GetAccThreadDescriptor()
    {
        if constexpr(ConvertToTensor)
        {
            // HRepeat x WRepeat x CRepeat x I1 x H1 x H2 x W1 x C1
            return make_naive_tensor_descriptor_packed(make_tuple(Number<HRepeat>{},
                                                                  Number<WRepeat>{},
                                                                  Number<CRepeat>{},
                                                                  I1,
                                                                  Number<NumAccImageSubTiles>{},
                                                                  I1,
                                                                  I1,
                                                                  Number<NumDataCompPerTile>{}));
        }
        else
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
    }

    __host__ __device__ static constexpr auto GetAccThreadDescLength()
    {
        if constexpr(ConvertToTensor)
        {
            return Sequence<HRepeat,
                            WRepeat,
                            CRepeat,
                            I1,
                            NumAccImageSubTiles,
                            I1,
                            I1,
                            NumDataCompPerTile>{};
        }
        else
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
    }

    template <typename AccBlockDesc_>
    __host__ __device__ static constexpr auto
    GetAccBlockWaveDescriptor(const AccBlockDesc_& accDesc)
    {
        if constexpr(ConvertToTensor)
        {
            static_assert(KPerBlock == CPerBlock, "");
            // H x W x C -> H0 x W0 x C0 x I0 x H1 x H2 x W1 x C1
            return transform_tensor_descriptor(
                accDesc,
                make_tuple(
                    make_unmerge_transform(make_tuple(Number<HPerBlock / HPerWconv>{},
                                                      Number<NumAccImageSubTiles>{},
                                                      Number<HPerWconv / NumAccImageSubTiles>{})),
                    make_unmerge_transform(
                        make_tuple(Number<WPerBlock / WPerWconv>{}, Number<WPerWconv>{})),
                    // TODO: replace with KPerBlock and KPerWconv
                    make_unmerge_transform(
                        make_tuple(Number<CPerBlock / CPerWconv>{}, Number<CPerWconv>{})),
                    make_insert_transform(I0)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<>{}),
                make_tuple(Sequence<0, 4, 5>{}, Sequence<1, 6>{}, Sequence<2, 7>{}, Sequence<3>{}));
        }
        else
        {
            // H x W x K -> H0 x W0 x K0 x H1 x H2 x W1 x K1 x K2
            return transform_tensor_descriptor(
                accDesc,
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
    }

    __device__ __host__ static auto CalculateAccThreadOriginDataIndex()
    {
#ifdef __HIP_DEVICE_COMPILE__
        if constexpr(ConvertToTensor)
        {

            const auto wave_idx          = GetWaveIdx();
            const auto waveId_h          = wave_idx[I0];
            const auto waveId_w          = wave_idx[I1];
            const auto waveId_k          = wave_idx[I2];
            const auto wconv_in_data_idx = wconv_conv.CalculateInDataThreadOriginDataIndex();
            // H0 x W0 x C0 x I0 x H1 x H2 x W1 x C1
            return make_tuple(waveId_h * HRepeat,
                              waveId_w * WRepeat,
                              waveId_k * KRepeat,
                              0,
                              0,
                              wconv_in_data_idx[I0],
                              wconv_in_data_idx[I1],
                              wconv_in_data_idx[I2]);
        }
        else
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
#else
        return make_tuple(0, 0, 0, 0, 0, 0, 0, 0);
#endif
    }

    using TupleWeiData = decltype(CalculateWeiDataThreadOriginDataIndex());
    using TupleInData  = decltype(CalculateInDataThreadOriginDataIndex());
    using TupleDsData  = decltype(CalculateDsThreadOriginDataIndex());
    __host__ __device__
    BlockwiseSubaConvWconv(TupleWeiData weight_origin = CalculateWeiDataThreadOriginDataIndex(),
                           TupleInData indata_origin  = CalculateInDataThreadOriginDataIndex(),
                           TupleDsData dsData_origin  = CalculateDsThreadOriginDataIndex())
        : weight_thread_copy_(weight_origin),
          indata_thread_copy_(indata_origin),
          ds_thread_copy_(dsData_origin)
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
    static constexpr DsBlockDesc ds_block_desc_;

    template <typename WeightBlockBuffer,
              typename InDataBlockBuffer,
              typename DsBlockBuffer,
              typename AccumThreadBuffer>
    __device__ void Run(const WeightBlockBuffer& weight_block_buf,
                        const InDataBlockBuffer& indata_block_buf,
                        const DsBlockBuffer& ds_block_buf,
                        AccumThreadBuffer& accum_thread_buf) const
    {
        auto weight_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, WeiDataType>(
            weight_thread_desc_.GetElementSpaceSize());

        static constexpr index_t NumDTensor = 2;
        auto ds_thread_buf                  = generate_tuple(
            [&](auto i) {
                return make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
                    ds_thread_desc_[Number<i>{}].GetElementSpaceSize());
            },
            Number<NumDTensor>{});

        using WeiDataVec    = typename decltype(wconv_conv)::WeiDataVec::type;
        using WeiDataTapVec = typename decltype(wconv_conv)::WeiDataTapVec::type;
        using InDataVec     = typename decltype(wconv_conv)::InDataVec::type;
        using BiasVec       = typename decltype(acc_sba)::BiasVec::type;
        using ScaleVec      = typename decltype(acc_sba)::ScaleVec::type;
        using BiasScaleVec  = typename decltype(acc_sba)::BiasScaleVec::type;

        if constexpr(ConvertToTensor)
        {
            static_assert(KRepeat * KPerWconv == CRepeat * CPerWconv, "");
            static_assert(FilterSize == 1, "");
            static_assert(InDataEnableLds == false, "");
            static_assert(std::is_same<AccDataType, InDataType>::value, "");
            constexpr index_t KCRepeat = KPerWconv / CPerWconv;
            static_for<0, HRepeat, 1>{}([&](auto h0) {
                static_for<0, WRepeat, 1>{}([&](auto w0) {
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        constexpr index_t accum_offset =
                            accum_thread_desc_.CalculateOffset(make_tuple(h0, w0, k0, I0));
                        auto& accum_vec =
                            accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{});
                        static_for<0, KCRepeat, 1>{}([&](auto c0) {
                            constexpr index_t indata_offset = indata_block_desc_.CalculateOffset(
                                make_tuple(w0, c0 + k0 * Number<KCRepeat>{}, h0, I0, I0, I0, I0));
                            accum_vec.template AsType<InDataVec>()(c0) =
                                *reinterpret_cast<const InDataVec*>(
                                    &indata_block_buf[Number<indata_offset>{}]);
                        });
                    });
                });
            });
            return;
        }

        const InDataVec* indata_thread_vec_ptr[4];
        static_assert(Iters <= 4 && FilterSize < 4, "");

        static_for<0, KRepeat, 1>{}([&](auto k0) {
            static_for<0, CRepeat, Iters>{}([&](auto c0) {
                const WeiDataVec* weight_thread_vec_ptr;
                typename decltype(wconv_conv)::WeiDataVec weight_thread_vec;
                // Load weights
                if constexpr(WeiDataEnableLds)
                {
                    if constexpr(FilterSize == 1)
                    {
                        constexpr auto WeightCPerIter = wconv_conv.GetNumWeightTapPerWave();
                        static_for<0, Iters, WeightCPerIter>{}([&](auto iter_idx) {
                            weight_thread_copy_.Run(weight_block_desc_,
                                                    make_tuple(k0, c0 + iter_idx, I0, I0, I0, I0),
                                                    weight_block_buf,
                                                    weight_thread_desc_,
                                                    make_tuple(I0, I0, I0, I0, I0, I0),
                                                    weight_thread_buf);
                            weight_thread_vec.template AsType<WeiDataTapVec>()(
                                iter_idx / Number<WeightCPerIter>{}) =
                                *reinterpret_cast<const WeiDataTapVec*>(&(weight_thread_buf[I0]));
                        });
                    }
                    else if constexpr(FilterSize == 3)
                    {
                        static_assert(Iters == 1, "");
                        constexpr index_t WeightTapPerWave = wconv_conv.GetNumWeightTapPerWave();
                        static_for<0, wconv_conv.GetNumWeightTap(), 1>{}([&](auto tape_idx) {
                            weight_thread_copy_.Run(
                                weight_block_desc_,
                                make_tuple(
                                    k0, c0, Number<WeightTapPerWave * tape_idx>{}, I0, I0, I0),
                                weight_block_buf,
                                weight_thread_desc_,
                                make_tuple(I0, I0, I0, I0, I0, I0),
                                weight_thread_buf);

                            weight_thread_vec.template AsType<WeiDataTapVec>()(Number<tape_idx>{}) =
                                *reinterpret_cast<const WeiDataTapVec*>(&weight_thread_buf[I0]);
                        });
                    }
                    weight_thread_vec_ptr = &weight_thread_vec.template AsType<WeiDataVec>()[I0];
                }
                else
                {
                    constexpr index_t wei_offset =
                        weight_block_desc_.CalculateOffset(make_tuple(k0, c0, I0, I0, I0, I0));
                    if constexpr((Iters == 4) && (wconv_conv.GetNumWeightTapPerWave() == 2))
                    {
                        constexpr index_t wei_offset2 = weight_block_desc_.CalculateOffset(
                            make_tuple(k0, c0 + I2, I0, I0, I0, I0));
                        weight_thread_vec.template AsType<WeiDataTapVec>()(I0) =
                            *reinterpret_cast<const WeiDataTapVec*>(
                                &weight_block_buf[Number<wei_offset>{}]);
                        weight_thread_vec.template AsType<WeiDataTapVec>()(I1) =
                            *reinterpret_cast<const WeiDataTapVec*>(
                                &weight_block_buf[Number<wei_offset2>{}]);
                        weight_thread_vec_ptr =
                            &weight_thread_vec.template AsType<WeiDataVec>()[I0];
                    }
                    else
                    {
                        weight_thread_vec_ptr = reinterpret_cast<const WeiDataVec*>(
                            &weight_block_buf[Number<wei_offset>{}]);
                    }
                }

                InDataVec indata_thread_vec[4];
                static_for<0, HRepeat, 1>{}([&](auto h0) {
                    static_for<0, WRepeat, 1>{}([&](auto w0) {
                        constexpr index_t accum_offset =
                            accum_thread_desc_.CalculateOffset(make_tuple(h0, w0, k0, I0));
                        if constexpr(InDataEnableLds)
                        {
                            // Load input tensor data
                            if constexpr(FilterSize == 1)
                            {
                                static_for<0, Iters, 1>{}([&](auto iter_idx) {
                                    auto indata_thread_buf =
                                        make_static_buffer_v3<AddressSpaceEnum::Vgpr, InDataType>(
                                            indata_thread_desc_.GetElementSpaceSize(),
                                            (InDataType*)(&indata_thread_vec[iter_idx]));
                                    indata_thread_copy_.Run(
                                        indata_block_desc_,
                                        make_tuple(w0, c0 + iter_idx, h0, I0, I0, I0, I0),
                                        indata_block_buf,
                                        indata_thread_desc_,
                                        make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                        indata_thread_buf);
                                    indata_thread_vec_ptr[iter_idx] = &indata_thread_vec[iter_idx];
                                });
                            }
                            else if constexpr(FilterSize == 3)
                            {
                                //  read tensor
                                static_for<0, FilterSize, 1>{}([&](auto array_idx) {
                                    auto indata_thread_buf =
                                        make_static_buffer_v3<AddressSpaceEnum::Vgpr, InDataType>(
                                            indata_thread_desc_.GetElementSpaceSize(),
                                            (InDataType*)(&indata_thread_vec[array_idx]));
                                    indata_thread_copy_.Run(
                                        indata_block_desc_,
                                        make_tuple(w0 + array_idx, c0, h0, I0, I0, I0, I0),
                                        indata_block_buf,
                                        indata_thread_desc_,
                                        make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                        indata_thread_buf);
                                    indata_thread_vec_ptr[array_idx] =
                                        &indata_thread_vec[array_idx];
                                });
                            }
                        }
                        else
                        {
                            // Load input tensor data
                            if constexpr(FilterSize == 1)
                            {
                                static_for<0, Iters, 1>{}([&](auto iter_idx) {
                                    constexpr index_t indata_offset =
                                        indata_block_desc_.CalculateOffset(
                                            make_tuple(w0, c0 + iter_idx, h0, I0, I0, I0, I0));
                                    indata_thread_vec_ptr[iter_idx] =
                                        reinterpret_cast<const InDataVec*>(
                                            &indata_block_buf[Number<indata_offset>{}]);
                                });
                            }
                            else if constexpr(FilterSize == 3)
                            {
                                static_for<0, FilterSize, 1>{}([&](auto array_idx) {
                                    constexpr index_t indata_offset =
                                        indata_block_desc_.CalculateOffset(
                                            make_tuple(w0 + array_idx, c0, h0, I0, I0, I0, I0));
                                    indata_thread_vec_ptr[array_idx] =
                                        reinterpret_cast<const InDataVec*>(
                                            &indata_block_buf[Number<indata_offset>{}]);
                                });
                            }
                        }
                        wconv_conv.wconv_instr.Run(
                            *weight_thread_vec_ptr,
                            indata_thread_vec_ptr,
                            accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{}),
                            I0);
                    });
                });
            });

            // Load scale/bias data
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                ds_thread_copy_.Run(ds_block_desc_[Number<i>{}],
                                    make_tuple(k0, I0, I0),
                                    ds_block_buf[Number<i>{}],
                                    ds_thread_desc_[Number<i>{}],
                                    make_tuple(I0, I0, I0),
                                    ds_thread_buf(i));
            });

            // run sba/uba
            constexpr auto accSbaInstance = ck::AccSba<AccDataType,
                                                       HPerWconv,
                                                       WPerWconv,
                                                       activeFun,
                                                       scaleBiasPacked,
                                                       uniformScale>();
            static_for<0, HRepeat, 1>{}([&](auto h0) {
                static_for<0, WRepeat, 1>{}([&](auto w0) {
                    constexpr index_t accum_offset =
                        accum_thread_desc_.CalculateOffset(make_tuple(h0, w0, k0, I0));
                    constexpr index_t bias_offset =
                        ds_thread_desc_[I0].CalculateOffset(make_tuple(I0, I0, I0));
                    constexpr index_t scale_offset =
                        ds_thread_desc_[I1].CalculateOffset(make_tuple(I0, I0, I0));

                    BiasVec bias_ = *reinterpret_cast<const BiasVec*>(
                        &(ds_thread_buf(I0)[Number<bias_offset>{}]));
                    ScaleVec scale_ = *reinterpret_cast<const ScaleVec*>(
                        &(ds_thread_buf(I1)[Number<scale_offset>{}]));
                    BiasScaleVec bias_scale_;

                    if(std::is_same<float, AccDataType>::value)
                    {
                        if constexpr(scaleBiasPacked)
                        {
                            auto laneId = get_thread_local_1d_id() & (WaveSize - 1);
                            if(laneId % 2)
                            {
                                bias_ = *reinterpret_cast<const BiasVec*>(
                                    &(ds_thread_buf(I1)[Number<scale_offset>{}]));
                            }
                            else
                            {
                                bias_ = *reinterpret_cast<const BiasVec*>(
                                    &(ds_thread_buf(I0)[Number<bias_offset>{}]));
                            }
                        }
                    }
                    else if(std::is_same<ck::half_t, AccDataType>::value ||
                            std::is_same<ck::bhalf_t, AccDataType>::value)
                    {
                        if constexpr(scaleBiasPacked)
                        {
                            bias_scale_[0] = *reinterpret_cast<const AccDataType*>(
                                &(ds_thread_buf(I1)[Number<bias_offset>{}]));
                            bias_scale_[1] = *reinterpret_cast<const AccDataType*>(
                                &(ds_thread_buf(I1)[Number<scale_offset>{}]));
                        }
                        else
                        {
                            bias_scale_ = bias_;
                        }
                    }

                    if constexpr(std::is_same<float, AccDataType>::value)
                    {
                        accSbaInstance.sba_instr.Run(
                            accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{}),
                            scale_,
                            bias_,
                            accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{}));
                    }
                    else if constexpr(std::is_same<half_t, AccDataType>::value ||
                                      std::is_same<bhalf_t, AccDataType>::value)
                    {
                        AccDataType halfScale_ = *reinterpret_cast<const AccDataType*>(
                            &(ds_thread_buf(I1)[Number<scale_offset>{}]));
                        if constexpr((HPerWconv == 4) && (WPerWconv == 2))
                        {
                            // 4xhalf for output of convolve
                            // 4xhalf for input of sba/uba
                            auto& c_vec =
                                accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{});
                            if constexpr(std::is_same<half_t, AccDataType>::value)
                            {
                                half2_t& sba_uba_output0 =
                                    c_vec.template AsType<half2_t>()(Number<0>{});
                                half2_t& sba_uba_output1 =
                                    c_vec.template AsType<half2_t>()(Number<1>{});
                                accSbaInstance.sba_instr.Run(
                                    accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{}),
                                    halfScale_,
                                    bias_scale_,
                                    sba_uba_output0,
                                    sba_uba_output1);
                            }
                            else
                            {
                                bhalf2_t& sba_uba_output0 =
                                    c_vec.template AsType<bhalf2_t>()(Number<0>{});
                                bhalf2_t& sba_uba_output1 =
                                    c_vec.template AsType<bhalf2_t>()(Number<1>{});
                                accSbaInstance.sba_instr.Run(
                                    accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{}),
                                    halfScale_,
                                    bias_scale_,
                                    sba_uba_output0,
                                    sba_uba_output1);
                            }
                        }
                        else
                        {
                            accSbaInstance.sba_instr.Run(
                                accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{}),
                                halfScale_,
                                bias_scale_,
                                accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{}));
                        }
                    }
                });
            });
        });
    };

    protected:
    // Thread descriptor for weight and input data
    static constexpr auto NumWeightSubTiles    = wconv_conv.GetNumSubTilesPerWeightTap();
    static constexpr auto NumWeightCompPerTile = wconv_conv.GetNumWeightCompPerTile();
    static constexpr auto weight_thread_desc_  = make_naive_tensor_descriptor_packed(
        make_tuple(I1, I1, I1, I1, Number<NumWeightSubTiles>{}, Number<NumWeightCompPerTile>{}));

    static constexpr auto NumDataSubTiles = wconv_conv.GetNumSubTilesPerImageTile();
    static constexpr auto NumDataTiles    = wconv_conv.GetNumImageTilesInVertical();
    static constexpr auto indata_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                       I1,
                                                       Number<NumDataTiles>{},
                                                       Number<NumDataSubTiles>{},
                                                       I1,
                                                       I1,
                                                       Number<NumDataCompPerTile>{}));
    static constexpr auto NumBias  = acc_sba.GetNumBiasComponents();
    static constexpr auto NumScale = acc_sba.GetNumScaleComponents();

    // C[H, W, K, NumAccComp]
    static constexpr auto accum_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<HRepeat>{},
                   Number<WRepeat>{},
                   Number<KRepeat>{},
                   Number<wconv_conv.GetNumAccumComponents()>{}));

    // D[H, W, K, NumDsComp]
    template <index_t Components>
    static constexpr auto MakeDsDescriptor()
    {
        return make_naive_tensor_descriptor_packed(make_tuple(I1, I1, Number<Components>{}));
    }

    static constexpr auto ds_bias_desc    = MakeDsDescriptor<NumBias>();
    static constexpr auto ds_scale_desc   = MakeDsDescriptor<NumScale>();
    static constexpr auto ds_thread_desc_ = make_tuple(ds_bias_desc, ds_scale_desc);

    // Initialize thread copy classes
    using WeiDataThreadLdsCopy = ThreadwiseTensorSliceTransfer_v4<
        WeiDataType,
        WeiDataType,
        decltype(weight_block_desc_),
        decltype(weight_thread_desc_),
        Sequence<1, 1, 1, 1, NumWeightSubTiles, NumWeightCompPerTile>,
        Sequence<0, 1, 2, 3, 4, 5>,
        5,
        NumWeightCompPerTile,
        NumWeightCompPerTile>;

    WeiDataThreadLdsCopy weight_thread_copy_;

    using InDataThreadLdsCopy = ThreadwiseTensorSliceTransfer_v4<
        InDataType,
        InDataType,
        decltype(indata_block_desc_),
        decltype(indata_thread_desc_),
        Sequence<1, 1, NumDataTiles, NumDataSubTiles, 1, 1, NumDataCompPerTile>,
        Sequence<0, 1, 2, 3, 4, 5, 6>,
        6,
        NumDataCompPerTile,
        NumDataCompPerTile>;
    InDataThreadLdsCopy indata_thread_copy_;

    using DsDataThreadLdsCopy =
        ThreadwiseTensorSliceTransfer_v4<AccDataType,
                                         AccDataType,
                                         remove_cvref_t<decltype(ds_block_desc_[I0])>,
                                         remove_cvref_t<decltype(ds_thread_desc_[I0])>,
                                         Sequence<1, 1, NumBias>, // ToDo for NumScale
                                         Sequence<0, 1, 2>,
                                         2,
                                         1,
                                         1>;
    DsDataThreadLdsCopy ds_thread_copy_;
};
} // namespace ck
