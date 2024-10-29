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
          index_t FilterSize,
          bool Transposed>
static constexpr index_t GetFilterIters()
{
    if constexpr(FilterSize == 2)
    {
        return Transposed ? 1 : 4;
    }
    else if constexpr(FilterSize == 3)
    {
        return 1;
    }
    else if constexpr(FilterSize == 1)
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
    else
    {
        static_assert(false, "never called");
    }
}

template <typename ThisThreadBlock,
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
          bool ConvertToTensor  = false,
          bool Transposed       = false,
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
 * NOTE: Assume stride is 2 if FilterSize = 2
 */
struct BlockwiseConvWconv
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};

    // Hardcode of WaveSize, since GFX13 conv only support wave32 mode
    static constexpr index_t WaveSize = 32;

    static constexpr auto NumOfThread     = ThisThreadBlock::GetNumOfThread();
    static constexpr bool EnableWaveGroup = ThisThreadBlock::InWaveGroup();

    static constexpr index_t WaveFilterSize = (FilterSize == 2) ? 1 : FilterSize;

    // Wave properties
    static constexpr index_t Iters = GetFilterIters<WeiDataType,
                                                    InDataType,
                                                    AccDataType,
                                                    CPerBlock,
                                                    HPerWconv,
                                                    WPerWconv,
                                                    FilterSize,
                                                    Transposed>();

    static constexpr auto wconv_conv = WconvConv<WeiDataType,
                                                 InDataType,
                                                 AccDataType,
                                                 HPerWconv,
                                                 WPerWconv,
                                                 WaveFilterSize,
                                                 DilationX,
                                                 DilationY,
                                                 Iters,
                                                 ThisThreadBlock::InWaveGroup(),
                                                 Aco>{};

    static constexpr index_t CPerWconv = wconv_conv.GetNumInputChannels();
    static constexpr index_t KPerWconv = wconv_conv.GetNumOutputChannels();

    // Output Size Per Wave: (HRepeat * HPerWconv) * (WRepeat * WPerWconv) * (KRepeat * KPerWconv)
    static constexpr index_t HWaves = HPerBlock / (HRepeat * HPerWconv);
    static constexpr index_t WWaves = WPerBlock / (WRepeat * WPerWconv);
    static constexpr index_t KWaves = NumOfThread / WaveSize / HWaves / WWaves;

    static constexpr index_t KRepeat = KPerBlock / KWaves / KPerWconv;
    static constexpr index_t CRepeat = CPerBlock / CPerWconv;

    static constexpr index_t DataTileHeight = 4;
    static constexpr index_t H_Pad          = (FilterSize == 3) ? DataTileHeight : 0;
    static constexpr index_t W_Pad          = (FilterSize == 3) ? WPerWconv : 0;
    static constexpr index_t HPerBlockIn    = HPerBlock + H_Pad * 2;
    static constexpr index_t WPerBlockIn    = WPerBlock + W_Pad * 2;

    static constexpr index_t NumSubTilePerImage      = wconv_conv.GetNumSubTilesPerImageTile();
    static constexpr index_t NumSubTilesPerWeightTap = wconv_conv.GetNumSubTilesPerWeightTap();
    static constexpr index_t NumWeightCompPerTile    = wconv_conv.GetNumWeightCompPerTile();
    static constexpr index_t NumWeightTap            = wconv_conv.GetNumWeightTap();
    static constexpr index_t HPerWave                = HRepeat * HPerWconv;
    static constexpr index_t WPerWave                = WRepeat * WPerWconv;
    static constexpr index_t CPerWave                = CPerBlock;
    static constexpr index_t KPerWave                = KPerBlock;
    static constexpr index_t HPerWaveIn              = HPerWave + H_Pad * 2;
    static constexpr index_t WPerWaveIn              = WPerWave + W_Pad * 2;
    static constexpr index_t OutputScale2            = (FilterSize == 2) ? (Transposed ? 4 : 1) : 2;
    static constexpr index_t HRepeatOut              = HRepeat * OutputScale2 / 2;
    static constexpr index_t WRepeatOut              = WRepeat * OutputScale2 / 2;
    static constexpr index_t HPerBlockOut            = HPerBlock * OutputScale2 / 2;
    static constexpr index_t WPerBlockOut            = WPerBlock * OutputScale2 / 2;
    static constexpr index_t HPerWaveOut             = HPerWave * OutputScale2 / 2;
    static constexpr index_t WPerWaveOut             = WPerWave * OutputScale2 / 2;

    // Accum descriptor info
    static constexpr auto NumAccComp        = wconv_conv.GetNumAccumComponents();
    static constexpr auto NumAccCompPerTile = NumAccComp / NumSubTilePerImage;
    static constexpr auto NumAccSwizzleComp = Aco ? 2 : 4;
    static constexpr auto NumAccCompSubTile =
        NumAccCompPerTile > NumAccSwizzleComp ? NumAccCompPerTile / NumAccSwizzleComp : 1;
    static constexpr auto NumDataCompPerTile = wconv_conv.GetNumDataCompPerTile();

    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                              AccDataType,
                              HRepeatOut * WPerBlockOut * KRepeat,
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

    // Input data descriptor help functions

    // Describe how the InData in global memory is viewed in block level.
    template <typename InGridDesc>
    __host__ __device__ static constexpr auto
    MakeInGridBlockDescriptor(const InGridDesc& in_grid_pad_desc)
    {
        if constexpr(InDataEnableLds)
        {
            // H x W x C
            return in_grid_pad_desc;
        }
        else
        {
            // H x W x C -> W0 x C0 x H0 x H1 x H2 x W1 x C1
            const auto H = in_grid_pad_desc.GetLength(I0);
            const auto W = in_grid_pad_desc.GetLength(I1);
            const auto C = in_grid_pad_desc.GetLength(I2);
            return transform_tensor_descriptor(
                in_grid_pad_desc,
                make_tuple(
                    make_unmerge_transform(make_tuple(H / HPerWconv,
                                                      Number<NumSubTilePerImage>{},
                                                      Number<HPerWconv / NumSubTilePerImage>{})),
                    make_unmerge_transform(make_tuple(W / WPerWconv, Number<WPerWconv>{})),
                    make_unmerge_transform(make_tuple(C / CPerWconv, Number<CPerWconv>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<2, 3, 4>{}, Sequence<0, 5>{}, Sequence<1, 6>{}));
        }
    }

    // Describe how data read from (LDS/VGPR) buffer, used by Block level classes.
    template <typename InBlockDesc_>
    __host__ __device__ static constexpr auto MakeInWaveDescriptor(const InBlockDesc_&)
    {
        constexpr auto in_wave_desc = [&]() {
            if constexpr(InDataEnableLds)
            {
                // H x W x C -> W0 x C0 x H0 x H1 x H2 x W1 x C1
                return transform_tensor_descriptor(
                    InBlockDesc_{},
                    make_tuple(make_unmerge_transform(
                                   make_tuple(Number<HPerBlockIn / HPerWconv>{},
                                              Number<NumSubTilePerImage>{},
                                              Number<HPerWconv / NumSubTilePerImage>{})),
                               make_unmerge_transform(make_tuple(Number<WPerBlockIn / WPerWconv>{},
                                                                 Number<WPerWconv>{})),
                               make_unmerge_transform(make_tuple(Number<CPerBlock / CPerWconv>{},
                                                                 Number<CPerWconv>{}))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<2, 3, 4>{}, Sequence<0, 5>{}, Sequence<1, 6>{}));
            }
            else
            {
                return InBlockDesc_{};
            }
        }();

        return in_wave_desc;
    }

    __host__ __device__ static constexpr auto MakeInBlockSliceCopyStep()
    {
        constexpr auto in_block_copy_step = [&]() {
            if constexpr(InDataEnableLds)
            {
                return Sequence<I0, I0, Number<CPerBlock>{}>{};
            }
            else
            {
                return Sequence<I0, Number<CPerBlock / CPerWconv>{}, I0, I0, I0, I0, I0>{};
            }
        }();

        return in_block_copy_step;
    }

    __host__ __device__ static constexpr auto GetInWaveDescLength()
    {
        return Sequence<WPerWaveIn / WPerWconv,
                        CPerWave / CPerWconv,
                        HPerWaveIn / HPerWconv,
                        NumSubTilePerImage,
                        1,
                        1,
                        NumDataCompPerTile>{};
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

    // Wei data descriptor help functions

    // Describe how the WeiData in global memory is viewed in block level.
    template <typename WeiGridDesc>
    __host__ __device__ static constexpr auto
    MakeWeiGridBlockDescriptor(const WeiGridDesc& wei_grid_desc)
    {
        if constexpr(WeiDataEnableLds)
        {
            // K x YX x C
            return wei_grid_desc;
        }
        else
        {
            // K x YX x C -> K0 x C0 x YX x K1 x C1 x C2
            const auto K = wei_grid_desc.GetLength(I0);
            const auto C = wei_grid_desc.GetLength(I2);
            return transform_tensor_descriptor(
                wei_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(K / KPerWconv, Number<KPerWconv>{})),
                           make_pass_through_transform(Number<FilterSize * FilterSize>{}),
                           make_unmerge_transform(
                               make_tuple(C / CPerWconv,
                                          Number<NumSubTilesPerWeightTap>{},
                                          Number<CPerWconv / NumSubTilesPerWeightTap>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0, 3>{}, Sequence<2>{}, Sequence<1, 4, 5>{}));
        }
    }

    // Describe how wei data read from (LDS/VGPR) buffer, used by Block level classes.
    template <typename WeiBlockDesc_>
    __host__ __device__ static constexpr auto MakeWeiWaveDescriptor(const WeiBlockDesc_&)
    {
        constexpr auto wei_wave_desc = [&]() {
            if constexpr(WeiDataEnableLds)
            {
                // K x YX x C -> K0 x C0 x YX x K1 x C1 x C2
                return transform_tensor_descriptor(
                    WeiBlockDesc_{},
                    make_tuple(make_unmerge_transform(make_tuple(Number<KPerBlock / KPerWconv>{},
                                                                 Number<KPerWconv>{})),
                               make_pass_through_transform(Number<FilterSize * FilterSize>{}),
                               make_unmerge_transform(
                                   make_tuple(Number<CPerBlock / CPerWconv>{},
                                              Number<NumSubTilesPerWeightTap>{},
                                              Number<CPerWconv / NumSubTilesPerWeightTap>{}))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0, 3>{}, Sequence<2>{}, Sequence<1, 4, 5>{}));
            }
            else
            {
                return WeiBlockDesc_{};
            }
        }();

        return wei_wave_desc;
    }

    __host__ __device__ static constexpr auto MakeWeiBlockSliceCopyStep()
    {
        constexpr auto wei_block_copy_step = [&]() {
            if constexpr(WeiDataEnableLds)
            {
                return Sequence<I0, I0, Number<CPerBlock>{}>{};
            }
            else
            {
                return Sequence<I0, Number<CPerBlock / CPerWconv>{}, I0, I0, I0, I0>{};
            }
        }();

        return wei_block_copy_step;
    }

    __host__ __device__ static constexpr auto GetWeiWaveDescLength()
    {
        constexpr index_t NumTapPerCopy = (FilterSize == 3) ? 1 : FilterSize * FilterSize;
        return Sequence<KPerWave / KPerWconv,
                        CPerWave / CPerWconv,
                        NumTapPerCopy,
                        1,
                        NumSubTilesPerWeightTap,
                        NumWeightCompPerTile>{};
    }

    __device__ __host__ static auto CalculateWeiDataThreadOriginDataIndex()
    {
#ifdef __HIP_DEVICE_COMPILE__
        const auto wave_idx      = GetWaveIdx();
        const auto waveId_k      = wave_idx[I2];
        const auto wconv_wei_idx = wconv_conv.CalculateWeiDataThreadOriginDataIndex();

        if constexpr(FilterSize == 2)
        {
            return make_tuple(waveId_k * KRepeat,
                              0,
                              wconv_wei_idx[I0],
                              wconv_wei_idx[I1],
                              wconv_wei_idx[I2],
                              wconv_wei_idx[I3]);
        }
        else if constexpr(Iters > 1)
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
    // Acc Data descriptor help functions

    // Describe how to read data from accum_thread_buf_ (VGPR)
    __host__ __device__ static constexpr auto GetAccThreadDescriptor()
    {
        if constexpr(ConvertToTensor)
        {
            // HRepeat x WRepeat x CRepeat x I1 x H1 x H2 x W1 x C1
            return make_naive_tensor_descriptor_packed(make_tuple(Number<HRepeat>{},
                                                                  Number<WRepeat>{},
                                                                  Number<CRepeat>{},
                                                                  I1,
                                                                  Number<NumSubTilePerImage>{},
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
                make_tuple(Number<HRepeatOut>{},
                           Number<WRepeatOut>{},
                           Number<KRepeat>{},
                           Number<NumSubTilePerImage>{},
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
                            NumSubTilePerImage,
                            I1,
                            I1,
                            NumDataCompPerTile>{};
        }
        else
        {
            return Sequence<HRepeatOut,
                            WRepeatOut,
                            KRepeat,
                            NumSubTilePerImage,
                            I1,
                            I1,
                            NumAccCompSubTile,
                            NumAccCompPerTile / NumAccCompSubTile>{};
        }
    }

    // Describe how data store to LDS buffer
    __host__ __device__ static constexpr auto GetAccBlockDescriptor()
    {
        constexpr auto acc_block_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<HPerBlockOut>{}, Number<WPerBlockOut>{}, Number<KPerBlock>{}));
        return acc_block_desc;
    }

    // Describe how to store accum data to LDS or Global memory.
    template <typename AccBlockDesc_>
    __host__ __device__ static constexpr auto GetAccWaveDescriptor(const AccBlockDesc_& accDesc)
    {
        if constexpr(ConvertToTensor)
        {
            static_assert(KPerBlock == CPerBlock, "");
            // H x W x C -> H0 x W0 x C0 x I0 x H1 x H2 x W1 x C1
            return transform_tensor_descriptor(
                accDesc,
                make_tuple(
                    make_unmerge_transform(make_tuple(Number<HPerBlock / HPerWconv>{},
                                                      Number<NumSubTilePerImage>{},
                                                      Number<HPerWconv / NumSubTilePerImage>{})),
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
                    make_unmerge_transform(make_tuple(Number<HPerBlockOut / HPerWconv>{},
                                                      Number<NumSubTilePerImage>{},
                                                      Number<HPerWconv / NumSubTilePerImage>{})),
                    make_unmerge_transform(
                        make_tuple(Number<WPerBlockOut / WPerWconv>{}, Number<WPerWconv>{})),
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

            return make_tuple(waveId_h * HRepeatOut,
                              waveId_w * WRepeatOut,
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

    using TupleWeiData    = decltype(CalculateWeiDataThreadOriginDataIndex());
    using TupleInData     = decltype(CalculateInDataThreadOriginDataIndex());
    using InDataWaveDesc  = decltype(MakeInWaveDescriptor(InDataBlockDesc{}));
    using WeiDataWaveDesc = decltype(MakeWeiWaveDescriptor(WeiDataBlockDesc{}));

    struct SharedMemTrait
    {
        // LDS allocation for A and B: be careful of alignment
        static constexpr auto max_lds_align = 8;

        static constexpr auto in_block_space_size_aligned =
            InDataEnableLds ? math::integer_least_multiple(InDataBlockDesc{}.GetElementSpaceSize(),
                                                           max_lds_align)
                            : 0;
        static constexpr auto wei_block_space_size_aligned =
            WeiDataEnableLds ? math::integer_least_multiple(
                                   WeiDataBlockDesc{}.GetElementSpaceSize(), max_lds_align)
                             : 0;

        static constexpr auto in_block_space_offset  = 0;
        static constexpr auto wei_block_space_offset = in_block_space_size_aligned;

        // LDS allocation for C shuffle in LDS
        static constexpr auto acc_block_space_size = GetAccBlockDescriptor().GetElementSpaceSize();

        static constexpr auto acc_block_space_offset = 0;

        static constexpr auto lds_size =
            math::max(acc_block_space_size * sizeof(AccDataType),
                      in_block_space_size_aligned * sizeof(InDataType) +
                          wei_block_space_size_aligned * sizeof(WeiDataType));
    };

    struct LaneSharedMemTrait
    {
        static constexpr auto max_lane_shared_align = 4;

        static constexpr auto in_block_space_size_aligned =
            EnableWaveGroup && (InDataEnableLds == false)
                ? math::integer_least_multiple(InDataWaveDesc{}.GetElementSpaceSize(),
                                               max_lane_shared_align)
                : 0;
        static constexpr auto wei_block_space_size_aligned =
            EnableWaveGroup && (WeiDataEnableLds == false)
                ? math::integer_least_multiple(WeiDataWaveDesc{}.GetElementSpaceSize(),
                                               max_lane_shared_align)
                : 0;

        static constexpr auto in_block_space_offset  = 0;
        static constexpr auto wei_block_space_offset = in_block_space_size_aligned;

        static constexpr auto lane_shared_size = in_block_space_size_aligned * sizeof(InDataType) +
                                                 wei_block_space_size_aligned * sizeof(WeiDataType);
    };

    template <bool HasMainLoop>
    static constexpr index_t GetLaneSharedMemCount()
    {
        return HasMainLoop ? 2 : 1;
    }

    __host__ __device__
    BlockwiseConvWconv(TupleWeiData weight_origin = CalculateWeiDataThreadOriginDataIndex(),
                       TupleInData indata_origin  = CalculateInDataThreadOriginDataIndex())
        : weight_thread_copy_(weight_origin), indata_thread_copy_(indata_origin)
    {
        static_assert(WeiDataBlockDesc::IsKnownAtCompileTime() &&
                          InDataBlockDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");
        static_assert(WeiDataWaveDesc::IsKnownAtCompileTime() &&
                          InDataWaveDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ThisThreadBlock::GetNumOfThread() == HWaves * WWaves * KWaves * WaveSize,
                      "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize\n");

        static_assert(HPerBlock % (HPerWconv * HRepeat) == 0 &&
                          WPerBlock % (WPerWconv * WRepeat) == 0 &&
                          KPerBlock % (KPerWconv * KRepeat) == 0,
                      "wrong!");

        static_assert(!(Transposed && (FilterSize != 2)),
                      "Only support strided conv2x2 transpose conv");
        static_assert(!((Transposed == false) && (FilterSize == 2) &&
                        ((HRepeat % 2 != 0) || (WRepeat % 2 != 0))),
                      "Repeat must be even for strided conv 2x2 conv");
        static_assert(HPerBlock % (HRepeat * HPerWconv) == 0, "");
        static_assert(WPerBlock % (WRepeat * WPerWconv) == 0, "");
        static_assert(KWaves > 0, "");
        static_assert(NumOfThread == (WaveSize * HWaves * WWaves * KWaves), "");
        static_assert(KRepeat >= 1, "");
        static_assert(CRepeat >= 1, "");
        static_assert(HPerWaveIn % HPerWconv == 0, "");
        static_assert(WPerWaveIn % WPerWconv == 0, "");
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
    static constexpr WeiDataWaveDesc weight_wave_desc_;
    static constexpr InDataWaveDesc indata_wave_desc_;

    template <typename WeightBlockBuffer, typename InDataBlockBuffer, typename AccumThreadBuffer>
    __device__ void RunEmulateConv2(const WeightBlockBuffer& weight_block_buf,
                                    const InDataBlockBuffer& indata_block_buf,
                                    AccumThreadBuffer& accum_thread_buf,
                                    bool isLast) const
    {
        auto weight_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, WeiDataType>(
            weight_thread_desc_.GetElementSpaceSize());
        using WeiDataVec    = typename decltype(wconv_conv)::WeiDataVec::type;
        using WeiDataTapVec = typename decltype(wconv_conv)::WeiDataTapVec::type;
        using InDataVec     = typename decltype(wconv_conv)::InDataVec::type;

        const InDataVec* indata_thread_vec_ptr[4];
        static_assert(Iters <= 4 && FilterSize == 2, "");

        constexpr auto NumYX          = FilterSize * FilterSize;
        constexpr index_t x_offset[4] = {0, 1, 0, 1};
        constexpr index_t y_offset[4] = {0, 0, 1, 1};
        constexpr index_t CStep       = Transposed == false ? 1 : Iters;
        constexpr index_t HStep       = Transposed == false ? 2 : 1;
        constexpr index_t WStep       = Transposed == false ? 2 : 1;

        static_for<0, KRepeat, 1>{}([&](auto k0) {
            static_for<0, CRepeat, CStep>{}([&](auto c0) {
                constexpr auto TapPerIter = wconv_conv.GetNumWeightTapPerWave();
                const WeiDataVec* weight_thread_vec_ptr[4];
                typename decltype(wconv_conv)::WeiDataVec weight_thread_vec[4];
                // Load weights
                if constexpr(WeiDataEnableLds)
                {
                    static_for<0, NumYX, TapPerIter>{}([&](auto i) {
                        weight_thread_copy_.Run(weight_wave_desc_,
                                                make_tuple(k0, c0, i, I0, I0, I0),
                                                weight_block_buf,
                                                weight_thread_desc_,
                                                make_tuple(I0, I0, I0, I0, I0, I0),
                                                weight_thread_buf);
                        if constexpr(Transposed)
                        {
                            weight_thread_vec[i / TapPerIter].template AsType<WeiDataVec>()(I0) =
                                *reinterpret_cast<const WeiDataVec*>(&(weight_thread_buf[I0]));
                            weight_thread_vec_ptr[i / TapPerIter] =
                                &weight_thread_vec[i / TapPerIter]
                                     .template AsType<WeiDataVec>()[I0];
                        }
                        else
                        {
                            weight_thread_vec[0].template AsType<WeiDataTapVec>()(
                                i / Number<TapPerIter>{}) =
                                *reinterpret_cast<const WeiDataTapVec*>(&(weight_thread_buf[I0]));
                            weight_thread_vec_ptr[0] =
                                &weight_thread_vec[0].template AsType<WeiDataVec>()[I0];
                        }
                    });
                }
                else
                {
                    if constexpr(Transposed)
                    {
                        static_for<0, NumYX, TapPerIter>{}([&](auto i) {
                            constexpr index_t wei_offset = weight_wave_desc_.CalculateOffset(
                                make_tuple(k0, c0, i, I0, I0, I0));
                            weight_thread_vec_ptr[i / TapPerIter] =
                                reinterpret_cast<const WeiDataVec*>(
                                    &weight_block_buf[Number<wei_offset>{}]);
                        });
                    }
                    else
                    {
                        constexpr index_t wei_offset =
                            weight_wave_desc_.CalculateOffset(make_tuple(k0, c0, I0, I0, I0, I0));
                        if constexpr(wconv_conv.GetNumWeightTapPerWave() == 2)
                        {
                            constexpr index_t wei_offset2 = weight_wave_desc_.CalculateOffset(
                                make_tuple(k0, c0, I2, I0, I0, I0));
                            weight_thread_vec[0].template AsType<WeiDataTapVec>()(I0) =
                                *reinterpret_cast<const WeiDataTapVec*>(
                                    &weight_block_buf[Number<wei_offset>{}]);
                            weight_thread_vec[0].template AsType<WeiDataTapVec>()(I1) =
                                *reinterpret_cast<const WeiDataTapVec*>(
                                    &weight_block_buf[Number<wei_offset2>{}]);
                            weight_thread_vec_ptr[0] =
                                &weight_thread_vec[0].template AsType<WeiDataVec>()[I0];
                        }
                        else
                        {
                            weight_thread_vec_ptr[0] = reinterpret_cast<const WeiDataVec*>(
                                &weight_block_buf[Number<wei_offset>{}]);
                        }
                    }
                }

                InDataVec indata_thread_vec[4];
                static_for<0, HRepeat, HStep>{}([&](auto h0) {
                    static_for<0, WRepeat, WStep>{}([&](auto w0) {
                        if constexpr(InDataEnableLds)
                        {
                            // Load input tensor data
                            if constexpr(Transposed == false)
                            {
                                static_assert(Iters == NumYX, "");
                                InDataVec indata_thread_vec_tmp[4];
                                static_for<0, NumYX, 1>{}([&](auto i) {
                                    auto indata_thread_buf =
                                        make_static_buffer_v3<AddressSpaceEnum::Vgpr, InDataType>(
                                            indata_thread_desc_.GetElementSpaceSize(),
                                            (InDataType*)(&indata_thread_vec_tmp[i]));
                                    indata_thread_copy_.Run(indata_wave_desc_,
                                                            make_tuple(Number<w0 + x_offset[i]>{},
                                                                       c0,
                                                                       Number<h0 + y_offset[i]>{},
                                                                       I0,
                                                                       I0,
                                                                       I0,
                                                                       I0),
                                                            indata_block_buf,
                                                            indata_thread_desc_,
                                                            make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                                            indata_thread_buf);
                                    indata_thread_vec_ptr[i] = &indata_thread_vec[i];
                                });
                                wconv_conv.UnshuffleStridedConv2Data(indata_thread_vec_tmp,
                                                                     indata_thread_vec);
                            }
                            else
                            {
                                auto indata_thread_buf =
                                    make_static_buffer_v3<AddressSpaceEnum::Vgpr, InDataType>(
                                        indata_thread_desc_.GetElementSpaceSize(),
                                        (InDataType*)(&indata_thread_vec[I0]));
                                indata_thread_copy_.Run(indata_wave_desc_,
                                                        make_tuple(w0, c0, h0, I0, I0, I0, I0),
                                                        indata_block_buf,
                                                        indata_thread_desc_,
                                                        make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                                        indata_thread_buf);
                                indata_thread_vec_ptr[I0] = &indata_thread_vec[I0];
                            }
                        }
                        else
                        {
                            // Load input tensor data
                            if constexpr(Transposed == false)
                            {
                                InDataVec indata_thread_vec_tmp[4];
                                static_for<0, NumYX, 1>{}([&](auto i) {
                                    constexpr index_t indata_offset =
                                        indata_wave_desc_.CalculateOffset(
                                            make_tuple(w0 + x_offset[i],
                                                       c0,
                                                       h0 + y_offset[i],
                                                       I0,
                                                       I0,
                                                       I0,
                                                       I0));
                                    indata_thread_vec_tmp[i] = *reinterpret_cast<const InDataVec*>(
                                        &indata_block_buf[Number<indata_offset>{}]);
                                    indata_thread_vec_ptr[i] = &indata_thread_vec[i];
                                });
                                wconv_conv.UnshuffleStridedConv2Data(indata_thread_vec_tmp,
                                                                     indata_thread_vec);
                            }
                            else
                            {
                                constexpr index_t indata_offset = indata_wave_desc_.CalculateOffset(
                                    make_tuple(w0, c0, h0, I0, I0, I0, I0));
                                indata_thread_vec_ptr[I0] = reinterpret_cast<const InDataVec*>(
                                    &indata_block_buf[Number<indata_offset>{}]);
                            }
                        }

                        if(Transposed == false)
                        {
                            constexpr index_t accum_offset = accum_thread_desc_.CalculateOffset(
                                make_tuple(Number<h0 / HStep>{}, Number<w0 / WStep>{}, k0, I0));
                            wconv_conv.wconv_instr.Run(
                                *weight_thread_vec_ptr[0],
                                indata_thread_vec_ptr,
                                accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{}),
                                I0);
                        }
                        else
                        {
                            static_for<0, NumYX, 1>{}([&](auto i) {
                                constexpr index_t tapIdx = i / wconv_conv.GetNumWeightTapPerWave();
                                constexpr bool isHigh    = i % wconv_conv.GetNumWeightTapPerWave();
                                constexpr index_t accum_offset = accum_thread_desc_.CalculateOffset(
                                    make_tuple(Number<h0 * 2 + y_offset[i]>{},
                                               Number<w0 * 2 + x_offset[i]>{},
                                               k0,
                                               I0));
                                constexpr auto Mod = []() {
                                    if constexpr(isHigh)
                                        return I1;
                                    else
                                        return I0;
                                }();
                                wconv_conv.wconv_instr.Run(
                                    *weight_thread_vec_ptr[tapIdx],
                                    indata_thread_vec_ptr,
                                    accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{}),
                                    Mod);
                            });
                        }
                    });
                });
            });
        });

        if constexpr(Transposed)
        {
            if(isLast)
            {
                using AccDataVec = typename decltype(wconv_conv)::AccDataVec::type;
                static_for<0, KRepeat, 1>{}([&](auto k0) {
                    static_for<0, HRepeat, 1>{}([&](auto h0) {
                        static_for<0, WRepeat, 1>{}([&](auto w0) {
                            // shuffle.
                            AccDataVec* accdata_thread_vec_ptr[NumYX];
                            static_for<0, NumYX, 1>{}([&](auto i) {
                                constexpr index_t accum_offset = accum_thread_desc_.CalculateOffset(
                                    make_tuple(Number<h0 * 2 + y_offset[i]>{},
                                               Number<w0 * 2 + x_offset[i]>{},
                                               k0,
                                               I0));
                                accdata_thread_vec_ptr[i] =
                                    &accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{})
                                         .template AsType<AccDataVec>()(Number<0>{});
                            });
                            wconv_conv.ShuffleConv2TransposedData(accdata_thread_vec_ptr);
                        });
                    });
                });
            }
        }
    }

    template <typename WeightBlockBuffer, typename InDataBlockBuffer, typename AccumThreadBuffer>
    __device__ void RunConv(const WeightBlockBuffer& weight_block_buf,
                            const InDataBlockBuffer& indata_block_buf,
                            AccumThreadBuffer& accum_thread_buf,
                            bool) const
    {
        auto weight_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, WeiDataType>(
            weight_thread_desc_.GetElementSpaceSize());
        using WeiDataVec    = typename decltype(wconv_conv)::WeiDataVec::type;
        using WeiDataTapVec = typename decltype(wconv_conv)::WeiDataTapVec::type;
        using InDataVec     = typename decltype(wconv_conv)::InDataVec::type;

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
                            constexpr index_t indata_offset = indata_wave_desc_.CalculateOffset(
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
        constexpr index_t CStep = Iters;

        static_for<0, KRepeat, 1>{}([&](auto k0) {
            static_for<0, CRepeat, CStep>{}([&](auto c0) {
                const WeiDataVec* weight_thread_vec_ptr;
                typename decltype(wconv_conv)::WeiDataVec weight_thread_vec;
                // Load weights
                if constexpr(WeiDataEnableLds)
                {
                    if constexpr(FilterSize == 1)
                    {
                        constexpr auto TapPerIter = wconv_conv.GetNumWeightTapPerWave();
                        static_for<0, Iters, TapPerIter>{}([&](auto i) {
                            weight_thread_copy_.Run(
                                weight_wave_desc_,
                                make_tuple(k0, Number<c0 + i>{}, I0, I0, I0, I0),
                                weight_block_buf,
                                weight_thread_desc_,
                                make_tuple(I0, I0, I0, I0, I0, I0),
                                weight_thread_buf);
                            weight_thread_vec.template AsType<WeiDataTapVec>()(
                                i / Number<TapPerIter>{}) =
                                *reinterpret_cast<const WeiDataTapVec*>(&(weight_thread_buf[I0]));
                        });
                        weight_thread_vec_ptr =
                            &weight_thread_vec.template AsType<WeiDataVec>()[I0];
                    }
                    else if constexpr(FilterSize == 3)
                    {
                        static_assert(Iters == 1, "");
                        constexpr index_t WeightTapPerWave = wconv_conv.GetNumWeightTapPerWave();
                        static_for<0, wconv_conv.GetNumWeightTap(), 1>{}([&](auto tape_idx) {
                            weight_thread_copy_.Run(
                                weight_wave_desc_,
                                make_tuple(
                                    k0, c0, Number<WeightTapPerWave * tape_idx>{}, I0, I0, I0),
                                weight_block_buf,
                                weight_thread_desc_,
                                make_tuple(I0, I0, I0, I0, I0, I0),
                                weight_thread_buf);

                            weight_thread_vec.template AsType<WeiDataTapVec>()(Number<tape_idx>{}) =
                                *reinterpret_cast<const WeiDataTapVec*>(&weight_thread_buf[I0]);
                        });
                        weight_thread_vec_ptr =
                            &weight_thread_vec.template AsType<WeiDataVec>()[I0];
                    }
                }
                else
                {
                    constexpr index_t wei_offset =
                        weight_wave_desc_.CalculateOffset(make_tuple(k0, c0, I0, I0, I0, I0));
                    if constexpr((Iters == 4) && (wconv_conv.GetNumWeightTapPerWave() == 2))
                    {

                        constexpr index_t wei_offset2 = weight_wave_desc_.CalculateOffset(
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
                                        indata_wave_desc_,
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
                                        indata_wave_desc_,
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
                                        indata_wave_desc_.CalculateOffset(
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
                                        indata_wave_desc_.CalculateOffset(
                                            make_tuple(w0 + array_idx, c0, h0, I0, I0, I0, I0));
                                    indata_thread_vec_ptr[array_idx] =
                                        reinterpret_cast<const InDataVec*>(
                                            &indata_block_buf[Number<indata_offset>{}]);
                                });
                            }
                        }

                        constexpr index_t accum_offset = accum_thread_desc_.CalculateOffset(
                            make_tuple(Number<h0>{}, Number<w0>{}, k0, I0));
                        wconv_conv.wconv_instr.Run(
                            *weight_thread_vec_ptr,
                            indata_thread_vec_ptr,
                            accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{}),
                            I0);
                    });
                });
            });
        });
    }

    template <typename WeightBlockBuffer, typename InDataBlockBuffer, typename AccumThreadBuffer>
    __device__ void Run(const WeightBlockBuffer& weight_block_buf,
                        const InDataBlockBuffer& indata_block_buf,
                        AccumThreadBuffer& accum_thread_buf,
                        bool isLast) const
    {
        if constexpr(FilterSize == 2)
        {
            RunEmulateConv2(weight_block_buf, indata_block_buf, accum_thread_buf, isLast);
        }
        else
        {
            RunConv(weight_block_buf, indata_block_buf, accum_thread_buf, isLast);
        }
    }

    protected:
    // Thread descriptor for weight and input data
    static constexpr auto weight_thread_desc_ = make_naive_tensor_descriptor_packed(make_tuple(
        I1, I1, I1, I1, Number<NumSubTilesPerWeightTap>{}, Number<NumWeightCompPerTile>{}));

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

    // C[H, W, K, NumAccComp]
    static constexpr auto accum_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<HRepeatOut>{},
                   Number<WRepeatOut>{},
                   Number<KRepeat>{},
                   Number<wconv_conv.GetNumAccumComponents()>{}));

    // Initialize thread copy classes
    using WeiDataThreadLdsCopy = ThreadwiseTensorSliceTransfer_v4<
        WeiDataType,
        WeiDataType,
        decltype(weight_wave_desc_),
        decltype(weight_thread_desc_),
        Sequence<1, 1, 1, 1, NumSubTilesPerWeightTap, NumWeightCompPerTile>,
        Sequence<0, 1, 2, 3, 4, 5>,
        5,
        NumWeightCompPerTile,
        NumWeightCompPerTile>;

    WeiDataThreadLdsCopy weight_thread_copy_;

    using InDataThreadLdsCopy = ThreadwiseTensorSliceTransfer_v4<
        InDataType,
        InDataType,
        decltype(indata_wave_desc_),
        decltype(indata_thread_desc_),
        Sequence<1, 1, NumDataTiles, NumDataSubTiles, 1, 1, NumDataCompPerTile>,
        Sequence<0, 1, 2, 3, 4, 5, 6>,
        6,
        NumDataCompPerTile,
        NumDataCompPerTile>;
    InDataThreadLdsCopy indata_thread_copy_;
};
} // namespace ck
