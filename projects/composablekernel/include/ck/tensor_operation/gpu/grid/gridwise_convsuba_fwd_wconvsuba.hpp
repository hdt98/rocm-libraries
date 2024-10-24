// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_conv_suba_pipeline.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_convsuba_wconvsuba.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_async_load.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseOp,
          typename InDataType,
          typename WeiDataType,
          typename DsPointer,
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename AccElementwiseOperation,
          typename InGridDesc,
          typename WeiGridDesc,
          typename DsGridDesc,
          typename AccGridDesc,
          typename Block2CTileMap,
          typename ComputePtrOffsetOfBatch,
          bool HasMainBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_grouped_convsuba_wconvsuba(const InDataType* __restrict__ p_in_grid,
                                          const WeiDataType* __restrict__ p_wei_grid,
                                          DsPointer p_ds_grid,
                                          AccDataType* __restrict__ p_acc_grid,
                                          const InElementwiseOperation in_element_op,
                                          const WeiElementwiseOperation wei_element_op,
                                          const AccElementwiseOperation acc_element_op,
                                          const index_t batch_count,
                                          const InGridDesc in_grid_desc,
                                          const WeiGridDesc wei_grid_desc,
                                          const DsGridDesc ds_grid_desc,
                                          const AccGridDesc acc_grid_desc,
                                          const Block2CTileMap block_2_ctile_map,
                                          const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx13__))
    // offset base pointer for each work-group
    static constexpr index_t NumDTensor = 2;
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t in_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
    const long_index_t wei_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
    const long_index_t acc_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx)));

    const auto ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);

    __shared__ char p_shared[GridwiseOp::SharedMemTrait::lds_size];

    DsPointer p_ds_grid_grp;
    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = p_ds_grid[i] + ds_batch_offset[i]; });

    GridwiseOp::template Run<HasMainBlockLoop>(p_in_grid + in_batch_offset,
                                               p_wei_grid + wei_batch_offset,
                                               p_ds_grid_grp,
                                               p_acc_grid + acc_batch_offset,
                                               p_shared,
                                               nullptr,
                                               in_grid_desc,
                                               wei_grid_desc,
                                               ds_grid_desc,
                                               acc_grid_desc,
                                               in_element_op,
                                               wei_element_op,
                                               acc_element_op,
                                               block_2_ctile_map);
#else
    ignore                                = p_in_grid;
    ignore                                = p_wei_grid;
    ignore                                = p_ds_grid_grp;
    ignore                                = p_acc_grid;
    ignore                                = in_grid_desc;
    ignore                                = wei_grid_desc;
    ignore                                = ds_grid_desc;
    ignore                                = acc_grid_desc;
    ignore                                = in_element_op;
    ignore                                = wei_element_op;
    ignore                                = acc_element_op;
    ignore                                = compute_ptr_offset_of_batch;
    ignore                                = block_2_ctile_map;
#endif
}

template <typename GridwiseOp,
          typename InDataType,
          typename WeiDataType,
          typename DsPointer,
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename AccElementwiseOperation,
          typename InGridDesc,
          typename WeiGridDesc,
          typename DsGridDesc,
          typename AccGridDesc,
          typename Block2CTileMap,
          typename ComputePtrOffsetOfBatch,
          bool HasMainBlockLoop>
__global__ void __exp_amd_wavegroup_kernel(4, 32, 256, 1, 1)
    kernel_grouped_convsuba_wconvsuba_wavegroup(
        const InDataType* __restrict__ p_in_grid,
        const WeiDataType* __restrict__ p_wei_grid,
        DsPointer p_ds_grid,
        AccDataType* __restrict__ p_acc_grid,
        const InElementwiseOperation in_element_op,
        const WeiElementwiseOperation wei_element_op,
        const AccElementwiseOperation acc_element_op,
        const index_t batch_count,
        const InGridDesc in_grid_desc,
        const WeiGridDesc wei_grid_desc,
        const DsGridDesc ds_grid_desc,
        const AccGridDesc acc_grid_desc,
        const Block2CTileMap block_2_ctile_map,
        const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx13__))
    // offset base pointer for each work-group
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t in_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
    const long_index_t wei_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
    const long_index_t acc_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx)));

    const long_index_t ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);
    DsPointer p_ds_grid_grp;
    static constexpr index_t NumDTensor = DsGridDesc::Size();
    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = p_ds_grid[i] + ds_batch_offset[i]; });

    __shared__ char p_shared[GridwiseOp::SharedMemTrait::lds_size];
    static __exp_amd_laneshared__ char
        p_lane_shared[GridwiseOp::LaneSharedMemTrait::lane_shared_size *
                      GridwiseOp::template GetLaneSharedMemCount<HasMainBlockLoop>()];

    static_assert(GridwiseOp::LaneSharedMemTrait::lane_shared_size <= 512 * 4, "");

    GridwiseOp::template Run<HasMainBlockLoop>(p_in_grid + in_batch_offset,
                                               p_wei_grid + wei_batch_offset,
                                               p_ds_grid_grp,
                                               p_acc_grid + acc_batch_offset,
                                               p_shared,
                                               p_lane_shared,
                                               in_grid_desc,
                                               wei_grid_desc,
                                               ds_grid_desc,
                                               acc_grid_desc,
                                               in_element_op,
                                               wei_element_op,
                                               acc_element_op,
                                               block_2_ctile_map);
#else
    ignore                                = p_in_grid;
    ignore                                = p_wei_grid;
    ignore                                = p_ds_grid_grp;
    ignore                                = p_acc_grid;
    ignore                                = in_grid_desc;
    ignore                                = wei_grid_desc;
    ignore                                = ds_grid_desc;
    ignore                                = acc_grid_desc;
    ignore                                = in_element_op;
    ignore                                = wei_element_op;
    ignore                                = acc_element_op;
    ignore                                = compute_ptr_offset_of_batch;
    ignore                                = block_2_ctile_map;
#endif
}

template <index_t BlockSize,
          typename InDataType,
          typename WeiDataType,
          typename DsDataType,
          typename AccDataType,
          typename InGridDesc,
          typename WeiGridDesc,
          typename DsGridDesc,
          typename AccGridDesc,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename AccElementwiseOperation,
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
          bool scaleBiasPacked,
          bool uniformScale,
          typename InBlockTransferThreadClusterLengths,
          index_t InBlockTransferSrcScalarPerVector,
          index_t InBlockTransferDstScalarPerVector,
          bool InEnableLds,
          bool InBlockLdsAddExtraM,
          typename WeiBlockTransferThreadClusterLengths,
          index_t WeiBlockTransferSrcScalarPerVector,
          index_t WeiBlockTransferDstScalarPerVector,
          bool WeiEnableLds,
          bool WeiBlockLdsAddExtraM,
          typename DsBlockTransferThreadClusterLengths,
          index_t DsBlockTransferSrcScalarPerVector,
          index_t DsBlockTransferDstScalarPerVector,
          bool DsEnableLds,
          bool DsBlockLdsAddExtraM,
          typename AccBlockTransferClusterLengths,
          index_t AccBlockTransferScalarPerVector,
          bool AccEnableLds,
          bool EnableAsync,
          index_t NumConvCPrefetchStage,
          bool EnableWaveGroup>
struct GridwiseConvSuba_Wconvsuba
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    static constexpr index_t WaveSize     = 32;
    static constexpr index_t NumWaveGroup = EnableWaveGroup ? 4 : 0;

    static constexpr index_t DsBlockSize = KPerBlock / DsBlockTransferSrcScalarPerVector;

    static constexpr index_t NumDTensor = DsDataType::Size();

    static_assert((EnableWaveGroup == false) || (BlockSize % (WaveSize * 4) == 0), "");

    using DsThreadBlock =
        typename std::conditional<EnableWaveGroup,
                                  ThisThreadBlockWaveGroup<DsBlockSize, WaveSize, NumWaveGroup>,
                                  ThisThreadBlock<DsBlockSize>>::type;

    using ThisThreadBlock =
        typename std::conditional<EnableWaveGroup,
                                  ThisThreadBlockWaveGroup<BlockSize, WaveSize, NumWaveGroup>,
                                  ThisThreadBlock<BlockSize>>::type;

    static constexpr index_t YX = FilterSize * FilterSize;
    using NumberYX              = Number<FilterSize * FilterSize>;

    static constexpr auto wconv_conv = WconvConv<WeiDataType,
                                                 InDataType,
                                                 AccDataType,
                                                 HPerWconv,
                                                 WPerWconv,
                                                 FilterSize,
                                                 DilationX,
                                                 DilationY,
                                                 1,
                                                 EnableWaveGroup>{};

    static constexpr auto acc_sba =
        AccSba<AccDataType, HPerWconv, WPerWconv, activeFun, scaleBiasPacked, uniformScale>{};

    using GridwiseConvPipe =
        remove_cvref_t<decltype(GridwiseConvPipeline_Selector<NumConvCPrefetchStage,
                                                              InEnableLds,
                                                              WeiEnableLds,
                                                              DsEnableLds,
                                                              EnableAsync,
                                                              EnableWaveGroup>())>;

    static constexpr index_t CPerWconv               = wconv_conv.GetNumInputChannels();
    static constexpr index_t KPerWconv               = wconv_conv.GetNumOutputChannels();
    static constexpr index_t NumWeightTap            = wconv_conv.GetNumWeightTap();
    static constexpr index_t NumSubTilesPerWeightTap = wconv_conv.GetNumSubTilesPerWeightTap();
    static constexpr index_t NumWeightCompPerTile    = wconv_conv.GetNumWeightCompPerTile();
    static constexpr index_t NumSubTilePerImage      = wconv_conv.GetNumSubTilesPerImageTile();
    static constexpr index_t NumDataCompPerTile      = wconv_conv.GetNumDataCompPerTile();
    static constexpr index_t DataTileHeight          = 4;
    static constexpr index_t H_Pad                   = (FilterSize == 3) ? DataTileHeight : 0;
    static constexpr index_t W_Pad                   = (FilterSize == 3) ? WPerWconv : 0;
    static constexpr index_t HPerBlockIn             = HPerBlock + H_Pad * 2;
    static constexpr index_t WPerBlockIn             = WPerBlock + W_Pad * 2;

    static constexpr index_t HPerWave   = HRepeat * HPerWconv;
    static constexpr index_t WPerWave   = WRepeat * WPerWconv;
    static constexpr index_t CPerWave   = CPerBlock;
    static constexpr index_t KPerWave   = KPerBlock;
    static constexpr index_t HPerWaveIn = HPerWave + H_Pad * 2;
    static constexpr index_t WPerWaveIn = WPerWave + W_Pad * 2;
    static_assert(HPerWaveIn % HPerWconv == 0, "");
    static_assert(WPerWaveIn % WPerWconv == 0, "");

    // Pad input and weight data grid description according to grid level options
    __host__ __device__ static constexpr auto
    MakeInGridPadDescriptor(const InGridDesc& in_grid_desc)
    {
        const auto in_grid_pad_desc = [&]() {
            if constexpr(FilterSize == 3)
            {
                const auto Hi = in_grid_desc.GetLength(I0);
                const auto Wi = in_grid_desc.GetLength(I1);
                const auto Ci = in_grid_desc.GetLength(I2);

                return transform_tensor_descriptor(
                    in_grid_desc,
                    make_tuple(make_pad_transform(Hi, H_Pad, H_Pad),
                               make_pad_transform(Wi, W_Pad, W_Pad),
                               make_pass_through_transform(Ci)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
            }
            else
            {
                return in_grid_desc;
            }
        }();

        if constexpr(InEnableLds)
        {
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

    __host__ __device__ static constexpr auto
    MakeWeiGridPadDescriptor(const WeiGridDesc& wei_grid_desc)
    {
        if constexpr(WeiEnableLds)
        {
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

    __host__ __device__ static constexpr auto
    MakeDsGridPadDescriptor(const DsGridDesc& ds_grid_desc)
    {
        if constexpr(DsEnableLds)
        {
            return ds_grid_desc;
        }
        else
        {
            // K0 x K1 X K2
            constexpr auto DsPerThread = acc_sba.GetNumBiasComponents();
            return generate_tuple(
                [&](auto i) {
                    const auto K = ds_grid_desc[i].GetLength(I0);
                    return transform_tensor_descriptor(ds_grid_desc[i],
                                                       make_tuple(make_unmerge_transform(make_tuple(
                                                           K / KPerWconv,
                                                           Number<KPerWconv / DsPerThread>{},
                                                           Number<DsPerThread>{}))),
                                                       make_tuple(Sequence<0>{}),
                                                       make_tuple(Sequence<0, 1, 2>{}));
                },
                Number<NumDTensor>{});
        }
    }

    // Describe how data store to (LDS/VGPR) buffer from Global memory
    __host__ __device__ static constexpr auto MakeInBlockDescriptor()
    {
        constexpr auto in_block_desc = [&]() {
            if constexpr(InEnableLds)
            {
                // H x W x C Per Block
                return make_naive_tensor_descriptor_packed(
                    make_tuple(Number<HPerBlockIn>{}, Number<WPerBlockIn>{}, Number<CPerBlock>{}));
            }
            else
            {
                // W0 x C0 x H0 x H1 x H2 x W1 x C1
                return make_naive_tensor_descriptor_packed(
                    make_tuple(Number<WPerWaveIn / WPerWconv>{},
                               Number<CPerWave / CPerWconv>{},
                               Number<HPerWaveIn / HPerWconv>{},
                               Number<NumSubTilePerImage>{},
                               I1,
                               I1,
                               Number<NumDataCompPerTile>{}));
            }
        }();

        return in_block_desc;
    }

    __host__ __device__ static constexpr auto MakeWeiBlockDescriptor()
    {
        constexpr auto wei_block_desc = [&]() {
            if constexpr(WeiEnableLds)
            {
                // K x YX x C per block
                return make_naive_tensor_descriptor_packed(make_tuple(
                    Number<KPerBlock>{}, Number<FilterSize * FilterSize>{}, Number<CPerBlock>{}));
            }
            else
            {
                // K0 x C0 x YX x K1 x C1 x C2
                return make_naive_tensor_descriptor_packed(
                    make_tuple(Number<KPerWave / KPerWconv>{},
                               Number<CPerWave / CPerWconv>{},
                               Number<NumWeightTap>{},
                               I1,
                               Number<NumSubTilesPerWeightTap>{},
                               Number<NumWeightCompPerTile>{}));
            }
        }();

        return wei_block_desc;
    }

    // Describe how data store to LDS buffer
    __host__ __device__ static constexpr auto GetAccBlockDescriptor()
    {
        constexpr auto acc_block_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<HPerBlock>{}, Number<WPerBlock>{}, Number<KPerBlock>{}));

        return acc_block_desc;
    }

    // Ds desc for source in blockwise copy
    __host__ __device__ static constexpr auto MakeDsBlockDescriptor()
    {
        constexpr auto ds_block_desc = [&]() {
            if constexpr(DsEnableLds)
            {
                return generate_tuple(
                    [&](auto i) {
                        return make_naive_tensor_descriptor_packed(make_tuple(Number<KPerBlock>{}));
                    },
                    Number<NumDTensor>{});
            }
            else
            {
                // KRepeat x K1 x K2
                constexpr auto DsPerThread = acc_sba.GetNumBiasComponents();
                return generate_tuple(
                    [&](auto i) {
                        return make_naive_tensor_descriptor_packed(
                            make_tuple(Number<KPerWave / KPerWconv>{},
                                       Number<KPerWconv / DsPerThread>{},
                                       Number<DsPerThread>{}));
                    },
                    Number<NumDTensor>{});
            }
        }();

        return ds_block_desc;
    }

    __host__ __device__ static constexpr auto MakeInBlockSliceCopyStep()
    {
        constexpr auto in_block_copy_step = [&]() {
            if constexpr(InEnableLds)
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

    __host__ __device__ static constexpr auto MakeWeiBlockSliceCopyStep()
    {
        constexpr auto wei_block_copy_step = [&]() {
            if constexpr(WeiEnableLds)
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

    // Describe how data read from (LDS/VGPR) buffer, used by Block level classes
    template <typename InBlockDesc_>
    __host__ __device__ static constexpr auto MakeInWaveDescriptor(const InBlockDesc_&)
    {
        constexpr auto in_wave_desc = [&]() {
            if constexpr(InEnableLds)
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

    template <typename WeiBlockDesc_>
    __host__ __device__ static constexpr auto MakeWeiWaveDescriptor(const WeiBlockDesc_&)
    {
        constexpr auto wei_wave_desc = [&]() {
            if constexpr(WeiEnableLds)
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

    // Describe how data read from (LDS/VGPR) buffer, used by Block level classes
    template <typename DsBlockDesc_>
    __host__ __device__ static constexpr auto MakeDsWaveDescriptor(const DsBlockDesc_&)
    {
        constexpr auto DsPerThread  = acc_sba.GetNumBiasComponents();
        constexpr auto ds_wave_desc = [&]() {
            if constexpr(DsEnableLds)
            {
                return generate_tuple(
                    [&](auto i) {
                        return transform_tensor_descriptor(
                            DsBlockDesc_{}[Number<i>{}],
                            make_tuple(
                                make_unmerge_transform(make_tuple(Number<KPerBlock / KPerWconv>{},
                                                                  Number<KPerWconv / DsPerThread>{},
                                                                  Number<DsPerThread>{}))),
                            make_tuple(Sequence<0>{}),
                            make_tuple(Sequence<0, 1, 2>{}));
                    },
                    Number<NumDTensor>{});
            }
            else
            {
                return DsBlockDesc_{};
            }
        }();

        return ds_wave_desc;
    }

    template <typename AccGridDec, typename AccThreadBuffer, typename BlockWiseConv>
    __host__ __device__ static void StoreAccData(const AccGridDec& acc_grid_desc,
                                                 AccDataType* __restrict__ p_acc_grid,
                                                 AccThreadBuffer& acc_thread_buf,
                                                 BlockWiseConv& blockwise_conv,
                                                 const AccElementwiseOperation& acc_element_op,
                                                 void* __restrict__ p_shared,
                                                 void* __restrict__,
                                                 index_t h_block_data_idx_on_grid,
                                                 index_t w_block_data_idx_on_grid,
                                                 index_t k_block_data_idx_on_grid)
    {
        auto acc_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_acc_grid, acc_grid_desc.GetElementSpaceSize());

        // C mapping in single thread.
        constexpr auto acc_thread_desc   = blockwise_conv.GetAccThreadDescriptor();
        constexpr auto acc_thread_length = blockwise_conv.GetAccThreadDescLength();

        // calculate origin of thread output tensor on global memory
        // blockwise conv acc starting index
        const auto acc_thread_mtx_on_block = blockwise_conv.CalculateAccThreadOriginDataIndex();

        if constexpr(AccEnableLds == false)
        {
            const auto acc_grid_wave_desc = blockwise_conv.GetAccBlockWaveDescriptor(acc_grid_desc);

            // Threadwise copy C from VGPR to global memory
            auto acc_thread_copy_vgpr_to_global =
                ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                   AccDataType,
                                                   decltype(acc_thread_desc),
                                                   decltype(acc_grid_wave_desc),
                                                   AccElementwiseOperation,
                                                   decltype(acc_thread_length),
                                                   Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                   7,
                                                   acc_thread_length[I7], // vector write pixel
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>{
                    acc_grid_wave_desc,
                    acc_thread_mtx_on_block + make_multi_index(h_block_data_idx_on_grid / HPerWconv,
                                                               w_block_data_idx_on_grid / WPerWconv,
                                                               k_block_data_idx_on_grid / KPerWconv,
                                                               I0,
                                                               I0,
                                                               I0,
                                                               I0,
                                                               I0),
                    acc_element_op};

            // each thread write its data from VGPR to global
            acc_thread_copy_vgpr_to_global.Run(acc_thread_desc,
                                               make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                               acc_thread_buf,
                                               acc_grid_wave_desc,
                                               acc_grid_buf);
        }
        else
        {
            // C mapping in single block
            // LDS descriptor, shuffle and write out in HRepeat x WRepeat x KRepeat times
            constexpr auto acc_block_desc = GetAccBlockDescriptor();
            constexpr auto acc_block_wave_desc =
                blockwise_conv.GetAccBlockWaveDescriptor(acc_block_desc);

            auto acc_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<AccDataType*>(p_shared) + SharedMemTrait::acc_block_space_offset,
                SharedMemTrait::acc_block_space_size);

            // Threadwise copy C from VGPR to LDS
            auto acc_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                   AccDataType,
                                                   decltype(acc_thread_desc),
                                                   decltype(acc_block_wave_desc),
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   decltype(acc_thread_length),
                                                   Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                   7,
                                                   acc_thread_length[I7], // vector write pixel
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>{
                    acc_block_wave_desc,
                    acc_thread_mtx_on_block,
                    ck::tensor_operation::element_wise::PassThrough{}};

            // blockwise copy C from LDS to global
            auto acc_block_copy_lds_to_global =
                ThreadGroupTensorSliceTransfer_v6r1<ThisThreadBlock,
                                                    AccElementwiseOperation,
                                                    InMemoryDataOperationEnum::Set,
                                                    Sequence<HPerBlock, WPerBlock, KPerBlock>,
                                                    AccBlockTransferClusterLengths,
                                                    Sequence<0, 1, 2>,
                                                    AccDataType,
                                                    AccDataType,
                                                    decltype(acc_block_desc),
                                                    decltype(acc_grid_desc),
                                                    Sequence<0, 1, 2>,
                                                    2,
                                                    AccBlockTransferScalarPerVector,
                                                    true,
                                                    false>{
                    acc_block_desc,
                    make_multi_index(0, 0, 0),
                    acc_grid_desc,
                    make_multi_index(h_block_data_idx_on_grid,
                                     w_block_data_idx_on_grid,
                                     k_block_data_idx_on_grid),
                    acc_element_op};

            // make sure it's safe to write to LDS
            block_sync_lds();

            // each thread write its data from VGPR to LDS
            acc_thread_copy_vgpr_to_lds.Run(acc_thread_desc,
                                            make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                            acc_thread_buf,
                                            acc_block_wave_desc,
                                            acc_block_buf);

            // make sure it's safe to read from LDS
            block_sync_lds();

            // each block copy its data from LDS to global
            acc_block_copy_lds_to_global.Run(
                acc_block_desc, acc_block_buf, acc_grid_desc, acc_grid_buf);
        }
    }

    // ck::Tuple<const D0DataType*, const D1DataType*, ...>
    static constexpr auto MakeDsGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                return static_cast<const DDataType*>(nullptr);
            },
            Number<NumDTensor>{});
    }

    using DsGridPointer = decltype(MakeDsGridPointer());

    template <typename Block2CTileMap>
    __host__ __device__ static constexpr bool CheckValidity(const InGridDesc& in_grid_desc,
                                                            const WeiGridDesc& wei_grid_desc,
                                                            const DsGridDesc& ds_grid_desc,
                                                            const AccGridDesc& acc_grid_desc,
                                                            const Block2CTileMap& block_2_ctile_map)
    {
        static_assert(HPerBlock % (HRepeat * HPerWconv) == 0, "");
        static_assert(WPerBlock % (WRepeat * WPerWconv) == 0, "");

        const auto GetInProblemsize = [&]() {
            return make_tuple(
                in_grid_desc.GetLength(I0), in_grid_desc.GetLength(I1), in_grid_desc.GetLength(I2));
        };

        const auto GetWeiProblemsize = [&]() {
            return make_tuple(wei_grid_desc.GetLength(I0),
                              wei_grid_desc.GetLength(I1),
                              wei_grid_desc.GetLength(I2));
        };

        const auto Ho = (FilterSize == 2) ? GetInProblemsize()[I0] / 2 : GetInProblemsize()[I0];
        const auto Wo = (FilterSize == 2) ? GetInProblemsize()[I1] / 2 : GetInProblemsize()[I1];
        const auto H  = (FilterSize == 2) ? GetInProblemsize()[I0] / 2 : GetInProblemsize()[I0];
        const auto W  = (FilterSize == 2) ? GetInProblemsize()[I1] / 2 : GetInProblemsize()[I1];
        const auto C  = GetInProblemsize()[I2];
        const auto K  = GetWeiProblemsize()[I0];

        bool valid = true;
        static_for<0, NumDTensor, 1>{}(
            [&](auto i) { valid = valid && (K == ds_grid_desc[i].GetLength(I0)); });
        if(!valid)
        {
            printf("GridwiseOp: D descriptor dimension check failure\n");
            return false;
        }

        if(!(Ho == acc_grid_desc.GetLength(I0) && Wo == acc_grid_desc.GetLength(I1) &&
             K == acc_grid_desc.GetLength(I2)) ||
           !(C == GetWeiProblemsize()[I2]))
        {
            printf("Tensor: HWC = %d x %d x %d, Filter: KXYC = %d x {%d, %d} x %d, Accum: HWK = %d "
                   "x %d x %d\n",
                   H,
                   W,
                   C,
                   K,
                   FilterSize,
                   FilterSize,
                   GetWeiProblemsize()[I2],
                   acc_grid_desc.GetLength(I0),
                   acc_grid_desc.GetLength(I1),
                   acc_grid_desc.GetLength(I2));
            printf("GridwiseOp err: ProblemSize check");
            return false;
        }

        if(!(H % HPerBlock == 0 && W % WPerBlock == 0 && K % KPerBlock == 0 && C % CPerBlock == 0))
        {
            printf("GridwiseOp err: ProblemSize division");
            return false;
        }

        // check gridwise conv pipeline
        const auto num_c_loop = C / CPerBlock;

        if(!GridwiseConvPipe::IsSupported(num_c_loop))
        {
            printf("GridwiseOp err: Pipeline not support this c_loop");
            return false;
        }

        if(!block_2_ctile_map.CheckValidity(acc_grid_desc))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        constexpr long_index_t TwoGB = (long_index_t{1} << 31);

        if(!(in_grid_desc.GetElementSpaceSize() * sizeof(InDataType) <= TwoGB &&
             wei_grid_desc.GetElementSpaceSize() * sizeof(WeiDataType) <= TwoGB))
        {
            return false;
        }

#if 0
        // Tensor & transform debugging code
        BlockwiseConv blockwise_conv = {};
        auto& acc_thread_buf = blockwise_conv.GetAccumThreadBuffer();
        const AccElementwiseOperation acc_element_op;
        StoreAccData(acc_grid_desc,
            nullptr,
            acc_thread_buf,
            blockwise_conv,
            acc_element_op,
            nullptr,
            nullptr,
            0,
            0,
            0);
#endif
        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainBlockLoop(index_t C)
    {
        const index_t num_loop = C / CPerBlock;

        return GridwiseConvPipe::CalculateHasMainLoop(num_loop);
    }

    // Return block_id to Acc tensor tile idx (k0, h0, w0) mapping
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2CTileMap(const AccGridDesc& c_grid_desc_h_w_k, index_t M01, index_t /* N01 */)
    {
        return BlockToCTileMap_KSplit_M00_N0_M01Adapt<HPerBlock, WPerBlock, AccGridDesc>(
            c_grid_desc_h_w_k, M01, c_grid_desc_h_w_k.GetLength(I2) / KPerBlock);
    }

    struct SharedMemTrait
    {
        // LDS allocation for A and B: be careful of alignment
        static constexpr auto max_lds_align = 8;

        static constexpr auto in_block_space_size_aligned =
            InEnableLds ? math::integer_least_multiple(
                              MakeInBlockDescriptor().GetElementSpaceSize(), max_lds_align)
                        : 0;
        static constexpr auto wei_block_space_size_aligned =
            WeiEnableLds ? math::integer_least_multiple(
                               MakeWeiBlockDescriptor().GetElementSpaceSize(), max_lds_align)
                         : 0;

        static constexpr auto bias_block_space_size_aligned =
            DsEnableLds
                ? math::integer_least_multiple(
                      MakeDsBlockDescriptor()[Number<0>{}].GetElementSpaceSize(), max_lds_align)
                : 0;

        static constexpr auto scale_block_space_size_aligned =
            DsEnableLds
                ? math::integer_least_multiple(
                      MakeDsBlockDescriptor()[Number<1>{}].GetElementSpaceSize(), max_lds_align)
                : 0;
        static constexpr auto ds_block_space_size_aligned =
            make_tuple(bias_block_space_size_aligned, scale_block_space_size_aligned);

        static constexpr auto in_block_space_offset = 0;
        static constexpr auto wei_block_space_offset =
            (in_block_space_offset + in_block_space_size_aligned) * sizeof(InDataType) /
            sizeof(WeiDataType);

        // LDS allocation for Ds in LDS
        static constexpr auto bias_block_space_offset =
            (wei_block_space_offset + wei_block_space_size_aligned) * sizeof(WeiDataType) /
            sizeof(AccDataType);
        static constexpr auto scale_block_space_offset =
            (bias_block_space_offset + bias_block_space_size_aligned) * sizeof(AccDataType) /
            sizeof(AccDataType);

        static constexpr auto ds_block_space_offset =
            make_tuple(bias_block_space_offset, scale_block_space_offset);

        // LDS allocation for C shuffle in LDS
        static constexpr auto acc_block_space_size = GetAccBlockDescriptor().GetElementSpaceSize();

        static constexpr auto acc_block_space_offset = 0;

        static constexpr auto lds_size =
            math::max(acc_block_space_size * sizeof(AccDataType),
                      in_block_space_size_aligned * sizeof(InDataType) +
                          wei_block_space_size_aligned * sizeof(WeiDataType) +
                          bias_block_space_size_aligned * sizeof(AccDataType) +
                          scale_block_space_size_aligned * sizeof(AccDataType));
    };

    struct LaneSharedMemTrait
    {
        static constexpr auto max_lane_shared_align = 4;

        static constexpr auto in_block_space_size_aligned =
            EnableWaveGroup && (InEnableLds == false)
                ? math::integer_least_multiple(MakeInBlockDescriptor().GetElementSpaceSize(),
                                               max_lane_shared_align)
                : 0;
        static constexpr auto wei_block_space_size_aligned =
            EnableWaveGroup && (WeiEnableLds == false)
                ? math::integer_least_multiple(MakeWeiBlockDescriptor().GetElementSpaceSize(),
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

    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(AccGridDesc{}, 1, 1))>;

    static constexpr auto in_block_desc  = MakeInBlockDescriptor();
    static constexpr auto wei_block_desc = MakeWeiBlockDescriptor();
    static constexpr auto ds_block_desc  = MakeDsBlockDescriptor();
#if FORCE_CONVERT_TO_TENSOR
    static constexpr bool ConvertToTensor = true;
#else
    static constexpr bool ConvertToTensor = false;
#endif
    using BlockwiseConv = BlockwiseSubaConvWconv<ThisThreadBlock,
                                                 WeiDataType,
                                                 InDataType,
                                                 DsDataType,
                                                 AccDataType,
                                                 decltype(MakeWeiWaveDescriptor(wei_block_desc)),
                                                 decltype(MakeInWaveDescriptor(in_block_desc)),
                                                 decltype(MakeDsWaveDescriptor(ds_block_desc)),
                                                 HPerBlock,
                                                 WPerBlock,
                                                 CPerBlock,
                                                 KPerBlock,
                                                 HRepeat,
                                                 WRepeat,
                                                 HPerWconv,
                                                 WPerWconv,
                                                 FilterSize,
                                                 DilationX,
                                                 DilationY,
                                                 activeFun,
                                                 scaleBiasPacked,
                                                 uniformScale,
                                                 WeiEnableLds,
                                                 InEnableLds,
                                                 DsEnableLds,
                                                 ConvertToTensor>;
    template <bool HasMainBlockLoop, typename Block2CTileMap = DefaultBlock2CTileMap>
    __device__ static void Run(const InDataType* __restrict__ p_in_grid,
                               const WeiDataType* __restrict__ p_wei_grid,
                               DsGridPointer p_ds_grid,
                               AccDataType* __restrict__ p_acc_grid,
                               void* __restrict__ p_shared,
                               void* __restrict__ p_lane_shared,
                               const InGridDesc& in_grid_desc,
                               const WeiGridDesc& wei_grid_desc,
                               const DsGridDesc& ds_grid_desc,
                               const AccGridDesc& acc_grid_desc,
                               const InElementwiseOperation& in_element_op,
                               const WeiElementwiseOperation& wei_element_op,
                               const AccElementwiseOperation& acc_element_op,
                               const Block2CTileMap& block_2_ctile_map)
    {
        /*******************************************************************************/
        // Memory buffer zone.
        const auto in_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_grid, in_grid_desc.GetElementSpaceSize());
        const auto wei_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_wei_grid, wei_grid_desc.GetElementSpaceSize());
        const auto ds_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_ds_grid[i], ds_grid_desc[i].GetElementSpaceSize());
            },
            Number<NumDTensor>{});

        const auto in_grid_pad_desc  = MakeInGridPadDescriptor(in_grid_desc);
        const auto wei_grid_pad_desc = MakeWeiGridPadDescriptor(wei_grid_desc);
        const auto ds_grid_pad_desc  = MakeDsGridPadDescriptor(ds_grid_desc);

        /*******************************************************************************/
        // BlockIdx.x -> [BlockId.k, BlockId.h, BlockId.w]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));
        if(!block_2_ctile_map.ValidCTileIndex(block_work_idx,
                                              make_tuple(acc_grid_desc.GetLength(I0),
                                                         acc_grid_desc.GetLength(I1),
                                                         acc_grid_desc.GetLength(I2))))
        {
            return;
        }

        // Store BlockId into SGPR
        const index_t k_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * KPerBlock);
        const index_t w_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I2] * WPerBlock);
        const index_t h_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * HPerBlock);

        /*******************************************************************************/
        // BlockLevel, Tensor and filter ThreadMapping in WCNN Source buffer, As Destinaion of
        // BlockWise_Copy
        const auto C = in_grid_desc.GetLength(I2);

        auto wave_idx = GetWconvWaveIdx<ThisThreadBlock,
                                        HPerBlock,
                                        WPerBlock,
                                        HRepeat,
                                        WRepeat,
                                        HPerWconv,
                                        WPerWconv>();

        auto in_block_trait = [&]() {
            // input data blockwise copy
            if constexpr(InEnableLds)
            {
                auto in_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<InDataType*>(p_shared),
                    SharedMemTrait::in_block_space_size_aligned);
                using InBlockTransferThreadClusterArrangeOrder = Sequence<0, 1, 2>;
                using InBlockTransferAccessOrder               = Sequence<0, 1, 2>;
                constexpr index_t InBlockTransferVectorDim     = 2;

                if constexpr(EnableAsync)
                {
                    auto in_blockwise_copy = ThreadGroupTensorSliceTransfer_AsyncLoad<
                        ThisThreadBlock,
                        Sequence<HPerBlockIn, WPerBlockIn, CPerBlock>,
                        InBlockTransferThreadClusterLengths,
                        InBlockTransferThreadClusterArrangeOrder,
                        InDataType,
                        InDataType,
                        decltype(in_grid_pad_desc),
                        decltype(in_block_desc),
                        InBlockTransferVectorDim,
                        InBlockTransferVectorDim,
                        InBlockTransferDstScalarPerVector>(
                        in_grid_pad_desc,
                        make_multi_index(h_block_data_idx_on_grid, w_block_data_idx_on_grid, 0),
                        in_block_desc,
                        make_multi_index(0, 0, 0));
                    return make_tuple(in_block_buf, in_blockwise_copy);
                }
                else
                {
                    auto in_blockwise_copy = ThreadGroupTensorSliceTransfer_v4r1<
                        ThisThreadBlock,
                        InElementwiseOperation,
                        ck::tensor_operation::element_wise::PassThrough,
                        InMemoryDataOperationEnum::Set,
                        Sequence<HPerBlockIn, WPerBlockIn, CPerBlock>,
                        InBlockTransferThreadClusterLengths,
                        InBlockTransferThreadClusterArrangeOrder,
                        InDataType,
                        InDataType,
                        decltype(in_grid_pad_desc),
                        decltype(in_block_desc),
                        InBlockTransferAccessOrder,
                        InBlockTransferAccessOrder,
                        InBlockTransferVectorDim,
                        InBlockTransferVectorDim,
                        InBlockTransferSrcScalarPerVector,
                        InBlockTransferDstScalarPerVector,
                        1,
                        1,
                        false,
                        true,
                        NumConvCPrefetchStage>(
                        in_grid_pad_desc,
                        make_multi_index(h_block_data_idx_on_grid, w_block_data_idx_on_grid, 0),
                        in_element_op,
                        in_block_desc,
                        make_multi_index(0, 0, 0),
                        ck::tensor_operation::element_wise::PassThrough{});
                    return make_tuple(in_block_buf, in_blockwise_copy);
                }
            }
            else
            {
                // Thread-wise copy
                // W0 x C0 x H0 x H1 x H2 x W1 x C1
                auto indata_slice_origin_idx = wconv_conv.CalculateInDataThreadOriginDataIndex();
                auto h0 = (h_block_data_idx_on_grid + wave_idx[I0] * HPerWave) / HPerWconv;
                auto w0 = (w_block_data_idx_on_grid + wave_idx[I1] * WPerWave) / WPerWconv;

                // Limitation: NumDim of Src and Dst descriptor should be identical
                auto in_blockwise_copy =
                    ThreadwiseTensorSliceTransfer_v2<InDataType,
                                                     InDataType,
                                                     decltype(in_grid_pad_desc),
                                                     decltype(in_block_desc),
                                                     Sequence<WPerWaveIn / WPerWconv,
                                                              CPerWave / CPerWconv,
                                                              HPerWaveIn / HPerWconv,
                                                              NumSubTilePerImage,
                                                              1,
                                                              1,
                                                              NumDataCompPerTile>,
                                                     Sequence<0, 1, 2, 3, 4, 5, 6>,
                                                     6,
                                                     NumDataCompPerTile,
                                                     1,
                                                     false>(
                        in_grid_pad_desc,
                        make_multi_index(w0,
                                         0,
                                         h0,
                                         0,
                                         indata_slice_origin_idx[I0],
                                         indata_slice_origin_idx[I1],
                                         indata_slice_origin_idx[I2]));

                if constexpr(EnableWaveGroup)
                {
                    static_assert(LaneSharedMemTrait::in_block_space_offset == 0, "");
                    return make_tuple(make_static_buffer_v4<AddressSpaceEnum::Vgpr, InDataType>(
                                          in_block_desc.GetElementSpaceSize(),
                                          static_cast<InDataType*>(p_lane_shared) +
                                              LaneSharedMemTrait::in_block_space_offset *
                                                  GetLaneSharedMemCount<HasMainBlockLoop>()),
                                      in_blockwise_copy);
                }
                else
                {
                    return make_tuple(make_static_buffer_v2<AddressSpaceEnum::Vgpr, InDataType>(
                                          in_block_desc.GetElementSpaceSize()),
                                      in_blockwise_copy);
                }
            }
        };

        auto wei_block_trait = [&]() {
            if constexpr(WeiEnableLds)
            {
                auto wei_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<WeiDataType*>(p_shared) + SharedMemTrait::wei_block_space_offset,
                    SharedMemTrait::wei_block_space_size_aligned);

                using WeiBlockTransferThreadClusterArrangeOrder = Sequence<0, 1, 2>;
                using WeiBlockTransferAccessOrder               = Sequence<0, 1, 2>;
                constexpr index_t WeiBlockTransferVectorDim     = 2;

                if constexpr(EnableAsync)
                {
                    using WeiThreadGroupTensorSliceTransfer =
                        ThreadGroupTensorSliceTransfer_AsyncLoad<
                            ThisThreadBlock,
                            Sequence<KPerBlock, 1, CPerBlock>,
                            WeiBlockTransferThreadClusterLengths,
                            WeiBlockTransferThreadClusterArrangeOrder,
                            WeiDataType,
                            WeiDataType,
                            decltype(wei_grid_desc),
                            decltype(wei_block_desc),
                            WeiBlockTransferVectorDim,
                            WeiBlockTransferVectorDim,
                            WeiBlockTransferSrcScalarPerVector>;

                    auto wei_blockwise_copy = generate_tuple(
                        [&](auto I) {
                            return WeiThreadGroupTensorSliceTransfer(
                                wei_grid_desc,
                                make_multi_index(k_block_data_idx_on_grid, I, 0),
                                wei_block_desc,
                                (FilterSize == 3)
                                    ? make_multi_index(0, wconv_conv.GetWeight3RemapTable()[I], 0)
                                    : make_multi_index(0, 0, 0));
                        },
                        NumberYX{});

                    return make_tuple(wei_block_buf, wei_blockwise_copy);
                }
                else
                {
                    using WeiThreadGroupTensorSliceTransfer = ThreadGroupTensorSliceTransfer_v4r1<
                        ThisThreadBlock,
                        WeiElementwiseOperation,
                        ck::tensor_operation::element_wise::PassThrough,
                        InMemoryDataOperationEnum::Set,
                        Sequence<KPerBlock, 1, CPerBlock>,
                        WeiBlockTransferThreadClusterLengths,
                        WeiBlockTransferThreadClusterArrangeOrder,
                        WeiDataType,
                        WeiDataType,
                        decltype(wei_grid_desc),
                        decltype(wei_block_desc),
                        WeiBlockTransferAccessOrder,
                        WeiBlockTransferAccessOrder,
                        WeiBlockTransferVectorDim,
                        WeiBlockTransferVectorDim,
                        WeiBlockTransferSrcScalarPerVector,
                        WeiBlockTransferDstScalarPerVector,
                        1,
                        1,
                        false,
                        true,
                        NumConvCPrefetchStage>;

                    auto wei_blockwise_copy = generate_tuple(
                        [&](auto I) {
                            return WeiThreadGroupTensorSliceTransfer(
                                wei_grid_desc,
                                make_multi_index(k_block_data_idx_on_grid, I, 0),
                                wei_element_op,
                                wei_block_desc,
                                (FilterSize == 3)
                                    ? make_multi_index(0, wconv_conv.GetWeight3RemapTable()[I], 0)
                                    : make_multi_index(0, 0, 0),
                                ck::tensor_operation::element_wise::PassThrough{});
                        },
                        NumberYX{});
                    return make_tuple(wei_block_buf, wei_blockwise_copy);
                }
            }
            else
            {
                constexpr index_t Iters = GetFilterIters<WeiDataType,
                                                         InDataType,
                                                         AccDataType,
                                                         CPerBlock,
                                                         HPerWconv,
                                                         WPerWconv,
                                                         FilterSize>();

                auto wei_slice_origin_idx = wconv_conv.CalculateWeiDataThreadOriginDataIndex();
                auto k0 = (k_block_data_idx_on_grid + wave_idx[I2] * KPerWave) / KPerWconv;

                // Limitation: NumDim of Src and Dst descriptor should be identical
                using WeiThreadGroupTensorSliceTransfer =
                    ThreadwiseTensorSliceTransfer_v2<WeiDataType,
                                                     WeiDataType,
                                                     decltype(wei_grid_pad_desc),
                                                     decltype(wei_block_desc),
                                                     Sequence<KPerWave / KPerWconv,
                                                              CPerWave / CPerWconv,
                                                              1,
                                                              1,
                                                              NumSubTilesPerWeightTap,
                                                              NumWeightCompPerTile>,
                                                     Sequence<0, 1, 2, 3, 4, 5>,
                                                     5,
                                                     NumWeightCompPerTile,
                                                     1,
                                                     false>;

                auto wei_blockwise_copy = generate_tuple(
                    [&](auto I) {
                        if constexpr(Iters > 1)
                        {
                            return WeiThreadGroupTensorSliceTransfer(
                                wei_grid_pad_desc,
                                make_multi_index(k0,
                                                 wei_slice_origin_idx[I0],
                                                 0,
                                                 wei_slice_origin_idx[I1],
                                                 wei_slice_origin_idx[I2],
                                                 wei_slice_origin_idx[I3]));
                        }
                        else
                        {
                            return WeiThreadGroupTensorSliceTransfer(
                                wei_grid_pad_desc,
                                make_multi_index(k0,
                                                 0,
                                                 I * wconv_conv.GetNumWeightTapPerWave() +
                                                     wei_slice_origin_idx[I0] *
                                                         wconv_conv.GetWeightSecondTapMapTable()[I],
                                                 wei_slice_origin_idx[I1],
                                                 wei_slice_origin_idx[I2],
                                                 wei_slice_origin_idx[I3]));
                        }
                    },
                    Number<NumWeightTap>{});

                if constexpr(EnableWaveGroup)
                {
                    return make_tuple(make_static_buffer_v4<AddressSpaceEnum::Vgpr, WeiDataType>(
                                          wei_block_desc.GetElementSpaceSize(),
                                          static_cast<WeiDataType*>(p_lane_shared) +
                                              LaneSharedMemTrait::wei_block_space_offset *
                                                  GetLaneSharedMemCount<HasMainBlockLoop>()),
                                      wei_blockwise_copy);
                }
                else
                {
                    return make_tuple(make_static_buffer_v2<AddressSpaceEnum::Vgpr, WeiDataType>(
                                          wei_block_desc.GetElementSpaceSize()),
                                      wei_blockwise_copy);
                }
            }
        };

        auto ds_block_trait = [&]() {
            if constexpr(DsEnableLds)
            {
                auto ds_array_block_buf = generate_tuple(
                    [&](auto i) {
                        return make_dynamic_buffer<AddressSpaceEnum::Lds>(
                            static_cast<AccDataType*>(p_shared) +
                                SharedMemTrait::ds_block_space_offset[Number<i>{}],
                            SharedMemTrait::ds_block_space_size_aligned[Number<i>{}]);
                    },
                    Number<NumDTensor>{});

                using DsBlockTransferThreadClusterArrangeOrder = Sequence<0>;
                using DsBlockTransferAccessOrder               = Sequence<0>;

                constexpr index_t DsBlockTransferVectorDim = 0;

                if constexpr(EnableAsync)
                {
                    auto ds_blockwise_copy = generate_tuple(
                        [&](auto i) {
                            using DsThreadGroupTensorSliceTransfer =
                                ThreadGroupTensorSliceTransfer_AsyncLoad<
                                    DsThreadBlock,
                                    Sequence<KPerBlock>,
                                    DsBlockTransferThreadClusterLengths,
                                    DsBlockTransferThreadClusterArrangeOrder,
                                    AccDataType,
                                    AccDataType,
                                    remove_cvref_t<decltype(ds_grid_pad_desc[Number<i>{}])>,
                                    remove_cvref_t<decltype(ds_block_desc[Number<i>{}])>,
                                    DsBlockTransferVectorDim,
                                    DsBlockTransferVectorDim,
                                    DsBlockTransferDstScalarPerVector>;
                            return DsThreadGroupTensorSliceTransfer(
                                ds_grid_pad_desc[Number<i>{}],
                                make_multi_index(k_block_data_idx_on_grid),
                                ds_block_desc[Number<i>{}],
                                make_multi_index(0));
                        },
                        Number<NumDTensor>{});
                    return make_tuple(ds_array_block_buf, ds_blockwise_copy);
                }
                else
                {

                    auto ds_blockwise_copy = generate_tuple(
                        [&](auto i) {
                            using DsThreadGroupTensorSliceTransfer =
                                ThreadGroupTensorSliceTransfer_v4r1<
                                    DsThreadBlock,
                                    ck::tensor_operation::element_wise::PassThrough,
                                    ck::tensor_operation::element_wise::PassThrough,
                                    InMemoryDataOperationEnum::Set,
                                    Sequence<KPerBlock>,
                                    DsBlockTransferThreadClusterLengths,
                                    DsBlockTransferThreadClusterArrangeOrder,
                                    AccDataType,
                                    AccDataType,
                                    remove_cvref_t<decltype(ds_grid_pad_desc[Number<i>{}])>,
                                    remove_cvref_t<decltype(ds_block_desc[Number<i>{}])>,
                                    DsBlockTransferAccessOrder,
                                    DsBlockTransferAccessOrder,
                                    DsBlockTransferVectorDim,
                                    DsBlockTransferVectorDim,
                                    DsBlockTransferSrcScalarPerVector,
                                    DsBlockTransferDstScalarPerVector,
                                    1,
                                    1,
                                    false,
                                    true,
                                    NumConvCPrefetchStage>;
                            return DsThreadGroupTensorSliceTransfer(
                                ds_grid_pad_desc[Number<i>{}],
                                make_multi_index(k_block_data_idx_on_grid),
                                ck::tensor_operation::element_wise::PassThrough{},
                                ds_block_desc[Number<i>{}],
                                make_multi_index(0),
                                ck::tensor_operation::element_wise::PassThrough{});
                        },
                        Number<NumDTensor>{});
                    return make_tuple(ds_array_block_buf, ds_blockwise_copy);
                }
            }
            else
            {
                using ds_count_vgpr_type =
                    decltype(make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
                        ds_block_desc[I0].GetElementSpaceSize()));
                Array<ds_count_vgpr_type, NumDTensor> ds_array_block_buf;
                static_for<0, NumDTensor, 1>{}([&](auto i) {
                    ds_array_block_buf(i) = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
                        ds_block_desc[Number<i>{}].GetElementSpaceSize());
                });

                auto k0 = (k_block_data_idx_on_grid + wave_idx[I2] * KPerWave) / KPerWconv;

                // Limitation: NumDim of Src and Dst descriptor should be identical

                auto ds_blockwise_copy = generate_tuple(
                    [&](auto i) {
                        constexpr index_t DsPerThread = acc_sba.GetNumBiasComponents();
                        auto ds_slice_origin_idx =
                            acc_sba.CalculateDsThreadOriginDataIndex(KPerWconv);
                        using DsThreadGroupTensorSliceTransfer = ThreadwiseTensorSliceTransfer_v2<
                            AccDataType,
                            AccDataType,
                            remove_cvref_t<decltype(ds_grid_pad_desc[Number<i>{}])>,
                            remove_cvref_t<decltype(ds_block_desc[Number<i>{}])>,
                            Sequence<Number<KPerWave / KPerWconv>{},
                                     Number<KPerWconv / DsPerThread>{},
                                     Number<DsPerThread>{}>,
                            Sequence<0, 1, 2>,
                            2,
                            DsPerThread,
                            1,
                            false>;

                        return DsThreadGroupTensorSliceTransfer(
                            ds_grid_pad_desc[Number<i>{}],
                            make_multi_index(k0, ds_slice_origin_idx[I0], ds_slice_origin_idx[I1]));
                    },
                    Number<NumDTensor>{});
                return make_tuple(ds_array_block_buf, ds_blockwise_copy);
            }
        };

        /*******************************************************************************/
        // CONV
        BlockwiseConv blockwise_conv = {};

        // Prepare Register for Accum
        auto& acc_thread_buf = blockwise_conv.GetAccumThreadBuffer();

        /*******************************************************************************/
        // Shift Per CPerBlock
        constexpr auto in_block_slice_copy_step  = MakeInBlockSliceCopyStep();
        constexpr auto wei_block_slice_copy_step = MakeWeiBlockSliceCopyStep();

        // Gridwise conv pipeline
        const index_t CBlockMainLoop = __builtin_amdgcn_readfirstlane(C / CPerBlock);

        auto in_block_buf      = in_block_trait()[I0];
        auto in_blockwise_copy = in_block_trait()[I1];

        auto wei_block_buf      = wei_block_trait()[I0];
        auto wei_blockwise_copy = wei_block_trait()[I1];

        auto ds_block_buf      = ds_block_trait()[I0];
        auto ds_blockwise_copy = ds_block_trait()[I1];

        GridwiseConvPipe::template Run<HasMainBlockLoop>(in_grid_pad_desc,
                                                         in_block_desc,
                                                         in_blockwise_copy,
                                                         in_grid_buf,
                                                         in_block_buf,
                                                         in_block_slice_copy_step,
                                                         wei_grid_pad_desc,
                                                         wei_block_desc,
                                                         wei_blockwise_copy,
                                                         wei_grid_buf,
                                                         wei_block_buf,
                                                         wei_block_slice_copy_step,
                                                         ds_grid_pad_desc,
                                                         ds_block_desc,
                                                         ds_blockwise_copy,
                                                         ds_grid_buf,
                                                         ds_block_buf,
                                                         blockwise_conv,
                                                         acc_thread_buf,
                                                         CBlockMainLoop);
        /*******************************************************************************/
        // Store accum buffer
        if((EnableWaveGroup == false) || (get_wave_id_in_wavegroup() == 1))
        {
            StoreAccData(acc_grid_desc,
                         p_acc_grid,
                         acc_thread_buf,
                         blockwise_conv,
                         acc_element_op,
                         p_shared,
                         p_lane_shared,
                         h_block_data_idx_on_grid,
                         w_block_data_idx_on_grid,
                         k_block_data_idx_on_grid);
        }
    }
};

} // namespace ck
