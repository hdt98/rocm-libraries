// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_conv_pipeline.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_conv_wconv.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_async.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseOp,
          typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename AccElementwiseOperation,
          typename InGridDesc,
          typename WeiGridDesc,
          typename AccGridDesc,
          typename Block2CTileMap,
          typename ComputePtrOffsetOfBatch,
          bool HasMainBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_grouped_conv_wconv(const InDataType* __restrict__ p_in_grid,
                                  const WeiDataType* __restrict__ p_wei_grid,
                                  AccDataType* __restrict__ p_acc_grid,
                                  const InElementwiseOperation in_element_op,
                                  const WeiElementwiseOperation wei_element_op,
                                  const AccElementwiseOperation acc_element_op,
                                  const index_t batch_count,
                                  const InGridDesc in_grid_desc,
                                  const WeiGridDesc wei_grid_desc,
                                  const AccGridDesc acc_grid_desc,
                                  const Block2CTileMap block_2_ctile_map,
                                  const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx13__))
    // offset base pointer for each work-group
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t in_batch_offset = amd_wave_read_first_lane(
        static_cast<int64_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
    const long_index_t wei_batch_offset = amd_wave_read_first_lane(
        static_cast<int64_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
    const long_index_t acc_batch_offset = amd_wave_read_first_lane(
        static_cast<int64_t>(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx)));

    __shared__ char p_shared[GridwiseOp::BlockwiseConv::SharedMemTrait::lds_size];

    GridwiseOp::template Run<HasMainBlockLoop>(p_in_grid + in_batch_offset,
                                               p_wei_grid + wei_batch_offset,
                                               p_acc_grid + acc_batch_offset,
                                               p_shared,
                                               nullptr,
                                               in_grid_desc,
                                               wei_grid_desc,
                                               acc_grid_desc,
                                               in_element_op,
                                               wei_element_op,
                                               acc_element_op,
                                               block_2_ctile_map);
#else
    ignore                                = p_in_grid;
    ignore                                = p_wei_grid;
    ignore                                = p_acc_grid;
    ignore                                = in_grid_desc;
    ignore                                = wei_grid_desc;
    ignore                                = acc_grid_desc;
    ignore                                = in_element_op;
    ignore                                = wei_element_op;
    ignore                                = acc_element_op;
    ignore                                = compute_ptr_offset_of_batch;
    ignore                                = block_2_ctile_map;
#endif
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wzero-length-array"
template <typename GridwiseOp,
          typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename AccElementwiseOperation,
          typename InGridDesc,
          typename WeiGridDesc,
          typename AccGridDesc,
          typename Block2CTileMap,
          typename ComputePtrOffsetOfBatch,
          bool HasMainBlockLoop>
__global__ void __exp_amd_wavegroup_kernel(4, 32, 256, 1, 1)
    kernel_grouped_conv_wconv_wavegroup(const InDataType* __restrict__ p_in_grid,
                                        const WeiDataType* __restrict__ p_wei_grid,
                                        AccDataType* __restrict__ p_acc_grid,
                                        const InElementwiseOperation in_element_op,
                                        const WeiElementwiseOperation wei_element_op,
                                        const AccElementwiseOperation acc_element_op,
                                        const index_t batch_count,
                                        const InGridDesc in_grid_desc,
                                        const WeiGridDesc wei_grid_desc,
                                        const AccGridDesc acc_grid_desc,
                                        const Block2CTileMap block_2_ctile_map,
                                        const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx13__))
    // offset base pointer for each work-group
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t in_batch_offset = amd_wave_read_first_lane(
        static_cast<int64_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
    const long_index_t wei_batch_offset = amd_wave_read_first_lane(
        static_cast<int64_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
    const long_index_t acc_batch_offset = amd_wave_read_first_lane(
        static_cast<int64_t>(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx)));

    __shared__ char p_shared[GridwiseOp::BlockwiseConv::SharedMemTrait::lds_size];
    static __exp_amd_laneshared__ char p_lane_shared
        [GridwiseOp::BlockwiseConv::LaneSharedMemTrait::lane_shared_size *
         GridwiseOp::BlockwiseConv::template GetLaneSharedMemCount<HasMainBlockLoop>()];

    static_assert(GridwiseOp::BlockwiseConv::LaneSharedMemTrait::lane_shared_size <= 512 * 4, "");

    GridwiseOp::template Run<HasMainBlockLoop>(p_in_grid + in_batch_offset,
                                               p_wei_grid + wei_batch_offset,
                                               p_acc_grid + acc_batch_offset,
                                               p_shared,
                                               p_lane_shared,
                                               in_grid_desc,
                                               wei_grid_desc,
                                               acc_grid_desc,
                                               in_element_op,
                                               wei_element_op,
                                               acc_element_op,
                                               block_2_ctile_map);
#else
    ignore                                = p_in_grid;
    ignore                                = p_wei_grid;
    ignore                                = p_acc_grid;
    ignore                                = in_grid_desc;
    ignore                                = wei_grid_desc;
    ignore                                = acc_grid_desc;
    ignore                                = in_element_op;
    ignore                                = wei_element_op;
    ignore                                = acc_element_op;
    ignore                                = compute_ptr_offset_of_batch;
    ignore                                = block_2_ctile_map;
#endif
}
#pragma clang diagnostic pop

template <index_t BlockSize,
          typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          typename InGridDesc,
          typename WeiGridDesc,
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
          typename AccBlockTransferClusterLengths,
          index_t AccBlockTransferScalarPerVector,
          bool AccEnableLds,
          bool EnableAsync,
          index_t NumConvCPrefetchStage,
          bool EnableWaveGroup,
          bool Transposed>
struct GridwiseConv_Wconv
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I7 = Number<7>{};

    static constexpr index_t WaveSize     = 32;
    static constexpr index_t NumWaveGroup = EnableWaveGroup ? 4 : 0;

    static_assert((EnableWaveGroup == false) || (BlockSize % (WaveSize * 4) == 0), "");

    using ThisThreadBlock =
        typename std::conditional<EnableWaveGroup,
                                  ThisThreadBlockWaveGroup<BlockSize, WaveSize, NumWaveGroup>,
                                  ThisThreadBlock<BlockSize>>::type;

    static constexpr index_t YX = FilterSize * FilterSize;
    using NumberYX              = Number<FilterSize * FilterSize>;

    static constexpr index_t WaveFilterSize = (FilterSize == 2) ? 1 : FilterSize;
    static constexpr auto wconv_conv        = WconvConv<WeiDataType,
                                                 InDataType,
                                                 AccDataType,
                                                 HPerWconv,
                                                 WPerWconv,
                                                 WaveFilterSize,
                                                 DilationX,
                                                 DilationY,
                                                 1,
                                                 EnableWaveGroup>{};

    static constexpr index_t CPerWconv               = wconv_conv.GetNumInputChannels();
    static constexpr index_t KPerWconv               = wconv_conv.GetNumOutputChannels();
    static constexpr index_t NumWeightTap            = wconv_conv.GetNumWeightTap();
    static constexpr index_t NumSubTilesPerWeightTap = wconv_conv.GetNumSubTilesPerWeightTap();
    static constexpr index_t NumWeightCompPerTile    = wconv_conv.GetNumWeightCompPerTile();
    static constexpr index_t NumSubTilePerImage      = wconv_conv.GetNumSubTilesPerImageTile();
    static constexpr index_t NumDataCompPerTile      = wconv_conv.GetNumDataCompPerTile();

    static constexpr index_t DataTileHeight = 4;
    static constexpr index_t H_Pad          = (FilterSize == 3) ? DataTileHeight : 0;
    static constexpr index_t W_Pad          = (FilterSize == 3) ? WPerWconv : 0;
    static constexpr index_t HPerBlockIn    = HPerBlock + H_Pad * 2;
    static constexpr index_t WPerBlockIn    = WPerBlock + W_Pad * 2;
    static constexpr index_t HPerWave       = HRepeat * HPerWconv;
    static constexpr index_t WPerWave       = WRepeat * WPerWconv;
    static constexpr index_t CPerWave       = CPerBlock;
    static constexpr index_t KPerWave       = KPerBlock;
    static constexpr index_t HPerWaveIn     = HPerWave + H_Pad * 2;
    static constexpr index_t WPerWaveIn     = WPerWave + W_Pad * 2;

    using GridwiseConvPipe =
        remove_cvref_t<decltype(GridwiseConvPipeline_Selector<NumConvCPrefetchStage,
                                                              InEnableLds,
                                                              WeiEnableLds,
                                                              EnableAsync,
                                                              EnableWaveGroup>())>;

#ifdef FORCE_CONVERT_TO_TENSOR
    static constexpr bool ConvertToTensor = true;
#else
    static constexpr bool ConvertToTensor = false;
#endif

    // Describe the layout of InData in block level (LDS or VGPR)
    __host__ __device__ static constexpr auto MakeInBlockDescriptor()
    {
        if constexpr(InEnableLds)
        {
            // H x W x C Per Block
            return make_naive_tensor_descriptor_packed(
                make_tuple(Number<HPerBlockIn>{}, Number<WPerBlockIn>{}, Number<CPerBlock>{}));
        }
        else
        {
            // W0 x C0 x H0 x H1 x H2 x W1 x C1
            return make_naive_tensor_descriptor_packed(make_tuple(Number<WPerWaveIn / WPerWconv>{},
                                                                  Number<CPerWave / CPerWconv>{},
                                                                  Number<HPerWaveIn / HPerWconv>{},
                                                                  Number<NumSubTilePerImage>{},
                                                                  I1,
                                                                  I1,
                                                                  Number<NumDataCompPerTile>{}));
        }
    }

    // Describe the layout of WeiData in block level (LDS or VGPR)
    __host__ __device__ static constexpr auto MakeWeiBlockDescriptor()
    {
        if constexpr(WeiEnableLds)
        {
            // K x YX x C per block
            return make_naive_tensor_descriptor_packed(make_tuple(
                Number<KPerBlock>{}, Number<FilterSize * FilterSize>{}, Number<CPerBlock>{}));
        }
        else
        {
            // K0 x C0 x YX x K1 x C1 x C2
            constexpr index_t NumXY = (FilterSize == 3) ? NumWeightTap : FilterSize * FilterSize;
            return make_naive_tensor_descriptor_packed(make_tuple(Number<KPerWave / KPerWconv>{},
                                                                  Number<CPerWave / CPerWconv>{},
                                                                  Number<NumXY>{},
                                                                  I1,
                                                                  Number<NumSubTilesPerWeightTap>{},
                                                                  Number<NumWeightCompPerTile>{}));
        }
    }

    using BlockwiseConv = BlockwiseConvWconv<ThisThreadBlock,
                                             WeiDataType,
                                             InDataType,
                                             AccDataType,
                                             decltype(MakeWeiBlockDescriptor()),
                                             decltype(MakeInBlockDescriptor()),
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
                                             WeiEnableLds,
                                             InEnableLds,
                                             ConvertToTensor,
                                             Transposed>;

    // Pad input and weight data grid description according to Filter size
    __host__ __device__ static constexpr auto
    MakeInGridPadDescriptor(const InGridDesc& in_grid_desc)
    {
        if constexpr(H_Pad > 0 || W_Pad > 0)
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
            const auto acc_grid_wave_desc = blockwise_conv.GetAccWaveDescriptor(acc_grid_desc);

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
            constexpr auto acc_block_desc = blockwise_conv.GetAccBlockDescriptor();
            constexpr auto acc_block_wave_desc =
                blockwise_conv.GetAccWaveDescriptor(acc_block_desc);

            auto acc_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<AccDataType*>(p_shared) +
                    BlockwiseConv::SharedMemTrait::acc_block_space_offset,
                BlockwiseConv::SharedMemTrait::acc_block_space_size);

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
            auto acc_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
                ThisThreadBlock,
                AccElementwiseOperation,
                InMemoryDataOperationEnum::Set,
                Sequence<BlockwiseConv::HPerBlockOut, BlockwiseConv::WPerBlockOut, KPerBlock>,
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
                false>{acc_block_desc,
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

    template <typename Block2CTileMap>
    __host__ __device__ static constexpr bool CheckValidity(const InGridDesc& in_grid_desc,
                                                            const WeiGridDesc& wei_grid_desc,
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

        const auto GetOutSize = [&](auto i) {
            if constexpr(FilterSize == 2)
            {
                return Transposed ? i * 2 : i / 2;
            }
            else
            {
                return i;
            }
        };

        const auto Ho = GetOutSize(GetInProblemsize()[I0]);
        const auto Wo = GetOutSize(GetInProblemsize()[I1]);
        const auto H  = GetInProblemsize()[I0];
        const auto W  = GetInProblemsize()[I1];
        const auto C  = GetInProblemsize()[I2];
        const auto K  = GetWeiProblemsize()[I0];

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
        return BlockToCTileMap_KSplit_M00_N0_M01Adapt<BlockwiseConv::HPerBlockOut,
                                                      BlockwiseConv::WPerBlockOut,
                                                      AccGridDesc>(
            c_grid_desc_h_w_k, M01, c_grid_desc_h_w_k.GetLength(I2) / KPerBlock);
    }

    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(AccGridDesc{}, 1, 1))>;

    static constexpr auto in_block_desc  = MakeInBlockDescriptor();
    static constexpr auto wei_block_desc = MakeWeiBlockDescriptor();

    template <bool HasMainBlockLoop, typename Block2CTileMap = DefaultBlock2CTileMap>
    __device__ static void Run(const InDataType* __restrict__ p_in_grid,
                               const WeiDataType* __restrict__ p_wei_grid,
                               AccDataType* __restrict__ p_acc_grid,
                               void* __restrict__ p_shared,
                               void* __restrict__ p_lane_shared,
                               const InGridDesc& in_grid_desc,
                               const WeiGridDesc& wei_grid_desc,
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

        const auto in_grid_block_desc =
            BlockwiseConv::MakeInGridBlockDescriptor(MakeInGridPadDescriptor(in_grid_desc));
        const auto wei_grid_block_desc = BlockwiseConv::MakeWeiGridBlockDescriptor(wei_grid_desc);

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

        const index_t w_out_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I2] * BlockwiseConv::WPerBlockOut);
        const index_t h_out_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * BlockwiseConv::HPerBlockOut);

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
                    BlockwiseConv::SharedMemTrait::in_block_space_size_aligned);
                using InBlockTransferThreadClusterArrangeOrder = Sequence<0, 1, 2>;
                using InBlockTransferAccessOrder               = Sequence<0, 1, 2>;
                constexpr index_t InBlockTransferVectorDim     = 2;
                if constexpr(EnableAsync)
                {
                    auto in_blockwise_copy = ThreadGroupTensorSliceTransferAsync<
                        ThisThreadBlock,
                        Sequence<HPerBlockIn, WPerBlockIn, CPerBlock>,
                        InBlockTransferThreadClusterLengths,
                        InBlockTransferThreadClusterArrangeOrder,
                        InDataType,
                        InDataType,
                        decltype(in_grid_block_desc),
                        decltype(in_block_desc),
                        InBlockTransferVectorDim,
                        InBlockTransferVectorDim,
                        InBlockTransferDstScalarPerVector,
                        false,
                        true>(
                        in_grid_block_desc,
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
                        decltype(in_grid_block_desc),
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
                        in_grid_block_desc,
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
                                                     decltype(in_grid_block_desc),
                                                     decltype(in_block_desc),
                                                     decltype(BlockwiseConv::GetInWaveDescLength()),
                                                     Sequence<0, 1, 2, 3, 4, 5, 6>,
                                                     6,
                                                     NumDataCompPerTile,
                                                     1,
                                                     false>(
                        in_grid_block_desc,
                        make_multi_index(w0,
                                         0,
                                         h0,
                                         0,
                                         indata_slice_origin_idx[I0],
                                         indata_slice_origin_idx[I1],
                                         indata_slice_origin_idx[I2]));

                if constexpr(EnableWaveGroup)
                {
                    static_assert(BlockwiseConv::LaneSharedMemTrait::in_block_space_offset == 0,
                                  "");
                    return make_tuple(
                        make_static_buffer_v4<AddressSpaceEnum::Vgpr, InDataType>(
                            in_block_desc.GetElementSpaceSize(),
                            static_cast<InDataType*>(p_lane_shared) +
                                BlockwiseConv::LaneSharedMemTrait::in_block_space_offset *
                                    BlockwiseConv::template GetLaneSharedMemCount<
                                        HasMainBlockLoop>()),
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
                    static_cast<WeiDataType*>(p_shared) +
                        BlockwiseConv::SharedMemTrait::wei_block_space_offset,
                    BlockwiseConv::SharedMemTrait::wei_block_space_size_aligned);

                using WeiBlockTransferThreadClusterArrangeOrder = Sequence<0, 1, 2>;
                using WeiBlockTransferAccessOrder               = Sequence<0, 1, 2>;
                constexpr index_t WeiBlockTransferVectorDim     = 2;
                constexpr index_t NumTapPerCopy                 = (FilterSize == 3) ? 1 : YX;
                constexpr auto NumWeiCopy                       = (FilterSize == 3) ? YX : 1;
                if constexpr(EnableAsync)
                {
                    using WeiThreadGroupTensorSliceTransfer = ThreadGroupTensorSliceTransferAsync<
                        ThisThreadBlock,
                        Sequence<KPerBlock, NumTapPerCopy, CPerBlock>,
                        WeiBlockTransferThreadClusterLengths,
                        WeiBlockTransferThreadClusterArrangeOrder,
                        WeiDataType,
                        WeiDataType,
                        decltype(wei_grid_desc),
                        decltype(wei_block_desc),
                        WeiBlockTransferVectorDim,
                        WeiBlockTransferVectorDim,
                        WeiBlockTransferSrcScalarPerVector,
                        false,
                        true>;

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
                        Number<NumWeiCopy>{});

                    return make_tuple(wei_block_buf, wei_blockwise_copy);
                }
                else
                {
                    using WeiThreadGroupTensorSliceTransfer = ThreadGroupTensorSliceTransfer_v4r1<
                        ThisThreadBlock,
                        WeiElementwiseOperation,
                        ck::tensor_operation::element_wise::PassThrough,
                        InMemoryDataOperationEnum::Set,
                        Sequence<KPerBlock, NumTapPerCopy, CPerBlock>,
                        WeiBlockTransferThreadClusterLengths,
                        WeiBlockTransferThreadClusterArrangeOrder,
                        WeiDataType,
                        WeiDataType,
                        decltype(wei_grid_block_desc),
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
                        Number<NumWeiCopy>{});
                    return make_tuple(wei_block_buf, wei_blockwise_copy);
                }
            }
            else
            {
                auto wei_slice_origin_idx = wconv_conv.CalculateWeiDataThreadOriginDataIndex();
                auto k0 = (k_block_data_idx_on_grid + wave_idx[I2] * KPerWave) / KPerWconv;

                // Limitation: NumDim of Src and Dst descriptor should be identical
                using WeiThreadGroupTensorSliceTransfer = ThreadwiseTensorSliceTransfer_v2<
                    WeiDataType,
                    WeiDataType,
                    decltype(wei_grid_block_desc),
                    decltype(wei_block_desc),
                    decltype(BlockwiseConv::GetWeiWaveDescLength()),
                    Sequence<0, 1, 2, 3, 4, 5>,
                    5,
                    NumWeightCompPerTile,
                    1,
                    false>;

                auto wei_blockwise_copy = generate_tuple(
                    [&](auto I) {
                        if constexpr(FilterSize != 3)
                        {
                            static_assert(NumWeightTap == 1, "");
                            if(FilterSize == 2)
                            {
                                // Use tap index in dim YX
                                return WeiThreadGroupTensorSliceTransfer(
                                    wei_grid_block_desc,
                                    make_multi_index(k0,
                                                     0,
                                                     wei_slice_origin_idx[I0],
                                                     wei_slice_origin_idx[I1],
                                                     wei_slice_origin_idx[I2],
                                                     wei_slice_origin_idx[I3]));
                            }
                            else
                            {
                                // Use tap index in dim C0
                                return WeiThreadGroupTensorSliceTransfer(
                                    wei_grid_block_desc,
                                    make_multi_index(k0,
                                                     wei_slice_origin_idx[I0],
                                                     0,
                                                     wei_slice_origin_idx[I1],
                                                     wei_slice_origin_idx[I2],
                                                     wei_slice_origin_idx[I3]));
                            }
                        }
                        else
                        {
                            return WeiThreadGroupTensorSliceTransfer(
                                wei_grid_block_desc,
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
                    return make_tuple(
                        make_static_buffer_v4<AddressSpaceEnum::Vgpr, WeiDataType>(
                            wei_block_desc.GetElementSpaceSize(),
                            static_cast<WeiDataType*>(p_lane_shared) +
                                BlockwiseConv::LaneSharedMemTrait::wei_block_space_offset *
                                    BlockwiseConv::template GetLaneSharedMemCount<
                                        HasMainBlockLoop>()),
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

        /*******************************************************************************/
        // CONV
        BlockwiseConv blockwise_conv = {};

        // Prepare Register for Accum
        auto& acc_thread_buf = blockwise_conv.GetAccumThreadBuffer();

        /*******************************************************************************/
        // Shift Per CPerBlock
        constexpr auto in_block_slice_copy_step  = BlockwiseConv::MakeInBlockSliceCopyStep();
        constexpr auto wei_block_slice_copy_step = BlockwiseConv::MakeWeiBlockSliceCopyStep();

        // Gridwise conv pipeline
        const index_t CBlockMainLoop = __builtin_amdgcn_readfirstlane(C / CPerBlock);

        auto in_block_buf      = in_block_trait()[I0];
        auto in_blockwise_copy = in_block_trait()[I1];

        auto wei_block_buf      = wei_block_trait()[I0];
        auto wei_blockwise_copy = wei_block_trait()[I1];
        GridwiseConvPipe::template Run<HasMainBlockLoop>(in_grid_block_desc,
                                                         in_block_desc,
                                                         in_blockwise_copy,
                                                         in_grid_buf,
                                                         in_block_buf,
                                                         in_block_slice_copy_step,
                                                         wei_grid_block_desc,
                                                         wei_block_desc,
                                                         wei_blockwise_copy,
                                                         wei_grid_buf,
                                                         wei_block_buf,
                                                         wei_block_slice_copy_step,
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
                         h_out_block_data_idx_on_grid,
                         w_out_block_data_idx_on_grid,
                         k_block_data_idx_on_grid);
        }
    }
};

} // namespace ck
