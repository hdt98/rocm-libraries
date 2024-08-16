// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_conv_pipeline.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_conv_wconv.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r2.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
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
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx11__) || defined(__gfx12__) || \
    defined(__gfx13__))
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

    __shared__ char p_shared[GridwiseOp::SharedMemTrait::lds_size];

    GridwiseOp::template Run<HasMainBlockLoop>(p_in_grid + in_batch_offset,
                                               p_wei_grid + wei_batch_offset,
                                               p_acc_grid + acc_batch_offset,
                                               p_shared,
                                               in_grid_desc,
                                               wei_grid_desc,
                                               acc_grid_desc,
                                               in_element_op,
                                               wei_element_op,
                                               acc_element_op,
                                               block_2_ctile_map);
#else
    ignore = p_in_grid;
    ignore = p_wei_grid;
    ignore = p_acc_grid;
    ignore = in_grid_desc;
    ignore = we_grid_desc;
    ignore = acc_grid_desc;
    ignore = in_element_op;
    ignore = wei_element_op;
    ignore = acc_element_op;
    ignore = compute_ptr_offset_of_batch;
    ignore = block_2_ctile_map;
#endif
}

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
          index_t NumConvCPrefetchStage = 1,
          LoopScheduler LoopSched       = make_default_loop_scheduler(),
          PipelineVersion PipelineVer   = PipelineVersion::v1>
struct GridwiseConv_Wconv
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    static constexpr index_t YX = FilterSize * FilterSize;
    using NumberYX              = Number<FilterSize * FilterSize>;

    static constexpr auto wconv_conv = WconvConv<WeiDataType,
                                                 InDataType,
                                                 AccDataType,
                                                 HPerWconv,
                                                 WPerWconv,
                                                 FilterSize,
                                                 DilationX,
                                                 DilationY>{};

    using GridwiseConvPipe =
        GridwiseConvPipeline_v1<NumConvCPrefetchStage, InEnableLds, WeiEnableLds>;

    static constexpr index_t CPerWconv                = wconv_conv.GetNumInputChannels();
    static constexpr index_t KPerWconv                = wconv_conv.GetNumOutputChannels();
    static constexpr index_t NumWeightTape            = wconv_conv.GetNumWeightTape();
    static constexpr index_t NumSubTilesPerWeightTape = wconv_conv.GetNumSubTilesPerWeightTape();
    static constexpr index_t NumWeightCompPerTile     = wconv_conv.GetNumWeightCompPerTile();
    static constexpr index_t NumSubTilePerImage       = wconv_conv.GetNumSubTilesPerImageTile();
    static constexpr index_t NumDataCompPerTile       = wconv_conv.GetNumDataCompPerTile();
    static constexpr index_t DataTileHeight           = 4;
    static constexpr index_t H_Pad                    = (FilterSize == 3) ? DataTileHeight : 0;
    static constexpr index_t W_Pad                    = (FilterSize == 3) ? WPerWconv : 0;
    static constexpr index_t HPerBlockIn              = HPerBlock + H_Pad * 2;
    static constexpr index_t WPerBlockIn              = WPerBlock + W_Pad * 2;

    static constexpr index_t HPerWave   = HRepeat * HPerWconv;
    static constexpr index_t WPerWave   = WRepeat * WPerWconv;
    static constexpr index_t CPerWave   = CPerBlock;
    static constexpr index_t KPerWave   = KPerBlock;
    static constexpr index_t HPerWaveIn = HPerWave + H_Pad * 2;
    static constexpr index_t WPerWaveIn = WPerWave + W_Pad * 2;
    static_assert(HPerWaveIn % HPerWconv == 0, "");
    static_assert(WPerWaveIn % WPerWconv == 0, "");

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
            // H x W x C -> H0 x W0 x C0 x H1 x H2 x W1 x C1
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
                make_tuple(Sequence<0, 3, 4>{}, Sequence<1, 5>{}, Sequence<2, 6>{}));
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
                                          Number<NumSubTilesPerWeightTape>{},
                                          Number<CPerWconv / NumSubTilesPerWeightTape>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0, 3>{}, Sequence<2>{}, Sequence<1, 4, 5>{}));
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
                // H0 x W0 x C0 x H1 x H2 x W1 x C1
                return make_naive_tensor_descriptor_packed(
                    make_tuple(Number<HPerWaveIn / HPerWconv>{},
                               Number<WPerWaveIn / WPerWconv>{},
                               Number<CPerWave / CPerWconv>{},
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
                               Number<NumWeightTape>{},
                               I1,
                               Number<NumSubTilesPerWeightTape>{},
                               Number<NumWeightCompPerTile>{}));
            }
        }();

        return wei_block_desc;
    }

    __host__ __device__ static constexpr auto MakeInBlockSliceCopyStep()
    {
        constexpr auto in_block_copy_step = [&]() {
            if constexpr(InEnableLds)
            {
                return make_multi_index(0, 0, CPerBlock);
            }
            else
            {
                return make_multi_index(0, 0, CPerBlock / CPerWconv, 0, 0, 0, 0);
            }
        }();

        return in_block_copy_step;
    }

    __host__ __device__ static constexpr auto MakeWeiBlockSliceCopyStep()
    {
        constexpr auto wei_block_copy_step = [&]() {
            if constexpr(WeiEnableLds)
            {
                return make_multi_index(0, 0, CPerBlock);
            }
            else
            {
                return make_multi_index(0, CPerBlock / CPerWconv, 0, 0, 0, 0);
            }
        }();

        return wei_block_copy_step;
    }

    // Describe how data read from (LDS/VGPR) buffer
    template <typename InBlockDesc_>
    __host__ __device__ static constexpr auto MakeInWaveDescriptor(const InBlockDesc_&)
    {
        constexpr auto in_wave_desc = [&]() {
            if constexpr(InEnableLds)
            {
                // H x W x C -> H0 x W0 x C0 x H1 x H2 x W1 x C1
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
                    make_tuple(Sequence<0, 3, 4>{}, Sequence<1, 5>{}, Sequence<2, 6>{}));
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
                // K x YX x C -> K0 x C0 x YX x K1 x  C1 x C2
                return transform_tensor_descriptor(
                    WeiBlockDesc_{},
                    make_tuple(make_unmerge_transform(make_tuple(Number<KPerBlock / KPerWconv>{},
                                                                 Number<KPerWconv>{})),
                               make_pass_through_transform(Number<FilterSize * FilterSize>{}),
                               make_unmerge_transform(
                                   make_tuple(Number<CPerBlock / CPerWconv>{},
                                              Number<NumSubTilesPerWeightTape>{},
                                              Number<CPerWconv / NumSubTilesPerWeightTape>{}))),
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

    __host__ __device__ static constexpr auto GetAccBlockDescriptor()
    {
        constexpr auto acc_block_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<HPerBlock>{}, Number<WPerBlock>{}, Number<KPerBlock>{}));

        return acc_block_desc;
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2CTileMap>
    __host__ __device__ static constexpr bool CheckValidity(const InGridDesc& in_grid_desc,
                                                            const WeiGridDesc& wei_grid_desc,
                                                            const AccGridDesc& acc_grid_desc,
                                                            const Block2CTileMap& block_2_ctile_map)
    {
        // static_assert(is_known_at_compile_time<remove_cv_t<decltype(K1)>>::value,
        //               "wrong! K1 need to be known at compile-time");
        //
        // static_assert((MPerBlock % (MPerWmma * MRepeat) == 0) &&
        //                   (NPerBlock % (NRepeat * NPerWmma)) == 0,
        //               "Invalid tuning param!");
#if 0
        Debug<Number<WeiBlockDesc_{}.GetLength(I2)>> BDIM2;
        Debug<Number<WeiBlockDesc_{}.GetLength(I1)>> BDIM1;
        Debug<Number<WeiBlockDesc_{}.GetLength(I0)>> BDIM0;
        constexpr index_t Bstride0 = WeiBlockDesc_{}.CalculateOffset(make_tuple(I1, I0, I0));
        constexpr index_t Bstride1 = WeiBlockDesc_{}.CalculateOffset(make_tuple(I0, I1, I0));
        constexpr index_t Bstride2 = WeiBlockDesc_{}.CalculateOffset(make_tuple(I0, I0, I1));
        Debug<Number<Bstride0>> BSTR0;
        Debug<Number<Bstride1>> BSTR1;
        Debug<Number<Bstride2>> BSTR2;


        Debug<Number<wei_wave_desc.GetLength(I5)>> DIM5;
        Debug<Number<wei_wave_desc.GetLength(I4)>> DIM4;
        Debug<Number<wei_wave_desc.GetLength(I3)>> DIM3;
        Debug<Number<wei_wave_desc.GetLength(I2)>> DIM2;
        Debug<Number<wei_wave_desc.GetLength(I1)>> DIM1;
        Debug<Number<wei_wave_desc.GetLength(I0)>> DIM0;

        constexpr index_t stride0 =
            wei_wave_desc.CalculateOffset(make_tuple(I1, I0, I0, I0, I0, I0));
        constexpr index_t stride1 =
            wei_wave_desc.CalculateOffset(make_tuple(I0, I1, I0, I0, I0, I0));
        constexpr index_t stride2 =
            wei_wave_desc.CalculateOffset(make_tuple(I0, I0, I1, I0, I0, I0));
        constexpr index_t stride3 =
            wei_wave_desc.CalculateOffset(make_tuple(I0, I0, I0, I1, I0, I0));
        constexpr index_t stride4 =
            wei_wave_desc.CalculateOffset(make_tuple(I0, I0, I0, I0, I1, I0));
        constexpr index_t stride5 =
            wei_wave_desc.CalculateOffset(make_tuple(I0, I0, I0, I0, I0, I1));
        Debug<Number<stride0>> STR0;
        Debug<Number<stride1>> STR1;
        Debug<Number<stride2>> STR2;
        Debug<Number<stride3>> STR3;
        Debug<Number<stride4>> STR4;
        Debug<Number<stride5>> STR5;
#endif
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

        const auto H = GetInProblemsize()[I0];
        const auto W = GetInProblemsize()[I1];
        const auto C = GetInProblemsize()[I2];
        const auto K = GetWeiProblemsize()[I0];

        if(!(H == acc_grid_desc.GetLength(I0) && W == acc_grid_desc.GetLength(I1) &&
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
        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainBlockLoop(index_t C)
    {
        const index_t num_loop = C / CPerBlock;

        return GridwiseConvPipe::CalculateHasMainLoop(num_loop);
    }

    __host__ __device__ static constexpr auto
    MakeAccGridDescriptor_Block(const AccGridDesc& acc_grid_desc)
    {
        const auto H = acc_grid_desc.GetLength(I0);
        const auto W = acc_grid_desc.GetLength(I1);
        const auto K = acc_grid_desc.GetLength(I2);

        const auto HBlock              = H / HPerBlock;
        const auto WBlock              = W / WPerBlock;
        const auto KBlock              = K / KPerBlock;
        const auto acc_grid_desc_block = transform_tensor_descriptor(
            acc_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(HBlock, Number<HPerBlock>{})),
                       make_unmerge_transform(make_tuple(WBlock, Number<WPerBlock>{})),
                       make_unmerge_transform(make_tuple(KBlock, Number<KPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0, 3>{}, Sequence<1, 4>{}, Sequence<2, 5>{}));

        return acc_grid_desc_block;
    }

    // return block_id to Acc tensor tile idx (k0, w0, h0) mapping
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2CTileMap(const AccGridDesc& c_grid_desc_m_n, index_t M01, index_t /* N01 */)
    {
        return BlockToCTileMap_KSplit_M00_N0_M01Adapt<WPerBlock, HPerBlock, AccGridDesc>(
            c_grid_desc_m_n, M01, c_grid_desc_m_n.GetLength(I2) / KPerBlock);
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

    // Debug<Number<SharedMemTrait::lds_size>> ddddd;
    using AccGridDescriptor_Block =
        remove_cvref_t<decltype(MakeAccGridDescriptor_Block(AccGridDesc{}))>;
    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(AccGridDesc{}, 1, 1))>;

    template <bool HasMainBlockLoop, typename Block2CTileMap = DefaultBlock2CTileMap>
    __device__ static void Run(const InDataType* __restrict__ p_in_grid,
                               const WeiDataType* __restrict__ p_wei_grid,
                               AccDataType* __restrict__ p_acc_grid,
                               void* __restrict__ p_shared,
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
        auto acc_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_acc_grid, acc_grid_desc.GetElementSpaceSize());

        const auto in_grid_pad_desc  = MakeInGridPadDescriptor(in_grid_desc);
        const auto wei_grid_pad_desc = MakeWeiGridPadDescriptor(wei_grid_desc);

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
        const index_t h_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * HPerBlock);
        const index_t w_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I2] * WPerBlock);
        /*******************************************************************************/
        // BlockLevel, Tensor and filter ThreadMapping in WCNN Source buffer, As Destinaion of
        // BlockWise_Copy
        const auto C = in_grid_desc.GetLength(I2);

        constexpr auto in_block_desc  = MakeInBlockDescriptor();
        constexpr auto wei_block_desc = MakeWeiBlockDescriptor();
        auto wave_idx                 = GetWconvWaveIdx<BlockSize,
                                        HPerBlock,
                                        WPerBlock,
                                        HRepeat,
                                        WRepeat,
                                        HPerWconv,
                                        WPerWconv,
                                        32>();

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
            else
            {
                // Thread-wise copy
                // H0 x W0 x C0 x H1 x H2 x W1 x C1
                auto in_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, InDataType>(
                    in_block_desc.GetElementSpaceSize());

                auto indata_slice_origin_idx = wconv_conv.CalculateInDataThreadOriginDataIndex();
                auto h0 = (h_block_data_idx_on_grid + wave_idx[I0] * HPerWave) / HPerWconv;
                auto w0 = (w_block_data_idx_on_grid + wave_idx[I1] * WPerWave) / WPerWconv;

                // Limitation: NumDim of Src and Dst descriptor should be identical
                auto in_blockwise_copy =
                    ThreadwiseTensorSliceTransfer_v2<InDataType,
                                                     InDataType,
                                                     decltype(in_grid_pad_desc),
                                                     decltype(in_block_desc),
                                                     Sequence<HPerWaveIn / HPerWconv,
                                                              WPerWaveIn / WPerWconv,
                                                              CPerWave / CPerWconv,
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
                        make_multi_index(h0,
                                         w0,
                                         0,
                                         0,
                                         indata_slice_origin_idx[I0],
                                         indata_slice_origin_idx[I1],
                                         indata_slice_origin_idx[I2]));
                return make_tuple(in_block_buf, in_blockwise_copy);
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
            else
            {
                constexpr index_t Iters = GetFilterIters<WeiDataType,
                                                         InDataType,
                                                         AccDataType,
                                                         CPerBlock,
                                                         HPerWconv,
                                                         WPerWconv,
                                                         FilterSize>();

                auto wei_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, WeiDataType>(
                    wei_block_desc.GetElementSpaceSize());
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
                                                              NumSubTilesPerWeightTape,
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
                                make_multi_index(
                                    k0,
                                    0,
                                    I * wconv_conv.GetNumWeightTapePerWave() +
                                        wei_slice_origin_idx[I0] *
                                            wconv_conv.GetWeightSecondTapeMapTable()[I],
                                    wei_slice_origin_idx[I1],
                                    wei_slice_origin_idx[I2],
                                    wei_slice_origin_idx[I3]));
                        }
                    },
                    Number<NumWeightTape>{});
                return make_tuple(wei_block_buf, wei_blockwise_copy);
            }
        };

        auto in_block_buf      = in_block_trait()[I0];
        auto in_blockwise_copy = in_block_trait()[I1];

        auto wei_block_buf      = wei_block_trait()[I0];
        auto wei_blockwise_copy = wei_block_trait()[I1];
        /*******************************************************************************/
        // CONV
        auto blockwise_conv = BlockwiseConvWconv<BlockSize,
                                                 WeiDataType,
                                                 InDataType,
                                                 AccDataType,
                                                 decltype(MakeWeiWaveDescriptor(wei_block_desc)),
                                                 decltype(MakeInWaveDescriptor(in_block_desc)),
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
                                                 InEnableLds>{};

        // Prepare Register for Accum
        auto acc_thread_buf = blockwise_conv.GetAccumThreadBuffer();

        /*******************************************************************************/
        // Shift Per CPerBlock
        constexpr auto in_block_slice_copy_step  = MakeInBlockSliceCopyStep();
        constexpr auto wei_block_slice_copy_step = MakeWeiBlockSliceCopyStep();

        // Gridwise conv pipeline
        const index_t CBlockMainLoop = __builtin_amdgcn_readfirstlane(C / CPerBlock);
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
                                                         blockwise_conv,
                                                         acc_thread_buf,
                                                         CBlockMainLoop);
        /*******************************************************************************/
        // Store accum buffer
        // Todo: replace it with ThreadwiseTensorSliceTransfer
        if constexpr(AccEnableLds == false)
        {
            const index_t laneId     = threadIdx.x % 32;
            const index_t accCompIdx = laneId * wconv_conv.GetNumAccumComponents() /
                                       wconv_conv.GetNumSubTilesPerImageTile();
            auto store_accum_buf = [&](index_t h0, index_t w0, index_t k0) {
                static constexpr auto I0 = Number<0>{};
                static constexpr auto I1 = Number<1>{};
                static constexpr auto I2 = Number<2>{};

                const index_t acc_H_stride = acc_grid_desc.CalculateOffset(make_tuple(I1, I0, I0));
                const index_t acc_W_stride = acc_grid_desc.CalculateOffset(make_tuple(I0, I1, I0));
                const index_t acc_K_stride = acc_grid_desc.CalculateOffset(make_tuple(I0, I0, I1));

                const index_t subW = (accCompIdx / wconv_conv.GetNumOutputChannels()) % WPerWconv;
                const index_t subH = (accCompIdx / wconv_conv.GetNumOutputChannels()) / WPerWconv;
                using AccDataVec   = typename decltype(wconv_conv)::AccDataVec;
                constexpr auto KRepeat = blockwise_conv.KRepeat;
                const auto waveId      = blockwise_conv.GetWaveIdx();
                static_for<0, HRepeat, 1>{}([&](auto h1) {
                    static_for<0, WRepeat, 1>{}([&](auto w1) {
                        static_for<0, KRepeat, 1>{}([&](auto k1) {
                            constexpr auto accum_offset =
                                (h1 * WRepeat * KRepeat + w1 * KRepeat + k1) *
                                wconv_conv.GetNumAccumComponents();
                            auto& c_vec =
                                acc_thread_buf.GetVectorTypeReference(Number<accum_offset>{});
                            if constexpr(wconv_conv.GetNumAccumComponents() == 4)
                            {
                                const index_t subK = accCompIdx % wconv_conv.GetNumOutputChannels();

                                const index_t offset =
                                    (h0 + waveId[I0] * HRepeat * HPerWconv + h1 * HPerWconv +
                                     subH) *
                                        acc_H_stride +
                                    (w0 + waveId[I1] * WRepeat * WPerWconv + w1 * WPerWconv +
                                     subW) *
                                        acc_W_stride +
                                    (k0 + waveId[I2] * KRepeat * KPerWconv + k1 * KPerWconv + subK);
                                *(typename AccDataVec::type*)(p_acc_grid + offset) =
                                    c_vec.template AsType<typename AccDataVec::type>()(Number<0>{});
                            }
                            else
                            {
                                static_assert(wconv_conv.GetNumAccumComponents() == 8,
                                              "unexpected value");
                                // ACO = 0, do swizzle after 4 channels.
                                using AccSwizzleVec = typename vector_type<AccDataType, 4>::type;
                                const index_t subK  = accCompIdx %
                                                     wconv_conv.GetNumOutputChannels() /
                                                     (wconv_conv.GetNumAccumComponents() * 2) *
                                                     (wconv_conv.GetNumAccumComponents() * 2);
                                const index_t offset =
                                    (h0 + waveId[I0] * HRepeat * HPerWconv + h1 * HPerWconv +
                                     subH) *
                                        acc_H_stride +
                                    (w0 + waveId[I1] * WRepeat * WPerWconv + w1 * WPerWconv +
                                     subW) *
                                        acc_W_stride +
                                    (k0 + waveId[I2] * KRepeat * KPerWconv + k1 * KPerWconv + subK);

                                index_t secOffset = 8;
                                if constexpr(wconv_conv.GetNumSubTilesPerImageTile() > 1)
                                {
                                    secOffset = HPerWconv /
                                                wconv_conv.GetNumSubTilesPerImageTile() *
                                                acc_H_stride;
                                }
                                const index_t swizzleOffset = (laneId & 1) * 4;
                                *(AccSwizzleVec*)(p_acc_grid + offset + swizzleOffset) =
                                    c_vec.template AsType<AccSwizzleVec>()(Number<0>{});
                                *(AccSwizzleVec*)(p_acc_grid + offset + swizzleOffset + secOffset) =
                                    c_vec.template AsType<AccSwizzleVec>()(Number<1>{});
                            }
                        });
                    });
                });
            };

            store_accum_buf(
                h_block_data_idx_on_grid, w_block_data_idx_on_grid, k_block_data_idx_on_grid);
        }
        else
        {
            // C mapping in single thread.
            constexpr auto acc_thread_desc   = blockwise_conv.GetAccThreadDescriptor();
            constexpr auto acc_thread_length = blockwise_conv.GetAccThreadDescLength();

            // C mapping in single block
            // LDS descriptor, shuffle and write out in HRepeat x WRepeat x KRepeat times
            constexpr auto acc_block_desc = GetAccBlockDescriptor();
            constexpr auto acc_block_wave_desc =
                blockwise_conv.GetAccBlockWaveDescriptor(acc_block_desc);

            auto acc_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<AccDataType*>(p_shared) + SharedMemTrait::acc_block_space_offset,
                SharedMemTrait::acc_block_space_size);

            // calculate origin of thread output tensor on global memory

            // blockwise conv acc starting index
            const auto acc_thread_mtx_on_block = blockwise_conv.CalculateAccThreadOriginDataIndex();

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

            // shuffle: blockwise copy C from LDS to global
            auto acc_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
                ThisThreadBlock,                           // ThreadGroup
                AccElementwiseOperation,                   // ElementwiseOperation,
                InMemoryDataOperationEnum::Set,            // DstInMemOp,
                Sequence<HPerBlock, WPerBlock, KPerBlock>, // BlockSliceLengths,
                AccBlockTransferClusterLengths,
                Sequence<0, 1, 2>, // typename ThreadClusterArrangeOrder,
                AccDataType,       // typename SrcData,
                AccDataType,       // typename DstData,
                decltype(acc_block_desc),
                decltype(acc_grid_desc),
                Sequence<0, 1, 2>,               // typename DimAccessOrder,
                2,                               // index_t VectorDim,
                AccBlockTransferScalarPerVector, // index_t ScalarPerVector,
                true,                            // bool ThreadTransferSrcResetCoordinateAfterRun,
                false>                           // bool ThreadTransferDstResetCoordinateAfterRun>
                {acc_block_desc,
                 make_multi_index(0, 0, 0),
                 acc_grid_desc,
                 make_multi_index(
                     h_block_data_idx_on_grid, w_block_data_idx_on_grid, k_block_data_idx_on_grid),
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
};

} // namespace ck
