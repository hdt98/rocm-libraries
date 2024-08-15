// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <index_t NumPrefetch, bool InDataEnableLds, bool WeiDataEnableLds>
struct GridwiseConvPipeline_v1;

template <typename T>
struct Debug;

// 1-stage prefetch
template <>
struct GridwiseConvPipeline_v1<1, true, true>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
    }

    template <bool HasMainLoop,
              typename InDataGridDesc,
              typename InDataBlockDesc,
              typename InDataBlockTransfer,
              typename InDataGridBuffer,
              typename InDataBlockBuffer,
              typename InDataBlockTransferStep,
              typename WeiDataGridDesc,
              typename WeiDataBlockDesc,
              typename WeiDataBlockTransfer,
              typename WeiDataGridBuffer,
              typename WeiDataBlockBuffer,
              typename WeiDataBlockTransferStep,
              typename BlockwiseConv,
              typename AccumThreadBuffer>
    __device__ static void Run(const InDataGridDesc& in_grid_desc,
                               const InDataBlockDesc& in_block_desc,
                               InDataBlockTransfer& in_blockwise_copy,
                               const InDataGridBuffer& in_grid_buf,
                               InDataBlockBuffer& in_block_buf,
                               const InDataBlockTransferStep& in_block_copy_step,
                               const WeiDataGridDesc& wei_grid_desc,
                               const WeiDataBlockDesc& wei_block_desc,
                               WeiDataBlockTransfer& wei_blockwise_copy,
                               const WeiDataGridBuffer& wei_grid_buf,
                               WeiDataBlockBuffer& wei_block_buf,
                               const WeiDataBlockTransferStep& wei_block_copy_step,
                               const BlockwiseConv& blockwise_conv,
                               AccumThreadBuffer& accum_thread_buf,
                               index_t num_loop)
    {
        constexpr index_t NumTape = wei_blockwise_copy.Size();
        using WeiDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I0])>>;

        // preload data into LDS
        static_for<0, NumTape, 1>{}([&](auto i) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                .RunRead(wei_grid_desc, wei_grid_buf);
        });

        in_blockwise_copy.RunRead(in_grid_desc, in_grid_buf);

        static_for<0, NumTape, 1>{}([&](auto i) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
        });
        in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

        // Initialize C
        accum_thread_buf.Clear();
        static_for<0, NumTape, 1>{}([&](auto i) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                .RunWrite(wei_block_desc, wei_block_buf);
        });

        in_blockwise_copy.RunWrite(in_block_desc, in_block_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                static_for<0, NumTape, 1>{}([&](auto i) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                        .RunRead(wei_grid_desc, wei_grid_buf);
                });

                block_sync_lds();

                in_blockwise_copy.RunRead(in_grid_desc, in_grid_buf);

                blockwise_conv.Run(wei_block_buf, in_block_buf, accum_thread_buf);

                block_sync_lds();

                static_for<0, NumTape, 1>{}([&](auto i) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                        .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                });

                in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

                static_for<0, NumTape, 1>{}([&](auto i) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                        .RunWrite(wei_block_desc, wei_block_buf);
                });

                in_blockwise_copy.RunWrite(in_block_desc, in_block_buf);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_conv.Run(wei_block_buf, in_block_buf, accum_thread_buf);
        }
    }
};

template <>
struct GridwiseConvPipeline_v1<1, false, true>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
    }

    template <bool HasMainLoop,
              typename InDataGridDesc,
              typename InDataBlockDesc,
              typename InDataBlockTransfer,
              typename InDataGridBuffer,
              typename InDataBlockBuffer,
              typename InDataBlockTransferStep,
              typename WeiDataGridDesc,
              typename WeiDataBlockDesc,
              typename WeiDataBlockTransfer,
              typename WeiDataGridBuffer,
              typename WeiDataBlockBuffer,
              typename WeiDataBlockTransferStep,
              typename BlockwiseConv,
              typename AccumThreadBuffer>
    __device__ static void Run(const InDataGridDesc& in_grid_desc,
                               const InDataBlockDesc& in_block_desc,
                               InDataBlockTransfer& in_blockwise_copy,
                               const InDataGridBuffer& in_grid_buf,
                               InDataBlockBuffer& in_block_buf,
                               const InDataBlockTransferStep& in_block_copy_step,
                               const WeiDataGridDesc& wei_grid_desc,
                               const WeiDataBlockDesc& wei_block_desc,
                               WeiDataBlockTransfer& wei_blockwise_copy,
                               const WeiDataGridBuffer& wei_grid_buf,
                               WeiDataBlockBuffer& wei_block_buf,
                               const WeiDataBlockTransferStep& wei_block_copy_step,
                               const BlockwiseConv& blockwise_conv,
                               AccumThreadBuffer& accum_thread_buf,
                               index_t num_loop)
    {
        constexpr index_t NumTape          = wei_blockwise_copy.Size();
        constexpr auto in_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
        auto in_block_buf_switch           = in_block_buf;

        using WeiDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I0])>>;

        // preload data into LDS
        static_for<0, NumTape, 1>{}([&](auto i) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                .RunRead(wei_grid_desc, wei_grid_buf);
        });

        in_blockwise_copy.Run(
            in_grid_desc, in_grid_buf, in_block_desc, in_block_origin_idx, in_block_buf);

        static_for<0, NumTape, 1>{}([&](auto i) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
        });
        in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

        // Initialize C
        accum_thread_buf.Clear();
        static_for<0, NumTape, 1>{}([&](auto i) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                .RunWrite(wei_block_desc, wei_block_buf);
        });

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                in_blockwise_copy.Run(in_grid_desc,
                                      in_grid_buf,
                                      in_block_desc,
                                      in_block_origin_idx,
                                      in_block_buf_switch);

                block_sync_lds();

                static_for<0, NumTape, 1>{}([&](auto i) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                        .RunRead(wei_grid_desc, wei_grid_buf);
                });

                blockwise_conv.Run(wei_block_buf, in_block_buf, accum_thread_buf);

                block_sync_lds();

                static_for<0, NumTape, 1>{}([&](auto i) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                        .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                });

                in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

                static_for<0, NumTape, 1>{}([&](auto i) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                        .RunWrite(wei_block_desc, wei_block_buf);
                });

                in_block_buf = in_block_buf_switch;
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_conv.Run(wei_block_buf, in_block_buf, accum_thread_buf);
        }
    }
};

template <>
struct GridwiseConvPipeline_v1<1, false, false>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
    }

    template <bool HasMainLoop,
              typename InDataGridDesc,
              typename InDataBlockDesc,
              typename InDataBlockTransfer,
              typename InDataGridBuffer,
              typename InDataBlockBuffer,
              typename InDataBlockTransferStep,
              typename WeiDataGridDesc,
              typename WeiDataBlockDesc,
              typename WeiDataBlockTransfer,
              typename WeiDataGridBuffer,
              typename WeiDataBlockBuffer,
              typename WeiDataBlockTransferStep,
              typename BlockwiseConv,
              typename AccumThreadBuffer>
    __device__ static void Run(const InDataGridDesc& in_grid_desc,
                               const InDataBlockDesc& in_block_desc,
                               InDataBlockTransfer& in_blockwise_copy,
                               const InDataGridBuffer& in_grid_buf,
                               InDataBlockBuffer& in_block_buf,
                               const InDataBlockTransferStep& in_block_copy_step,
                               const WeiDataGridDesc& wei_grid_desc,
                               const WeiDataBlockDesc& wei_block_desc,
                               WeiDataBlockTransfer& wei_blockwise_copy,
                               const WeiDataGridBuffer& wei_grid_buf,
                               WeiDataBlockBuffer& wei_block_buf,
                               const WeiDataBlockTransferStep& wei_block_copy_step,
                               const BlockwiseConv& blockwise_conv,
                               AccumThreadBuffer& accum_thread_buf,
                               index_t num_loop)
    {
        constexpr index_t NumTape          = wei_blockwise_copy.Size();
        constexpr auto in_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
        auto in_block_buf_switch           = in_block_buf;
        auto wei_block_buf_switch          = wei_block_buf;
        constexpr auto wei_remap_table     = blockwise_conv.GetWeightRemapTable();

        using WeiDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I0])>>;

        // preload data into LDS
        static_for<0, NumTape, 1>{}([&](auto i) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                .Run(wei_grid_desc,
                     wei_grid_buf,
                     wei_block_desc,
                     make_tuple(I0, I0, wei_remap_table[i], I0, I0, I0),
                     wei_block_buf);
        });

        in_blockwise_copy.Run(
            in_grid_desc, in_grid_buf, in_block_desc, in_block_origin_idx, in_block_buf);

        static_for<0, NumTape, 1>{}([&](auto i) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
        });
        in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

        // Initialize C
        accum_thread_buf.Clear();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                static_for<0, NumTape, 1>{}([&](auto i) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                        .Run(wei_grid_desc,
                             wei_grid_buf,
                             wei_block_desc,
                             make_tuple(I0, I0, wei_remap_table[i], I0, I0, I0),
                             wei_block_buf_switch);
                });
                in_blockwise_copy.Run(in_grid_desc,
                                      in_grid_buf,
                                      in_block_desc,
                                      in_block_origin_idx,
                                      in_block_buf_switch);

                block_sync_lds();

                blockwise_conv.Run(wei_block_buf, in_block_buf, accum_thread_buf);

                block_sync_lds();

                static_for<0, NumTape, 1>{}([&](auto i) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                        .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                });

                in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

                in_block_buf  = in_block_buf_switch;
                wei_block_buf = wei_block_buf_switch;
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_conv.Run(wei_block_buf, in_block_buf, accum_thread_buf);
        }
    }
};

template <>
struct GridwiseConvPipeline_v1<1, true, false>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
    }

    template <bool HasMainLoop,
              typename InDataGridDesc,
              typename InDataBlockDesc,
              typename InDataBlockTransfer,
              typename InDataGridBuffer,
              typename InDataBlockBuffer,
              typename InDataBlockTransferStep,
              typename WeiDataGridDesc,
              typename WeiDataBlockDesc,
              typename WeiDataBlockTransfer,
              typename WeiDataGridBuffer,
              typename WeiDataBlockBuffer,
              typename WeiDataBlockTransferStep,
              typename BlockwiseConv,
              typename AccumThreadBuffer>
    __device__ static void Run(const InDataGridDesc& in_grid_desc,
                               const InDataBlockDesc& in_block_desc,
                               InDataBlockTransfer& in_blockwise_copy,
                               const InDataGridBuffer& in_grid_buf,
                               InDataBlockBuffer& in_block_buf,
                               const InDataBlockTransferStep& in_block_copy_step,
                               const WeiDataGridDesc& wei_grid_desc,
                               const WeiDataBlockDesc& wei_block_desc,
                               WeiDataBlockTransfer& wei_blockwise_copy,
                               const WeiDataGridBuffer& wei_grid_buf,
                               WeiDataBlockBuffer& wei_block_buf,
                               const WeiDataBlockTransferStep& wei_block_copy_step,
                               const BlockwiseConv& blockwise_conv,
                               AccumThreadBuffer& accum_thread_buf,
                               index_t num_loop)
    {
        constexpr index_t NumTape      = wei_blockwise_copy.Size();
        constexpr auto wei_remap_table = blockwise_conv.GetWeightRemapTable();
        auto wei_block_buf_switch      = wei_block_buf;

        using WeiDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I0])>>;

        // preload data into LDS
        static_for<0, NumTape, 1>{}([&](auto i) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                .Run(wei_grid_desc,
                     wei_grid_buf,
                     wei_block_desc,
                     make_tuple(I0, I0, wei_remap_table[i], I0, I0, I0),
                     wei_block_buf);
        });

        in_blockwise_copy.RunRead(in_grid_desc, in_grid_buf);

        static_for<0, NumTape, 1>{}([&](auto i) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
        });
        in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

        // Initialize C
        accum_thread_buf.Clear();

        in_blockwise_copy.RunWrite(in_block_desc, in_block_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                static_for<0, NumTape, 1>{}([&](auto i) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                        .Run(wei_grid_desc,
                             wei_grid_buf,
                             wei_block_desc,
                             make_tuple(I0, I0, wei_remap_table[i], I0, I0, I0),
                             wei_block_buf_switch);
                });

                block_sync_lds();

                in_blockwise_copy.RunRead(in_grid_desc, in_grid_buf);

                blockwise_conv.Run(wei_block_buf, in_block_buf, accum_thread_buf);

                block_sync_lds();

                static_for<0, NumTape, 1>{}([&](auto i) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                        .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                });

                in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

                in_blockwise_copy.RunWrite(in_block_desc, in_block_buf);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_conv.Run(wei_block_buf, in_block_buf, accum_thread_buf);
        }
    }
};
} // namespace ck
