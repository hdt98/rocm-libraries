// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck/utility/amd_semaphore.hpp"
#include "ck/utility/amd_named_barrier.hpp"

#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"

namespace ck {
// Gridwise convolution pipeline with wavegroup enabled
// wave 0: load data
// wave 1: do convolution
// NOTE: Prefetch doesn't supported when LDS is enabled.
template <index_t NumPrefetch,
          bool InDataEnableLds,
          bool WeiDataEnableLds,
          bool DsDataEnableLds,
          bool EnableAsync>
struct GridwiseConvPipeline_v2
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
              typename DsDataGridDesc,
              typename DsDataBlockDesc,
              typename DsDataBlockTransfer,
              typename DsDataGridBuffer,
              typename DsDataBlockBuffer,
              typename BlockwiseConv,
              typename AccumThreadBuffer>
    __device__ static void Run(const InDataGridDesc& in_grid_desc,
                               const InDataBlockDesc& in_block_desc,
                               InDataBlockTransfer& in_blockwise_copy,
                               const InDataGridBuffer& in_grid_buf,
                               InDataBlockBuffer& in_block_buf,
                               const InDataBlockTransferStep&,
                               const WeiDataGridDesc& wei_grid_desc,
                               const WeiDataBlockDesc& wei_block_desc,
                               WeiDataBlockTransfer& wei_blockwise_copy,
                               const WeiDataGridBuffer& wei_grid_buf,
                               WeiDataBlockBuffer& wei_block_buf,
                               const WeiDataBlockTransferStep&,
                               const DsDataGridDesc& ds_grid_desc,
                               const DsDataBlockDesc& ds_block_desc,
                               DsDataBlockTransfer& ds_blockwise_copy,
                               const DsDataGridBuffer& ds_grid_buf,
                               DsDataBlockBuffer& ds_block_buf,
                               const BlockwiseConv& blockwise_conv,
                               AccumThreadBuffer& accum_thread_buf,
                               index_t num_loop)
    {
        constexpr auto wei_block_copy_step = to_multi_index(WeiDataBlockTransferStep{});
        constexpr auto in_block_copy_step  = to_multi_index(InDataBlockTransferStep{});

        // sync between data load wave (0) and conv wave (1)
        WavegroupSemaphore<1, 1> semaLoad;
        WavegroupSemaphore<1, 2> semaLds;
        WavegroupSemaphore<0, 1> semaRun;

        // sync for all wave with id = 0 in a group
        NamedBarrier<1> barrierLds;

        constexpr index_t NumTap           = wei_blockwise_copy.Size();
        constexpr auto in_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
        constexpr auto wei_remap_table     = blockwise_conv.GetWeightRemapTable();

        using WeiDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I0])>>;

        if(get_wave_id_in_wavegroup() == 1)
        {
            // Initialize C
            accum_thread_buf.Clear();
        }

        if(get_wave_id_in_wavegroup() == 0)
        {
            barrierLds.init(4);
            barrierLds.join();
        }

        semaLoad.init();
        semaLds.init();
        semaRun.init(1, 1, true);
        // wait semaphore, named-barrier init
        __syncthreads();

        // pre-fetch data
        if(get_wave_id_in_wavegroup() == 0)
        {
            if constexpr(WeiDataEnableLds == false)
            {
                static_for<0, NumTap, 1>{}([&](auto i) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                        .Run(wei_grid_desc,
                             wei_grid_buf,
                             wei_block_desc,
                             make_tuple(I0, I0, wei_remap_table[i], I0, I0, I0),
                             wei_block_buf);
                });
                if constexpr(HasMainLoop)
                {
                    static_for<0, NumTap, 1>{}([&](auto i) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                            .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                    });
                    wei_block_buf.SwitchBuffer();
                }
            }

            if constexpr(InDataEnableLds == false)
            {
                in_blockwise_copy.Run(
                    in_grid_desc, in_grid_buf, in_block_desc, in_block_origin_idx, in_block_buf);
                if constexpr(HasMainLoop)
                {
                    in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);
                    in_block_buf.SwitchBuffer();
                }
            }

            if constexpr(DsDataEnableLds == false)
            {
                constexpr index_t NumDs = ds_blockwise_copy.Size();
                using DsDataBlockTransfer0 =
                    std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[I0])>>;

                static_for<0, NumDs, 1>{}([&](auto i) {
                    const_cast<DsDataBlockTransfer0&>(ds_blockwise_copy[i])
                        .Run(ds_grid_desc[Number<i>{}],
                             ds_grid_buf[Number<i>{}],
                             ds_block_desc[Number<i>{}],
                             make_tuple(I0, I0, I0),
                             ds_block_buf(i));
                });
            }
            semaLoad.template signal<SemaphoreAddressSpaceGlobal>();
        }

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                if(get_wave_id_in_wavegroup() == 0)
                {
                    semaRun.template wait<0>(); // sync within wavegroup
                    barrierLds.signal();        // sync in workgroup
                    barrierLds.wait();
                    if constexpr(WeiDataEnableLds)
                    {
                        if constexpr(EnableAsync)
                        {
                            static_for<0, NumTap, 1>{}([&](auto i) {
                                const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                                    .Run(
                                        wei_grid_desc, wei_grid_buf, wei_block_desc, wei_block_buf);
                            });
                        }
                        else
                        {
                            static_for<0, NumTap, 1>{}([&](auto i) {
                                const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                                    .RunRead(wei_grid_desc, wei_grid_buf);
                            });
                            static_for<0, NumTap, 1>{}([&](auto i) {
                                const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                                    .RunWrite(wei_block_desc, wei_block_buf);
                            });
                        }
                    }
                    else
                    {
                        static_for<0, NumTap, 1>{}([&](auto i) {
                            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                                .Run(wei_grid_desc,
                                     wei_grid_buf,
                                     wei_block_desc,
                                     make_tuple(I0, I0, wei_remap_table[i], I0, I0, I0),
                                     wei_block_buf);
                        });
                        wei_block_buf.SwitchBuffer();
                    }

                    if constexpr(InDataEnableLds)
                    {
                        if constexpr(EnableAsync)
                        {
                            in_blockwise_copy.Run(
                                in_grid_desc, in_grid_buf, in_block_desc, in_block_buf);
                        }
                        else
                        {
                            in_blockwise_copy.RunRead(in_grid_desc, in_grid_buf);
                            in_blockwise_copy.RunWrite(in_block_desc, in_block_buf);
                        }
                    }
                    else
                    {
                        in_blockwise_copy.Run(in_grid_desc,
                                              in_grid_buf,
                                              in_block_desc,
                                              in_block_origin_idx,
                                              in_block_buf);
                        in_block_buf.SwitchBuffer();
                    }

                    static_for<0, NumTap, 1>{}([&](auto i) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                            .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                    });
                    in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

                    barrierLds.sync_lds<EnableAsync>();

                    semaLds.template signal<0>();
                    semaLoad.template signal<SemaphoreAddressSpaceGlobal>();
                }

                if(get_wave_id_in_wavegroup() == 1)
                {
                    semaLds.template wait<0>();
                    semaLoad.template wait<0>();
                    blockwise_conv.Run(
                        wei_block_buf, in_block_buf, ds_block_buf, accum_thread_buf, false);
                    if constexpr(InDataEnableLds == false)
                    {
                        in_block_buf.SwitchBuffer();
                    }
                    if constexpr(WeiDataEnableLds == false)
                    {
                        wei_block_buf.SwitchBuffer();
                    }

                    semaRun.template signal<0>();
                }

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            if(get_wave_id_in_wavegroup() == 0)
            {
                semaRun.template wait<0>();
                if constexpr(WeiDataEnableLds)
                {
                    if constexpr(EnableAsync)
                    {
                        static_for<0, NumTap, 1>{}([&](auto i) {
                            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                                .Run(wei_grid_desc, wei_grid_buf, wei_block_desc, wei_block_buf);
                        });
                    }
                    else
                    {
                        static_for<0, NumTap, 1>{}([&](auto i) {
                            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                                .RunRead(wei_grid_desc, wei_grid_buf);
                        });
                        static_for<0, NumTap, 1>{}([&](auto i) {
                            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                                .RunWrite(wei_block_desc, wei_block_buf);
                        });
                    }
                }

                if constexpr(InDataEnableLds)
                {
                    if constexpr(EnableAsync)
                    {
                        in_blockwise_copy.Run(
                            in_grid_desc, in_grid_buf, in_block_desc, in_block_buf);
                    }
                    else
                    {
                        in_blockwise_copy.RunRead(in_grid_desc, in_grid_buf);
                        in_blockwise_copy.RunWrite(in_block_desc, in_block_buf);
                    }
                }

                barrierLds.sync_lds<EnableAsync>();
                semaLds.template signal<0>();
            }

            if(get_wave_id_in_wavegroup() == 1)
            {
                semaLds.template wait<0>();
                semaLoad.template wait<0>();
                blockwise_conv.Run(
                    wei_block_buf, in_block_buf, ds_block_buf, accum_thread_buf, true);
            }
        }
    }
};

template <bool EnableAsync>
struct GridwiseConvPipeline_v2<1, false, false, false, EnableAsync>
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
              typename DsDataGridDesc,
              typename DsDataBlockDesc,
              typename DsDataBlockTransfer,
              typename DsDataGridBuffer,
              typename DsDataBlockBuffer,
              typename BlockwiseConv,
              typename AccumThreadBuffer>
    __device__ static void Run(const InDataGridDesc& in_grid_desc,
                               const InDataBlockDesc& in_block_desc,
                               InDataBlockTransfer& in_blockwise_copy,
                               const InDataGridBuffer& in_grid_buf,
                               InDataBlockBuffer& in_block_buf,
                               const InDataBlockTransferStep&,
                               const WeiDataGridDesc& wei_grid_desc,
                               const WeiDataBlockDesc& wei_block_desc,
                               WeiDataBlockTransfer& wei_blockwise_copy,
                               const WeiDataGridBuffer& wei_grid_buf,
                               WeiDataBlockBuffer& wei_block_buf,
                               const WeiDataBlockTransferStep&,
                               const DsDataGridDesc& ds_grid_desc,
                               const DsDataBlockDesc& ds_block_desc,
                               DsDataBlockTransfer& ds_blockwise_copy,
                               const DsDataGridBuffer& ds_grid_buf,
                               DsDataBlockBuffer& ds_block_buf,
                               const BlockwiseConv& blockwise_conv,
                               AccumThreadBuffer& accum_thread_buf,
                               index_t num_loop)
    {
        constexpr auto wei_block_copy_step = to_multi_index(WeiDataBlockTransferStep{});
        constexpr auto in_block_copy_step  = to_multi_index(InDataBlockTransferStep{});

        // sync between data load wave (0) and conv wave (1)
        WavegroupSemaphore<1, 1> semaLoad;
        WavegroupSemaphore<0, 1> semaRun;

        constexpr index_t NumTap           = wei_blockwise_copy.Size();
        constexpr auto in_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
        constexpr auto wei_remap_table     = blockwise_conv.GetWeightRemapTable();

        using WeiDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I0])>>;

        if(get_wave_id_in_wavegroup() == 1)
        {
            // Initialize C
            accum_thread_buf.Clear();
        }

        semaLoad.init();
        semaRun.init(1, 1, true);
        // wait semaphore init
        __syncthreads();

        // pre-fetch data
        if(get_wave_id_in_wavegroup() == 0)
        {
            static_for<0, NumTap, 1>{}([&](auto i) {
                const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                    .Run(wei_grid_desc,
                         wei_grid_buf,
                         wei_block_desc,
                         make_tuple(I0, I0, wei_remap_table[i], I0, I0, I0),
                         wei_block_buf);
            });

            in_blockwise_copy.Run(
                in_grid_desc, in_grid_buf, in_block_desc, in_block_origin_idx, in_block_buf);

            constexpr index_t NumDs = ds_blockwise_copy.Size();
            using DsDataBlockTransfer0 =
                std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[I0])>>;
            static_for<0, NumDs, 1>{}([&](auto i) {
                const_cast<DsDataBlockTransfer0&>(ds_blockwise_copy[i])
                    .Run(ds_grid_desc[Number<i>{}],
                         ds_grid_buf[Number<i>{}],
                         ds_block_desc[Number<i>{}],
                         make_tuple(I0, I0, I0),
                         ds_block_buf(i));
            });

            if constexpr(HasMainLoop)
            {
                static_for<0, NumTap, 1>{}([&](auto i) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                        .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                });
                in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);
                in_block_buf.SwitchBuffer();
                wei_block_buf.SwitchBuffer();
            }

            semaLoad.template signal<SemaphoreAddressSpaceGlobal>();
        }

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                if(get_wave_id_in_wavegroup() == 0)
                {
                    semaRun.template wait<0>();
                    static_for<0, NumTap, 1>{}([&](auto i) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                            .Run(wei_grid_desc,
                                 wei_grid_buf,
                                 wei_block_desc,
                                 make_tuple(I0, I0, wei_remap_table[i], I0, I0, I0),
                                 wei_block_buf);
                    });
                    wei_block_buf.SwitchBuffer();

                    in_blockwise_copy.Run(in_grid_desc,
                                          in_grid_buf,
                                          in_block_desc,
                                          in_block_origin_idx,
                                          in_block_buf);
                    in_block_buf.SwitchBuffer();

                    static_for<0, NumTap, 1>{}([&](auto i) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                            .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                    });
                    in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

                    semaLoad.template signal<SemaphoreAddressSpaceGlobal>();
                }

                if(get_wave_id_in_wavegroup() == 1)
                {
                    semaLoad.template wait<0>();
                    blockwise_conv.Run(
                        wei_block_buf, in_block_buf, ds_block_buf, accum_thread_buf, false);
                    in_block_buf.SwitchBuffer();
                    wei_block_buf.SwitchBuffer();
                    semaRun.template signal<0>();
                }

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            if(get_wave_id_in_wavegroup() == 1)
            {
                semaLoad.template wait<0>();
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "global");
                blockwise_conv.Run(
                    wei_block_buf, in_block_buf, ds_block_buf, accum_thread_buf, true);
            }
        }
    }
};

template <bool EnableAsync>
struct GridwiseConvPipeline_v2<1, true, true, true, EnableAsync>
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
              typename DsDataGridDesc,
              typename DsDataBlockDesc,
              typename DsDataBlockTransfer,
              typename DsDataGridBuffer,
              typename DsDataBlockBuffer,
              typename BlockwiseConv,
              typename AccumThreadBuffer>
    __device__ static void Run(const InDataGridDesc& in_grid_desc,
                               const InDataBlockDesc& in_block_desc,
                               InDataBlockTransfer& in_blockwise_copy,
                               const InDataGridBuffer& in_grid_buf,
                               InDataBlockBuffer& in_block_buf,
                               const InDataBlockTransferStep&,
                               const WeiDataGridDesc& wei_grid_desc,
                               const WeiDataBlockDesc& wei_block_desc,
                               WeiDataBlockTransfer& wei_blockwise_copy,
                               const WeiDataGridBuffer& wei_grid_buf,
                               WeiDataBlockBuffer& wei_block_buf,
                               const WeiDataBlockTransferStep&,
                               const DsDataGridDesc& ds_grid_desc,
                               const DsDataBlockDesc& ds_block_desc,
                               DsDataBlockTransfer& ds_blockwise_copy,
                               const DsDataGridBuffer& ds_grid_buf,
                               DsDataBlockBuffer& ds_block_buf,
                               const BlockwiseConv& blockwise_conv,
                               AccumThreadBuffer& accum_thread_buf,
                               index_t num_loop)
    {
        constexpr auto wei_block_copy_step = to_multi_index(WeiDataBlockTransferStep{});
        constexpr auto in_block_copy_step  = to_multi_index(InDataBlockTransferStep{});

        // sync between data load wave (0) and conv wave (1)
        WavegroupSemaphore<1, 1> semaLds;
        WavegroupSemaphore<0, 1> semaRun;
        NamedBarrier<1> barrierLds;

        constexpr index_t NumTap = wei_blockwise_copy.Size();

        using WeiDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I0])>>;

        if(get_wave_id_in_wavegroup() == 1)
        {
            // Initialize C
            accum_thread_buf.Clear();
        }

        if(get_wave_id_in_wavegroup() == 0)
        {
            barrierLds.init(4);
            barrierLds.join();
        }

        semaLds.init();
        semaRun.init(1, 1, true);
        // wait semaphore init
        __syncthreads();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                if(get_wave_id_in_wavegroup() == 0)
                {
                    semaRun.template wait<0>();
                    barrierLds.signal();
                    barrierLds.wait();
                    if constexpr(EnableAsync)
                    {
                        static_for<0, NumTap, 1>{}([&](auto i) {
                            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                                .Run(wei_grid_desc, wei_grid_buf, wei_block_desc, wei_block_buf);
                        });
                    }
                    else
                    {
                        static_for<0, NumTap, 1>{}([&](auto i) {
                            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                                .RunRead(wei_grid_desc, wei_grid_buf);
                        });
                        static_for<0, NumTap, 1>{}([&](auto i) {
                            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                                .RunWrite(wei_block_desc, wei_block_buf);
                        });
                    }

                    if constexpr(EnableAsync)
                    {
                        in_blockwise_copy.Run(
                            in_grid_desc, in_grid_buf, in_block_desc, in_block_buf);
                    }
                    else
                    {
                        in_blockwise_copy.RunRead(in_grid_desc, in_grid_buf);
                        in_blockwise_copy.RunWrite(in_block_desc, in_block_buf);
                    }

                    static_for<0, NumTap, 1>{}([&](auto i) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                            .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                    });
                    in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);
                    barrierLds.sync_lds<EnableAsync>();

                    semaLds.template signal<0>();
                }

                if(get_wave_id_in_wavegroup() == 1)
                {
                    semaLds.template wait<0>();
                    blockwise_conv.Run(
                        wei_block_buf, in_block_buf, ds_block_buf, accum_thread_buf, false);
                    semaRun.template signal<0>();
                }
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            if(get_wave_id_in_wavegroup() == 0)
            {
                semaRun.template wait<0>();
                barrierLds.signal();
                barrierLds.wait();
                if constexpr(EnableAsync)
                {
                    static_for<0, NumTap, 1>{}([&](auto i) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                            .Run(wei_grid_desc, wei_grid_buf, wei_block_desc, wei_block_buf);
                    });
                }
                else
                {
                    static_for<0, NumTap, 1>{}([&](auto i) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                            .RunRead(wei_grid_desc, wei_grid_buf);
                    });
                    static_for<0, NumTap, 1>{}([&](auto i) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                            .RunWrite(wei_block_desc, wei_block_buf);
                    });
                }

                if constexpr(EnableAsync)
                {
                    in_blockwise_copy.Run(in_grid_desc, in_grid_buf, in_block_desc, in_block_buf);
                }
                else
                {
                    in_blockwise_copy.RunRead(in_grid_desc, in_grid_buf);
                    in_blockwise_copy.RunWrite(in_block_desc, in_block_buf);
                }

                static_for<0, NumTap, 1>{}([&](auto i) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[i])
                        .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                });
                in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

                constexpr index_t NumDs = ds_blockwise_copy.Size();
                using DsDataBlockTransfer0 =
                    std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[I0])>>;
                if constexpr(EnableAsync)
                {
                    static_for<0, NumDs, 1>{}([&](auto i) {
                        const_cast<DsDataBlockTransfer0&>(ds_blockwise_copy[i])
                            .Run(ds_grid_desc[Number<i>{}],
                                 ds_grid_buf[Number<i>{}],
                                 ds_block_desc[Number<i>{}],
                                 ds_block_buf(i));
                    });
                }
                else
                {
                    static_for<0, NumDs, 1>{}([&](auto i) {
                        const_cast<DsDataBlockTransfer0&>(ds_blockwise_copy[i])
                            .RunRead(ds_grid_desc[Number<i>{}], ds_grid_buf[Number<i>{}]);
                    });
                    static_for<0, NumDs, 1>{}([&](auto i) {
                        const_cast<DsDataBlockTransfer0&>(ds_blockwise_copy[i])
                            .RunWrite(ds_block_desc[Number<i>{}], ds_block_buf(i));
                    });
                }

                barrierLds.sync_lds<EnableAsync>();

                semaLds.template signal<0>();
            }

            if(get_wave_id_in_wavegroup() == 1)
            {
                semaLds.template wait<0>();
                blockwise_conv.Run(
                    wei_block_buf, in_block_buf, ds_block_buf, accum_thread_buf, true);
            }
        }
    }
};

} // namespace ck
