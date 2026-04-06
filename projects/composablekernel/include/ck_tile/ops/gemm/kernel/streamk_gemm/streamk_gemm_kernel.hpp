// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/concat.hpp"
#include "streamk_gemm_coherency.hpp"

namespace ck_tile {

/**
 * @brief The Stream K GEMM kernel host arguments.
 *
 * @par Overview
 *      This structure is passed to @ref StreamKKernel "StreamKKernel" when creating the kernel
 *      arguments object. It contains all necessary information required to build proper kernel
 *      arguments and launch the kernel on GPU. This structure defines the GEMM problem
 *      configuration by stating all required information like M,N,K sizes and respective strides.
 */
template <index_t NumATensor, index_t NumBTensor, index_t NumDTensor>
struct StreamKHostArgs : public ck_tile::UniversalGemmHostArgs<NumATensor, NumBTensor, NumDTensor>
{
    CK_TILE_HOST explicit StreamKHostArgs(const std::array<const void*, NumATensor>& as_ptr_,
                                          const std::array<const void*, NumBTensor>& bs_ptr_,
                                          const std::array<const void*, NumDTensor>& ds_ptr_,
                                          void* c_ptr_,
                                          index_t M_,
                                          index_t N_,
                                          index_t K_,
                                          const std::array<index_t, NumATensor>& stride_As_,
                                          const std::array<index_t, NumBTensor>& stride_Bs_,
                                          const std::array<index_t, NumDTensor>& stride_Ds_,
                                          index_t stride_C_)
        : UniversalGemmHostArgs<NumATensor, NumBTensor, NumDTensor>(as_ptr_,
                                                                    bs_ptr_,
                                                                    ds_ptr_,
                                                                    c_ptr_,
                                                                    /*k_batch_ =*/1,
                                                                    M_,
                                                                    N_,
                                                                    K_,
                                                                    stride_As_,
                                                                    stride_Bs_,
                                                                    stride_Ds_,
                                                                    stride_C_)
    {
    }
};

/**
 * @brief The Stream K GEMM kernel class.
 *
 * @par Overview
 *      This class is responsible for the Stream-K kernel, making use of UniversalGemm.
 *	 The main kernel functions are the operator() functions. There is one for Persistent
 *	 and one for Non-Persistent data parallel sections of the Stream-K algorithm.
 *
 *	 Both the Non-Persistent and Persistent kernels make use of `BaseGemm()` and
 *	 `StreamKGemm()`. `BaseGemm()` computes offsets into the A,B,C tensors, then calls
 *	 `RunGemm()` which runs the GEMM pipeline and epilogue. `StreamKGemm()` performs the
 *	 main Stream-K algorithm. Each iteration of the Stream-K loop calls `BaseGemm()`.
 */
template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct StreamKKernel
{
    /**
     *@brief Inject the UniversalGemmKernel base class to support execution of all necessary
     *functions.
     */
    using UniversalGemmKernel =
        UniversalGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;

    static constexpr index_t kBlockSize = UniversalGemmKernel::kBlockSize;
    static constexpr bool PersistentDP  = UniversalGemmKernel::PersistentKernel;

    using TilePartitioner  = TilePartitioner_;
    using GemmPipeline     = GemmPipeline_;
    using EpiloguePipeline = EpiloguePipeline_;
    using WarpGemm         = typename GemmPipeline::BlockGemm::WarpGemm;
    using BlockGemmShape   = typename GemmPipeline::BlockGemmShape;

    static_assert(
        TilePartitioner::PERSISTENT == PersistentDP,
        "Persistent flag from TilePartitioner must match Persistent flag from UniversalGemm.");

    /**
     * @brief  Specify the layout configurations for A, B, C and D
     */
    using AsLayout = remove_cvref_t<typename GemmPipeline::AsLayout>;
    using BsLayout = remove_cvref_t<typename GemmPipeline::BsLayout>;
    using CLayout  = remove_cvref_t<typename GemmPipeline::CLayout>;
    using DsLayout = remove_cvref_t<typename EpiloguePipeline::DsLayout>;

    /**
     * @brief  Specify the data type configurations for A, B, C and D
     */
    using AsDataType  = remove_cvref_t<typename GemmPipeline::AsDataType>;
    using BsDataType  = remove_cvref_t<typename GemmPipeline::BsDataType>;
    using CDataType   = remove_cvref_t<typename EpiloguePipeline::ODataType>;
    using DsDataType  = remove_cvref_t<typename EpiloguePipeline::DsDataType>;
    using AccDataType = typename EpiloguePipeline::AccDataType;

    /**
     * @brief  Specify the element-wise operation types for A and B
     */
    using AElementWise = remove_cvref_t<typename GemmPipeline::AElementWise>;
    using BElementWise = remove_cvref_t<typename GemmPipeline::BElementWise>;

    /**
     * @brief The sizes of NumATensor, NumBTensor and NumDTensor is set by the user.
     */
    static constexpr index_t NumATensor = AsDataType::size();
    static constexpr index_t NumBTensor = BsDataType::size();
    static constexpr index_t NumDTensor = DsDataType::size();

    template <typename T>
    static constexpr bool is_tuple_v = is_detected<is_tuple, T>::value;
    /**
     *@brief AsLayout and AsDataType are expected to be tuples, not scalars.
     */
    static_assert(is_tuple_v<AsLayout> && is_tuple_v<AsDataType>,
                  "AsLayout and AsDataType must be tuples.");

    /**
     *@brief BsLayout and BsDataType are expected to be tuples, not scalars.
     */
    static_assert(is_tuple_v<BsLayout> && is_tuple_v<BsDataType>,
                  "BsLayout and BsDataType must be tuples.");

    /**
     *@brief CLayout and CDataType are expected to be scalars, not a tuple.
     */
    static_assert(!is_tuple_v<CLayout> && !is_tuple_v<CDataType>,
                  "CLayout and CDataType must be scalars.");

    /**
     *@brief DsLayout and DsDataType are expected to be tuples, not scalars.
     */
    static_assert(is_tuple_v<DsLayout> && is_tuple_v<DsDataType> &&
                      DsLayout::size() == DsDataType::size() && DsLayout::size() > 0,
                  "DsLayout and DsDataType must be tuples and must have the same size.");

    using ADataType = remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<0, BsDataType>>;
    using DDataType = remove_cvref_t<std::tuple_element_t<0, DsDataType>>;

    struct StreamKKernelArgs : ck_tile::UniversalGemmKernelArgs<NumATensor, NumBTensor, NumDTensor>
    {
        StreamKKernelArgs(const StreamKHostArgs<NumATensor, NumBTensor, NumDTensor>& host_args,
                          index_t max_active_wgs)
            : UniversalGemmKernelArgs<NumATensor, NumBTensor, NumDTensor>{host_args.as_ptr,
                                                                          host_args.bs_ptr,
                                                                          host_args.ds_ptr,
                                                                          host_args.e_ptr,
                                                                          host_args.M,
                                                                          host_args.N,
                                                                          host_args.K,
                                                                          host_args.stride_As,
                                                                          host_args.stride_Bs,
                                                                          host_args.stride_Ds,
                                                                          host_args.stride_E,
                                                                          host_args.k_batch},
              // The workspace pointer is set to nullptr because we must first
              // instantiate the TilePartitioner to get the necessary size
              workspace_ptr{nullptr},
              tile_partitioner{
                  TilePartitioner{host_args.M, host_args.N, host_args.K, max_active_wgs}}

        {
        }
        /**
         * @brief  A pointer to a buffer in device memory for accumulating partial via reduction
         * strategy.
         */
        void* workspace_ptr;
        /**
         * @brief  An instance of the TilePartioner class for assisting with mapping workgroups to
         * the C tensor.
         */
        TilePartitioner tile_partitioner;
    };

    using KernelArgs = StreamKKernelArgs;
    using Kernel     = StreamKKernel<TilePartitioner, GemmPipeline, EpiloguePipeline>;
    using StreamKOps = StreamKReductionOps<TilePartitioner, GemmPipeline, StreamKKernelArgs>;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        using P_ = GemmPipeline;
        using WarpTile = typename BlockGemmShape::WarpTile;

        return concat('_', "streamk", gemm_prec_str<ADataType, BDataType>(),
                      concat('x', P_::MPerBlock, P_::NPerBlock, P_::KPerBlock),
                      concat('x', WarpTile::at(number<0>{}), WarpTile::at(number<1>{}), WarpTile::at(number<2>{})),
                      concat('x', P_::GetVectorSizeA(), P_::GetVectorSizeB(), P_::GetVectorSizeC()),
                      concat('x', P_::kPadM, P_::kPadN, P_::kPadK));
        // clang-format on
    }

    /**
     * @brief Compute the grid size for the Stream K kernel using the tile_partitioner.
     * @return The grid size.
     */
    CK_TILE_HOST static auto GridSize(const TilePartitioner& tile_partitioner) -> dim3
    {
        return tile_partitioner.grid_size();
    }

    /**
     * @brief Get the maximum occupancy grid size for the persistent kernel on the current device.
     * @return The maximum occupancy grid size.
     * @note This function queries the maximum occupancy of the kernel using
     * `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
     */
    CK_TILE_HOST static auto MaxOccupancyGridSize(const stream_config& s) -> dim3
    {
        return UniversalGemmKernel::MaxOccupancyGridSize(s);
    }

    CK_TILE_HOST static constexpr auto BlockSize() -> dim3
    {
        return UniversalGemmKernel::BlockSize();
    }

    /**
     * @brief Constructs kernel arguments for the Stream-K kernel.
     * @param host_args Stream-K host arguments.
     * @param num_cu Number of compute units (CUs). The default is the number of CUs on the device.
     * The caller may select their own to assist with test reproducibility, etc.
     * @param occupancy The maximum number of active blocks per CU for this kernel. The caller may
     * select their own to assist with test reproducibility, etc.
     * @return The kernel arguments for Stream-K.
     */
    CK_TILE_HOST static StreamKKernelArgs
    MakeKernelArgs(const StreamKHostArgs<NumATensor, NumBTensor, NumDTensor>& host_args,
                   int num_cu    = NumCU(),
                   int occupancy = Occupancy())
    {
        const index_t max_active_wgs = num_cu * occupancy;

        return StreamKKernelArgs{host_args, max_active_wgs};
    }

    template <bool UseDefaultScheduler = true>
    CK_TILE_DEVICE static void
    RunGemm(const std::array<const ADataType*, UniversalGemmKernel::NumATensor>& as_ptr,
            const std::array<const BDataType*, UniversalGemmKernel::NumBTensor>& bs_ptr,
            const std::array<const void*, UniversalGemmKernel::NumDTensor>& ds_ptr,
            CDataType* c_ptr,
            void* smem_ptr_0,
            const typename UniversalGemmKernel::KernelArgs& kargs,
            const index_t num_loop,
            const index_t block_idx_m,
            const index_t block_idx_n,
            const index_t k_size)
    {
        // Create block windows using specialized methods
        const auto& as_block_window =
            UniversalGemmKernel::MakeABlockWindows(as_ptr, kargs, k_size, block_idx_m);
        const auto& bs_block_window =
            UniversalGemmKernel::MakeBBlockWindows(bs_ptr, kargs, k_size, block_idx_n);
        const auto& ds_block_window =
            UniversalGemmKernel::MakeDBlockWindows(ds_ptr, kargs, block_idx_m, block_idx_n);

        // Since num_loop can vary per WG and per iteration of the Stream-K while loop, we compute
        // has_hot_loop and tail_num here. This is a similar pattern used by grouped GEMM. In this
        // case, we call the GemmPipeline's operator() function that takes both has_hot_loop and
        // tail_num.
        // const bool has_hot_loop   = GemmPipeline::BlockHasHotloop(num_loop);
        // const TailNumber tail_num = GemmPipeline::GetBlockLoopTailNum(num_loop);

        // Run GEMM cooperatively by whole workgroup.
        // Pass the full as_block_window and bs_block_window arrays along with element-wise functors
        // to support multi-ABD operations
        const auto& c_block_tile = GemmPipeline{}.template operator()(
            as_block_window, AElementWise{}, bs_block_window, BElementWise{}, num_loop, smem_ptr_0);

        if(UseDefaultScheduler || (get_warp_id() == 0))
        {
            // Run Epilogue Pipeline
            auto c_block_window =
                UniversalGemmKernel::template MakeCBlockWindows<TilePartitioner::MemoryOperation>(
                    c_ptr, kargs, block_idx_m, block_idx_n);

            EpiloguePipeline{}(c_block_window, c_block_tile, ds_block_window, smem_ptr_0);
        }
    }

    CK_TILE_HOST static bool IsSupportedArgument(const StreamKKernelArgs& kargs)
    {
        return UniversalGemmKernel::IsSupportedArgument(kargs);
    }

    /**
     * @brief Computes the buffer size needed to store accumulation results for Stream K.
     * @return The buffer size needed.
     */
    CK_TILE_HOST static uint32_t GetWorkSpaceSize(const StreamKKernelArgs& kargs)
    {
        return kargs.tile_partitioner.get_workspace_size(sizeof(AccDataType));
    }
    /**
     *@brief Sets the kargs' current workspace_ptr to the given workspace_ptr.
     * @note Assumes that the given workspace_ptr points to allocated device memory.
     */
    CK_TILE_HOST static void SetWorkSpacePointer(StreamKKernelArgs& kargs, void* workspace_ptr)
    {
        kargs.workspace_ptr = workspace_ptr;
    }

    /**
     * @brief Computes offsets into A, B, and C tensors then runs the GEMM pipeline and epilogue.
     * @param kargs Stream-K kernel arguments.
     * @param tile_idx The 1D tile index in the C tensor for this workgroup.
     * @param num_loop The number of iterations (at the macro tile level) in the K dimension this
     * workgroup will perform in the C tile.
     * @param i_k_as The K offsets in the A tensors.
     * @param i_k_bs The K offsets in the B tensors.
     * @param k_size The portion of the K dimension this workgroup processes in the assigned
     * `tile_idx`.
     * @param smem_ptr_0 Pointer to LDS.
     */
    CK_TILE_DEVICE void BaseGemm(StreamKKernelArgs& kargs,
                                 index_t tile_idx,
                                 index_t num_loop,
                                 const std::array<index_t, NumATensor>& i_k_as,
                                 const std::array<index_t, NumBTensor>& i_k_bs,
                                 index_t k_size,
                                 void* smem_ptr_0) const
    {
        const auto c_macro_tile_idx = kargs.tile_partitioner.get_output_tile_index(tile_idx);
        index_t i_m = c_macro_tile_idx[UniversalGemmKernel::I0] * TilePartitioner::MPerBlock;
        index_t i_n = c_macro_tile_idx[UniversalGemmKernel::I1] * TilePartitioner::NPerBlock;

        std::array<const ADataType*, NumATensor> as_ptr;
        static_for<0, NumATensor, 1>{}([&](auto i) {
            as_ptr[i] = static_cast<const ADataType*>(kargs.as_ptr[i]) + i_k_as[i];
        });

        std::array<const BDataType*, NumBTensor> bs_ptr;
        static_for<0, NumBTensor, 1>{}([&](auto i) {
            bs_ptr[i] = static_cast<const BDataType*>(kargs.bs_ptr[i]) + i_k_bs[i];
        });

        CDataType* c_ptr = static_cast<CDataType*>(kargs.e_ptr);

        // Run the GEMM pipeline and Epilogue.
        RunGemm(as_ptr, bs_ptr, kargs.ds_ptr, c_ptr, smem_ptr_0, kargs, num_loop, i_m, i_n, k_size);
    }

    /**
     * @brief Runs the main Stream - K algorithm.
     * @param kargs Stream - K kernel arguments.
     * @param cta_idx The current Stream - K workgroup's index.
     * @param smem_ptr_0 Pointer to LDS.
     * @note It is assumed that the first Stream - K workgroup has a `cta_idx` of zero. If a
     * non-persistent data-parallel (DP) section is used, then a Stream-K workgroup's `cta_idx`
     * *should be something like `blockIdx.x` minus number of DP workgroups.
     */
    CK_TILE_DEVICE
    void StreamKGemm(StreamKKernelArgs& kargs, index_t cta_idx, void* smem_ptr_0) const
    {
        const StreamKOps sk_ops{};
        index_t iter_start, iter_end;
        kargs.tile_partitioner.get_iter_boundaries(iter_start, iter_end, cta_idx);

        while(iter_start < iter_end)
        {
            // Get the 1D tile index in the C tensor that this workgroup will work in for this
            // iteration of the loop.
            index_t tile_idx =
                amd_wave_read_first_lane(kargs.tile_partitioner.get_tile_index(iter_start));

            // Get the start and end boundaries for the current tile.
            index_t tile_iter_start, tile_iter_end;
            kargs.tile_partitioner.get_tile_boundaries(tile_iter_start, tile_iter_end, tile_idx);

            // Get the start and end iteration within the current tile for the workgroup.
            index_t local_iter_start = amd_wave_read_first_lane(
                kargs.tile_partitioner.get_local_iter(iter_start, tile_iter_start));
            index_t local_iter_end =
                amd_wave_read_first_lane(kargs.tile_partitioner.get_local_iter_end(
                    tile_iter_start, iter_end, tile_iter_end));

            // Get the iteration length.
            index_t num_loop_sk = local_iter_end - local_iter_start;

            // Determine the total size along the K dimension the workgroup is using in this
            // iteration (used to construct tensor views).
            index_t k_size = num_loop_sk * TilePartitioner::KPerBlock;

            // Get the K offsets for the A and B tensors
            auto [i_k_as, i_k_bs] =
                GetKOffsets<AsLayout, BsLayout>(local_iter_start, kargs.stride_As, kargs.stride_Bs);

            if constexpr(TilePartitioner::ReductionStrategy == StreamKReductionStrategy::Atomic)
            {
                BaseGemm(kargs, tile_idx, num_loop_sk, i_k_as, i_k_bs, k_size, smem_ptr_0);
            }
            else if(TilePartitioner::ReductionStrategy == StreamKReductionStrategy::Linear ||
                    TilePartitioner::ReductionStrategy == StreamKReductionStrategy::Tree)
            {
                const auto c_macro_tile_idx =
                    kargs.tile_partitioner.get_output_tile_index(tile_idx);
                index_t i_m =
                    c_macro_tile_idx[UniversalGemmKernel::I0] * TilePartitioner::MPerBlock;
                index_t i_n =
                    c_macro_tile_idx[UniversalGemmKernel::I1] * TilePartitioner::NPerBlock;

                std::array<const ADataType*, NumATensor> as_ptr;
                static_for<0, NumATensor, 1>{}([&](auto i) {
                    as_ptr[i] = static_cast<const ADataType*>(kargs.as_ptr[i]) + i_k_as[i];
                });

                std::array<const BDataType*, NumBTensor> bs_ptr;
                static_for<0, NumBTensor, 1>{}([&](auto i) {
                    bs_ptr[i] = static_cast<const BDataType*>(kargs.bs_ptr[i]) + i_k_bs[i];
                });
                CDataType* c_ptr = static_cast<CDataType*>(kargs.e_ptr);

                // Create block windows using specialized methods
                const auto& as_block_window =
                    UniversalGemmKernel::MakeABlockWindows(as_ptr, kargs, k_size, i_m);
                const auto& bs_block_window =
                    UniversalGemmKernel::MakeBBlockWindows(bs_ptr, kargs, k_size, i_n);
                const auto& ds_block_window = UniversalGemmKernel::MakeDBlockWindows(
                    kargs.ds_ptr,
                    static_cast<const typename UniversalGemmKernel::KernelArgs&>(kargs),
                    i_m,
                    i_n);

                // Since num_loop can vary per WG and per iteration of the Stream-K while loop,
                // we compute has_hot_loop and tail_num here. This is a similar pattern used by
                // grouped GEMM. In this case, we call the GemmPipeline's operator() function
                // that takes both has_hot_loop and tail_num.
                // const bool has_hot_loop   = GemmPipeline::BlockHasHotloop(num_loop_sk);
                // const TailNumber tail_num = GemmPipeline::GetBlockLoopTailNum(num_loop_sk);

                // Run GEMM cooperatively by whole workgroup.
                const auto& c_block_tile = GemmPipeline{}.template operator()(as_block_window,
                                                                              AElementWise{},
                                                                              bs_block_window,
                                                                              BElementWise{},
                                                                              num_loop_sk,
                                                                              smem_ptr_0);
                // const auto& c_block_tile =
                // GemmPipeline{}(as_block_window[UniversalGemmKernel::I0],
                //                                           bs_block_window[UniversalGemmKernel::I0],
                //                                           num_loop_sk,
                //                                           has_hot_loop,
                //                                           tail_num,
                //                                           smem_ptr_0);

                auto tile_started = iter_start == tile_iter_start;
                auto tile_ended   = iter_end >= tile_iter_end;

                if constexpr(TilePartitioner::ReductionStrategy == StreamKReductionStrategy::Linear)
                {
                    if(!tile_started)
                    {
                        sk_ops.StorePartial(kargs, cta_idx, c_block_tile);
                        sk_ops.SignalStorePartialDone(kargs, cta_idx);
                    }
                    else
                    {
                        auto accum_block_tile = c_block_tile;
                        if(!tile_ended)
                        {
                            const index_t iter_per_tile =
                                kargs.tile_partitioner.get_iters_per_tile();
                            const index_t iter_per_cta =
                                kargs.tile_partitioner.get_iters_per_sk_cta();
                            const index_t extra_iters = kargs.tile_partitioner.get_extra_iters();
                            int accum_iters           = local_iter_end - local_iter_start;
                            int next_cta              = cta_idx + 1;

                            while(accum_iters < iter_per_tile)
                            {
                                sk_ops.WaitStorePartialDone(kargs, next_cta);

                                using BlockType = remove_cvref_t<decltype(c_block_tile)>;
                                sk_ops.AddBlockTile(
                                    accum_block_tile,
                                    sk_ops.template LoadPartial<typename BlockType::DataType>(
                                        kargs, next_cta, c_block_tile.get_tile_distribution()));

                                accum_iters += iter_per_cta + (next_cta < extra_iters);
                                ++next_cta;
                            }
                        }

                        auto c_block_window = UniversalGemmKernel::template MakeCBlockWindows<
                            TilePartitioner::MemoryOperation>(c_ptr, kargs, i_m, i_n);
                        EpiloguePipeline{}(
                            c_block_window, accum_block_tile, ds_block_window, smem_ptr_0);
                    }
                }
                else // Tree Reduction
                {
                    auto accum_block_tile      = c_block_tile;
                    index_t tile_local_cta_idx = amd_wave_read_first_lane(
                        kargs.tile_partitioner.get_tile_local_cta_index(tile_iter_start, cta_idx));

                    index_t stride = amd_wave_read_first_lane(1);

                    for(;; stride <<= 1)
                    {
                        const index_t partner_cta_idx = amd_wave_read_first_lane(cta_idx + stride);
                        const index_t partner_start_iter = amd_wave_read_first_lane(
                            kargs.tile_partitioner.get_start_iter(partner_cta_idx));
                        bool partner_in_tile =
                            amd_wave_read_first_lane(partner_start_iter < tile_iter_end);

                        // If the partner of the workgroup who started the tile is not in this tile,
                        // then the work for this tile is done and results can be stored in the C
                        // tensor.
                        if(tile_started && !partner_in_tile)
                        {
                            auto c_block_window = UniversalGemmKernel::template MakeCBlockWindows<
                                TilePartitioner::MemoryOperation>(c_ptr, kargs, i_m, i_n);
                            EpiloguePipeline{}(
                                c_block_window, accum_block_tile, ds_block_window, smem_ptr_0);
                            break;
                        }

                        // It's this workgroup's turn to read from partials.
                        if(tile_local_cta_idx % (stride << 1) == 0)
                        {
                            // If this workgroup's partner is in the tile then it can read from
                            // partials and accumulate results.
                            if(partner_in_tile)
                            {
                                sk_ops.WaitStorePartialDone(kargs, partner_cta_idx);
                                using BlockType = remove_cvref_t<decltype(c_block_tile)>;
                                sk_ops.AddBlockTile(
                                    accum_block_tile,
                                    sk_ops.template LoadPartial<typename BlockType::DataType>(
                                        kargs,
                                        partner_cta_idx,
                                        c_block_tile.get_tile_distribution()));
                            }
                        }
                        // Otherwise, it's this workgroup's turn to write to partials. All
                        // workgroups, except the workgroup who starts the tile, will write to
                        // partials.
                        else
                        {
                            sk_ops.StorePartial(kargs, cta_idx, accum_block_tile);
                            sk_ops.SignalStorePartialDone(kargs, cta_idx);
                            // Once the workgroup writes to partials, it has no more work to do for
                            // this tile.
                            break;
                        }
                    }
                }
            }
            else
            {
                static_assert(
                    "An implementation does not exist for the chosen reduction strategy.");
            }

            // Prepare for next Stream-K loop iteration.
            iter_start = tile_iter_end;
            block_sync_lds();
        }
    }

    /**
     * @brief Entry point for the Stream-K kernel.
     *
     * @par Overview
     *     Uses StreamKDispatch to handle both persistent and non-persistent DP sections.
     *     Non-persistent: dedicated DP workgroups process full tiles, then dedicated SK
     *     workgroups share remaining K-iterations.
     *     Persistent: each workgroup loops over DP tiles (round-robin), then proceeds
     *     to SK work.
     */
    CK_TILE_DEVICE void operator()(StreamKKernelArgs kargs) const
    {
        __shared__ char smem_ptr_0[UniversalGemmKernel::GetSmemSize()];
        const index_t dp_num_loop = kargs.tile_partitioner.get_iters_per_tile();

        StreamKDispatch(
            kargs.tile_partitioner,
            [&](index_t tile_idx) {
                BaseGemm(kargs, tile_idx, dp_num_loop, {0}, {0}, kargs.K, smem_ptr_0);
            },
            [&](index_t sk_cta_idx) { StreamKGemm(kargs, sk_cta_idx, smem_ptr_0); });
    }

    private:
    /**
     * @brief Computes the K offsets in the A and B tensors given iter_offset, where iter_offset
     * is the starting macro tile index in the K dimension for the workgroup.
     * @return A tuple containing the offsets into the A and B tensors accounting for the
     * layouts of A and B.
     * @note The default case is that A is assumed to be row major and B is assumed to be column
     * major.
     */
    template <typename AsLayout, typename BsLayout>
    CK_TILE_DEVICE static tuple<std::array<index_t, NumATensor>, std::array<index_t, NumBTensor>>
    GetKOffsets(index_t iter_offset,
                const std::array<index_t, NumATensor>& stride_as,
                const std::array<index_t, NumBTensor>& stride_bs)
    {

        std::array<index_t, NumATensor> a_k_offsets;
        std::array<index_t, NumBTensor> b_k_offsets;

        index_t base_offset = iter_offset * TilePartitioner::KPerBlock;

        static_for<0, NumATensor, 1>{}([&](auto i) {
            using AiLayout = std::tuple_element_t<i.value, AsLayout>;
            index_t stride_offset_a;
            if constexpr(std::is_same_v<AiLayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
            {
                stride_offset_a = stride_as[i];
            }
            else
            {
                stride_offset_a = 1;
            }
            a_k_offsets[i] = base_offset * stride_offset_a;
        });

        static_for<0, NumBTensor, 1>{}([&](auto i) {
            using BiLayout = std::tuple_element_t<i.value, BsLayout>;
            index_t stride_offset_b;
            if constexpr(std::is_same_v<BiLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                stride_offset_b = stride_bs[i];
            }
            else
            {
                stride_offset_b = 1;
            }

            b_k_offsets[i] = base_offset * stride_offset_b;
        });

        return ck_tile::make_tuple(a_k_offsets, b_k_offsets);
    }

    CK_TILE_HOST static int NumCU()
    {
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        ck_tile::hip_check_error(hipGetDevice(&dev));
        ck_tile::hip_check_error(hipGetDeviceProperties(&dev_prop, dev));
        int num_cu = dev_prop.multiProcessorCount;

        return num_cu;
    }

    /**
     * @brief Computes the occupancy (i.e. maximum number of active blocks per CU) for the
     * kernel
     * @return The occupancy
     * @note This function queries the maximum occupancy of the kernel using
     * `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
     */
    CK_TILE_HOST static int Occupancy()
    {
        int occupancy;

        // Since occupancy of 1 is valid for stream k, we set min_num_block_per_cu to 1
        constexpr int min_block_per_cu = 1;
        const auto kernel              = kentry<min_block_per_cu, Kernel, KernelArgs>;

        ck_tile::hip_check_error(
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, kBlockSize, 0));

        return max(occupancy, 1);
    }
};
} // namespace ck_tile
