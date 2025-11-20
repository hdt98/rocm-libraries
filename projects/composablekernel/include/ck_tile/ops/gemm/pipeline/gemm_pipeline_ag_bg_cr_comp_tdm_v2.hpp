// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_default_policy.hpp"

#include "ck_tile/core/arch/barrier.hpp"

namespace ck_tile {

template <typename Problem, typename Policy>
struct GemmPipelineAgBgCrCompTDMV1;

/**
 * @brief Compute optimized pipeline version using TDM(tensor data mover)
 *
 * This pipeline introduces load from global memory to LDS using TDM and uses wave
 * specialization.
 *
 */
template <typename Problem, typename Policy = GemmPipelineAgBgCrCompTDMDefaultPolicy<true>>
struct GemmPipelineAgBgCrCompTDMV2 : public GemmPipelineAgBgCrCompTDMV1<Problem, Policy>
{
    using Base             = GemmPipelineAgBgCrCompTDMV1<Problem, Policy>;
    using PipelineImplBase = typename Base::PipelineImplBase;

    static constexpr bool HasHotLoop = Base::HasHotLoop;
    static constexpr auto TailNum    = Base::TailNum;
    static constexpr auto Scheduler  = Base::Scheduler;

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static_assert(BlockSize == get_warp_size() * 4, "pipeline requires 4 waves per workgroup");

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t smem_size         = Policy::template GetSmemSize<Problem>();
        constexpr index_t barrier_smem_size = 8; // 64 bits per barrier
        // because smem_size is 16 bytes aligned, so just add barrier_smem_size directly,
        // barrier_smem should already be 8 byte aligned
        return 2 * (smem_size + barrier_smem_size);
    }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public PipelineImplBase
    {
        using Base                             = PipelineImplBase;
        using OuterClass                       = GemmPipelineAgBgCrCompTDMV2<Problem, Policy>;
        using ADataType                        = typename OuterClass::ADataType;
        using BDataType                        = typename OuterClass::BDataType;
        using ALayout                          = typename OuterClass::ALayout;
        using BLayout                          = typename OuterClass::BLayout;
        using AsLayout                         = typename OuterClass::AsLayout;
        using BsLayout                         = typename OuterClass::BsLayout;
        using BlockGemm                        = typename OuterClass::BlockGemm;
        using I0                               = typename OuterClass::I0;
        using I1                               = typename OuterClass::I1;
        using I2                               = typename OuterClass::I2;
        static constexpr bool UseClusterLaunch = OuterClass::UseClusterLaunch;
        static constexpr index_t MPerBlock     = OuterClass::MPerBlock;
        static constexpr index_t NPerBlock     = OuterClass::NPerBlock;
        static constexpr index_t KPerBlock     = OuterClass::KPerBlock;

        template <index_t warp_id,
                  bool HasHotLoop,
                  TailNumber TailNum,
                  typename AsDramBlockWindowTmp,
                  typename BsDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction,
                  typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                                is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                            bool>* = nullptr>
        CK_TILE_DEVICE auto
        wave_specialized_func(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                              const AElementFunction& a_element_func,
                              const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                              const BElementFunction& b_element_func,
                              index_t num_loop,
                              void* __restrict__ p_smem) const
        {
            constexpr index_t smem_size = Policy::template GetSmemSize<Problem>();

            auto&& [a_lds_block0, b_lds_block0] = Base::GetABLdsTensorViews(p_smem);
            auto&& [a_lds_block1, b_lds_block1] =
                Base::GetABLdsTensorViews(static_cast<char*>(p_smem) + smem_size);
            // currently lds config is set to 29; so phase width is 3
            LdsAtomicBarrier<3>* barriers[2];
            barriers[0] = reinterpret_cast<LdsAtomicBarrier<3>*>(
                static_cast<char*>(p_smem) + 2 * smem_size); // after both LDS buffers
            barriers[1] = reinterpret_cast<LdsAtomicBarrier<3>*>(
                static_cast<char*>(p_smem) + 2 * smem_size +
                sizeof(LdsAtomicBarrier<3>)); // after first barrier
            if constexpr(warp_id == 0)
            {
                barriers[0]->init(1);
                barriers[1]->init(1);
            }

            TDMConfig tdm_config_a[2];
            TDMConfig tdm_config_b[2];

            // enable atomic_barrier in TDM to make sure data is visible in LDS before wave reads
            // them; tdm_config_a[0] for wave 0, tdm_config_a[1] for wave 2;
            // tdm_config_b[0] for wave 1, tdm_config_b[1] for wave 3
            tdm_config_a[0].atomic_barrier_enable = true;
            tdm_config_b[0].atomic_barrier_enable = true;

            tdm_config_a[0].atomic_barrier_address =
                static_cast<uint16_t>(reinterpret_cast<uintptr_t>(barriers[0])) >> 3;
            tdm_config_b[0].atomic_barrier_address =
                static_cast<uint16_t>(reinterpret_cast<uintptr_t>(barriers[0])) >> 3;

            tdm_config_a[1].atomic_barrier_enable = true;
            tdm_config_b[1].atomic_barrier_enable = true;

            tdm_config_a[1].atomic_barrier_address =
                static_cast<uint16_t>(reinterpret_cast<uintptr_t>(barriers[1])) >> 3;
            tdm_config_b[1].atomic_barrier_address =
                static_cast<uint16_t>(reinterpret_cast<uintptr_t>(barriers[1])) >> 3;

            if constexpr(UseClusterLaunch)
            {
                dim3 block_id_in_cluster{amd_wave_read_first_lane(get_cluster_workgroup_id_x()),
                                         amd_wave_read_first_lane(get_cluster_workgroup_id_y()),
                                         amd_wave_read_first_lane(get_cluster_workgroup_id_z())};
                tdm_config_a[0].workgroup_mask =
                    Policy::template GetTDMWorkgroupMask<MultiCastDirection::kM, Problem>(
                        block_id_in_cluster);
                tdm_config_b[0].workgroup_mask =
                    Policy::template GetTDMWorkgroupMask<MultiCastDirection::kN, Problem>(
                        block_id_in_cluster);

                tdm_config_a[1].workgroup_mask =
                    Policy::template GetTDMWorkgroupMask<MultiCastDirection::kM, Problem>(
                        block_id_in_cluster);
                tdm_config_b[1].workgroup_mask =
                    Policy::template GetTDMWorkgroupMask<MultiCastDirection::kN, Problem>(
                        block_id_in_cluster);
            }

            static_assert(1 == std::tuple_size_v<AsDramBlockWindowTmp>);
            static_assert(1 == std::tuple_size_v<BsDramBlockWindowTmp>);

            using ADramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, AsDramBlockWindowTmp>>;
            using BDramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, BsDramBlockWindowTmp>>;
            // TODO currently fused elementwise are not supported
            static_assert(std::is_same_v<remove_cvref_t<decltype(a_element_func)>,
                                         element_wise::PassThrough>);
            static_assert(std::is_same_v<remove_cvref_t<decltype(b_element_func)>,
                                         element_wise::PassThrough>);
            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "Data Type conflict on A and B matrix input data type.");

            constexpr bool is_a_col_major =
                std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

            // TODO currently only support A matrix row major, B matrix col major; if A matrix is
            // col major or B is row major, need to combine with transpose load api
            static_assert(!(is_a_col_major || is_b_row_major),
                          "only support A matrix is row major, B matrix is col major!");

            static_assert(is_a_col_major
                              ? (KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "A block window has incorrect lengths for defined ALayout!");
            static_assert(is_b_row_major
                              ? (KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "B block window has incorrect lengths for defined BLayout!");

            ////////////// global window & register /////////////////
            // A DRAM tile window(s) for load
            auto a_tile_windows = generate_tuple(
                [&](auto idx) {
                    return make_tile_window(
                        a_dram_block_window_tmp[number<idx>{}].get_bottom_tensor_view(),
                        make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                        a_dram_block_window_tmp[number<idx>{}].get_window_origin(),
                        Policy::template MakeADramTileDistribution<Problem>());
                },
                number<AsLayout::size()>{});
            // B DRAM window(s) for load
            auto b_tile_windows = generate_tuple(
                [&](auto idx) {
                    return make_tile_window(
                        b_dram_block_window_tmp[number<idx>{}].get_bottom_tensor_view(),
                        make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                        b_dram_block_window_tmp[number<idx>{}].get_window_origin(),
                        Policy::template MakeBDramTileDistribution<Problem>());
                },
                number<BsLayout::size()>{});

            // LDS tile windows for storing, one per LDS buffer
            auto a_copy_lds_window0 = make_tile_window(
                a_lds_block0, make_tuple(number<MPerBlock>{}, number<KPerBlock>{}), {0, 0});

            auto a_copy_lds_window1 = make_tile_window(
                a_lds_block1, make_tuple(number<MPerBlock>{}, number<KPerBlock>{}), {0, 0});

            auto b_copy_lds_window0 = make_tile_window(
                b_lds_block0, make_tuple(number<NPerBlock>{}, number<KPerBlock>{}), {0, 0});

            auto b_copy_lds_window1 = make_tile_window(
                b_lds_block1, make_tuple(number<NPerBlock>{}, number<KPerBlock>{}), {0, 0});

            // initialize DRAM window steps, used to advance the DRAM windows
            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

            constexpr ADramTileWindowStep a_dram_tile_window_step_stride =
                is_a_col_major ? make_array(KPerBlock * 2, 0) : make_array(0, KPerBlock * 2);
            constexpr BDramTileWindowStep b_dram_tile_window_step_stride =
                is_b_row_major ? make_array(KPerBlock * 2, 0) : make_array(0, KPerBlock * 2);

            constexpr auto ALdsTileDistr = decltype(make_static_tile_distribution(
                BlockGemm::MakeABlockDistributionEncode())){};
            constexpr auto BLdsTileDistr = decltype(make_static_tile_distribution(
                BlockGemm::MakeBBlockDistributionEncode())){};

            using ALdsTile = decltype(make_static_distributed_tensor<ADataType>(ALdsTileDistr));
            using BLdsTile = decltype(make_static_distributed_tensor<BDataType>(BLdsTileDistr));

            if constexpr(warp_id == 0)
            {
                Base::GlobalPrefetchTDM(
                    tdm_config_a[0], a_copy_lds_window0, a_tile_windows[number<0>{}]);
            }
            if constexpr(warp_id == 1)
            {
                Base::GlobalPrefetchTDM(
                    tdm_config_b[0], b_copy_lds_window0, b_tile_windows[number<0>{}]);
            }
            if constexpr(warp_id == 2)
            {
                move_tile_window(a_tile_windows[number<0>{}], a_dram_tile_window_step);
                Base::GlobalPrefetchTDM(
                    tdm_config_a[1], a_copy_lds_window1, a_tile_windows[number<0>{}]);
            }
            if constexpr(warp_id == 3)
            {
                move_tile_window(b_tile_windows[number<0>{}], b_dram_tile_window_step);
                Base::GlobalPrefetchTDM(
                    tdm_config_b[1], b_copy_lds_window1, b_tile_windows[number<0>{}]);
            }

            // initialize block gemm
            auto block_gemm = BlockGemm();

            // initialize C block tile
            auto c_block_tile = block_gemm.MakeCBlockTile();
            clear_tile(c_block_tile);

            // register tiles; double buffering -> a register tile corresponds to a LDS tile window
            ALdsTile a_block_tile0;
            ALdsTile a_block_tile1;

            BLdsTile b_block_tile0;
            BLdsTile b_block_tile1;

            // LDS tile windows for reading;
            // they share the data pointer with the LDS windows for storing
            // but also associate with a distribution to produce a register tile when reading
            auto a_lds_ld_window0 =
                make_tile_window(a_lds_block0,
                                 make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                                 {0, 0},
                                 ALdsTileDistr);
            auto a_lds_ld_window1 =
                make_tile_window(a_lds_block1,
                                 make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                                 {0, 0},
                                 ALdsTileDistr);
            auto b_lds_ld_window0 =
                make_tile_window(b_lds_block0,
                                 make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                                 {0, 0},
                                 BLdsTileDistr);
            auto b_lds_ld_window1 =
                make_tile_window(b_lds_block1,
                                 make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                                 {0, 0},
                                 BLdsTileDistr);

            static_assert(!(is_tile_window_linear_v<decltype(a_lds_ld_window0)>) &&
                              !(is_tile_window_linear_v<decltype(a_lds_ld_window1)>) &&
                              !(is_tile_window_linear_v<decltype(b_lds_ld_window0)>) &&
                              !(is_tile_window_linear_v<decltype(b_lds_ld_window1)>),
                          "LDS windows must not be linear");
            uint32_t phase[2]             = {7, 7};
            constexpr uint32_t PHASE_MASK = 0x7;

            phase[0] = (phase[0] - 1) & PHASE_MASK;
            barriers[0]->wait(phase[0]);
            // read A(0), B(0) from LDS window(0) to pipeline registers(0)
            Base::LocalPrefetch(a_block_tile0, a_lds_ld_window0);
            Base::LocalPrefetch(b_block_tile0, b_lds_ld_window0);
            // LDS window(0) contents are overwritten below by global prefetch, need to sync

            if constexpr(HasHotLoop)
            {
                index_t i_global_read = amd_wave_read_first_lane(2);
                do
                {
                    // ping
                    {
                        block_sync_lds();
                        if constexpr(warp_id == 0)
                        {
                            move_tile_window(a_tile_windows[number<0>{}],
                                             a_dram_tile_window_step_stride);
                            Base::GlobalPrefetchTDM(
                                tdm_config_a[0], a_copy_lds_window0, a_tile_windows[number<0>{}]);
                        }
                        if constexpr(warp_id == 1)
                        {
                            move_tile_window(b_tile_windows[number<0>{}],
                                             b_dram_tile_window_step_stride);
                            Base::GlobalPrefetchTDM(
                                tdm_config_b[0], b_copy_lds_window0, b_tile_windows[number<0>{}]);
                        }
                        block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
                    }
                    // pong
                    {
                        phase[1] = (phase[1] - 1) & PHASE_MASK;
                        barriers[1]->wait(phase[1]);
                        Base::LocalPrefetch(a_block_tile1, a_lds_ld_window1);
                        Base::LocalPrefetch(b_block_tile1, b_lds_ld_window1);
                        block_sync_lds();
                        if constexpr(warp_id == 2)
                        {
                            move_tile_window(a_tile_windows[number<0>{}],
                                             a_dram_tile_window_step_stride);
                            Base::GlobalPrefetchTDM(
                                tdm_config_a[1], a_copy_lds_window1, a_tile_windows[number<0>{}]);
                        }
                        if constexpr(warp_id == 3)
                        {
                            move_tile_window(b_tile_windows[number<0>{}],
                                             b_dram_tile_window_step_stride);
                            Base::GlobalPrefetchTDM(
                                tdm_config_b[1], b_copy_lds_window1, b_tile_windows[number<0>{}]);
                        }
                        block_gemm(c_block_tile, a_block_tile1, b_block_tile1);
                        phase[0] = (phase[0] - 1) & PHASE_MASK;
                        barriers[0]->wait(phase[0]);
                        Base::LocalPrefetch(a_block_tile0, a_lds_ld_window0);
                        Base::LocalPrefetch(b_block_tile0, b_lds_ld_window0);
                    }
                    i_global_read += 2;
                } while(i_global_read < num_loop);
            }

            // 2 block gemms remaining
            if constexpr(TailNum == TailNumber::Two)
            {
                {
                    // read A(num_loop), B(num_loop) from LDS window(1) to pipeline registers(1)
                    phase[1] = (phase[1] - 1) & PHASE_MASK;
                    barriers[1]->wait(phase[1]);
                    block_sync_lds();
                    Base::LocalPrefetch(a_block_tile1, a_lds_ld_window1);
                    Base::LocalPrefetch(b_block_tile1, b_lds_ld_window1);
                    // C(num_loop-1) = A(num_loop-1) @ B(num_loop-1)
                    block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
                }
                {
                    // C(num_loop) = A(num_loop) @ B(num_loop)
                    block_gemm(c_block_tile, a_block_tile1, b_block_tile1);
                }
            }
            else
            {
                block_sync_lds();
                block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
            }
            return c_block_tile;
        }

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename AsDramBlockWindowTmp,
                  typename BsDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction,
                  typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                                is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                            bool>* = nullptr>
        CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       index_t num_loop,
                                       void* __restrict__ p_smem) const
        {
            const index_t warp_id = get_warp_id();

            if(warp_id == 0)
            {
                return wave_specialized_func<0, HasHotLoop, TailNum>(a_dram_block_window_tmp,
                                                                     a_element_func,
                                                                     b_dram_block_window_tmp,
                                                                     b_element_func,
                                                                     num_loop,
                                                                     p_smem);
            }
            else if(warp_id == 1)
            {
                return wave_specialized_func<1, HasHotLoop, TailNum>(a_dram_block_window_tmp,
                                                                     a_element_func,
                                                                     b_dram_block_window_tmp,
                                                                     b_element_func,
                                                                     num_loop,
                                                                     p_smem);
            }
            else if(warp_id == 2)
            {
                return wave_specialized_func<2, HasHotLoop, TailNum>(a_dram_block_window_tmp,
                                                                     a_element_func,
                                                                     b_dram_block_window_tmp,
                                                                     b_element_func,
                                                                     num_loop,
                                                                     p_smem);
            }
            else
            {
                return wave_specialized_func<3, HasHotLoop, TailNum>(a_dram_block_window_tmp,
                                                                     a_element_func,
                                                                     b_dram_block_window_tmp,
                                                                     b_element_func,
                                                                     num_loop,
                                                                     p_smem);
            }
        }
    };

    public:
    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            a_element_func,
            b_dram_block_window_tmp,
            b_element_func,
            num_loop,
            p_smem);
    }

    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const index_t num_loop,
                                   void* __restrict__ p_smem) const
    {
        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            [](const Base::ADataType& a) { return a; },
            b_dram_block_window_tmp,
            [](const Base::BDataType& b) { return b; },
            num_loop,
            p_smem);
    }
};
} // namespace ck_tile
