// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"

namespace ck_tile {

// MHC-specific GEMM Pipeline V3 with optional fusion function support
// This pipeline extends the standard GemmPipelineAgBgCrCompV3 to support
// an optional fusion function that can be applied to the A tile after loading
// from DRAM but before storing to LDS. This enables single-pass fusion of
// operations like norm computation with the GEMM operation.
//
// Usage:
//   auto pipeline = MhcGemmPipelineAgBgCrCompV3Fused<Problem>{};
//
//   // With fusion function (e.g., for norm computation)
//   auto fusion_func = [&](const auto& a_tile) {
//       // Process a_tile for norm computation
//   };
//   auto c_tile = pipeline(a_dram_window, a_element_func,
//                         b_dram_window, b_element_func,
//                         num_loop, smem, fusion_func);
//
//   // Without fusion function (standard GEMM)
//   auto c_tile = pipeline(a_dram_window, a_element_func,
//                         b_dram_window, b_element_func,
//                         num_loop, smem);

template <typename Problem, typename Policy = UniversalGemmPipelineAgBgCrPolicy>
struct MhcGemmPipelineAgBgCrCompV3Fused : public BaseGemmPipelineAgBgCrCompV3<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrCompV3<Problem>;
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using AsDataType = remove_cvref_t<typename Problem::AsDataTypeTuple>;
    using BsDataType = remove_cvref_t<typename Problem::BsDataTypeTuple>;
    using CDataType  = remove_cvref_t<typename Problem::CDataType>;

    using AElementWise   = remove_cvref_t<typename Problem::AElementWise>;
    using BElementWise   = remove_cvref_t<typename Problem::BElementWise>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using AsLayout = remove_cvref_t<typename Problem::AsLayoutTuple>;
    using BsLayout = remove_cvref_t<typename Problem::BsLayoutTuple>;
    using CLayout  = remove_cvref_t<typename Problem::CLayout>;

    using ALayout = remove_cvref_t<std::tuple_element_t<0, AsLayout>>;
    using BLayout = remove_cvref_t<std::tuple_element_t<0, BsLayout>>;

    using ADataType = remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<0, BsDataType>>;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;
    using I0        = number<0>;
    using I1        = number<1>;
    using I2        = number<2>;

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr bool Async = false;

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return Policy::template GetVectorSizeA<Problem, IsWave32Host>();
    }
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return Policy::template GetVectorSizeB<Problem, IsWave32Host>();
    }
    static constexpr index_t GetVectorSizeC() { return Policy::template GetVectorSizeC<Problem>(); }

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;
    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;
    static constexpr index_t Preshuffle    = Problem::Preshuffle;

    static constexpr auto Scheduler = Problem::Scheduler;

    static constexpr auto is_a_load_tr_v = bool_constant<PipelineImplBase::is_a_load_tr>{};
    static constexpr auto is_b_load_tr_v = bool_constant<PipelineImplBase::is_b_load_tr>{};

    using Base::PrefetchStages;
    using Base::UsePersistentKernel;

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName()
    {
        return "MHC_COMPUTE_V3_FUSED";
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
        constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});
        return concat('_',
                      "pipeline_MhcAgBgCrCompV3Fused",
                      concat('x', MPerBlock, NPerBlock, KPerBlock),
                      BlockSize,
                      concat('x', GetVectorSizeA(), GetVectorSizeB(), GetVectorSizeC()),
                      concat('x', WaveNumM, WaveNumN),
                      concat('x', kPadM, kPadN, kPadK),
                      Problem::GetName());
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    CK_TILE_HOST static std::string Print()
    {
        constexpr index_t MPerXDL = BlockGemm::WarpGemm::kM;
        constexpr index_t NPerXDL = BlockGemm::WarpGemm::kN;
        constexpr index_t KPerXDL = BlockGemm::WarpGemm::WarpGemmAttribute::Impl::kK;

        constexpr index_t WaveSize = get_warp_size();
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
        constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});

        constexpr index_t A_LDS_Read_Width = GetSmemPackA();
        constexpr index_t B_LDS_Read_Width = GetSmemPackB();

        constexpr index_t A_LDS_Write_Width = GetSmemPackA();
        constexpr index_t B_LDS_Write_Width = GetSmemPackB();

        constexpr index_t A_Buffer_Load_Inst_Num =
            MPerBlock * KPerBlock / (BlockSize * GetVectorSizeA());
        constexpr index_t B_Buffer_Load_Inst_Num =
            NPerBlock * KPerBlock / (BlockSize * GetVectorSizeB());

        constexpr index_t A_LDS_Write_Inst_Num =
            MPerBlock * KPerBlock / (BlockSize * A_LDS_Write_Width);
        constexpr index_t B_LDS_Write_Inst_Num =
            NPerBlock * KPerBlock / (BlockSize * B_LDS_Write_Width);

        constexpr index_t A_LDS_Read_Inst_Num =
            WaveNumN * MPerBlock * KPerBlock / (BlockSize * A_LDS_Read_Width);
        constexpr index_t B_LDS_Read_Inst_Num =
            WaveNumM * NPerBlock * KPerBlock / (BlockSize * B_LDS_Read_Width);

        constexpr index_t C_MFMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
                                            (BlockSize / WaveSize) / (MPerXDL * NPerXDL * KPerXDL);

        auto str = std::stringstream{};

        str << "MHC GEMM Pipeline V3 with Fusion Support\n"
            << "A/B vector size: " << GetVectorSizeA() << ", " << GetVectorSizeB() << "\n"
            << "A/B LDS read/write width: " << A_LDS_Read_Width << ", " << B_LDS_Read_Width << "\n"
            << "A/B buffer load inst: " << A_Buffer_Load_Inst_Num << ", " << B_Buffer_Load_Inst_Num
            << "\n"
            << "A/B LDS write inst: " << A_LDS_Write_Inst_Num << ", " << B_LDS_Write_Inst_Num
            << "\n"
            << "A/B LDS read inst: " << A_LDS_Read_Inst_Num << ", " << B_LDS_Read_Inst_Num << "\n"
            << "C MFMA inst: " << C_MFMA_Inst_Num << "\n"
            << "KPack: " << BlockGemm::Traits::KPack << "\n"
            << "PrefetchStages: " << PrefetchStages << "\n";
        return str.str();
    }

    // Empty fusion function type for default case
    struct NoFusion
    {
        template <typename Tile>
        CK_TILE_DEVICE void operator()(const Tile&) const
        {
        }
    };

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        CK_TILE_DEVICE static constexpr auto HotLoopScheduler()
        {
            constexpr index_t MPerXDL = BlockGemm::WarpGemm::kM;
            constexpr index_t NPerXDL = BlockGemm::WarpGemm::kN;
            constexpr index_t KPerXDL = BlockGemm::WarpGemm::WarpGemmAttribute::Impl::kK;

            constexpr index_t WaveSize = get_warp_size();
            constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
            constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});

            constexpr index_t A_LDS_Read_Width = GetSmemPackA();
            constexpr index_t B_LDS_Read_Width = GetSmemPackB();

            constexpr index_t A_LDS_Write_Width = GetSmemPackA();
            constexpr index_t B_LDS_Write_Width = GetSmemPackB();

            constexpr index_t A_Buffer_Load_Inst_Num =
                MPerBlock * KPerBlock / (BlockSize * GetVectorSizeA());
            constexpr index_t B_Buffer_Load_Inst_Num =
                NPerBlock * KPerBlock / (BlockSize * GetVectorSizeB());

            constexpr index_t A_LDS_Write_Inst_Num =
                MPerBlock * KPerBlock / (BlockSize * A_LDS_Write_Width);
            constexpr index_t B_LDS_Write_Inst_Num =
                NPerBlock * KPerBlock / (BlockSize * B_LDS_Write_Width);

            constexpr index_t A_LDS_Read_Inst_Num =
                WaveNumN * MPerBlock * KPerBlock / (BlockSize * A_LDS_Read_Width);
            constexpr index_t B_LDS_Read_Inst_Num =
                WaveNumM * NPerBlock * KPerBlock / (BlockSize * B_LDS_Read_Width);

            constexpr index_t C_MFMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
                                                (BlockSize / WaveSize) /
                                                (MPerXDL * NPerXDL * KPerXDL);

            constexpr auto num_ds_read_inst_a =
                A_LDS_Read_Width * sizeof(ADataType) / APackedSize == 16 ? A_LDS_Read_Inst_Num
                                                                         : A_LDS_Read_Inst_Num / 2;
            constexpr auto num_ds_read_inst_b =
                B_LDS_Read_Width * sizeof(BDataType) / BPackedSize == 16 ? B_LDS_Read_Inst_Num
                                                                         : B_LDS_Read_Inst_Num / 2;

            constexpr auto num_ds_write_inst_a = A_LDS_Write_Inst_Num;
            constexpr auto num_ds_write_inst_b = B_LDS_Write_Inst_Num;

            constexpr auto num_buffer_load_inst_a = A_Buffer_Load_Inst_Num;
            constexpr auto num_buffer_load_inst_b = B_Buffer_Load_Inst_Num;

            constexpr auto num_mfma_inst = C_MFMA_Inst_Num;

            constexpr auto mfma_cycle = NPerXDL == 16 ? 16 : 32;
            constexpr auto ds_read_a_issue_cycle =
                A_LDS_Read_Width * sizeof(ADataType) / APackedSize == 16 ? 8 : 4;
            constexpr auto ds_read_b_issue_cycle =
                B_LDS_Read_Width * sizeof(BDataType) / BPackedSize == 16 ? 8 : 4;
            constexpr auto ds_read_a_mfma_rate =
                (mfma_cycle - 4 + 2 * ds_read_a_issue_cycle - 1) / (2 * ds_read_a_issue_cycle);
            constexpr auto ds_read_b_mfma_rate =
                (mfma_cycle - 4 + 2 * ds_read_b_issue_cycle - 1) / (2 * ds_read_b_issue_cycle);

            constexpr auto num_dsread_a_mfma =
                (num_ds_read_inst_a + ds_read_a_mfma_rate - 1) / ds_read_a_mfma_rate;
            constexpr auto num_dsread_b_mfma =
                (num_ds_read_inst_b + ds_read_b_mfma_rate - 1) / ds_read_b_mfma_rate;

            constexpr auto num_mfma_stage1 =
                num_mfma_inst - (num_dsread_a_mfma + num_dsread_b_mfma);
            constexpr auto num_mfma_per_issue =
                num_mfma_stage1 / (num_buffer_load_inst_a + num_buffer_load_inst_b);
            constexpr auto num_dswrite_per_issue_a = num_ds_write_inst_a / num_buffer_load_inst_a;
            constexpr auto num_dswrite_per_issue_b = num_ds_write_inst_b / num_buffer_load_inst_b;

            static_for<0, num_buffer_load_inst_a, 1>{}([&](auto i) {
                ignore = i;
                static_for<0, num_dswrite_per_issue_a, 1>{}([&](auto idswrite) {
                    ignore = idswrite;
                    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                });
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);
                __builtin_amdgcn_sched_group_barrier(
                    0x008, num_mfma_per_issue - num_dswrite_per_issue_a, 0);
            });
            static_for<0, num_buffer_load_inst_b, 1>{}([&](auto i) {
                ignore = i;
                static_for<0, num_dswrite_per_issue_b, 1>{}([&](auto idswrite) {
                    ignore = idswrite;
                    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                });
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);
                __builtin_amdgcn_sched_group_barrier(
                    0x008, num_mfma_per_issue - num_dswrite_per_issue_b, 0);
            });

            static_for<0, num_dsread_a_mfma, 1>{}([&](auto i) {
                if constexpr((num_ds_read_inst_a - (i + 1) * ds_read_a_mfma_rate) >=
                             ds_read_a_mfma_rate)
                {
                    __builtin_amdgcn_sched_group_barrier(0x100, ds_read_a_mfma_rate, 0);
                }
                else
                {
                    __builtin_amdgcn_sched_group_barrier(
                        0x100,
                        num_ds_read_inst_a - (num_dsread_a_mfma - 1) * ds_read_a_mfma_rate,
                        0);
                }
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
            });

            static_for<0, num_dsread_b_mfma, 1>{}([&](auto i) {
                if constexpr((num_ds_read_inst_b - (i + 1) * ds_read_b_mfma_rate) >=
                             ds_read_b_mfma_rate)
                {
                    __builtin_amdgcn_sched_group_barrier(0x100, ds_read_b_mfma_rate, 0);
                }
                else
                {
                    __builtin_amdgcn_sched_group_barrier(
                        0x100,
                        num_ds_read_inst_b - (num_dsread_b_mfma - 1) * ds_read_b_mfma_rate,
                        0);
                }
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
            });
        }

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename AsDramBlockWindowTmp,
                  typename BsDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction,
                  typename FusionFunction          = NoFusion,
                  typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                                is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                            bool>* = nullptr>
        CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       index_t num_loop,
                                       void* p_smem,
                                       const FusionFunction& fusion_func = NoFusion{}) const
        {
            using ADramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, AsDramBlockWindowTmp>>;
            using BDramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, BsDramBlockWindowTmp>>;

            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "A/B Dram block window should have the same data type as appropriate "
                "([A|B]DataType) defined in Problem definition!");

            constexpr bool is_a_col_major =
                std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

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

            // A/B tiles in LDS
            auto&& [a_lds_block, b_lds_block] = Base::GetABLdsTensorViews(p_smem);

            // Tile distribution for load from lds
            constexpr auto a_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto b_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            // A DRAM/LDS windows
            auto&& [a_copy_dram_window, a_copy_lds_window, a_lds_gemm_window] =
                Base::GetAWindows(a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr);

            // B DRAM/LDS windows
            auto&& [b_copy_dram_window, b_copy_lds_window, b_lds_gemm_window] =
                Base::GetBWindows(b_dram_block_window_tmp, b_lds_block, b_lds_load_tile_distr);

            // Block GEMM
            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();

            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

            // Initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // Load tile with elementwise function
            auto elementwise_As_res =
                load_tile_with_elementwise(a_copy_dram_window, a_element_func);

            // Apply fusion function to A tile (e.g., for norm computation)
            if constexpr(!std::is_same_v<FusionFunction, NoFusion>)
            {
                fusion_func(elementwise_As_res);
            }

            move_tile_window(a_copy_dram_window, a_dram_tile_window_step);

            auto elementwise_Bs_res =
                load_tile_with_elementwise(b_copy_dram_window, b_element_func);
            move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

            // LDS write 0
            if constexpr(is_a_col_major && !is_a_load_tr_v())
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffle_tmp, elementwise_As_res);
                Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp);
            }
            else
            {
                Base::LocalPrefill(a_copy_lds_window, elementwise_As_res);
            }
            if constexpr(is_b_row_major && !is_b_load_tr_v())
            {
                auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(b_shuffle_tmp, elementwise_Bs_res);
                Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp);
            }
            else
            {
                Base::LocalPrefill(b_copy_lds_window, elementwise_Bs_res);
            }

            // global read 1
            elementwise_As_res = load_tile_with_elementwise(a_copy_dram_window, a_element_func);

            // Apply fusion function to A tile
            if constexpr(!std::is_same_v<FusionFunction, NoFusion>)
            {
                fusion_func(elementwise_As_res);
            }

            move_tile_window(a_copy_dram_window, a_dram_tile_window_step);

            elementwise_Bs_res = load_tile_with_elementwise(b_copy_dram_window, b_element_func);
            move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

            block_sync_lds();
            block_gemm.LocalPrefetch(
                a_lds_gemm_window, b_lds_gemm_window, is_a_load_tr_v, is_b_load_tr_v);

            __builtin_amdgcn_sched_barrier(0);

            // main body
            if constexpr(HasHotLoop)
            {
                index_t i = 0;
                do
                {
                    block_sync_lds();

                    if constexpr(is_a_col_major && !is_a_load_tr_v())
                    {
                        auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                            Policy::template MakeShuffledARegTileDistribution<Problem>());
                        transpose_tile2d(a_shuffle_tmp, elementwise_As_res);
                        Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp);
                    }
                    else
                    {
                        Base::LocalPrefill(a_copy_lds_window, elementwise_As_res);
                    }
                    if constexpr(is_b_row_major && !is_b_load_tr_v())
                    {
                        auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                            Policy::template MakeShuffledBRegTileDistribution<Problem>());
                        transpose_tile2d(b_shuffle_tmp, elementwise_Bs_res);
                        Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp);
                    }
                    else
                    {
                        Base::LocalPrefill(b_copy_lds_window, elementwise_Bs_res);
                    }

                    elementwise_As_res =
                        load_tile_with_elementwise(a_copy_dram_window, a_element_func);

                    // Apply fusion function to A tile
                    if constexpr(!std::is_same_v<FusionFunction, NoFusion>)
                    {
                        fusion_func(elementwise_As_res);
                    }

                    move_tile_window(a_copy_dram_window, a_dram_tile_window_step);

                    elementwise_Bs_res =
                        load_tile_with_elementwise(b_copy_dram_window, b_element_func);
                    move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

                    block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

                    block_sync_lds();

                    block_gemm.LocalPrefetch(
                        a_lds_gemm_window, b_lds_gemm_window, is_a_load_tr_v, is_b_load_tr_v);
                    HotLoopScheduler();
                    __builtin_amdgcn_sched_barrier(0);

                    i += 1;
                } while(i < (num_loop - 1));
            }
            // tail
            if constexpr(TailNum == TailNumber::Odd)
            {
                block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
            }
            else
            {
                block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
                block_sync_lds();

                if constexpr(is_a_col_major && !is_a_load_tr_v())
                {
                    auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                        Policy::template MakeShuffledARegTileDistribution<Problem>());
                    transpose_tile2d(a_shuffle_tmp, elementwise_As_res);
                    Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp);
                }
                else
                {
                    Base::LocalPrefill(a_copy_lds_window, elementwise_As_res);
                }
                if constexpr(is_b_row_major && !is_b_load_tr_v())
                {
                    auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                        Policy::template MakeShuffledBRegTileDistribution<Problem>());
                    transpose_tile2d(b_shuffle_tmp, elementwise_Bs_res);
                    Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp);
                }
                else
                {
                    Base::LocalPrefill(b_copy_lds_window, elementwise_Bs_res);
                }
                block_sync_lds();
                block_gemm.LocalPrefetch(
                    a_lds_gemm_window, b_lds_gemm_window, is_a_load_tr_v, is_b_load_tr_v);
                block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
            }
            return c_block_tile;
        }
    };

    // Main operator() with fusion function support
    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction,
              typename FusionFunction          = NoFusion,
              typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* p_smem,
                                   const FusionFunction& fusion_func = NoFusion{}) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);

        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp,
                a_element_func,
                b_dram_block_window_tmp,
                b_element_func,
                num_loop,
                p_smem,
                fusion_func);
        };

        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }

    // Overload without fusion function (standard GEMM)
    template <typename AsDramBlockWindowTmp,
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
                                   void* p_smem) const
    {
        return operator()(a_dram_block_window_tmp,
                          a_element_func,
                          b_dram_block_window_tmp,
                          b_element_func,
                          num_loop,
                          p_smem,
                          NoFusion{});
    }

    // Single input overloads (non-tuple)
    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction,
              typename FusionFunction          = NoFusion,
              typename std::enable_if_t<!is_detected<is_tuple, ADramBlockWindowTmp>::value &&
                                            !is_detected<is_tuple, BDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* p_smem,
                                   const FusionFunction& fusion_func = NoFusion{}) const
    {
        return operator()(ck_tile::make_tuple(a_dram_block_window_tmp),
                          a_element_func,
                          ck_tile::make_tuple(b_dram_block_window_tmp),
                          b_element_func,
                          num_loop,
                          p_smem,
                          fusion_func);
    }

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction,
              typename std::enable_if_t<!is_detected<is_tuple, ADramBlockWindowTmp>::value &&
                                            !is_detected<is_tuple, BDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return operator()(a_dram_block_window_tmp,
                          a_element_func,
                          b_dram_block_window_tmp,
                          b_element_func,
                          num_loop,
                          p_smem,
                          NoFusion{});
    }
};

} // namespace ck_tile
