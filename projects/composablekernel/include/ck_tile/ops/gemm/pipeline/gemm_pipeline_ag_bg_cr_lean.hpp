// CK Tile project-local "Lean" GEMM pipeline.
//
// Mirrors GemmPipelineAgBgCrCompV3 but drops the BlockGemm::LocalPrefetch
// register-tile prefetch shadow registers. Instead of holding A and B
// fragments live across K iterations, we read LDS inline with the MFMAs.
// Net effect: lower VGPR usage at the cost of slightly worse MFMA-vs-LDS-read
// overlap. On small fixed-shape conv workloads where the universal pipeline
// is launch/occupancy bound rather than MFMA bound, this trades favorably.
//
// Same template surface as GemmPipelineAgBgCrCompV3 so we can drop it into the
// existing GroupedConvolutionForwardKernel without modifications.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"

namespace ck_tile {

template <typename Problem>
struct BaseGemmPipelineAgBgCrLean
{
    static constexpr index_t PrefetchStages   = 2;
    static constexpr index_t PrefillStages    = 1;
    static constexpr index_t GlobalBufferNum  = 1;
    static constexpr bool UsePersistentKernel = Problem::Traits::UsePersistentKernel;

    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        if(BlockHasHotloop(num_loop) || num_loop == 3) return TailNumber::Odd;
        if(num_loop == 2) return TailNumber::Even;
        return TailNumber::Odd;
    }

    template <size_t I = 0, typename RunFunction>
    CK_TILE_HOST_DEVICE static auto
    TailHandler(const RunFunction& run_func, bool has_hot_loop, TailNumber tail_number)
    {
        const bool has_hot_loop_first_lane      = amd_wave_read_first_lane(has_hot_loop);
        const TailNumber tail_number_first_lane = amd_wave_read_first_lane(tail_number);

        constexpr auto scenarios = std::array<std::pair<bool, ck_tile::TailNumber>, 3>{
            std::make_pair(true,  TailNumber::Odd),
            std::make_pair(false, TailNumber::Odd),
            std::make_pair(false, TailNumber::Even),
        };
        if(has_hot_loop_first_lane == scenarios[I].first &&
           tail_number_first_lane == scenarios[I].second)
            return run_func(bool_constant<scenarios[I].first>{}, constant<scenarios[I].second>{});
        else if constexpr(I + 1 < scenarios.size())
            return TailHandler<I + 1>(run_func, has_hot_loop, tail_number);

#if defined(__HIP_DEVICE_COMPILE__)
        __builtin_unreachable();
#else
        throw std::logic_error("Invalid TailNumber for Lean pipeline");
#endif
    }
};

template <typename Problem, typename Policy = UniversalGemmPipelineAgBgCrPolicy>
struct GemmPipelineAgBgCrLean : public BaseGemmPipelineAgBgCrLean<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrLean<Problem>;
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

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName() { return "LEAN"; }
    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
        constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});
        return concat('_', "pipeline_AgBgCrLean",
                      concat('x', MPerBlock, NPerBlock, KPerBlock), BlockSize,
                      concat('x', GetVectorSizeA(), GetVectorSizeB(), GetVectorSizeC()),
                      concat('x', WaveNumM, WaveNumN),
                      concat('x', kPadM, kPadN, kPadK),
                      Problem::GetName());
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    // Renamed inner implementation to avoid ambiguity in dispatcher chain.
    template <bool HasHotLoop,
              TailNumber TailNum,
              typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction>
    CK_TILE_DEVICE auto run_impl(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                 const AElementFunction& a_element_func,
                                 const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                 const BElementFunction& b_element_func,
                                 index_t num_loop,
                                 void* p_smem) const
    {
        using ADramBlockWindowTmp =
            remove_cvref_t<std::tuple_element_t<number<0>{}, AsDramBlockWindowTmp>>;
        using BDramBlockWindowTmp =
            remove_cvref_t<std::tuple_element_t<number<0>{}, BsDramBlockWindowTmp>>;

        constexpr bool is_a_col_major =
            std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
        constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

        // A/B tiles in LDS
        auto&& [a_lds_block, b_lds_block] = PipelineImplBase{}.GetABLdsTensorViews(p_smem);

        constexpr auto a_lds_load_tile_distr =
            make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
        constexpr auto b_lds_load_tile_distr =
            make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

        auto&& [a_copy_dram_window, a_copy_lds_window, a_lds_gemm_window] =
            PipelineImplBase{}.GetAWindows(a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr);

        auto&& [b_copy_dram_window, b_copy_lds_window, b_lds_gemm_window] =
            PipelineImplBase{}.GetBWindows(b_dram_block_window_tmp, b_lds_block, b_lds_load_tile_distr);

        auto block_gemm   = BlockGemm();
        auto c_block_tile = block_gemm.MakeCBlockTile();

        using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
        using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;

        constexpr ADramTileWindowStep a_dram_tile_window_step =
            is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
        constexpr BDramTileWindowStep b_dram_tile_window_step =
            is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

        // initialize C
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        // ----------------------------------------------------------
        // Prologue: load tile 0 -> regs -> LDS, then load tile 1 to regs.
        // No register-tile prefetch (saves VGPRs).
        // ----------------------------------------------------------
        auto elementwise_As_res =
            load_tile_with_elementwise(a_copy_dram_window, a_element_func);
        move_tile_window(a_copy_dram_window, a_dram_tile_window_step);
        auto elementwise_Bs_res =
            load_tile_with_elementwise(b_copy_dram_window, b_element_func);
        move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

        if constexpr(is_a_col_major && !is_a_load_tr_v())
        {
            auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                Policy::template MakeShuffledARegTileDistribution<Problem>());
            transpose_tile2d(a_shuffle_tmp, elementwise_As_res);
            PipelineImplBase{}.LocalPrefill(a_copy_lds_window, a_shuffle_tmp);
        }
        else
        {
            PipelineImplBase{}.LocalPrefill(a_copy_lds_window, elementwise_As_res);
        }
        if constexpr(is_b_row_major && !is_b_load_tr_v())
        {
            auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                Policy::template MakeShuffledBRegTileDistribution<Problem>());
            transpose_tile2d(b_shuffle_tmp, elementwise_Bs_res);
            PipelineImplBase{}.LocalPrefill(b_copy_lds_window, b_shuffle_tmp);
        }
        else
        {
            PipelineImplBase{}.LocalPrefill(b_copy_lds_window, elementwise_Bs_res);
        }

        // global read 1
        elementwise_As_res = load_tile_with_elementwise(a_copy_dram_window, a_element_func);
        move_tile_window(a_copy_dram_window, a_dram_tile_window_step);
        elementwise_Bs_res = load_tile_with_elementwise(b_copy_dram_window, b_element_func);
        move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

        block_sync_lds();
        // Initial LDS->regs prefetch for tile 0 (matches V3 prologue).
        block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window,
                                 is_a_load_tr_v, is_b_load_tr_v);
        __builtin_amdgcn_sched_barrier(0);

        // ----------------------------------------------------------
        // Hot loop: each iter does compute on current LDS tile (using
        // the just-prefetched warp tiles), then overwrites LDS with the
        // next tile from gmem regs, and refreshes the prefetch tiles.
        // Unlike V3, we do not stretch the prefetch pipeline by 1 extra
        // stage -- the prefetch is local to each iter, so the compiler
        // can keep the warp tiles as scratch regs.
        // ----------------------------------------------------------
        if constexpr(HasHotLoop)
        {
            index_t i = 0;
            do
            {
                // compute on current LDS tile (uses warp_tile_ regs from prefetch)
                block_gemm(c_block_tile,
                           a_lds_gemm_window,
                           b_lds_gemm_window,
                           is_a_load_tr_v,
                           is_b_load_tr_v);
                block_sync_lds();

                if constexpr(is_a_col_major && !is_a_load_tr_v())
                {
                    auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                        Policy::template MakeShuffledARegTileDistribution<Problem>());
                    transpose_tile2d(a_shuffle_tmp, elementwise_As_res);
                    PipelineImplBase{}.LocalPrefill(a_copy_lds_window, a_shuffle_tmp);
                }
                else
                {
                    PipelineImplBase{}.LocalPrefill(a_copy_lds_window, elementwise_As_res);
                }
                if constexpr(is_b_row_major && !is_b_load_tr_v())
                {
                    auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                        Policy::template MakeShuffledBRegTileDistribution<Problem>());
                    transpose_tile2d(b_shuffle_tmp, elementwise_Bs_res);
                    PipelineImplBase{}.LocalPrefill(b_copy_lds_window, b_shuffle_tmp);
                }
                else
                {
                    PipelineImplBase{}.LocalPrefill(b_copy_lds_window, elementwise_Bs_res);
                }

                elementwise_As_res =
                    load_tile_with_elementwise(a_copy_dram_window, a_element_func);
                move_tile_window(a_copy_dram_window, a_dram_tile_window_step);

                elementwise_Bs_res =
                    load_tile_with_elementwise(b_copy_dram_window, b_element_func);
                move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

                block_sync_lds();
                // Refresh the warp_tile_ prefetch from the LDS we just wrote.
                block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window,
                                         is_a_load_tr_v, is_b_load_tr_v);
                __builtin_amdgcn_sched_barrier(0);
                i += 1;
            } while(i < (num_loop - 1));
        }

        // ----------------------------------------------------------
        // Tail. The last LocalPrefetch already filled the warp tiles
        // for the final LDS tile. One block_gemm consumes them.
        // ----------------------------------------------------------
        block_gemm(c_block_tile,
                   a_lds_gemm_window,
                   b_lds_gemm_window,
                   is_a_load_tr_v,
                   is_b_load_tr_v);
        return c_block_tile;
    }

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
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](auto hot_loop_, auto tail_num_) {
            return this->template run_impl<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp,
                a_element_func,
                b_dram_block_window_tmp,
                b_element_func,
                num_loop,
                p_smem);
        };
        return Base::TailHandler(Run, has_hot_loop, tail_number);
    }

    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   bool has_hot_loop,
                                   TailNumber tail_number,
                                   void* p_smem) const
    {
        ck_tile::element_wise::PassThrough a_id{};
        ck_tile::element_wise::PassThrough b_id{};
        const auto Run = [&](auto hot_loop_, auto tail_num_) {
            return this->template run_impl<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp, a_id,
                b_dram_block_window_tmp, b_id,
                num_loop, p_smem);
        };
        return Base::TailHandler(Run, has_hot_loop, tail_number);
    }

    // 4-arg tuple overload: dispatch through the bool/tail entry point so we
    // never need element functions on this path.
    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);
        return this->operator()(a_dram_block_window_tmp,
                                b_dram_block_window_tmp,
                                num_loop, has_hot_loop, tail_number, p_smem);
    }

    // ------------------------------------------------------------------
    // Single-window overloads: the GroupedConvolutionForwardKernel passes
    // a plain tile_window (not a tuple). Wrap in make_tuple and delegate.
    // SFINAE-constrained to non-tuple inputs so they don't compete with the
    // tuple overloads above.
    // ------------------------------------------------------------------
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
        return this->operator()(ck_tile::make_tuple(a_dram_block_window_tmp), a_element_func,
                                ck_tile::make_tuple(b_dram_block_window_tmp), b_element_func,
                                num_loop, p_smem);
    }

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename std::enable_if_t<!is_detected<is_tuple, ADramBlockWindowTmp>::value &&
                                            !is_detected<is_tuple, BDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   bool has_hot_loop,
                                   TailNumber tail_number,
                                   void* p_smem) const
    {
        return this->operator()(ck_tile::make_tuple(a_dram_block_window_tmp),
                                ck_tile::make_tuple(b_dram_block_window_tmp),
                                num_loop, has_hot_loop, tail_number, p_smem);
    }

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename std::enable_if_t<!is_detected<is_tuple, ADramBlockWindowTmp>::value &&
                                            !is_detected<is_tuple, BDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return this->operator()(ck_tile::make_tuple(a_dram_block_window_tmp),
                                ck_tile::make_tuple(b_dram_block_window_tmp),
                                num_loop, p_smem);
    }
};

} // namespace ck_tile
