// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_coherency.hpp"

namespace ck_tile {
enum StreamKReductionStrategy : uint32_t
{
    Atomic = 0u,
    Linear = 1u,
    Tree   = 2u
};

/// @brief StreamK reduction helpers: partial store/load, flag signaling, and tile accumulation.
///        Shared by StreamK GEMM and StreamK conv bwd weight kernels.
template <typename TilePartitioner_, typename GemmPipeline_>
struct StreamKReductionOps
{
    using TilePartitioner = remove_cvref_t<TilePartitioner_>;
    using BlockGemm       = typename GemmPipeline_::BlockGemm;
    using WarpGemm        = typename BlockGemm::WarpGemm;
    using BlockGemmShape  = typename GemmPipeline_::BlockGemmShape;

    CK_TILE_DEVICE static constexpr index_t GetVectorSizePartials()
    {
        return WarpGemm::WarpGemmAttribute::Impl::kCM1PerLane;
    }

    CK_TILE_DEVICE static constexpr auto MakePartialsDistribution()
    {
        constexpr index_t m_warp = BlockGemmShape::BlockWarps::at(number<0>{});
        constexpr index_t n_warp = BlockGemmShape::BlockWarps::at(number<1>{});

        constexpr index_t m_iter_per_warp = TilePartitioner::MPerBlock / (m_warp * WarpGemm::kM);
        constexpr index_t n_iter_per_warp = TilePartitioner::NPerBlock / (n_warp * WarpGemm::kN);

        constexpr auto partials_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<m_iter_per_warp, m_warp>, sequence<n_iter_per_warp, n_warp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr index_t vector_size         = GetVectorSizePartials();
        constexpr index_t m_warp_repeat       = WarpGemm::WarpGemmAttribute::Impl::kCM0PerLane;
        constexpr index_t warp_tile_n_threads = WarpGemm::kN / vector_size;
        constexpr index_t warp_tile_m_threads = get_warp_size() / warp_tile_n_threads;

        constexpr auto partials_inner_dstr_encoding =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<m_warp_repeat, warp_tile_m_threads>,
                                             sequence<warp_tile_n_threads, vector_size>>,
                                       tuple<sequence<1, 2>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{};

        constexpr auto partials_dstr_encode = detail::make_embed_tile_distribution_encoding(
            partials_outer_dstr_encoding, partials_inner_dstr_encoding);

        return make_static_tile_distribution(partials_dstr_encode);
    }

    template <typename OAccTile, typename KernelArgs>
    CK_TILE_DEVICE void
    StorePartial(const KernelArgs& kargs, index_t cta_idx, const OAccTile& c_block_tile) const
    {
        const auto c_block_tile_buffer_size = TilePartitioner::MPerBlock *
                                              TilePartitioner::NPerBlock *
                                              sizeof(typename OAccTile::DataType);
        void* partial_buffer_ptr = static_cast<char*>(kargs.workspace_ptr) +
                                   kargs.tile_partitioner.get_flags_buffer_size() +
                                   cta_idx * c_block_tile_buffer_size;

        const auto& partial_tensor_view = make_naive_tensor_view<
            address_space_enum::global,
            memory_operation_enum::set,
            StreamKCoherency<decltype(core::arch::get_compiler_target())>::BUFFER_COHERENCE>(
            static_cast<typename OAccTile::DataType*>(partial_buffer_ptr),
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            make_tuple(TilePartitioner::NPerBlock, 1),
            number<GetVectorSizePartials()>{},
            number<1>{});

        auto partial_tile_window = make_tile_window(
            partial_tensor_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {0, 0},
            MakePartialsDistribution());

        auto c_with_partials_dist = make_static_distributed_tensor<typename OAccTile::DataType>(
            MakePartialsDistribution(), c_block_tile.get_thread_buffer());

        store_tile(partial_tile_window, c_with_partials_dist);
        s_waitcnt</*vmcnt*/ 0, waitcnt_arg::kMaxExpCnt, waitcnt_arg::kMaxLgkmCnt>();
        __builtin_amdgcn_s_barrier();
    }

    template <typename DataType, typename OAccTileDist, typename KernelArgs>
    CK_TILE_DEVICE auto LoadPartial(const KernelArgs& kargs,
                                    index_t cta_idx,
                                    const OAccTileDist& c_block_tile_dist) const
    {
        const auto c_block_tile_buffer_size =
            TilePartitioner::MPerBlock * TilePartitioner::NPerBlock * sizeof(DataType);
        void* partial_buffer_ptr = static_cast<char*>(kargs.workspace_ptr) +
                                   kargs.tile_partitioner.get_flags_buffer_size() +
                                   cta_idx * c_block_tile_buffer_size;

        const auto& partial_tensor_view = make_naive_tensor_view<
            address_space_enum::global,
            memory_operation_enum::set,
            StreamKCoherency<decltype(core::arch::get_compiler_target())>::BUFFER_COHERENCE>(
            static_cast<DataType*>(partial_buffer_ptr),
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            make_tuple(TilePartitioner::NPerBlock, 1),
            number<GetVectorSizePartials()>{},
            number<1>{});

        auto partial_tile_window = make_tile_window(
            partial_tensor_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {0, 0},
            MakePartialsDistribution());

        auto partials_tile = load_tile(partial_tile_window);

        auto partials_tile_with_c_distr = make_static_distributed_tensor<DataType>(
            c_block_tile_dist, partials_tile.get_thread_buffer());

        return partials_tile_with_c_distr;
    }

    template <typename OAccTile>
    CK_TILE_DEVICE static void AddBlockTile(OAccTile& in_out_block_tile,
                                            const OAccTile& in_block_tile)
    {
        using BlockType        = remove_cvref_t<decltype(in_out_block_tile)>;
        constexpr auto o_spans = BlockType::get_distributed_spans();
        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto idx     = make_tuple(idx0, idx1);
                in_out_block_tile(idx) = in_out_block_tile[idx] + in_block_tile[idx];
            });
        });
    }

    template <typename KernelArgs>
    CK_TILE_DEVICE static void SignalStorePartialDone(const KernelArgs& kargs, index_t cta_idx)
    {
        auto* sk_flags_ptr = static_cast<index_t*>(kargs.workspace_ptr);
        index_t offset     = cta_idx * sizeof(index_t);

        asm volatile("s_store_dword %0, %1, %2 glc\n\t"
                     "s_waitcnt lgkmcnt(0)"
                     :
                     : "s"(1), "s"(sk_flags_ptr), "s"(offset)
                     : "memory");
    }

    template <typename KernelArgs>
    CK_TILE_DEVICE static void WaitStorePartialDone(const KernelArgs& kargs, index_t cta_idx)
    {
        auto* sk_flags_ptr = static_cast<index_t*>(kargs.workspace_ptr);
        index_t result;
        index_t offset = cta_idx * sizeof(index_t);

        do
        {
            asm volatile("s_load_dword %0, %1, %2 glc\n\t"
                         "s_waitcnt lgkmcnt(0)"
                         : "=s"(result)
                         : "s"(sk_flags_ptr), "s"(offset)
                         : "memory");
        } while(result != 1);
    }
};

} // namespace ck_tile
