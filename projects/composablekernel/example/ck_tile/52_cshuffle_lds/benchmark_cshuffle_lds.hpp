// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file benchmark_cshuffle_lds.hpp
 * @brief LDS benchmark setup for CShuffleEpilogue.
 *
 * Provides Setup adapters that extract LDS descriptor and distribution
 * from CShuffleEpilogue for use with generic tile benchmark kernels.
 */

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/utility/tile_load_store_microkernels.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {

/**
 * @brief Create CShuffleEpilogue type from benchmark parameters.
 */
template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename ODataType,
          index_t kM,
          index_t kN,
          index_t MWave,
          index_t NWave,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t KPerXdl>
using BenchmarkEpilogue = CShuffleEpilogue<CShuffleEpilogueProblem<ADataType,
                                                                   BDataType,
                                                                   tuple<>,
                                                                   AccDataType,
                                                                   ODataType,
                                                                   tuple<>,
                                                                   tensor_layout::gemm::RowMajor,
                                                                   element_wise::PassThrough,
                                                                   kM,
                                                                   kN,
                                                                   MWave,
                                                                   NWave,
                                                                   MPerXdl,
                                                                   NPerXdl,
                                                                   KPerXdl,
                                                                   false>>;

/**
 * @brief Debug helper to print XOR swizzle parameters at runtime.
 */
template <typename Epilogue>
struct XorDebugInfo
{
    CK_TILE_DEVICE static void print()
    {
        if(threadIdx.x == 0 && blockIdx.x == 0)
        {
            // Get values from Epilogue
            constexpr int MPerXdl = Epilogue::Problem::MPerXdl;
            constexpr int NPerIter = Epilogue::NPerIterationShuffle;
            constexpr int DataTypeSize = sizeof(typename Epilogue::ODataType);
            constexpr int BaseVectorLen = Epilogue::GetVectorSizeC();

            // Compute log2 at runtime
            int log2_base = 0;
            for(int v = BaseVectorLen; v > 1; v >>= 1) ++log2_base;

#if defined(CK_GFX950_SUPPORT)
            // gfx950: 64-bank LDS needs 3-bit XOR, which requires VectorLen=4
            bool needs_reduction = (MPerXdl == 16) &&
                                   ((1 << (log2_base + 3)) > NPerIter) &&  // +3 for 3-bit XOR
                                   (BaseVectorLen > 4);
            int vec_len = needs_reduction ? 4 : BaseVectorLen;
#else
            // gfx942: 32-bank LDS works with 2-bit XOR, VectorLen=8 sufficient
            bool needs_reduction = (MPerXdl == 16) &&
                                   ((1 << (log2_base + 2)) > NPerIter) &&
                                   (BaseVectorLen > 8);
            int vec_len = needs_reduction ? 8 : BaseVectorLen;
#endif

            int log2_vec = 0;
            for(int v = vec_len; v > 1; v >>= 1) ++log2_vec;
            int col_bit_start = log2_vec;

#if defined(CK_GFX950_SUPPORT)
            int xor_bits = 3;
            int is_gfx950 = 1;
#else
            int xor_bits = 2;
            int is_gfx950 = 0;
#endif
            bool has_enough = (1 << (col_bit_start + xor_bits)) <= NPerIter;
            bool xor_applied = (MPerXdl == 16) && has_enough;

            printf("XOR Debug: ODataSize=%d, MPerXdl=%d, NPerIter=%d\n",
                   DataTypeSize, MPerXdl, NPerIter);
            printf("  BaseVectorLen=%d, VectorLen=%d (reduced=%d)\n",
                   BaseVectorLen, vec_len, (int)needs_reduction);
            printf("  col_bit_start=%d, xor_bits=%d, gfx950=%d\n",
                   col_bit_start, xor_bits, is_gfx950);
            printf("  has_enough_col_bits=%d (need N>=%d, have N=%d)\n",
                   (int)has_enough, (1 << (col_bit_start + xor_bits)), NPerIter);
            printf("  XOR_APPLIED=%d\n", (int)xor_applied);
        }
    }
};

/**
 * @brief Setup for LDS store benchmark - adapts CShuffleEpilogue for tile benchmark.
 */
template <typename Epilogue>
struct LdsStoreSetup
{
    using ODataType                     = typename Epilogue::ODataType;
    static constexpr index_t kBlockSize = Epilogue::kBlockSize;
    static constexpr index_t kBytes =
        Epilogue::MPerIterationShuffle * Epilogue::NPerIterationShuffle * sizeof(ODataType);
    static constexpr auto lds_desc =
        Epilogue::template MakeLdsBlockDescriptor<typename Epilogue::Problem>();
    static constexpr auto distr =
        make_static_tile_distribution(Epilogue::MakeLdsDistributionEncode());

    CK_TILE_DEVICE static auto create()
    {
        // Print XOR debug info once per kernel
        XorDebugInfo<Epilogue>::print();

        alignas(16) __shared__ char smem[Epilogue::GetSmemSize()];

        auto lds_view =
            make_tensor_view<address_space_enum::lds>(reinterpret_cast<ODataType*>(smem), lds_desc);

        auto window = make_tile_window(lds_view,
                                       make_tuple(number<Epilogue::MPerIterationShuffle>{},
                                                  number<Epilogue::NPerIterationShuffle>{}),
                                       {0, 0},
                                       distr);

        auto tile = make_static_distributed_tensor<ODataType>(distr);

        return make_tuple(window, tile);
    }
};

/**
 * @brief Setup for LDS load benchmark - adapts CShuffleEpilogue for tile benchmark.
 */
template <typename Epilogue>
struct LdsLoadSetup
{
    using ODataType                     = typename Epilogue::ODataType;
    static constexpr index_t kBlockSize = Epilogue::kBlockSize;
    static constexpr index_t kBytes =
        Epilogue::MPerIterationShuffle * Epilogue::NPerIterationShuffle * sizeof(ODataType);
    static constexpr auto lds_desc =
        Epilogue::template MakeLdsBlockDescriptor<typename Epilogue::Problem>();

    using ReadPattern =
        tile_distribution_encoding_pattern_2d<Epilogue::kBlockSize,
                                              Epilogue::MPerIterationShuffle,
                                              Epilogue::NPerIterationShuffle,
                                              Epilogue::GetVectorSizeC(),
                                              tile_distribution_pattern::thread_raked>;
    static constexpr auto read_distr = ReadPattern::make_2d_static_tile_distribution();

    CK_TILE_DEVICE static auto create()
    {
        alignas(16) __shared__ char smem[Epilogue::GetSmemSize()];

        auto lds_view =
            make_tensor_view<address_space_enum::lds>(reinterpret_cast<ODataType*>(smem), lds_desc);

        return make_tile_window(lds_view,
                                make_tuple(number<Epilogue::MPerIterationShuffle>{},
                                           number<Epilogue::NPerIterationShuffle>{}),
                                {0, 0},
                                read_distr);
    }
};

} // namespace ck_tile
