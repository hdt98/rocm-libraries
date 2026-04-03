// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "block_gemm_asmem_bsmem_creg_policy.hpp"

namespace ck_tile {

// BlockGemmASmemBSmemCReg is the warp-coordination layer. It takes A and B already
// in LDS and C already allocated in registers, then:
//   1. Divides the block tile among warps (each warp owns a subset of M and N rows).
//   2. Iterates over the (K, M, N) warp sub-tiles in a nested loop.
//   3. For each (kIter, mIter, nIter): loads A and B warp tiles from LDS, calls WarpGemm,
//      accumulates into the C register tile.
//
// Two operator() overloads:
//   void operator()(c, a, b)  -- accumulate form: c += a * b  (used inside the K loop)
//   auto operator()(a, b)     -- init form:       c  = a * b  (used for standalone calls)
//
// A is block window on shared memory (LDS)
// B is block window on shared memory (LDS)
// C is block distributed tensor (registers / VGPRs)
template <typename Problem, typename Policy = BlockGemmASmemBSmemCRegPolicy>
struct BlockGemmASmemBSmemCReg
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    // WarpGemm is the hardware MFMA wrapper (e.g. WarpGemmMfmaF16F16F32M32N32K8).
    // MWarp and NWarp are the warp grid dimensions within the block.
    // With defaults: MWarp=4 (4 warps cover M), NWarp=1 (all warps share same N range).
    using WarpGemm = remove_cvref_t<
        decltype(Policy::template GetWarpGemmMWarpNWarp<Problem>().template get<0>())>;
    static constexpr index_t MWarp =
        Policy::template GetWarpGemmMWarpNWarp<Problem>().template get<1>();
    static constexpr index_t NWarp =
        Policy::template GetWarpGemmMWarpNWarp<Problem>().template get<2>();

    // AWarpDstr, BWarpDstr, CWarpDstr: tile distribution types for A, B, C at the warp level.
    // These encode which thread (lane) in the warp is responsible for which element,
    // as determined by the MFMA hardware contract.
    using AWarpDstr = typename WarpGemm::AWarpDstr;
    using BWarpDstr = typename WarpGemm::BWarpDstr;
    using CWarpDstr = typename WarpGemm::CWarpDstr;

    using AWarpTensor = typename WarpGemm::AWarpTensor;
    using BWarpTensor = typename WarpGemm::BWarpTensor;
    using CWarpTensor = typename WarpGemm::CWarpTensor;

    // a_warp_y_lengths / b_warp_y_lengths / c_warp_y_lengths:
    // Extents of the per-thread Y dimensions in the warp tile distribution.
    // Used for get_y_sliced_thread_data / set_y_sliced_thread_data to splice C sub-tiles.
    static constexpr auto a_warp_y_lengths =
        to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    static constexpr auto b_warp_y_lengths =
        to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    static constexpr auto c_warp_y_lengths =
        to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

    // Zero-index sequences used as starting offsets when slicing warp sub-tiles out of C.
    static constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
    static constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
    static constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

    // -----------------------------------------------------------------------
    // Accumulate form: c += a * b
    //
    // Called from the pipeline's K loop. c_block_tensor is passed in from outside
    // and accumulates across K iterations.
    //
    // With default parameters (kMPerBlock=256, kNPerBlock=128, kKPerBlock=32, MWarp=4, NWarp=1,
    // WarpGemm::kM=32, WarpGemm::kN=32, WarpGemm::kK=8):
    //   MIterPerWarp = 256 / (4 * 32) = 2   -- each warp covers 2 M-tiles of size 32
    //   NIterPerWarp = 128 / (1 * 32) = 4   -- each warp covers 4 N-tiles of size 32
    //   KIterPerWarp = 32  /  8       = 4   -- each K-slice covers 4 MFMA K-tiles
    //
    // Total MFMA calls per warp: KIterPerWarp * MIterPerWarp * NIterPerWarp = 4 * 2 * 4 = 32
    // Total MFMA calls per block: 32 * 4 warps = 128
    template <typename CBlockTensor, typename ABlockWindowTmp, typename BBlockWindowTmp>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   [[maybe_unused]] const ABlockWindowTmp& a_block_window_tmp,
                                   [[maybe_unused]] const BBlockWindowTmp& b_block_window_tmp) const
    {
        static_assert(std::is_same_v<ADataType, typename ABlockWindowTmp::DataType> &&
                          std::is_same_v<BDataType, typename BBlockWindowTmp::DataType> &&
                          std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                      "wrong!");

        constexpr index_t MPerBlock = ABlockWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t KPerBlock = ABlockWindowTmp{}.get_window_lengths()[number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        // Compute iteration counts for each dimension.
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WarpGemm::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WarpGemm::kK;

        // The M/N/K extent covered by the block per iteration of each loop.
        constexpr index_t MPerBlockPerIter = MPerBlock / MIterPerWarp;
        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        // Determine which warp this thread belongs to (in M and N dimensions).
        // iMWarp: this warp's M stripe index (0..MWarp-1)
        // iNWarp: this warp's N stripe index (0..NWarp-1)
        const index_t iMWarp = get_warp_id() / NWarp;
        const index_t iNWarp = get_warp_id() % NWarp;

        // --- Pre-construct all A warp windows ---
        //
        // Create a 2D array a_warp_windows[mIter][kIter], each a tile window into LDS
        // for this warp's (mIter, kIter) sub-tile of A.
        //
        // AWarpDstrEncoding is the warp-level tile distribution encoding from the MFMA
        // hardware contract: thread j (lane j) is responsible for element (m=j, k=0..7)
        // in the A warp tile (split across the thread pair j and j+32).
        // This encoding is identical for A and B because the MFMA hardware contract is
        // symmetric when M=N=32 (kAMLane == kBNLane). See block_gemm_asmem_bsmem_creg_policy.hpp.
        //
        // statically_indexed_array + static_for are compile-time constructs. The compiler
        // fully unrolls these loops, computes all window origins at compile time, and
        // produces a flat array of constant register addresses -- no loop overhead at runtime.
        auto a_warp_window_tmp = make_tile_window(
            a_block_window_tmp.get_bottom_tensor_view(),
            make_tuple(number<WarpGemm::kM>{}, number<WarpGemm::kK>{}),
            {a_block_window_tmp.get_window_origin().at(number<0>{}) + iMWarp * WarpGemm::kM,
             a_block_window_tmp.get_window_origin().at(number<1>{})},
            make_static_tile_distribution(typename WarpGemm::AWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(a_warp_window_tmp), KIterPerWarp>,
            MIterPerWarp>
            a_warp_windows;

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                a_warp_windows(mIter)(kIter) = a_warp_window_tmp;
                move_tile_window(a_warp_windows(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        // --- Pre-construct all B warp windows ---
        //
        // Same structure as A warp windows, with N and K as the two dimensions.
        // BWarpDstrEncoding encodes thread j -> (n=j, k=0..7), symmetric to A.
        auto b_warp_window_tmp = make_tile_window(
            b_block_window_tmp.get_bottom_tensor_view(),
            make_tuple(number<WarpGemm::kN>{}, number<WarpGemm::kK>{}),
            {b_block_window_tmp.get_window_origin().at(number<0>{}) + iNWarp * WarpGemm::kN,
             b_block_window_tmp.get_window_origin().at(number<1>{})},
            make_static_tile_distribution(typename WarpGemm::BWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(b_warp_window_tmp), KIterPerWarp>,
            NIterPerWarp>
            b_warp_windows;

        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_warp_windows(nIter)(kIter) = b_warp_window_tmp;
                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        // --- Hot loop: K outer -> M middle -> N inner ---
        //
        // Loop order rationale:
        //   K is outer: each K-slice must be fully consumed before moving to the next.
        //   M is middle: loading A once per (kIter, mIter) and reusing it across all nIter.
        //   N is inner: B is loaded once per (kIter, mIter, nIter) -- never reused.
        //               This means A is reused NIterPerWarp times; B is never reused.
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // Load the A warp tile for this (kIter, mIter) from LDS into registers.
                // This tile is reused across all NIterPerWarp iterations of the inner loop.
                AWarpTensor a_warp_tensor;
                a_warp_tensor = load_tile(a_warp_windows(mIter)(kIter));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // Load the B warp tile for this (kIter, nIter) from LDS.
                    // B is loaded fresh for every (kIter, mIter, nIter) -- it is not reused.
                    BWarpTensor b_warp_tensor;
                    b_warp_tensor = load_tile(b_warp_windows(nIter)(kIter));

                    // Splice the C warp tile for (mIter, nIter) out of the C block tensor.
                    // get_y_sliced_thread_data extracts the per-thread register slice
                    // corresponding to this warp's (mIter, nIter) C sub-tile.
                    // This is a pure register-to-register operation -- no LDS access.
                    CWarpTensor c_warp_tensor;

                    c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    // Call WarpGemm: c_warp_tensor += a_warp_tensor * b_warp_tensor
                    // This compiles directly to one __builtin_amdgcn_mfma_f32_32x32x8f16
                    // instruction (or the bf16 equivalent).
                    WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                    // Write the updated C warp slice back into the C block tensor's register
                    // buffer.
                    c_block_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());
                });
            });
        });
    }

    // -----------------------------------------------------------------------
    // Init form: c = a * b  (allocates and returns a fresh C tile)
    //
    // Used when no prior accumulator exists. Constructs the C block tensor internally,
    // runs the same warp loop, and returns the result.
    template <typename ABlockWindowTmp, typename BBlockWindowTmp>
    CK_TILE_DEVICE auto operator()([[maybe_unused]] const ABlockWindowTmp& a_block_window_tmp,
                                   [[maybe_unused]] const BBlockWindowTmp& b_block_window_tmp) const
    {
        static_assert(std::is_same_v<ADataType, typename ABlockWindowTmp::DataType> &&
                          std::is_same_v<BDataType, typename BBlockWindowTmp::DataType>,
                      "wrong!");

        constexpr index_t MPerBlock = ABlockWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t KPerBlock = ABlockWindowTmp{}.get_window_lengths()[number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WarpGemm::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WarpGemm::kK;

        constexpr index_t MPerBlockPerIter = MPerBlock / MIterPerWarp;
        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        const index_t iMWarp = get_warp_id() / NWarp;
        const index_t iNWarp = get_warp_id() % NWarp;

        // Construct A warp windows (same as the accumulate form)
        auto a_warp_window_tmp = make_tile_window(
            a_block_window_tmp.get_bottom_tensor_view(),
            make_tuple(number<WarpGemm::kM>{}, number<WarpGemm::kK>{}),
            {a_block_window_tmp.get_window_origin().at(number<0>{}) + iMWarp * WarpGemm::kM,
             a_block_window_tmp.get_window_origin().at(number<1>{})},
            make_static_tile_distribution(typename WarpGemm::AWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(a_warp_window_tmp), KIterPerWarp>,
            MIterPerWarp>
            a_warp_windows;

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                a_warp_windows(mIter)(kIter) = a_warp_window_tmp;
                move_tile_window(a_warp_windows(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        // Construct B warp windows (same as the accumulate form)
        auto b_warp_window_tmp = make_tile_window(
            b_block_window_tmp.get_bottom_tensor_view(),
            make_tuple(number<WarpGemm::kN>{}, number<WarpGemm::kK>{}),
            {b_block_window_tmp.get_window_origin().at(number<0>{}) + iNWarp * WarpGemm::kN,
             b_block_window_tmp.get_window_origin().at(number<1>{})},
            make_static_tile_distribution(typename WarpGemm::BWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(b_warp_window_tmp), KIterPerWarp>,
            NIterPerWarp>
            b_warp_windows;

        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_warp_windows(nIter)(kIter) = b_warp_window_tmp;
                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        static_assert(std::is_same_v<CDataType, typename WarpGemm::CDataType>, "wrong!");

        // Construct the C block tensor distribution encoding.
        // The outer encoding places (MIterPerWarp, MWarp) and (NIterPerWarp, NWarp) as the
        // two outer dimensions, then embeds the WarpGemm's per-warp C distribution inside.
        // The result is a single tile distribution that covers the entire block tile of C,
        // with each thread owning its exact register slice.
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});

        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);

        auto c_block_tensor = make_static_distributed_tensor<CDataType>(c_block_dstr);

        // Hot loop (same structure as accumulate form)
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                AWarpTensor a_warp_tensor;
                a_warp_tensor = load_tile(a_warp_windows(mIter)(kIter));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    BWarpTensor b_warp_tensor;
                    b_warp_tensor = load_tile(b_warp_windows(nIter)(kIter));

                    CWarpTensor c_warp_tensor;

                    // Warp GEMM
                    if constexpr(KIterPerWarp == 0)
                    {
                        // First (and only) K iteration: initialize c rather than accumulate.
                        // c = a * b  (no prior value to add to)
                        c_warp_tensor = WarpGemm{}(a_warp_tensor, b_warp_tensor);
                    }
                    else
                    {
                        // Subsequent K iterations: accumulate into existing c.
                        // c += a * b
                        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
                    }

                    // Write C warp tensor into C block tensor
                    c_block_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());
                });
            });
        });

        return c_block_tensor;
    }
};

} // namespace ck_tile
