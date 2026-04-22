// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp"
#include "ck_tile/ops/gemm_mx/kernel/scale_pointer.hpp"

namespace ck_tile {

template <typename ScaleM    = MXScalePointer<e8m0_t, -1>,
          typename ScaleN    = MXScalePointer<e8m0_t, -1>,
          index_t NumATensor = 1,
          index_t NumBTensor = 1,
          index_t NumDTensor = 0>
struct MXGemmKernelArgs : UniversalGemmKernelArgs<NumATensor, NumBTensor, NumDTensor>
{
    using Base = UniversalGemmKernelArgs<NumATensor, NumBTensor, NumDTensor>;

    CK_TILE_HOST MXGemmKernelArgs(const std::array<const void*, NumATensor>& as_ptr_,
                                  const std::array<const void*, NumBTensor>& bs_ptr_,
                                  const std::array<const void*, NumDTensor>& ds_ptr_,
                                  void* e_ptr_,
                                  index_t k_batch_,
                                  index_t M_,
                                  index_t N_,
                                  index_t K_,
                                  const std::array<index_t, NumATensor>& stride_As_,
                                  const std::array<index_t, NumBTensor>& stride_Bs_,
                                  const std::array<index_t, NumDTensor>& stride_Ds_,
                                  index_t stride_E_,
                                  ScaleM scale_m_ptr_,
                                  ScaleN scale_n_ptr_)
        : Base{as_ptr_,
               bs_ptr_,
               ds_ptr_,
               e_ptr_,
               M_,
               N_,
               K_,
               stride_As_,
               stride_Bs_,
               stride_Ds_,
               stride_E_,
               k_batch_},
          scale_m_ptr(scale_m_ptr_),
          scale_n_ptr(scale_n_ptr_)
    {
    }

    ScaleM scale_m_ptr;
    ScaleN scale_n_ptr;
};

template <typename TilePartitioner_, typename MXGemmPipeline_, typename EpiloguePipeline_>
struct MXGemmKernel : UniversalGemmKernel<TilePartitioner_, MXGemmPipeline_, EpiloguePipeline_>
{
    using Underlying = UniversalGemmKernel<TilePartitioner_, MXGemmPipeline_, EpiloguePipeline_>;

    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using MXGemmPipeline   = remove_cvref_t<MXGemmPipeline_>;
    using BlockGemmShape   = remove_cvref_t<typename MXGemmPipeline::BlockGemmShape>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using ALayout          = remove_cvref_t<typename MXGemmPipeline::ALayout>;
    using BLayout          = remove_cvref_t<typename MXGemmPipeline::BLayout>;
    using ELayout          = remove_cvref_t<typename MXGemmPipeline::CLayout>;
    using DsLayout         = remove_cvref_t<typename EpiloguePipeline::DsLayout>;
    using DsDataType       = remove_cvref_t<typename EpiloguePipeline::DsDataType>;
    static constexpr index_t KernelBlockSize  = MXGemmPipeline::BlockSize;
    static constexpr bool UsePersistentKernel = MXGemmPipeline::UsePersistentKernel;

    // Below type is actually accumulation data type - the output of block GEMM.
    using EDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();
    static constexpr auto I4 = number<4>();
    static constexpr auto I5 = number<5>();

    static constexpr index_t NumATensor = Underlying::AsDataType::size();
    static constexpr index_t NumBTensor = Underlying::BsDataType::size();
    static constexpr index_t NumDTensor = Underlying::DsDataType::size();

    using ADataType = remove_cvref_t<std::tuple_element_t<I0, typename Underlying::AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<I0, typename Underlying::BsDataType>>;

    static constexpr auto MThreadPerXdl = BlockGemmShape::WarpTile::at(number<0>{});
    static constexpr auto NThreadPerXdl = BlockGemmShape::WarpTile::at(number<1>{});
    static constexpr auto KThreadPerXdl = 64 / MThreadPerXdl;

    static constexpr auto APackedSize = numeric_traits<ADataType>::PackedSize;
    static constexpr auto BPackedSize = numeric_traits<BDataType>::PackedSize;

    // XdlPack: desired packing of e8m0_t scale values into int32_t
    static constexpr index_t MXdlPack = 2;
    static constexpr index_t NXdlPack = 2;
    static constexpr index_t KXdlPack = 2;

    // Effective pack sizes: fall back to 1 when dimension is too small
    using BlockWarps_                      = typename BlockGemmShape::BlockWarps;
    static constexpr index_t MPerBlock_    = BlockGemmShape::kM;
    static constexpr index_t NPerBlock_    = BlockGemmShape::kN;
    static constexpr index_t KPerBlock_    = BlockGemmShape::kK;
    static constexpr index_t MWarp_        = BlockWarps_::at(number<0>{});
    static constexpr index_t NWarp_        = BlockWarps_::at(number<1>{});
    static constexpr index_t KPerXdl_      = BlockGemmShape::WarpTile::at(number<2>{});
    static constexpr index_t MIterPerWarp_ = MPerBlock_ / (MWarp_ * MThreadPerXdl);
    static constexpr index_t NIterPerWarp_ = NPerBlock_ / (NWarp_ * NThreadPerXdl);
    static constexpr index_t KIterPerWarp_ = KPerBlock_ / KPerXdl_;

    static constexpr index_t MXdlPackEff =
        (MIterPerWarp_ >= MXdlPack && MIterPerWarp_ % MXdlPack == 0) ? MXdlPack : 1;
    static constexpr index_t NXdlPackEff =
        (NIterPerWarp_ >= NXdlPack && NIterPerWarp_ % NXdlPack == 0) ? NXdlPack : 1;
    static constexpr index_t KXdlPackEff =
        (KIterPerWarp_ >= KXdlPack && KIterPerWarp_ % KXdlPack == 0) ? KXdlPack : 1;

    static constexpr int kBlockPerCu = 1;

    // Scale block size (same constant used by MXGemmPipeline): each e8m0 scale covers 32 K elements
    static constexpr index_t ScaleBlockSize = 32;

    // Padding flags pulled from pipeline so the kernel can pad the (unscaled) C and scale views
    // consistently with the A/B views that the pipeline already pads via
    // Underlying::MakeA/BBlockWindows.
    static constexpr bool kPadM = MXGemmPipeline::kPadM;
    static constexpr bool kPadN = MXGemmPipeline::kPadN;
    static constexpr bool kPadK = MXGemmPipeline::kPadK;

    static_assert(DsLayout::size() == DsDataType::size(),
                  "The size of DsLayout and DsDataType should be the same");

    // ------------------------------------------------------------------
    // Compile-time padding-support invariants for the MX comp-async pipeline.
    //
    //   - K padding is NOT supported: async_load_tile issues vector buffer reads whose
    //     OOB check is per-vector-start, so a vector that straddles the K pad boundary
    //     pulls in data from the adjacent row / next K tile rather than zero. The packed
    //     scale tile has the same vector-load property. Until the async path learns how
    //     to do per-element pad masking, we forbid kPadK at compile time.
    //
    //   - kPadM / kPadN are supported only when the GEMM has at least one full block
    //     along that dimension; the CShuffleEpilogue's LDS shuffle uses thread positions
    //     that do not all participate when the entire dimension is smaller than a tile
    //     (resulting in zeros being written into in-range output rows). The "entire
    //     dimension < tile" case is rejected at runtime in IsSupportedArgument; we
    //     cannot statically catch it because M and N are runtime values.
    // ------------------------------------------------------------------
    static_assert(!kPadK,
                  "MX GEMM (comp-async pipeline): K padding (kPadK = true) is not supported. "
                  "The async vector loads do not mask elements that straddle the K pad "
                  "boundary, so partial K tiles produce silently wrong results. Choose K so "
                  "that K is a multiple of KPerBlock * k_batch.");

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "mx_gemm", gemm_prec_str<ADataType, BDataType>, MXGemmPipeline::GetName());
        // clang-format on
    }

    template <typename ScaleM, typename ScaleN>
    using KernelArgs = MXGemmKernelArgs<ScaleM, ScaleN, NumATensor, NumBTensor, NumDTensor>;

    template <typename ScaleM, typename ScaleN>
    CK_TILE_HOST static auto MakeKernelArgs(const std::array<const void*, NumATensor>& as_ptr,
                                            const std::array<const void*, NumBTensor>& bs_ptr,
                                            const std::array<const void*, NumDTensor>& ds_ptr,
                                            void* e_ptr,
                                            index_t k_batch,
                                            index_t M,
                                            index_t N,
                                            index_t K,
                                            const std::array<index_t, NumATensor>& stride_As,
                                            const std::array<index_t, NumBTensor>& stride_Bs,
                                            const std::array<index_t, NumDTensor>& stride_Ds,
                                            index_t stride_E,
                                            ScaleM scale_m_ptr,
                                            ScaleN scale_n_ptr)
    {
        return KernelArgs<ScaleM, ScaleN>(as_ptr,
                                          bs_ptr,
                                          ds_ptr,
                                          e_ptr,
                                          k_batch,
                                          M,
                                          N,
                                          K,
                                          stride_As,
                                          stride_Bs,
                                          stride_Ds,
                                          stride_E,
                                          scale_m_ptr,
                                          scale_n_ptr);
    }

    template <class ScaleM, class ScaleN>
    CK_TILE_HOST static constexpr auto GridSize(const KernelArgs<ScaleM, ScaleN>& kargs)
    {
        const int total_work_tile_cnt = TilePartitioner::GridSize(kargs.M, kargs.N);

        if constexpr(UsePersistentKernel)
        {
            hipDeviceProp_t prop;
            int deviceId = 0; // default device

            int dync_smem_size       = 0;
            int maxActiveBlocksPerCU = 0;

            if(hipGetDeviceProperties(&prop, deviceId) != hipSuccess)
                throw std::runtime_error(std::string("hipGetDeviceProperties failed: ") +
                                         hipGetErrorName(hipGetLastError()));

            if(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                   &maxActiveBlocksPerCU,
                   reinterpret_cast<void*>(
                       kentry<1, MXGemmKernel, remove_cvref_t<decltype(kargs)>>),
                   KernelBlockSize,
                   dync_smem_size) != hipSuccess)
                throw std::runtime_error(
                    std::string("hipOccupancyMaxActiveBlocksPerMultiprocessor failed: ") +
                    hipGetErrorName(hipGetLastError()));

            const int persistent_block_size = prop.multiProcessorCount * maxActiveBlocksPerCU;
            const int actual_grid_size      = min(persistent_block_size, total_work_tile_cnt);

            // blockIdx.z selects the K split. For split-K, each k_id gets its own set of
            // persistent blocks looping over the MxN tile space.
            return dim3(actual_grid_size, 1, kargs.k_batch);
        }
        else
        {
            // Non-persistent: grid is (MxN tiles) x 1 x k_batch. blockIdx.z selects the K split.
            return dim3(total_work_tile_cnt, 1, kargs.k_batch);
        }
    }

    template <class ScaleM, class ScaleN>
    CK_TILE_HOST static bool IsSupportedArgument(const KernelArgs<ScaleM, ScaleN>& kargs)
    {
        // Reject unsupported combinations early; the MX pipeline silently produces wrong
        // results otherwise (OOB reads, partial-tile shuffle artifacts, mis-aligned splits).
        // See the static_assert block at the top of MXGemmKernel for the rationale behind
        // each constraint.
        const bool log = ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING));

        if(kargs.k_batch < 1)
        {
            if(log)
                CK_TILE_ERROR("MX GEMM: k_batch must be >= 1.");
            return false;
        }

        // M / N must be a multiple of the block tile when padding is disabled.
        if(!kPadM && (kargs.M % TilePartitioner::MPerBlock != 0))
        {
            if(log)
                CK_TILE_ERROR("MX GEMM: M must be a multiple of MPerBlock when kPadM is false. "
                              "Enable kPadM on the GEMM config to run this shape.");
            return false;
        }
        if(!kPadN && (kargs.N % TilePartitioner::NPerBlock != 0))
        {
            if(log)
                CK_TILE_ERROR("MX GEMM: N must be a multiple of NPerBlock when kPadN is false. "
                              "Enable kPadN on the GEMM config to run this shape.");
            return false;
        }

        // CShuffleEpilogue cannot run with a single partial tile along M or N: the shuffle's
        // LDS write/read pattern leaves some in-range output rows/cols at zero. Reject these
        // pathological shapes whether or not kPadM/kPadN is enabled.
        if(kargs.M < TilePartitioner::MPerBlock)
        {
            if(log)
                CK_TILE_ERROR("MX GEMM: M must be >= MPerBlock. Partial-only M tiles are not "
                              "supported by the MX CShuffleEpilogue.");
            return false;
        }
        if(kargs.N < TilePartitioner::NPerBlock)
        {
            if(log)
                CK_TILE_ERROR("MX GEMM: N must be >= NPerBlock. Partial-only N tiles are not "
                              "supported by the MX CShuffleEpilogue.");
            return false;
        }

        // K padding is unconditionally rejected (kPadK is also a compile-time error -- see the
        // static_assert at the top of MXGemmKernel). Every split must consume an exact number
        // of K tiles, otherwise the async vector loads read garbage past the K boundary.
        const index_t k_tile = TilePartitioner::KPerBlock;
        if(kargs.K % (k_tile * kargs.k_batch) != 0)
        {
            if(log)
                CK_TILE_ERROR(
                    "MX GEMM: K must be a multiple of KPerBlock * k_batch. The MX comp-async "
                    "pipeline does not currently support K padding (vector loads across the K "
                    "pad boundary read garbage); pick aligned K dimensions or change k_batch.");
            return false;
        }

        // Scales are granular in K: each packed int32_t covers ScaleBlockSize * KXdlPackEff
        // consecutive K elements. Every split-K boundary must land on that granularity so that
        // each split can compute a packed-scale K offset. K1 is the WarpTile K, which is a
        // multiple of that granularity for all shipped configs, but be defensive.
        constexpr index_t scale_granularity_k = ScaleBlockSize * KXdlPackEff;
        if(kargs.k_batch > 1)
        {
            // splitk_batch_offset allocates K in units of K1 (warp-tile K). If K1 itself is
            // not a multiple of the scale granularity, split-K is not safe.
            constexpr index_t K1 = BlockGemmShape::WarpTile::at(number<2>{});
            static_assert(K1 % scale_granularity_k == 0,
                          "MX GEMM: WarpTile K must be a multiple of ScaleBlockSize * KXdlPack "
                          "to support split-K.");
            // Defensive runtime check: K must split evenly along K1 boundaries so that each
            // k_id consumes a whole number of warp-tile K chunks (and therefore a whole
            // number of packed-scale K elements).
            if(kargs.K % (K1 * kargs.k_batch) != 0)
            {
                if(log)
                    CK_TILE_ERROR("MX GEMM: with k_batch > 1, K must be a multiple of WarpTile_K * "
                                  "k_batch so that every split lands on a packed-scale boundary.");
                return false;
            }
        }

        return Underlying::IsSupportedArgument(
            static_cast<const typename Underlying::KernelArgs&>(kargs));
    }

    using SplitKBatchOffset = typename Underlying::SplitKBatchOffset;

    // Create C block window following UniversalGemmKernel pattern
    template <memory_operation_enum DstInMemOp = memory_operation_enum::set,
              typename ScaleM,
              typename ScaleN>
    CK_TILE_DEVICE static auto MakeCBlockWindows(EDataType* e_ptr,
                                                 const KernelArgs<ScaleM, ScaleN>& kargs,
                                                 const index_t i_m,
                                                 const index_t i_n)
    {
        // Create tensor view for E/C tensor
        constexpr index_t vector_size = EpiloguePipeline::GetVectorSizeC();
        const auto& e_tensor_view     = [&]() -> auto {
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(kargs.stride_E, 1),
                    number<vector_size>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(1, kargs.stride_E),
                    number<1>{},
                    number<vector_size>{});
            }
        }();

        // Pad both dims so OOB C writes (including partial trailing tiles where M < MPerBlock
        // or N < NPerBlock) are masked by the pad transform.
        const auto& e_pad_view = pad_tensor_view(
            e_tensor_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            sequence<kPadM, kPadN>{});

        // Create block window
        auto c_block_window = make_tile_window(
            e_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_m, i_n});

        return c_block_window;
    }

    // Create scale A block windows with packed int32_t layout.
    // Host packs (MXdlPack x KXdlPack) e8m0_t values into a single int32_t, producing a
    // packed tensor of shape [M/MXdlPackEff, K/ScaleBlockSize/KXdlPackEff].
    //
    //   k_elem_offset: starting K element index for this block (0 unless split-K).
    //                  Must be a multiple of ScaleBlockSize * KXdlPackEff.
    template <typename ScaleM, typename ScaleN>
    CK_TILE_DEVICE static auto MakeScaleABlockWindows(const KernelArgs<ScaleM, ScaleN>& kargs,
                                                      const index_t i_m,
                                                      const index_t k_elem_offset = 0)
    {
        auto scale_a = kargs.scale_m_ptr;

        static constexpr int BlockScaleSize = ScaleM::GranularityK;
        const auto scale_k_packed           = kargs.K / BlockScaleSize / KXdlPackEff;
        const auto scale_m_packed           = kargs.M / MXdlPackEff;

        const auto scale_a_tensor_view = make_naive_tensor_view<address_space_enum::global>(
            reinterpret_cast<const int32_t*>(scale_a.ptr),
            make_tuple(scale_m_packed, scale_k_packed),
            make_tuple(scale_k_packed, 1));

        // Pad the scale view so that partial trailing tiles along M and K are handled safely
        // (OOB scale loads return zero; with A/B also zero on the padded region the contribution
        // is zero regardless of scale value). Mirrors the A/B view padding done by the pipeline.
        const auto scale_a_pad_view = pad_tensor_view(
            scale_a_tensor_view,
            make_tuple(number<TilePartitioner::MPerBlock / MXdlPackEff>{},
                       number<TilePartitioner::KPerBlock / BlockScaleSize / KXdlPackEff>{}),
            sequence<kPadM, kPadK>{});

        const index_t k_scale_offset = k_elem_offset / BlockScaleSize / KXdlPackEff;

        auto scale_a_block_window = make_tile_window(
            scale_a_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock / MXdlPackEff>{},
                       number<TilePartitioner::KPerBlock / BlockScaleSize / KXdlPackEff>{}),
            {i_m / MXdlPackEff, k_scale_offset});

        return scale_a_block_window;
    }

    template <typename ScaleM, typename ScaleN>
    CK_TILE_DEVICE static auto MakeScaleBBlockWindows(const KernelArgs<ScaleM, ScaleN>& kargs,
                                                      const index_t i_n,
                                                      const index_t k_elem_offset = 0)
    {
        auto scale_b = kargs.scale_n_ptr;

        static constexpr int BlockScaleSize = ScaleN::GranularityK;
        const auto scale_k_packed           = kargs.K / BlockScaleSize / KXdlPackEff;
        const auto scale_n_packed           = kargs.N / NXdlPackEff;

        const auto scale_b_tensor_view = make_naive_tensor_view<address_space_enum::global>(
            reinterpret_cast<const int32_t*>(scale_b.ptr),
            make_tuple(scale_n_packed, scale_k_packed),
            make_tuple(scale_k_packed, 1));

        const auto scale_b_pad_view = pad_tensor_view(
            scale_b_tensor_view,
            make_tuple(number<TilePartitioner::NPerBlock / NXdlPackEff>{},
                       number<TilePartitioner::KPerBlock / BlockScaleSize / KXdlPackEff>{}),
            sequence<kPadN, kPadK>{});

        const index_t k_scale_offset = k_elem_offset / BlockScaleSize / KXdlPackEff;

        auto scale_b_block_window = make_tile_window(
            scale_b_pad_view,
            make_tuple(number<TilePartitioner::NPerBlock / NXdlPackEff>{},
                       number<TilePartitioner::KPerBlock / BlockScaleSize / KXdlPackEff>{}),
            {i_n / NXdlPackEff, k_scale_offset});

        return scale_b_block_window;
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set,
              class ScaleM,
              class ScaleN>
    CK_TILE_DEVICE static void RunMxGemm(const std::array<const ADataType*, NumATensor>& as_ptr,
                                         const std::array<const BDataType*, NumBTensor>& bs_ptr,
                                         const std::array<const void*, NumDTensor>& ds_ptr,
                                         EDataType* e_ptr,
                                         void* smem_ptr_ping,
                                         void* smem_ptr_pong,
                                         const KernelArgs<ScaleM, ScaleN>& kargs,
                                         const SplitKBatchOffset& splitk_batch_offset,
                                         const index_t i_m,
                                         const index_t i_n,
                                         const index_t k_elem_offset = 0)
    {
        // Create block windows directly, following the new pattern from UniversalGemmKernel
        // i_m and i_n are element offsets (iM * MPerBlock, iN * NPerBlock), not tile indices
        const auto& a_block_window =
            Underlying::MakeABlockWindows(as_ptr, kargs, splitk_batch_offset.splitted_k, i_m);
        const auto& b_block_window =
            Underlying::MakeBBlockWindows(bs_ptr, kargs, splitk_batch_offset.splitted_k, i_n);
        const auto& d_block_window = Underlying::MakeDBlockWindows(ds_ptr, kargs, i_m, i_n);

        // Create scale block windows. For split-K (k_batch > 1), k_elem_offset advances the
        // scale origin into the correct packed-K slice for this k_id; otherwise it is zero.
        const auto& scale_a_block_window = MakeScaleABlockWindows(kargs, i_m, k_elem_offset);
        const auto& scale_b_block_window = MakeScaleBBlockWindows(kargs, i_n, k_elem_offset);

        const index_t num_loop = TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k);

        static_assert(ScaleM::GranularityK == ScaleN::GranularityK // have the same granK
                          || ScaleM::GranularityMN == -1           // or ScaleA is disable
                          || ScaleN::GranularityMN == -1,          // or ScaleB is disable
                      "ScaleM and ScaleN should have the same GranularityK");

        const auto& c_block_tile = MXGemmPipeline{}(a_block_window[number<0>{}],
                                                    b_block_window[number<0>{}],
                                                    scale_a_block_window,
                                                    scale_b_block_window,
                                                    num_loop,
                                                    smem_ptr_ping,
                                                    smem_ptr_pong);

        // Run Epilogue Pipeline - create C block window with the requested memory op.
        auto c_block_window = MakeCBlockWindows<DstInMemOp>(e_ptr, kargs, i_m, i_n);
        EpiloguePipeline{}(c_block_window, c_block_tile, d_block_window, smem_ptr_ping);
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPingSize()
    {
        return max(MXGemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPongSize()
    {
        return MXGemmPipeline::GetSmemSize();
    }

    // Compute the K-element offset for a given split-K batch id. Matches the formula used by
    // Underlying::SplitKBatchOffset so that scale windows stay in lock-step with A/B windows.
    CK_TILE_DEVICE static index_t GetSplitKElemOffset(index_t K, index_t k_batch, index_t k_id)
    {
        constexpr auto K1       = BlockGemmShape::WarpTile::at(number<2>{});
        const index_t num_all   = K / K1;
        index_t num_full        = num_all % k_batch;
        num_full                = (num_full == 0) ? k_batch : num_full;
        const index_t num_iters = max(integer_divide_ceil(num_all, k_batch), 1);
        const index_t full_k    = num_iters * K1;
        const index_t partial_k = (num_iters - 1) * K1;
        return min(k_id, num_full) * full_k + max(k_id - num_full, 0) * partial_k;
    }

    template <class ScaleM, class ScaleN>
    CK_TILE_DEVICE void operator()(KernelArgs<ScaleM, ScaleN> kargs,
                                   int partition_idx = get_block_id()) const
    {
        const int total_work_tile_cnt =
            amd_wave_read_first_lane(TilePartitioner::GridSize(kargs.M, kargs.N));

        // Allocate shared memory for ping pong buffers
        __shared__ char smem_ptr_ping[GetSmemPingSize()];
        __shared__ char smem_ptr_pong[GetSmemPongSize()];

        // k_id selects the split-K batch id. blockIdx.z is 0 only when k_batch == 1; when
        // k_batch > 1, blockIdx.z selects the active split-K batch in both persistent and
        // non-persistent modes.
        const index_t k_id = amd_wave_read_first_lane(blockIdx.z);

        // Support both persistent and non-persistent modes
        do
        {
            const auto [iM, iN] =
                TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(partition_idx);
            const index_t i_m = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
            const index_t i_n = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);

            // SplitKBatchOffset defaults k_id to blockIdx.z, which is what we want here.
            const SplitKBatchOffset splitk_batch_offset(
                static_cast<const typename Underlying::KernelArgs&>(kargs));

            const index_t k_elem_offset =
                amd_wave_read_first_lane(GetSplitKElemOffset(kargs.K, kargs.k_batch, k_id));

            EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);

            std::array<const ADataType*, NumATensor> as_ptr;
            static_for<0, NumATensor, 1>{}([&](auto i) {
                as_ptr[i] = static_cast<const ADataType*>(kargs.as_ptr[i]) +
                            splitk_batch_offset.as_k_split_offset[i] / APackedSize;
            });

            std::array<const BDataType*, NumBTensor> bs_ptr;
            static_for<0, NumBTensor, 1>{}([&](auto i) {
                bs_ptr[i] = static_cast<const BDataType*>(kargs.bs_ptr[i]) +
                            splitk_batch_offset.bs_k_split_offset[i] / BPackedSize;
            });

            // Dispatch epilogue: when k_batch > 1 each split accumulates a partial result into
            // the same C tile, so we need atomic add (universal_gemm_kernel pattern). For atomic
            // add of fp16/bf16 the epilogue requires even vector size -- guard against invalid
            // combinations the same way UniversalGemmKernel does.
            if(kargs.k_batch == 1)
            {
                RunMxGemm<memory_operation_enum::set>(as_ptr,
                                                      bs_ptr,
                                                      kargs.ds_ptr,
                                                      e_ptr,
                                                      smem_ptr_ping,
                                                      smem_ptr_pong,
                                                      kargs,
                                                      splitk_batch_offset,
                                                      i_m,
                                                      i_n,
                                                      /*k_elem_offset=*/0);
            }
            else
            {
                if constexpr(EpiloguePipeline::GetVectorSizeC() % 2 == 0 ||
                             !is_any_of<EDataType, fp16_t, bf16_t>::value)
                {
                    RunMxGemm<memory_operation_enum::atomic_add>(as_ptr,
                                                                 bs_ptr,
                                                                 kargs.ds_ptr,
                                                                 e_ptr,
                                                                 smem_ptr_ping,
                                                                 smem_ptr_pong,
                                                                 kargs,
                                                                 splitk_batch_offset,
                                                                 i_m,
                                                                 i_n,
                                                                 k_elem_offset);
                }
            }
            partition_idx += gridDim.x;
        } while(UsePersistentKernel && partition_idx < total_work_tile_cnt);
    }
};

} // namespace ck_tile
