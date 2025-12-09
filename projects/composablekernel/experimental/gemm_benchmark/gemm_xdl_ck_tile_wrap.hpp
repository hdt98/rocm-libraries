// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/tensor_operation/gpu/device/device_gemm_v2.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "../../example/ck_tile/03_gemm/gemm_utils.hpp"
#include "../../example/ck_tile/03_gemm/run_gemm_example.inc"
#include "../../example/ck_tile/03_gemm/universal_gemm_invoker.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayoutCk,
          typename BLayoutCk,
          typename CLayoutCk,
          typename ADataTypeCk,
          typename BDataTypeCk,
          typename CDataTypeCk,
          typename GemmAccDataTypeCk,
          typename CShuffleDataTypeCk,
          typename AElementwiseOperationCk,
          typename BElementwiseOperationCk,
          typename CElementwiseOperationCk,
          typename GemmSpec,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t KPerXDL,
          index_t MWarp,
          index_t NWarp,
          index_t KWarp,
          index_t CShuffleNXdlPerWavePerShuffle = 1,
          typename ComputeDataTypeCk            = ADataTypeCk,
          index_t ClusterSizeM                  = 1,
          index_t ClusterSizeN                  = 1,
          ck_tile::GemmPipelineScheduler PipelineScheduler =
              ck_tile::GemmPipelineScheduler::Intrawave,
          ck_tile::GemmPipeline PipelineVer = ck_tile::GemmPipeline::COMPUTE_V3,
          index_t MinimumOccupancy          = 0>
struct DeviceGemm_Xdl_CkTileWrap : public DeviceGemmV2<ALayoutCk,
                                                       BLayoutCk,
                                                       CLayoutCk,
                                                       ADataTypeCk,
                                                       BDataTypeCk,
                                                       CDataTypeCk,
                                                       AElementwiseOperationCk,
                                                       BElementwiseOperationCk,
                                                       CElementwiseOperationCk>
{
    template <typename CkGemmLayout>
    static constexpr auto GetCkTileGemmLayout()
    {
        if constexpr(is_same_v<CkGemmLayout, ck::tensor_layout::gemm::RowMajor>)
        {
            return ck_tile::tensor_layout::gemm::RowMajor{};
        }
        else if constexpr(is_same_v<CkGemmLayout, ck::tensor_layout::gemm::ColumnMajor>)
        {
            return ck_tile::tensor_layout::gemm::ColumnMajor{};
        }
        else
        {
            static_assert(false);
        }
    }
    template <typename CkDataType>
    static constexpr auto GetCkTileDataType()
    {
        if constexpr(is_same_v<CkDataType, ck::half_t>)
        {
            return ck_tile::fp16_t{};
        }
        else if constexpr(is_same_v<CkDataType, ck::bhalf_t>)
        {
            return ck_tile::bf16_t{};
        }
        else if constexpr(is_same_v<CkDataType, ck::f8_t>)
        {
            return ck_tile::fp8_t{};
        }
        else if constexpr(is_same_v<CkDataType, ck::bf8_t>)
        {
            return ck_tile::bf8_t{};
        }
        else if constexpr(is_same_v<CkDataType, ck::pk_i4_t>)
        {
            return ck_tile::pk_int4_t{};
        }
        else if constexpr(is_same_v<CkDataType, ck::f4x2_pk_t>)
        {
            return ck_tile::pk_fp4_t{};
        }
        else
        {
            return CkDataType{};
        }
    }

    template <typename DataType>
    static constexpr auto GetPackedSize()
    {
        if constexpr(is_same_v<DataType, ck_tile::pk_int4_t> ||
                     is_same_v<DataType, ck_tile::pk_fp4_t>)
            return 2;
        else
            return 1;
    }

    template <typename CkElementwiseOperation>
    static constexpr auto GetCkTileElementwiseOperation()
    {
        if constexpr(is_same_v<CkElementwiseOperation,
                               ck::tensor_operation::element_wise::PassThrough>)
        {
            return ck_tile::element_wise::PassThrough{};
        }
        else
        {
            static_assert(0);
            return ck_tile::element_wise::PassThrough{};
        }
    }
    using ALayout          = decltype(GetCkTileGemmLayout<ALayoutCk>());
    using BLayout          = decltype(GetCkTileGemmLayout<BLayoutCk>());
    using CLayout          = decltype(GetCkTileGemmLayout<CLayoutCk>());
    using ADataType        = decltype(GetCkTileDataType<ADataTypeCk>());
    using BDataType        = decltype(GetCkTileDataType<BDataTypeCk>());
    using CDataType        = decltype(GetCkTileDataType<CDataTypeCk>());
    using GemmAccDataType  = decltype(GetCkTileDataType<GemmAccDataTypeCk>());
    using CShuffleDataType = decltype(GetCkTileDataType<CShuffleDataTypeCk>());
    using ComputeDataType  = decltype(GetCkTileDataType<ComputeDataTypeCk>());
    using AElementwiseOperation =
        decltype(GetCkTileElementwiseOperation<AElementwiseOperationCk>());
    using BElementwiseOperation =
        decltype(GetCkTileElementwiseOperation<BElementwiseOperationCk>());
    using CElementwiseOperation =
        decltype(GetCkTileElementwiseOperation<CElementwiseOperationCk>());

    struct GemmConfig
    {
        static constexpr auto I0 = Number<0>{};
        static constexpr auto I1 = Number<1>{};
        static constexpr auto I2 = Number<2>{};

        static constexpr bool kPadM = GemmSpec()[0];
        static constexpr bool kPadN = GemmSpec()[1];
        static constexpr bool kPadK = GemmSpec()[2];

        static constexpr ck_tile::index_t M_Tile = MPerBlock;
        static constexpr ck_tile::index_t N_Tile = NPerBlock;
        static constexpr ck_tile::index_t K_Tile = KPerBlock;

        static constexpr ck_tile::index_t M_Warp = MWarp;
        static constexpr ck_tile::index_t N_Warp = NWarp;
        static constexpr ck_tile::index_t K_Warp = KWarp;

        static constexpr ck_tile::index_t M_Warp_Tile = MPerXDL;
        static constexpr ck_tile::index_t N_Warp_Tile = NPerXDL;
        static constexpr ck_tile::index_t K_Warp_Tile = KPerXDL;

        static constexpr bool TransposeC =
            std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::RowMajor>;
        static constexpr bool UseStructuredSparsity = false;

        static constexpr auto Scheduler = PipelineScheduler;
        // COMPUTE_V3 is mapped to BASIC_V2 in universal_gemm_invoker.hpp
        static constexpr ck_tile::GemmPipeline Pipeline =
            (PipelineVer == ck_tile::GemmPipeline::COMPUTE_V3) ? ck_tile::GemmPipeline::BASIC_V2
                                                               : PipelineVer;
        static constexpr int kBlockPerCu =
            MinimumOccupancy
                ? MinimumOccupancy
                : (PipelineScheduler == ck_tile::GemmPipelineScheduler::Interwave ? 2 : 1);
        static constexpr ck_tile::index_t NumWaveGroups =
            Pipeline == ck_tile::GemmPipeline::COMPUTE_V5 ? 2 : 1;
        static constexpr bool DoubleSmemBuffer =
            Pipeline == ck_tile::GemmPipeline::COMPUTE_V4 ||
            Pipeline == ck_tile::GemmPipeline::COMPUTE_ASYNC ||
            Pipeline == ck_tile::GemmPipeline::COMPUTE_TDM_V1 ||
            Pipeline == ck_tile::GemmPipeline::COMPUTE_TDM_V2;

        static constexpr bool PermuteA         = false;
        static constexpr bool PermuteB         = false;
        static constexpr bool Preshuffle       = false;
        static constexpr bool TiledMMAPermuteN = false;

        static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        static constexpr ck_tile::index_t TileParitionerM01      = 4;

        static constexpr ck_tile::index_t kClusterSizeM       = ClusterSizeM;
        static constexpr ck_tile::index_t kClusterSizeN       = ClusterSizeN;
        static constexpr ck_tile::index_t BlockedXDLN_PerWarp = CShuffleNXdlPerWavePerShuffle;
    };

    template <typename DeviceArch_>
    static constexpr index_t GetEstimateVgprCount(DeviceArch_)
    {
        constexpr index_t WaveSize =
            (is_same_v<DeviceArch_, gfx950_t> || is_same_v<DeviceArch_, gfx9_t>) ? 64 : 32;
        constexpr index_t AVgprSize = MPerBlock * KPerBlock / MWarp / WaveSize * sizeof(ADataType) /
                                      GetPackedSize<ADataType>() / sizeof(uint32_t);
        constexpr index_t BVgprSize = NPerBlock * KPerBlock / NWarp / WaveSize * sizeof(BDataType) /
                                      GetPackedSize<BDataType>() / sizeof(uint32_t);
        constexpr index_t AccVgprSize = MPerBlock * NPerBlock / (MWarp * NWarp * WaveSize) *
                                        sizeof(GemmAccDataType) / sizeof(uint32_t);
        if constexpr(PipelineVer == ck_tile::GemmPipeline::BASIC_V1)
        {
            return AVgprSize + BVgprSize + AccVgprSize;
        }
        else if constexpr((PipelineVer == ck_tile::GemmPipeline::BASIC_V2) ||
                          (PipelineVer == ck_tile::GemmPipeline::COMPUTE_V3) ||
                          (PipelineVer == ck_tile::GemmPipeline::MEMORY) ||
                          (PipelineVer == ck_tile::GemmPipeline::COMPUTE_ASYNC))
        {
            return 2 * (AVgprSize + BVgprSize) + AccVgprSize;
        }
        else if constexpr(PipelineVer == ck_tile::GemmPipeline::COMPUTE_V4)
        {
            return 3 * (AVgprSize + BVgprSize) + AccVgprSize;
        }
        else if constexpr(PipelineVer == ck_tile::GemmPipeline::COMPUTE_TDM_V1 ||
                          PipelineVer == ck_tile::GemmPipeline::COMPUTE_TDM_V2)
        {
            return math::min(2 * (AVgprSize + BVgprSize), 256) + AccVgprSize;
        }
        else
        {
            // invalid pipeline version
            static_assert(0);
        }
    }

    static constexpr index_t GetEstimateSmemSize()
    {
        constexpr index_t MSize =
            MPerBlock * KPerBlock * sizeof(ComputeDataType) / GetPackedSize<ComputeDataType>();
        constexpr index_t NSize =
            NPerBlock * KPerBlock * sizeof(ComputeDataType) / GetPackedSize<ComputeDataType>();
        if constexpr(PipelineVer == ck_tile::GemmPipeline::COMPUTE_V4 ||
                     PipelineVer == ck_tile::GemmPipeline::COMPUTE_ASYNC ||
                     PipelineVer == ck_tile::GemmPipeline::COMPUTE_TDM_V1 ||
                     PipelineVer == ck_tile::GemmPipeline::COMPUTE_TDM_V2)
        {
            return 2 * (MSize + NSize);
        }
        else
        {
            return MSize + NSize;
        }
    }

#if CK_TILE_USE_WMMA
#if defined(CK_USE_GFX1250)
    using DeviceArch = gfx125_t;
#else
    using DeviceArch = gfx120_t;
#endif
#else
#if defined(CK_GFX950_SUPPORT)
    using DeviceArch = gfx950_t;
#else
    using DeviceArch = gfx9_t;
#endif
#endif

    template <typename DeviceArch_>
    static constexpr bool IsValidCompilationParameter(DeviceArch_ arch)
    {
        if constexpr(GemmConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_TDM_V2)
        {
            if constexpr(GemmConfig::M_Warp * GemmConfig::N_Warp != 4)
            {
                return false;
            }
        }
        if constexpr(GemmConfig::Pipeline == ck_tile::GemmPipeline::BASIC_V1)
        {
            if constexpr(is_same_v<ALayoutCk, ck::tensor_layout::gemm::ColumnMajor> ||
                         is_same_v<BLayoutCk, ck::tensor_layout::gemm::RowMajor>)
            {
                return false;
            }
        }
        if constexpr(GemmConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_V4 ||
                     GemmConfig::Pipeline == ck_tile::GemmPipeline::MEMORY ||
                     GemmConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_ASYNC ||
                     GemmConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_TDM_V2 ||
                     GemmConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_TDM_V1)
        {
            if constexpr(is_same_v<BDataType, ck_tile::pk_int4_t>)
            {
                return false;
            }
        }

        if constexpr(MinimumOccupancy != 0)
        {
            constexpr auto EstimateVgprCount  = GetEstimateVgprCount(arch);
            constexpr auto AvailableVgprCount = get_max_vgpr_count(arch) / MinimumOccupancy /
                                                (math::integer_divide_ceil(MWarp * NWarp, 4));
            if constexpr(EstimateVgprCount > (AvailableVgprCount + AvailableVgprCount / 4))
            {
                return false;
            }
        }

        constexpr index_t LdsSize = GetEstimateSmemSize();
        if constexpr(LdsSize > get_lds_size(arch))
        {
            return false;
        }
        return true;
    }

    struct Argument : public tensor_operation::device::BaseArgument
    {
        __host__ Argument(const ADataTypeCk* p_a_grid_,
                          const BDataTypeCk* p_b_grid_,
                          CDataTypeCk* p_c_grid_,
                          index_t M_,
                          index_t N_,
                          index_t K_,
                          index_t StrideA_,
                          index_t StrideB_,
                          index_t StrideC_,
                          index_t k_batch_)
            : host_arg(p_a_grid_,
                       p_b_grid_,
                       p_c_grid_,
                       k_batch_,
                       M_,
                       N_,
                       K_,
                       StrideA_,
                       StrideB_,
                       StrideC_)
        {
        }

        ck_tile::GemmHostArgs host_arg;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& s = StreamConfig{})
        {
            if constexpr(IsValidCompilationParameter(DeviceArch{}))
            {
                return UniversalInvoker::gemm<GemmConfig,
                                              ADataType,
                                              BDataType,
                                              ck_tile::tuple<>,
                                              GemmAccDataType,
                                              CDataType,
                                              ALayout,
                                              BLayout,
                                              ck_tile::tuple<>,
                                              CLayout,
                                              false,
                                              ck_tile::element_wise::PassThrough,
                                              ComputeDataType>(
                    arg.host_arg,
                    ck_tile::stream_config{s.stream_id_,
                                           s.time_kernel_,
                                           s.log_level_,
                                           s.cold_niters_,
                                           s.nrepeat_,
                                           true,
                                           s.flush_cache,
                                           s.rotating_count});
            }
            else
            {
                return 0;
            }
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if constexpr(IsValidCompilationParameter(DeviceArch{}))
        {
            return UniversalInvoker::gemm<GemmConfig,
                                          ADataType,
                                          BDataType,
                                          ck_tile::tuple<>,
                                          GemmAccDataType,
                                          CDataType,
                                          ALayout,
                                          BLayout,
                                          ck_tile::tuple<>,
                                          CLayout,
                                          false,
                                          ck_tile::element_wise::PassThrough,
                                          ComputeDataType>(
                       arg.host_arg, ck_tile::stream_config{}, true) != 0.0f;
        }
        else
        {
            return false;
        }
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    index_t GetKPerBlock() override { return KPerBlock; }

    bool GetPermuteA() override { return false; }
    bool GetPermuteB() override { return false; }

    static auto MakeArgument(const ADataTypeCk* p_a,
                             const BDataTypeCk* p_b,
                             CDataTypeCk* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             index_t KBatch,
                             AElementwiseOperationCk,
                             BElementwiseOperationCk,
                             CElementwiseOperationCk)
    {
        return Argument{p_a, p_b, p_c, M, N, K, StrideA, StrideB, StrideC, KBatch};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      index_t KBatch,
                                                      AElementwiseOperationCk,
                                                      BElementwiseOperationCk,
                                                      CElementwiseOperationCk) override
    {
        return std::make_unique<Argument>(static_cast<const ADataTypeCk*>(p_a),
                                          static_cast<const BDataTypeCk*>(p_b),
                                          static_cast<CDataTypeCk*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          KBatch);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        std::map<ck_tile::GemmPipelineScheduler, std::string> PipelineSchedulerToString{
            {ck_tile::GemmPipelineScheduler::Intrawave, "Intrawave"},
            {ck_tile::GemmPipelineScheduler::Interwave, "Interwave"},
            {ck_tile::GemmPipelineScheduler::Default, "Default"}};

        std::map<ck_tile::GemmPipeline, std::string> PipelineToString{
            {ck_tile::GemmPipeline::COMPUTE_ASYNC, "COMPUTE_ASYNC"},
            {ck_tile::GemmPipeline::COMPUTE_V3, "COMPUTE_V3"},
            {ck_tile::GemmPipeline::COMPUTE_V4, "COMPUTE_V4"},
            {ck_tile::GemmPipeline::COMPUTE_V5, "COMPUTE_V5"},
            {ck_tile::GemmPipeline::COMPUTE_V6, "COMPUTE_V6"},
            {ck_tile::GemmPipeline::MEMORY, "MEMORY"},
            {ck_tile::GemmPipeline::BASIC_V1, "BASIC_V1"},
            {ck_tile::GemmPipeline::BASIC_V2, "BASIC_V2"},
            {ck_tile::GemmPipeline::PRESHUFFLE_V2, "PRESHUFFLE_V2"},
            {ck_tile::GemmPipeline::COMPUTE_TDM_V1, "COMPUTE_TDM_V1"},
            {ck_tile::GemmPipeline::COMPUTE_TDM_V2, "COMPUTE_TDM_V2"}};

        auto str = std::stringstream();
        // clang-format off
        str << "DeviceGemm_Xdl_CkTileWrap"
            << "<"
            << std::string(ALayoutCk::name)[0]
            << std::string(BLayoutCk::name)[0]
            << std::string(CLayoutCk::name)[0] << ", "
            << get_type_name<ADataTypeCk>() << ", "
            << get_type_name<BDataTypeCk>() << ", "
            << get_type_name<GemmAccDataTypeCk>() << ", "
            << get_type_name<CDataTypeCk>() << ", "
            << "GemmSepc<" << GemmSpec{}[0] << ", " << GemmSpec{}[1] << ", " << GemmSpec{}[2] << ">, "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << KPerXDL << ", "
            << MWarp << ", "
            << NWarp << ", "
            << KWarp << ", "
            << PipelineSchedulerToString[PipelineScheduler]  << ", "
            << PipelineToString[PipelineVer] << ">";
        // clang-format on

        return str.str();
    }
    REGISTER_EXTRA_PRINTING_METHODS
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
