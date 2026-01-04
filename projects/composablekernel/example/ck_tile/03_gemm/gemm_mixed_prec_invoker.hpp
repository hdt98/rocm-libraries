// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "gemm_utils.hpp"

struct MixedPrecInvoker
{
    template <typename GemmConfig,
              typename ADataType,
              typename BDataType,
              typename DsDataType,
              typename AccDataType,
              typename CDataType,
              typename ALayout,
              typename BLayout,
              typename DsLayout,
              typename ELayout,
              bool Persistent,
              typename CDEElementWise,
              typename ComputeDataType = ADataType>
    static float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
    {
        static_assert(std::is_same_v<ADataType, ck_tile::fp8_t> ||
                          std::is_same_v<ADataType, ck_tile::bf8_t>,
                      "MixedPrecInvoker: ADataType must be fp8_t or bf8_t");
        static_assert(std::is_same_v<BDataType, ck_tile::pk_fp4_t>,
                      "MixedPrecInvoker: BDataType must be pk_fp4_t");

        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
            ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
            ck_tile::sequence<GemmConfig::M_Warp_Tile,
                              GemmConfig::N_Warp_Tile,
                              GemmConfig::K_Warp_Tile>>;

        using TilePartitioner =
            ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                       GemmConfig::TileParitionerGroupNum,
                                                       GemmConfig::TileParitionerM01>;

        using GemmUniversalTraits =
            ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                             GemmConfig::kPadN,
                                             GemmConfig::kPadK,
                                             GemmConfig::DoubleSmemBuffer,
                                             ALayout,
                                             BLayout,
                                             ELayout,
                                             GemmConfig::TransposeC,
                                             GemmConfig::UseStructuredSparsity,
                                             Persistent,
                                             GemmConfig::NumWaveGroups,
                                             GemmConfig::Preshuffle>;

        static_assert(sizeof(ComputeDataType) >= sizeof(BDataType),
                      "mixed_prec_flatmm requires ADataType is a wider type than BDataType");

        using UniversalGemmProblem =
            ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                  BDataType,
                                                  AccDataType,
                                                  GemmShape,
                                                  GemmUniversalTraits,
                                                  GemmConfig::Scheduler,
                                                  ck_tile::element_wise::PassThrough,
                                                  ck_tile::element_wise::PassThrough,
                                                  ComputeDataType>;

        using GemmPipeline = typename PipelineTypeTraits<
            GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

        const auto Run = [&](const auto memory_operation_) {
            constexpr auto memory_operation = memory_operation_.value;

            using GemmEpilogue = typename EpilogueTypeTraits<
                GemmConfig::Pipeline,
                ck_tile::CShuffleEpilogueProblem<
                    ADataType,
                    BDataType,
                    DsDataType,
                    AccDataType,
                    CDataType,
                    DsLayout,
                    ELayout,
                    CDEElementWise,
                    TilePartitioner::MPerBlock,
                    TilePartitioner::NPerBlock,
                    GemmConfig::M_Warp,
                    GemmConfig::N_Warp,
                    GemmConfig::M_Warp_Tile,
                    GemmConfig::N_Warp_Tile,
                    GemmConfig::K_Warp_Tile,
                    UniversalGemmProblem::TransposeC,
                    memory_operation,
                    GemmConfig::NumWaveGroups,
                    false,                           /*FixedVectorSize_*/
                    1,                               /*VectorSizeC_*/
                    false,                           /*TiledMMAPermuteN_*/
                    GemmConfig::BlockedXDLN_PerWarp, /*BlockedXDLN_PerWarp_*/
                    GemmConfig::DoubleSmemBuffer,    /*DoubleSmemBuffer*/
                    ComputeDataType>>::Epilogue;

            using GemmKernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

            const auto kargs = GemmKernel::MakeKernelArgs(args);

            const dim3 grids  = Persistent ? GemmKernel::MaxOccupancyGridSize(s)
                                           : GemmKernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = GemmKernel::BlockSize();

            if(!GemmKernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error(
                    "Mixed precision GEMM: Arguments not supported! "
                    "This may be due to hardware limitations with pk_fp4_t vector loads.");
            }

            if(s.log_level_ > 0)
            {
                std::cout << "Launching kernel with args:" << GemmKernel::GetName() << "\n"
                          << "Shape: " << GemmShape::GetName() << "\n"
                          << "problem: " << UniversalGemmProblem::GetName() << "\n"
                          << "pipeline: " << GemmPipeline::GetName() << "\n"
                          << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                          << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z
                          << "}" << std::endl;
            }

            // Declare rotating_mem_ptr here so it stays in scope until it is needed
            std::unique_ptr<ck_tile::RotatingMemWrapper<ADataType, BDataType>> rotating_mem_ptr;
            std::function<void()> preprocess;

            auto clear_gemm_output = [&]() {
                if(args.k_batch > 1)
                    hipGetErrorString(hipMemsetAsync(
                        args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_));
            };

            if(s.flush_cache_)
            {
                std::cout << "Flushing cache..." << std::endl;
                constexpr ck_tile::index_t APackedSize =
                    ck_tile::numeric_traits<ADataType>::PackedSize;
                constexpr ck_tile::index_t BPackedSize =
                    ck_tile::numeric_traits<BDataType>::PackedSize;

                ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
                    args.M, args.K, args.stride_A, is_row_major(ALayout{})));
                ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
                    args.K, args.N, args.stride_B, is_row_major(BLayout{})));

                auto size_a_buffer = a_m.get_element_space_size_in_bytes() / APackedSize;
                auto size_b_buffer = b_n.get_element_space_size_in_bytes() / BPackedSize;

                rotating_mem_ptr =
                    std::make_unique<ck_tile::RotatingMemWrapper<ADataType, BDataType>>(
                        kargs.as_ptr[0],
                        kargs.bs_ptr[0],
                        s.rotating_count_,
                        size_a_buffer,
                        size_b_buffer);
                rotating_mem_ptr->Print();

                preprocess = [&]() {
                    ck_tile::flush_icache();
                    rotating_mem_ptr->Next();
                    clear_gemm_output();
                };
                rotating_mem_ptr->Print();

                preprocess = [&]() {
                    ck_tile::flush_icache();
                    rotating_mem_ptr->Next();
                    clear_gemm_output();
                };
            }
            else
            {
                preprocess = clear_gemm_output;
            }

            return ck_tile::launch_kernel_time_mask(s,
                                                    preprocess,
                                                    ck_tile::make_kernel<GemmConfig::kBlockPerCu>(
                                                        GemmKernel{}, grids, blocks, 0, kargs));
        };

        if(args.k_batch == 1)
        {
            return Run(MemoryOpSet{});
        }
        else
        {
            return Run(MemoryOpAtomicAdd{});
        }
    }
};
