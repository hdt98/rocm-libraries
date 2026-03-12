// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_tile_partitioner.hpp"
#include "ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp"

using namespace ck_tile;

// Tile config matching TestConvConfig from the existing test
struct StreamKTestConvConfig
{
    static constexpr index_t VectorSizeA = 4;
    static constexpr index_t VectorSizeB = 8;
    static constexpr index_t VectorSizeC = 8;

    static constexpr index_t M_Tile = 128;
    static constexpr index_t N_Tile = 128;
    static constexpr index_t K_Tile = 32;

    static constexpr index_t M_Warp = 2;
    static constexpr index_t N_Warp = 2;
    static constexpr index_t K_Warp = 1;

    static constexpr index_t M_Warp_Tile = 16;
    static constexpr index_t N_Warp_Tile = 16;
    static constexpr index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer    = false;
    static constexpr GemmPipeline Pipeline    = GemmPipeline::COMPUTE_V3;
    static constexpr index_t NumWaveGroups    = 1;
    static constexpr index_t NumGroupsToMerge = 1;
    static constexpr auto Scheduler           = GemmPipelineScheduler::Intrawave;
};

// Build kernel type — parameterized by TilePartitioner
template <typename PrecType,
          typename ConvConfig,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename TilePartitioner_,
          index_t NDimSpatial = 2>
struct BuildStreamKKernel
{
    using GemmShape = TileGemmShape<
        sequence<ConvConfig::M_Tile, ConvConfig::N_Tile, ConvConfig::K_Tile>,
        sequence<ConvConfig::M_Warp, ConvConfig::N_Warp, ConvConfig::K_Warp>,
        sequence<ConvConfig::M_Warp_Tile, ConvConfig::N_Warp_Tile, ConvConfig::K_Warp_Tile>>;

    using ConvTraits = GroupedConvTraits<NDimSpatial,
                                         ConvolutionSpecialization::Default,
                                         InLayout,
                                         WeiLayout,
                                         tuple<>,
                                         OutLayout,
                                         ConvConfig::VectorSizeA,
                                         ConvConfig::VectorSizeB,
                                         ConvConfig::VectorSizeC,
                                         ConvConfig::NumGroupsToMerge>;

    using GemmUniversalTraits =
        TileGemmUniversalTraits<ConvTraits::FixedGemmParams::kPadM,
                                ConvTraits::FixedGemmParams::kPadN,
                                ConvTraits::FixedGemmParams::kPadK,
                                ConvConfig::DoubleSmemBuffer,
                                typename ConvTraits::AsLayoutBwdWeight,
                                typename ConvTraits::BsLayoutBwdWeight,
                                typename ConvTraits::CLayoutBwdWeight,
                                ConvTraits::FixedGemmParams::TransposeC,
                                ConvTraits::FixedGemmParams::UseStructuredSparsity,
                                ConvTraits::FixedGemmParams::Persistent,
                                ConvConfig::NumWaveGroups>;

    using GemmPipelineProblem_ =
        GemmPipelineProblem<PrecType,
                            PrecType,
                            float,
                            GemmShape,
                            typename ConvTraits::template GroupedConvImplicitGemmTraitsBwdWeight<
                                ConvConfig::NumWaveGroups>,
                            element_wise::PassThrough,
                            element_wise::PassThrough,
                            PrecType,
                            ConvTraits::FixedGemmParams::FixedVectorSize,
                            ConvTraits::VectorSizeA,
                            ConvTraits::VectorSizeB>;

    using UniversalGemmProblem =
        UniversalGemmPipelineProblem<PrecType,
                                     PrecType,
                                     float,
                                     GemmShape,
                                     GemmUniversalTraits,
                                     ConvConfig::Scheduler,
                                     element_wise::PassThrough,
                                     element_wise::PassThrough,
                                     PrecType,
                                     ConvTraits::FixedGemmParams::FixedVectorSize,
                                     ConvTraits::VectorSizeA,
                                     ConvTraits::VectorSizeB>;

    using GemmPipeline_ = GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

    using EpilogueProblem = CShuffleEpilogueProblem<PrecType,
                                                    PrecType,
                                                    tuple<>,
                                                    float,
                                                    PrecType,
                                                    typename ConvTraits::ImplicitGemmDsLayout,
                                                    typename ConvTraits::FixedGemmParams::ELayout,
                                                    element_wise::PassThrough,
                                                    TilePartitioner_::MPerBlock,
                                                    TilePartitioner_::NPerBlock,
                                                    ConvConfig::M_Warp,
                                                    ConvConfig::N_Warp,
                                                    ConvConfig::M_Warp_Tile,
                                                    ConvConfig::N_Warp_Tile,
                                                    ConvConfig::K_Warp_Tile,
                                                    ConvTraits::FixedGemmParams::TransposeC,
                                                    ConvConfig::NumWaveGroups,
                                                    ConvTraits::FixedGemmParams::FixedVectorSize,
                                                    ConvTraits::VectorSizeC>;

    using Epilogue = CShuffleEpilogue<EpilogueProblem>;

    using type = GroupedConvolutionBackwardWeightKernel<ConvTraits,
                                                        TilePartitioner_,
                                                        GemmPipeline_,
                                                        Epilogue>;
};

// Helper to create 2D host args
static GroupedConvBwdWeightHostArgs create_streamk_host_args(index_t G,
                                                             index_t N,
                                                             index_t K,
                                                             index_t C,
                                                             index_t Y,
                                                             index_t X,
                                                             index_t Hi,
                                                             index_t Wi,
                                                             index_t stride_y,
                                                             index_t stride_x,
                                                             index_t dilation_y,
                                                             index_t dilation_x,
                                                             index_t left_pad_y,
                                                             index_t left_pad_x,
                                                             index_t right_pad_y,
                                                             index_t right_pad_x,
                                                             index_t k_batch = 1)
{
    auto conv_param = conv::ConvParam{2,
                                      G,
                                      N,
                                      K,
                                      C,
                                      {Y, X},
                                      {Hi, Wi},
                                      {stride_y, stride_x},
                                      {dilation_y, dilation_x},
                                      {left_pad_y, left_pad_x},
                                      {right_pad_y, right_pad_x}};

    return GroupedConvBwdWeightHostArgs{conv_param, nullptr, nullptr, {}, nullptr, k_batch};
}

// --- Test: Type trait ---
TEST(StreamKConvBwdWeight, TypeTraitDetection)
{
    using GemmShape =
        TileGemmShape<sequence<128, 128, 32>, sequence<2, 2, 1>, sequence<16, 16, 16>>;

    using SplitKPartitioner = GemmSpatiallyLocalTilePartitioner<GemmShape, 8, 4>;
    using StreamKPartitioner =
        StreamKTilePartitioner<GemmShape, StreamKReductionStrategy::Linear, false>;

    EXPECT_FALSE(is_streamk_partitioner<SplitKPartitioner>::value);
    EXPECT_TRUE(is_streamk_partitioner<StreamKPartitioner>::value);
}

// --- Test: StreamK kernel args construction ---
TEST(StreamKConvBwdWeight, KernelArgsConstruction)
{
    using GemmShape =
        TileGemmShape<sequence<128, 128, 32>, sequence<2, 2, 1>, sequence<16, 16, 16>>;

    using StreamKPartitioner =
        StreamKTilePartitioner<GemmShape, StreamKReductionStrategy::Linear, false>;

    using Kernel = typename BuildStreamKKernel<half_t,
                                               StreamKTestConvConfig,
                                               tensor_layout::convolution::NHWGC,
                                               tensor_layout::convolution::GKYXC,
                                               tensor_layout::convolution::NHWGK,
                                               StreamKPartitioner>::type;

    EXPECT_TRUE(Kernel::IsStreamK);

    // Create host args for a simple 2D conv
    // G=1, N=4, K=128, C=128, 3x3 filter, 16x16 input, stride=1, pad=1
    auto host_args =
        create_streamk_host_args(1, 4, 128, 128, 3, 3, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1);

    // MakeKernelArgs with explicit num_cu and occupancy for reproducibility
    auto kargs = Kernel::MakeKernelArgs(host_args, /*num_cu=*/4, /*occupancy=*/1);

    EXPECT_EQ(kargs.k_batch, 1);
    EXPECT_GT(kargs.GemmM, 0);
    EXPECT_GT(kargs.GemmN, 0);
    EXPECT_GT(kargs.GemmK, 0);
    EXPECT_GT(kargs.tile_partitioner.get_grid(), 0);
}

// --- Test: GridSize for StreamK ---
TEST(StreamKConvBwdWeight, GridSize)
{
    using GemmShape =
        TileGemmShape<sequence<128, 128, 32>, sequence<2, 2, 1>, sequence<16, 16, 16>>;

    using StreamKPartitioner =
        StreamKTilePartitioner<GemmShape, StreamKReductionStrategy::Linear, false>;

    using Kernel = typename BuildStreamKKernel<half_t,
                                               StreamKTestConvConfig,
                                               tensor_layout::convolution::NHWGC,
                                               tensor_layout::convolution::GKYXC,
                                               tensor_layout::convolution::NHWGK,
                                               StreamKPartitioner>::type;

    auto host_args =
        create_streamk_host_args(1, 4, 128, 128, 3, 3, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1);

    auto kargs = Kernel::MakeKernelArgs(host_args, /*num_cu=*/4, /*occupancy=*/1);

    auto grid = Kernel::GridSize(kargs);

    // gridDim.x = partitioner grid (num_cu * occupancy = 4)
    // gridDim.y = GemmBatch (groups)
    // gridDim.z = 1 (StreamK doesn't use blockIdx.z for split-K)
    // grid.x = dp_ctas + sk_ctas for non-persistent partitioner
    auto sk_grid = kargs.tile_partitioner.grid_size();
    EXPECT_EQ(grid.x, sk_grid.x);
    EXPECT_EQ(grid.y, static_cast<unsigned int>(kargs.GemmBatch));
    EXPECT_EQ(grid.z, 1u);
}

// --- Test: WorkSpace size ---
TEST(StreamKConvBwdWeight, WorkSpaceSize)
{
    using GemmShape =
        TileGemmShape<sequence<128, 128, 32>, sequence<2, 2, 1>, sequence<16, 16, 16>>;

    using StreamKPartitioner =
        StreamKTilePartitioner<GemmShape, StreamKReductionStrategy::Linear, false>;

    using Kernel = typename BuildStreamKKernel<half_t,
                                               StreamKTestConvConfig,
                                               tensor_layout::convolution::NHWGC,
                                               tensor_layout::convolution::GKYXC,
                                               tensor_layout::convolution::NHWGK,
                                               StreamKPartitioner>::type;

    auto host_args =
        create_streamk_host_args(1, 4, 128, 128, 3, 3, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1);

    auto kargs = Kernel::MakeKernelArgs(host_args, /*num_cu=*/4, /*occupancy=*/1);

    auto ws_size = Kernel::GetWorkSpaceSize(kargs);
    EXPECT_GT(ws_size, 0);
}

// --- Test: SplitK kernel does not require workspace ---
TEST(StreamKConvBwdWeight, SplitKNoWorkspace)
{
    using GemmShape =
        TileGemmShape<sequence<128, 128, 32>, sequence<2, 2, 1>, sequence<16, 16, 16>>;

    using SplitKPartitioner = GemmSpatiallyLocalTilePartitioner<GemmShape, 8, 4>;

    using Kernel = typename BuildStreamKKernel<half_t,
                                               StreamKTestConvConfig,
                                               tensor_layout::convolution::NHWGC,
                                               tensor_layout::convolution::GKYXC,
                                               tensor_layout::convolution::NHWGK,
                                               SplitKPartitioner>::type;

    EXPECT_FALSE(Kernel::IsStreamK);

    auto host_args =
        create_streamk_host_args(1, 4, 128, 128, 3, 3, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1);

    auto kargs   = Kernel::MakeKernelArgs(host_args);
    auto ws_size = Kernel::GetWorkSpaceSize(kargs);
    EXPECT_EQ(ws_size, 0);
}

// --- End-to-end GPU test: StreamK vs Split-K=1 reference ---

// Helper to build a Split-K kernel (reference) with the same tile config
template <typename PrecType,
          typename ConvConfig,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          index_t NDimSpatial = 2>
struct BuildSplitKKernel
{
    using GemmShape = TileGemmShape<
        sequence<ConvConfig::M_Tile, ConvConfig::N_Tile, ConvConfig::K_Tile>,
        sequence<ConvConfig::M_Warp, ConvConfig::N_Warp, ConvConfig::K_Warp>,
        sequence<ConvConfig::M_Warp_Tile, ConvConfig::N_Warp_Tile, ConvConfig::K_Warp_Tile>>;

    using ConvTraits = GroupedConvTraits<NDimSpatial,
                                         ConvolutionSpecialization::Default,
                                         InLayout,
                                         WeiLayout,
                                         tuple<>,
                                         OutLayout,
                                         ConvConfig::VectorSizeA,
                                         ConvConfig::VectorSizeB,
                                         ConvConfig::VectorSizeC,
                                         ConvConfig::NumGroupsToMerge>;

    using TilePartitioner = GemmSpatiallyLocalTilePartitioner<GemmShape, 8, 4>;

    using GemmUniversalTraits =
        TileGemmUniversalTraits<ConvTraits::FixedGemmParams::kPadM,
                                ConvTraits::FixedGemmParams::kPadN,
                                ConvTraits::FixedGemmParams::kPadK,
                                ConvConfig::DoubleSmemBuffer,
                                typename ConvTraits::AsLayoutBwdWeight,
                                typename ConvTraits::BsLayoutBwdWeight,
                                typename ConvTraits::CLayoutBwdWeight,
                                ConvTraits::FixedGemmParams::TransposeC,
                                ConvTraits::FixedGemmParams::UseStructuredSparsity,
                                ConvTraits::FixedGemmParams::Persistent,
                                ConvConfig::NumWaveGroups>;

    using UniversalGemmProblem =
        UniversalGemmPipelineProblem<PrecType,
                                     PrecType,
                                     float,
                                     GemmShape,
                                     GemmUniversalTraits,
                                     ConvConfig::Scheduler,
                                     element_wise::PassThrough,
                                     element_wise::PassThrough,
                                     PrecType,
                                     ConvTraits::FixedGemmParams::FixedVectorSize,
                                     ConvTraits::VectorSizeA,
                                     ConvTraits::VectorSizeB>;

    using GemmPipeline_ = GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

    using EpilogueProblem = CShuffleEpilogueProblem<PrecType,
                                                    PrecType,
                                                    tuple<>,
                                                    float,
                                                    PrecType,
                                                    typename ConvTraits::ImplicitGemmDsLayout,
                                                    typename ConvTraits::FixedGemmParams::ELayout,
                                                    element_wise::PassThrough,
                                                    TilePartitioner::MPerBlock,
                                                    TilePartitioner::NPerBlock,
                                                    ConvConfig::M_Warp,
                                                    ConvConfig::N_Warp,
                                                    ConvConfig::M_Warp_Tile,
                                                    ConvConfig::N_Warp_Tile,
                                                    ConvConfig::K_Warp_Tile,
                                                    ConvTraits::FixedGemmParams::TransposeC,
                                                    ConvConfig::NumWaveGroups,
                                                    ConvTraits::FixedGemmParams::FixedVectorSize,
                                                    ConvTraits::VectorSizeC>;

    using Epilogue = CShuffleEpilogue<EpilogueProblem>;

    using type = GroupedConvolutionBackwardWeightKernel<ConvTraits,
                                                        TilePartitioner,
                                                        GemmPipeline_,
                                                        Epilogue>;
};

#include "ck_tile/host/convolution_host_tensor_descriptor_helper.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/fill.hpp"
#include "ck_tile/host/reference/reference_grouped_conv_bwd_weight.hpp"

using InLayout  = tensor_layout::convolution::NHWGC;
using WeiLayout = tensor_layout::convolution::GKYXC;
using OutLayout = tensor_layout::convolution::NHWGK;
using PrecType  = half_t;

using SplitKKernel =
    typename BuildSplitKKernel<PrecType, StreamKTestConvConfig, InLayout, WeiLayout, OutLayout>::
        type;

using StreamKGemmShape =
    TileGemmShape<sequence<128, 128, 32>, sequence<2, 2, 1>, sequence<16, 16, 16>>;
using StreamKPartitioner =
    StreamKTilePartitioner<StreamKGemmShape, StreamKReductionStrategy::Linear, false>;
using StreamKKernel_ = typename BuildStreamKKernel<PrecType,
                                                   StreamKTestConvConfig,
                                                   InLayout,
                                                   WeiLayout,
                                                   OutLayout,
                                                   StreamKPartitioner>::type;

// Run a single conv bwd weight problem with both Split-K=1 and StreamK,
// and compare outputs.
static bool run_streamk_vs_splitk_test(index_t G,
                                       index_t N,
                                       index_t K,
                                       index_t C,
                                       index_t Y,
                                       index_t X,
                                       index_t Hi,
                                       index_t Wi,
                                       index_t num_cu,
                                       index_t occupancy)
{
    constexpr index_t NDimSpatial = 2;

    auto conv_param = conv::ConvParam{NDimSpatial,
                                      G,
                                      N,
                                      K,
                                      C,
                                      {Y, X},
                                      {Hi, Wi},
                                      {1, 1},  // strides
                                      {1, 1},  // dilations
                                      {1, 1},  // left pads
                                      {1, 1}}; // right pads

    // Create host tensor descriptors
    const auto in_desc =
        conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
    const auto wei_desc =
        conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);
    const auto out_desc =
        conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    // Host tensors
    HostTensor<PrecType> input(in_desc);
    HostTensor<PrecType> output(out_desc);
    HostTensor<PrecType> weight_ref(wei_desc);
    HostTensor<PrecType> weight_streamk(wei_desc);

    // Fill with small integers for reproducibility
    FillUniformDistribution<PrecType>{-1.f, 1.f}(input);
    FillUniformDistribution<PrecType>{-1.f, 1.f}(output);

    // Device buffers
    DeviceMem input_dev(input.get_element_space_size_in_bytes());
    DeviceMem output_dev(output.get_element_space_size_in_bytes());
    DeviceMem weight_ref_dev(weight_ref.get_element_space_size_in_bytes());
    DeviceMem weight_streamk_dev(weight_streamk.get_element_space_size_in_bytes());

    input_dev.ToDevice(input.data());
    output_dev.ToDevice(output.data());

    // ========= Reference: Split-K=1 =========
    {
        weight_ref_dev.SetZero();

        GroupedConvBwdWeightHostArgs host_args(conv_param,
                                               input_dev.GetDeviceBuffer(),
                                               weight_ref_dev.GetDeviceBuffer(),
                                               {},
                                               output_dev.GetDeviceBuffer(),
                                               /*k_batch=*/1);

        auto kargs = SplitKKernel::MakeKernelArgs(host_args);
        if(!SplitKKernel::IsSupportedArgument(kargs))
        {
            std::cout << "Split-K kernel does not support this shape, skipping\n";
            return true;
        }

        const dim3 grids  = SplitKKernel::GridSize(kargs);
        const dim3 blocks = SplitKKernel::BlockSize();

        auto kernel_func = make_kernel<1>(SplitKKernel{}, grids, blocks, 0, kargs);
        launch_kernel(stream_config{nullptr, false}, kernel_func);
        hip_check_error(hipDeviceSynchronize());
    }

    // ========= StreamK =========
    {
        weight_streamk_dev.SetZero();

        GroupedConvBwdWeightHostArgs host_args(conv_param,
                                               input_dev.GetDeviceBuffer(),
                                               weight_streamk_dev.GetDeviceBuffer(),
                                               {},
                                               output_dev.GetDeviceBuffer(),
                                               /*k_batch=*/1);

        auto kargs = StreamKKernel_::MakeKernelArgs(host_args, num_cu, occupancy);

        // Allocate and set workspace
        auto ws_size = StreamKKernel_::GetWorkSpaceSize(kargs);
        DeviceMem workspace_dev(ws_size);
        workspace_dev.SetZero();
        StreamKKernel_::SetWorkSpacePointer(kargs, workspace_dev.GetDeviceBuffer());

        const dim3 grids  = StreamKKernel_::GridSize(kargs);
        const dim3 blocks = StreamKKernel_::BlockSize();

        auto kernel_func = make_kernel<1>(StreamKKernel_{}, grids, blocks, 0, kargs);
        launch_kernel(stream_config{nullptr, false}, kernel_func);
        hip_check_error(hipDeviceSynchronize());
    }

    // Copy results back
    weight_ref_dev.FromDevice(weight_ref.data());
    weight_streamk_dev.FromDevice(weight_streamk.data());

    // Compare
    return check_err(weight_streamk, weight_ref, "StreamK vs SplitK mismatch", 1e-3, 1e-3);
}

// Small shape: should fit in 1 output tile with M=128,N=128
// G=1, N=4, K=128, C=128, 1x1 filter (stride=1,pad=1 but filter=1 means no spatial mixing)
// This should produce GemmM=K=128, GemmN=C=128 — exactly 1 output tile
TEST(StreamKConvBwdWeight, EndToEnd_SmallShape)
{
    // G=1, N=4, K=128, C=128, 3x3, 16x16 input
    // GemmM = K = 128, GemmN = C*Y*X = 128*3*3 = 1152
    // Multiple output tiles; enough K for StreamK to split
    bool pass = run_streamk_vs_splitk_test(
        /*G=*/1,
        /*N=*/4,
        /*K=*/128,
        /*C=*/128,
        /*Y=*/3,
        /*X=*/3,
        /*Hi=*/16,
        /*Wi=*/16,
        /*num_cu=*/2,
        /*occupancy=*/1);
    EXPECT_TRUE(pass);
}

// Medium shape
TEST(StreamKConvBwdWeight, EndToEnd_MediumShape)
{
    // G=1, N=8, K=256, C=128, 3x3, 16x16 input
    bool pass = run_streamk_vs_splitk_test(
        /*G=*/1,
        /*N=*/8,
        /*K=*/256,
        /*C=*/128,
        /*Y=*/3,
        /*X=*/3,
        /*Hi=*/16,
        /*Wi=*/16,
        /*num_cu=*/4,
        /*occupancy=*/1);
    EXPECT_TRUE(pass);
}

// Single-group, more SK work (higher grid relative to tiles)
TEST(StreamKConvBwdWeight, EndToEnd_MoreSKWork)
{
    // G=1, N=4, K=128, C=128, 3x3, 16x16, grid=4
    // This gives 9 tiles, 4 CUs → more SK work
    bool pass = run_streamk_vs_splitk_test(
        /*G=*/1,
        /*N=*/4,
        /*K=*/128,
        /*C=*/128,
        /*Y=*/3,
        /*X=*/3,
        /*Hi=*/16,
        /*Wi=*/16,
        /*num_cu=*/4,
        /*occupancy=*/1);
    EXPECT_TRUE(pass);
}

// Multi-group
TEST(StreamKConvBwdWeight, EndToEnd_MultiGroup)
{
    // G=2, N=4, K=128, C=128, 3x3, 16x16 input
    bool pass = run_streamk_vs_splitk_test(
        /*G=*/2,
        /*N=*/4,
        /*K=*/128,
        /*C=*/128,
        /*Y=*/3,
        /*X=*/3,
        /*Hi=*/16,
        /*Wi=*/16,
        /*num_cu=*/4,
        /*occupancy=*/1);
    EXPECT_TRUE(pass);
}
