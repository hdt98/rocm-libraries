// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_tile_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"
#include "ck_tile/builder/testing/conv/fwd.hpp"
#include "ck_tile/builder/testing/conv/ck_tile.hpp"
#include "ck_tile/builder/testing/conv/reference.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "testing_utils.hpp"

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;
namespace cku = ck_tile::builder::test_utils;

using ck_tile::test::MatchesReference;

constexpr auto SIGNATURE =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::FORWARD,
                       .data_type              = ckb::DataType::FP16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NHWGK}}};

constexpr auto TILE_ALGORITHM = cku::ConvAlgorithm_Tile_GroupedConvolutionKernel{}
                                    .with_tile_specializations(ckb::TileConvSpecialization::DEFAULT)
                                    .with_tile_thread_block(cku::TileThreadBlock_64x64x64)
                                    .with_tile_block_gemm(cku::TileBlockGemmDesc_16x16_v3_intrawave)
                                    .with_tile_transfer(cku::TileTransfer_4x4x4)
                                    .with_tile_optimizations(ckt::TileOptimizations{
                                        .num_groups_to_merge = 1,
                                        .split_image         = false,
                                        .explicit_gemm       = false,
                                    });

using Builder   = ckb::ConvBuilder<SIGNATURE, TILE_ALGORITHM>;
using TileConv  = Builder::Instance;
using Reference = ckb::ConvBuilder<SIGNATURE, ckt::ConvAlgorithm_Reference{}>::Instance;

TEST(Fwd2DFp16_TileV3_NHWGC, Create)
{
    const auto expected_type_string = "grouped_convolution_forward";
    cku::run_ck_tile_test<Builder>({
        expected_type_string,
        "fp16",
        "NHWGC_GKYXC_NHWGK",
        "64x64x64",
        "2x2",
        "16x16x16",
        //    "4x4x4", // TODO: Enable this check
        "Default",
        "Intrawave",
        "CShuffleEpilogue",
        "pipeline_AgBgCrCompV3",
        "DoubleSmemBuffer_0",
        "NumWaveGroups_1",
        "MergedGroups_1",
        "SplitImage_0",
        "ExplicitGemm_0",
    });
}

TEST(Fwd2DFp16_TileV3_NHWGC, EndToEnd)
{
    if(!ck_tile::get_device_name().starts_with("gfx9"))
    {
        GTEST_SKIP() << "unsupported architecture";
    }

    ckt::Args<SIGNATURE> args = {
        .lengths =
            {
                .batch_size      = 16,
                .groups          = 1,
                .input_channels  = 32,
                .output_channels = 48,
                .image =
                    {
                        .width  = 56,
                        .height = 64,
                    },
                .filter =
                    {
                        .width  = 3,
                        .height = 5,
                    },
            },
        .filter_strides     = {.width = 1, .height = 1},
        .filter_dilation    = {.width = 1, .height = 1},
        .input_left_pad     = {.width = 0, .height = 0},
        .input_right_pad    = {.width = 0, .height = 0},
        .a_elementwise_op   = {},
        .b_elementwise_op   = {},
        .cde_elementwise_op = {},
    };

    auto inputs    = ckt::alloc_inputs(args);
    auto outputs   = ckt::alloc_outputs(args);
    auto reference = ckt::alloc_outputs(args);

    ckt::init_inputs(args, inputs.get());

    auto tile_conv = TileConv{};
    EXPECT_THAT(ckt::run(tile_conv, args, inputs.get(), outputs.get()),
                ck_tile::test::SuccessfulRun());

    auto ref_conv = Reference{};
    EXPECT_THAT(ckt::run(ref_conv, args, inputs.get(), reference.get()),
                ck_tile::test::SuccessfulRun());

    EXPECT_THAT(outputs.get(), MatchesReference(args, reference.get()));
}

TEST(Fwd2DFp16_TileV3_NHWGC, EndToEnd_ExplicitStrides)
{
    if(!ck_tile::get_device_name().starts_with("gfx9"))
    {
        GTEST_SKIP() << "unsupported architecture";
    }

    ckt::Args<SIGNATURE> args = {
        .lengths =
            {
                .batch_size      = 16,
                .groups          = 1,
                .input_channels  = 32,
                .output_channels = 48,
                .image =
                    {
                        .width  = 56,
                        .height = 64,
                    },
                .filter =
                    {
                        .width  = 3,
                        .height = 5,
                    },
            },
        .filter_strides     = {.width = 1, .height = 1},
        .filter_dilation    = {.width = 1, .height = 1},
        .input_left_pad     = {.width = 0, .height = 0},
        .input_right_pad    = {.width = 0, .height = 0},
        .a_elementwise_op   = {},
        .b_elementwise_op   = {},
        .cde_elementwise_op = {},
    };

    // Extract the strides that PackedOrderedLayout would auto-compute,
    // then set them as explicit values to exercise the has_value() path.
    args.input_strides  = args.make_input_descriptor().get_strides();
    args.weight_strides = args.make_weight_descriptor().get_strides();
    args.output_strides = args.make_output_descriptor().get_strides();

    auto inputs    = ckt::alloc_inputs(args);
    auto outputs   = ckt::alloc_outputs(args);
    auto reference = ckt::alloc_outputs(args);

    ckt::init_inputs(args, inputs.get());

    auto tile_conv = TileConv{};
    EXPECT_THAT(ckt::run(tile_conv, args, inputs.get(), outputs.get()),
                ck_tile::test::SuccessfulRun());

    auto ref_conv = Reference{};
    EXPECT_THAT(ckt::run(ref_conv, args, inputs.get(), reference.get()),
                ck_tile::test::SuccessfulRun());

    EXPECT_THAT(outputs.get(), MatchesReference(args, reference.get()));
}

TEST(Fwd2DFp16_TileV3_NHWGC, EndToEnd_ExplicitStrides_Stride2)
{
    if(!ck_tile::get_device_name().starts_with("gfx9"))
    {
        GTEST_SKIP() << "unsupported architecture";
    }

    ckt::Args<SIGNATURE> args = {
        .lengths =
            {
                .batch_size      = 16,
                .groups          = 1,
                .input_channels  = 32,
                .output_channels = 48,
                .image =
                    {
                        .width  = 56,
                        .height = 64,
                    },
                .filter =
                    {
                        .width  = 3,
                        .height = 5,
                    },
            },
        .filter_strides     = {.width = 2, .height = 2},
        .filter_dilation    = {.width = 1, .height = 1},
        .input_left_pad     = {.width = 0, .height = 0},
        .input_right_pad    = {.width = 0, .height = 0},
        .a_elementwise_op   = {},
        .b_elementwise_op   = {},
        .cde_elementwise_op = {},
    };

    args.input_strides  = args.make_input_descriptor().get_strides();
    args.weight_strides = args.make_weight_descriptor().get_strides();
    args.output_strides = args.make_output_descriptor().get_strides();

    auto inputs    = ckt::alloc_inputs(args);
    auto outputs   = ckt::alloc_outputs(args);
    auto reference = ckt::alloc_outputs(args);

    ckt::init_inputs(args, inputs.get());

    auto tile_conv = TileConv{};
    EXPECT_THAT(ckt::run(tile_conv, args, inputs.get(), outputs.get()),
                ck_tile::test::SuccessfulRun());

    auto ref_conv = Reference{};
    EXPECT_THAT(ckt::run(ref_conv, args, inputs.get(), reference.get()),
                ck_tile::test::SuccessfulRun());

    EXPECT_THAT(outputs.get(), MatchesReference(args, reference.get()));
}
