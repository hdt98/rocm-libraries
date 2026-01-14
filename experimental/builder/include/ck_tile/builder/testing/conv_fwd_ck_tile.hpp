// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/conv_fwd.hpp"
#include "ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

/// This file contains the implementation details for invoking/testing
/// grouped forward convolution operations using CK Tile kernels.
/// The main item is the `run()` function, which is used to invoke the
/// CK Tile grouped convolution forward kernel instances.

namespace ck_tile::builder::test {

template <typename Conv, auto SIGNATURE>
concept TileConvInstance = requires {
    typename Conv::CDElementwise;
    typename Conv::GroupedConvFwdKernelArgsSpecialized;
    {
        Conv::MakeKernelArgs(
            std::declval<ck_tile::GroupedConvFwdHostArgs<typename Conv::CDElementwise>>())
    } -> std::convertible_to<typename Conv::GroupedConvFwdKernelArgsSpecialized>;
};

/// @brief `run()` specialization for forward convolution and CK Tile.
///
/// @tparam SIGNATURE Forward convolution signature.
/// @throws std::runtime_error if the arguments weren't actually valid for the
/// operation. This should be caught and reported by the testing framework.
///
/// @see run()
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE>
void run(TileConvInstance<SIGNATURE> auto& conv,
         const Args<SIGNATURE>& args,
         const Inputs<SIGNATURE>& inputs,
         const Outputs<SIGNATURE>& outputs)
{
    constexpr auto spatial_dim = SIGNATURE.spatial_dim;

    // For now, CK Tile EndToEnd only supports packed tensors.
    // (Explicit custom strides are added as API in Args, but tile kernels
    // are not yet wired for that in the testing framework.)
    if(!args.make_input_descriptor().is_packed())
        throw std::runtime_error("TODO: Support non-packed input tensor in CK Tile runner");
    if(!args.make_weight_descriptor().is_packed())
        throw std::runtime_error("TODO: Support non-packed weight tensor in CK Tile runner");
    if(!args.make_output_descriptor().is_packed())
        throw std::runtime_error("TODO: Support non-packed output tensor in CK Tile runner");

    const auto problem = args.make_conv_problem();

    const std::vector<ck_tile::long_index_t> input_spatial(problem.input_spatial.begin(),
                                                           problem.input_spatial.end());
    const std::vector<ck_tile::long_index_t> filter_spatial(problem.filter_spatial.begin(),
                                                            problem.filter_spatial.end());
    const std::vector<ck_tile::long_index_t> conv_strides(problem.conv_strides.begin(),
                                                          problem.conv_strides.end());
    const std::vector<ck_tile::long_index_t> conv_dilations(problem.conv_dilations.begin(),
                                                            problem.conv_dilations.end());
    const std::vector<ck_tile::long_index_t> left_pads(problem.left_pads.begin(),
                                                       problem.left_pads.end());
    const std::vector<ck_tile::long_index_t> right_pads(problem.right_pads.begin(),
                                                        problem.right_pads.end());

    // CK Tile host args are built around ck_tile::conv::ConvParam.
    ck_tile::conv::ConvParam conv_param(static_cast<ck_tile::long_index_t>(spatial_dim),
                                        problem.G,
                                        problem.N,
                                        problem.K,
                                        problem.C,
                                        filter_spatial,
                                        input_spatial,
                                        conv_strides,
                                        conv_dilations,
                                        left_pads,
                                        right_pads);

    using Kernel        = std::remove_cvref_t<decltype(conv)>;
    using CDElementwise = typename Kernel::CDElementwise;

    ck_tile::GroupedConvFwdHostArgs<CDElementwise> host_args(conv_param,
                                                             inputs.input,
                                                             inputs.weight,
                                                             {},
                                                             outputs.output,
                                                             /*k_batch=*/1,
                                                             CDElementwise{});

    auto kargs = Kernel::MakeKernelArgs(host_args);

    if(!Kernel::IsSupportedArgument(kargs))
    {
        throw std::runtime_error("invalid argument");
    }

    const dim3 grids  = Kernel::GridSize(kargs);
    const dim3 blocks = Kernel::BlockSize();

    (void)ck_tile::launch_kernel(ck_tile::stream_config{},
                                 ck_tile::make_kernel(Kernel{}, grids, blocks, 0, kargs));
}

} // namespace ck_tile::builder::test
